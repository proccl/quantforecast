"""
訓練器模塊
統一處理模型訓練、驗證、早停、學習率調度
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
import copy

from src.config import TrainingConfig
from src.models.patchtst import PatchTST

logger = logging.getLogger(__name__)


class Trainer:
    """統一訓練邏輯"""
    
    def __init__(
        self,
        model: PatchTST,
        config: TrainingConfig,
        device: torch.device
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 根據模型類型選擇損失函數
        self.head_type = getattr(model, 'head_type', 'regression')
        if self.head_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.HuberLoss(delta=0.1)
        
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        self.best_val_metric = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state: Optional[dict] = None
        
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def _build_optimizer(self) -> optim.Optimizer:
        """構建優化器"""
        if self.config.optimizer == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"不支持的優化器: {self.config.optimizer}")
    
    def _build_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """構建學習率調度器"""
        if self.config.scheduler == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            )
        return None
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        執行訓練
        
        Returns:
            訓練歷史記錄
        """
        logger.info(f"開始訓練，設備: {self.device}")
        logger.info(f"訓練輪數: {self.config.epochs}, 早停耐心: {self.config.early_stopping_patience}")
        
        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch [{epoch+1:3d}/{self.config.epochs}] "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} (Acc: {val_acc:.2%}) | LR: {current_lr:.6f}"
                )
            
            # 早停檢查
            should_stop = self._check_early_stopping(val_acc, val_loss)
            if should_stop:
                logger.info(f"早停觸發 (best acc: {self.best_val_metric:.2%})")
                break
            
            # 學習率調度
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
        
        # 載入最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("✓ 已載入最佳模型狀態")
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """單個訓練 epoch（支持樣本加權）"""
        self.model.train()
        total_loss = 0.0
        total_weight = 0.0
        n_batches = 0
        
        for batch in train_loader:
            x = batch['x'].to(self.device)
            y_return = batch['y_return'].to(self.device)
            y_direction = batch['y_direction'].to(self.device)
            # 獲取樣本權重（如果沒有，默認為1）
            weights = batch.get('weight', torch.ones_like(y_return)).to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            
            # 根據模型類型計算損失
            if self.head_type == 'classification':
                # 分類：使用 CrossEntropyLoss，目標是方向類別 (0/1)
                loss = self.criterion(pred, y_direction.squeeze().long())
                weighted_loss = loss  # 分類暫不支持樣本加權
            else:
                # 回歸：使用 HuberLoss，目標是收益率
                # 使用 reduction='none' 來獲取逐樣本損失
                loss_per_sample = nn.functional.huber_loss(pred, y_return, delta=0.1, reduction='none')
                weighted_loss = (loss_per_sample * weights).mean()
            
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += weighted_loss.item() * weights.sum().item()
            total_weight += weights.sum().item()
            n_batches += 1
        
        return total_loss / max(total_weight, 1e-8)
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """單個驗證 epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(self.device)
                y_return = batch['y_return'].to(self.device)
                y_direction = batch['y_direction'].to(self.device)
                
                pred = self.model(x)
                
                # 根據模型類型計算損失和準確率
                if self.head_type == 'classification':
                    # 分類：使用 CrossEntropyLoss
                    loss = self.criterion(pred, y_direction.squeeze().long())
                    # 預測類別
                    pred_class = torch.argmax(pred, dim=1)
                    correct += (pred_class == y_direction.squeeze()).sum().item()
                else:
                    # 回歸：使用 HuberLoss
                    loss = self.criterion(pred, y_return)
                    pred_direction = (pred.squeeze() > 0).long()
                    correct += (pred_direction == y_direction.squeeze()).sum().item()
                
                total_loss += loss.item()
                total += y_direction.numel()
                n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)
        
        return avg_loss, accuracy
    
    def _check_early_stopping(self, val_acc: float, val_loss: float) -> bool:
        """檢查是否觸發早停"""
        if not self.config.early_stopping:
            return False
        
        improved = val_acc > self.best_val_metric + self.config.early_stopping_min_delta
        
        if improved:
            self.best_val_metric = val_acc
            self.best_val_loss = val_loss
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.config.early_stopping_patience
    
    def save_checkpoint(self, path: str, model_config: Optional[Dict] = None) -> None:
        """保存模型檢查點"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_metric,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        if model_config is not None:
            checkpoint['model_config'] = model_config
        torch.save(checkpoint, path)
        logger.info(f"✓ 模型已保存: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """載入模型檢查點"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_metric = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', {})
        logger.info(f"✓ 模型已載入: {path}")
