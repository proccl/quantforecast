"""
評估器模塊
統一模型評估邏輯
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict
import logging

from src.models.patchtst import PatchTST

logger = logging.getLogger(__name__)


class Evaluator:
    """統一評估邏輯"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.criterion = torch.nn.HuberLoss(delta=0.1)
    
    def evaluate(self, model: PatchTST, test_loader: DataLoader) -> Dict[str, float]:
        """
        在測試集上評估模型
        
        Returns:
            評估指標字典
        """
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0
        
        all_preds = []
        all_targets = []
        all_directions = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(self.device)
                y_return = batch['y_return'].to(self.device)
                y_direction = batch['y_direction'].to(self.device)
                
                pred = model(x)
                loss = self.criterion(pred, y_return)
                
                total_loss += loss.item()
                
                pred_direction = (pred.squeeze() > 0).long()
                correct += (pred_direction == y_direction.squeeze()).sum().item()
                total += y_direction.numel()
                
                all_preds.append(pred.cpu())
                all_targets.append(y_return.cpu())
                all_directions.append(y_direction.cpu())
                
                n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)
        
        # 計算 MSE 和 MAE
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        mse = torch.nn.functional.mse_loss(all_preds, all_targets).item()
        mae = torch.nn.functional.l1_loss(all_preds, all_targets).item()
        
        results = {
            'directional_accuracy': accuracy,
            'loss': avg_loss,
            'mse': mse,
            'mae': mae
        }
        
        logger.info("測試集評估結果:")
        for key, value in results.items():
            if key == 'directional_accuracy':
                logger.info(f"  {key}: {value:.2%}")
            else:
                logger.info(f"  {key}: {value:.6f}")
        
        return results
