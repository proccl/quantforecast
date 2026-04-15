#!/usr/bin/env python3
"""
使用舊版超參數重新訓練 PatchTST 模型
並生成訓練過程可視化圖表
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from datetime import datetime
import json
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader as TorchDataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data.loader import DataLoader as DataLoaderClass
from src.data.features import FeatureEngineer
from src.data.preprocessor import Preprocessor, TimeSeriesDataset
from src.models.patchtst import PatchTST
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator

# 舊版最佳超參數 (main branch)
OLD_BEST_PARAMS = {
    "d_model": 64,
    "n_heads": 8,
    "n_layers": 3,
    "dropout": 0.2,
    "lr": 0.0002051338263087451,
    "patch_len": 5,
    "batch_size": 32,
    "seq_len": 20,
    "pred_len": 5
}

def train_with_logging():
    """訓練模型並記錄詳細過程"""
    
    print("=" * 70)
    print("【使用舊版超參數重新訓練 PatchTST】")
    print("=" * 70)
    print(f"\n超參數配置:")
    for k, v in OLD_BEST_PARAMS.items():
        print(f"  {k}: {v}")
    print("=" * 70)
    
    # 配置
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n設備: {device}")
    
    # 加載數據
    print("\n【1/4】加載數據...")
    data_loader = DataLoaderClass(config.data)
    df = data_loader.load()
    print(f"  原始數據: {len(df)} 條")
    
    # 特徵工程
    print("\n【2/4】特徵工程...")
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_features(df)
    df_clean = df_features.dropna()
    print(f"  清洗後: {len(df_clean)} 條")
    print(f"  特徵數: {len([c for c in df_clean.columns if c not in ['date', 'target_return_5d', 'target_direction']])}")
    
    # 預處理
    print("\n【3/4】數據分割...")
    preprocessor = Preprocessor(config.data)
    train_df, val_df, test_df = preprocessor.split(df_clean)
    
    feature_cols = [c for c in df_clean.columns if c not in ['date', 'target_return_5d', 'target_direction']]
    train_ds, val_ds, test_ds = preprocessor.create_datasets(train_df, val_df, test_df, feature_cols)
    
    print(f"  訓練集: {len(train_ds)} 樣本")
    print(f"  驗證集: {len(val_ds)} 樣本")
    print(f"  測試集: {len(test_ds)} 樣本")
    
    # 創建模型
    print("\n【4/4】創建模型...")
    model = PatchTST(
        n_features=len(feature_cols),
        seq_len=OLD_BEST_PARAMS['seq_len'],
        pred_len=OLD_BEST_PARAMS['pred_len'],
        patch_len=OLD_BEST_PARAMS['patch_len'],
        stride=OLD_BEST_PARAMS['patch_len'] // 2,
        d_model=OLD_BEST_PARAMS['d_model'],
        n_heads=OLD_BEST_PARAMS['n_heads'],
        n_layers=OLD_BEST_PARAMS['n_layers'],
        d_ff=OLD_BEST_PARAMS['d_model'] * 2,
        dropout=OLD_BEST_PARAMS['dropout'],
        head_type='classification',  # 使用分類頭
        use_revin=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  總參數: {total_params:,}")
    print(f"  可訓練參數: {trainable_params:,}")
    
    # 數據加載器
    batch_size = OLD_BEST_PARAMS['batch_size']
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size
    )
    
    # 訓練配置
    train_cfg = config.training
    train_cfg.lr = OLD_BEST_PARAMS['lr']
    train_cfg.epochs = 100
    train_cfg.early_stopping_patience = 15
    
    # 自定義訓練器以記錄詳細指標
    print("\n" + "=" * 70)
    print("【開始訓練】")
    print("=" * 70)
    
    trainer = Trainer(model, train_cfg, device)
    
    # 手動訓練循環以記錄更多指標
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_direction_acc': [],
        'val_direction_acc': [],
        'learning_rate': [],
        'epoch': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(train_cfg.epochs):
        # 訓練階段
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            x = batch['x'].to(device)
            y_direction = batch['y_direction'].to(device)
            
            trainer.optimizer.zero_grad()
            
            pred_direction = model(x)
            
            # 方向損失
            loss = nn.CrossEntropyLoss()(pred_direction, y_direction.squeeze().long())
            
            loss.backward()
            trainer.optimizer.step()
            
            train_losses.append(loss.item())
            
            # 計算方向準確率
            pred_dir_class = torch.argmax(pred_direction, dim=1)
            train_correct += (pred_dir_class == y_direction).sum().item()
            train_total += y_direction.size(0)
        
        # 驗證階段
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y_direction = batch['y_direction'].to(device)
                
                pred_direction = model(x)
                
                loss = nn.CrossEntropyLoss()(pred_direction, y_direction.squeeze().long())
                
                val_losses.append(loss.item())
                
                pred_dir_class = torch.argmax(pred_direction, dim=1)
                val_correct += (pred_dir_class == y_direction).sum().item()
                val_total += y_direction.size(0)
        
        # 記錄指標
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_direction_acc'].append(train_acc)
        history['val_direction_acc'].append(val_acc)
        history['learning_rate'].append(trainer.optimizer.param_groups[0]['lr'])
        
        # 打印進度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{train_cfg.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2%}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.early_stopping_patience:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                break
    
    # 載入最佳模型
    model.load_state_dict(best_model_state)
    
    # 測試集評估
    print("\n" + "=" * 70)
    print("【測試集評估】")
    print("=" * 70)
    
    model.eval()
    test_correct = 0
    test_total = 0
    test_losses = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            y_direction = batch['y_direction'].to(device)
            
            pred_direction = model(x)
            loss = nn.CrossEntropyLoss()(pred_direction, y_direction.squeeze().long())
            test_losses.append(loss.item())
            
            pred_dir_class = torch.argmax(pred_direction, dim=1)
            test_correct += (pred_dir_class == y_direction.squeeze()).sum().item()
            test_total += y_direction.size(0)
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    test_loss = np.mean(test_losses)
    
    print(f"  測試集方向準確率: {test_acc:.2%}")
    print(f"  測試集 Loss: {test_loss:.4f}")
    
    test_results = {
        'directional_accuracy': test_acc,
        'loss': test_loss
    }
    
    # 保存結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存訓練歷史
    history_path = f"results/training_history_{timestamp}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ 訓練歷史已保存: {history_path}")
    
    # 保存模型
    model_config = {
        'seq_len': OLD_BEST_PARAMS['seq_len'],
        'pred_len': OLD_BEST_PARAMS['pred_len'],
        'd_model': OLD_BEST_PARAMS['d_model'],
        'n_heads': OLD_BEST_PARAMS['n_heads'],
        'n_layers': OLD_BEST_PARAMS['n_layers'],
        'dropout': OLD_BEST_PARAMS['dropout'],
        'patch_len': OLD_BEST_PARAMS['patch_len'],
        'stride': OLD_BEST_PARAMS['patch_len'] // 2,
        'd_ff': OLD_BEST_PARAMS['d_model'] * 2
    }
    
    model_path = f"models/patchtst_old_params_{timestamp}.pth"
    trainer.save_checkpoint(model_path, model_config)
    print(f"✓ 模型已保存: {model_path}")
    
    # 繪製訓練結果圖
    plot_training_results(history, test_results, timestamp)
    
    return history, test_results

def plot_training_results(history, test_results, timestamp):
    """
    繪製訓練結果圖表 (深色主題，與 complete_backtest_results 風格一致)
    """
    print("\n" + "=" * 70)
    print("【生成訓練結果圖表】")
    print("=" * 70)
    
    # 創建圖表
    fig = plt.figure(figsize=(16, 12), facecolor='#1a1a1a')
    
    # 大標題
    training_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"Training Results | {training_time} | Test Acc: {test_results['directional_accuracy']:.1%}",
        color="#ffffff", fontsize=16, fontweight="bold", y=0.995
    )
    
    # 設置深色主題
    plt.rcParams['axes.facecolor'] = '#2d2d2d'
    plt.rcParams['axes.edgecolor'] = '#666666'
    plt.rcParams['axes.labelcolor'] = '#cccccc'
    plt.rcParams['text.color'] = '#cccccc'
    plt.rcParams['xtick.color'] = '#cccccc'
    plt.rcParams['ytick.color'] = '#cccccc'
    plt.rcParams['grid.color'] = '#444444'
    plt.rcParams['figure.facecolor'] = '#1a1a1a'
    
    epochs = history['epoch']
    
    # 1. Loss 曲線
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(epochs, history['train_loss'], color='#4a9eff', linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color='#ff6b6b', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 學習率曲線
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(epochs, history['learning_rate'], color='#51cf66', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Learning Rate', fontsize=11)
    ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 方向準確率曲線
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(epochs, history['train_direction_acc'], color='#4a9eff', linewidth=2, label='Train Acc')
    ax3.plot(epochs, history['val_direction_acc'], color='#ff6b6b', linewidth=2, label='Val Acc')
    ax3.axhline(y=0.5, color='#666666', linestyle='--', alpha=0.7, label='Random (50%)')
    ax3.axhline(y=test_results['directional_accuracy'], color='#51cf66', linestyle='-', 
                linewidth=2, alpha=0.8, label=f"Test Acc ({test_results['directional_accuracy']:.1%})")
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Directional Accuracy', fontsize=11)
    ax3.set_title('Directional Accuracy', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. 學習率和最終指標
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # 計算最終指標
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_acc = history['train_direction_acc'][-1]
    final_val_acc = history['val_direction_acc'][-1]
    best_val_acc = max(history['val_direction_acc'])
    best_epoch = history['val_direction_acc'].index(best_val_acc) + 1
    
    summary_text = f"""Training Summary

Hyperparameters:
  d_model: {OLD_BEST_PARAMS['d_model']}
  n_heads: {OLD_BEST_PARAMS['n_heads']}
  n_layers: {OLD_BEST_PARAMS['n_layers']}
  dropout: {OLD_BEST_PARAMS['dropout']}
  lr: {OLD_BEST_PARAMS['lr']:.6f}
  batch_size: {OLD_BEST_PARAMS['batch_size']}

Final Metrics:
  Train Loss: {final_train_loss:.4f}
  Val Loss: {final_val_loss:.4f}
  Train Acc: {final_train_acc:.2%}
  Val Acc: {final_val_acc:.2%}

Best Validation:
  Epoch: {best_epoch}
  Acc: {best_val_acc:.2%}

Test Set:
  Direction Acc: {test_results['directional_accuracy']:.2%}

Total Epochs: {len(epochs)}
"""
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', color='white',
             family='monospace', bbox=dict(boxstyle='round', facecolor='#2d2d2d', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存圖表
    output_path = f"results/training_results_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"✓ 訓練結果圖表已保存: {output_path}")
    
    return output_path

if __name__ == '__main__':
    train_with_logging()
