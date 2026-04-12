import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime

from step3_data_preprocessing import TimeSeriesDataset, RevIN
from step4_patchtst_model import PatchTST

print("=" * 70)
print("【最終模型訓練】使用優化後的超參數")
print("=" * 70)

# 載入優化後的配置
with open('best_hyperparameters.json', 'r') as f:
    best_config = json.load(f)

print("\n【優化後的超參數】")
print("-" * 50)
print(f"d_model: {best_config['d_model']}")
print(f"n_heads: {best_config['n_heads']}")
print(f"n_layers: {best_config['n_layers']}")
print(f"dropout: {best_config['dropout']}")
print(f"patch_len: {best_config['patch_len']}")
print(f"stride: {best_config['stride']}")
print(f"d_ff: {best_config['d_ff']}")
print(f"learning_rate: {best_config['learning_rate']}")

# 載入數據配置
with open('data_config.pkl', 'rb') as f:
    data_config = pickle.load(f)

SEQ_LEN = data_config['seq_len']  # 20
PRED_LEN = data_config['pred_len']  # 5
N_FEATURES = data_config['n_features']  # 17
feature_cols = data_config['feature_cols']
BATCH_SIZE = 64

print(f"\n【數據配置】")
print("-" * 50)
print(f"序列長度: {SEQ_LEN} 天")
print(f"預測長度: {PRED_LEN} 天")
print(f"特徵數: {N_FEATURES}")

# ============================================
# 載入並處理數據
# ============================================
print("\n【載入真實數據】")
print("-" * 50)

df = pd.read_csv('../data/xiaomi_real.csv')
df['date'] = pd.to_datetime(df['date'])

# 特徵工程
def compute_features(df):
    for span in [5, 10, 20]:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        df[f'ema_{span}_ratio'] = df['close'] / df[f'ema_{span}'] - 1
    
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    df['atr_ratio'] = df['atr_14'] / df['close']
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
    df['volatility_20d'] = df['return_1d'].rolling(window=20).std()
    df['target_return_5d'] = df['close'].pct_change(5).shift(-5)
    df['target_direction'] = (df['target_return_5d'] > 0).astype(int)
    
    return df

df = compute_features(df)
df_clean = df[feature_cols + ['target_return_5d', 'target_direction', 'date']].dropna()

print(f"✓ 數據時間範圍: {df_clean['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df_clean['date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"✓ 清洗後樣本: {len(df_clean)}")

# 數據劃分
train_ratio = 0.7
val_ratio = 0.15
n_total = len(df_clean)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

train_df = df_clean.iloc[:n_train].copy()
val_df = df_clean.iloc[n_train:n_train+n_val].copy()
test_df = df_clean.iloc[n_train+n_val:].copy()

print(f"\n【數據集劃分】")
print("-" * 50)
print(f"訓練集: {len(train_df)} 樣本 ({train_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {train_df['date'].iloc[-1].strftime('%Y-%m-%d')})")
print(f"驗證集: {len(val_df)} 樣本 ({val_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {val_df['date'].iloc[-1].strftime('%Y-%m-%d')})")
print(f"測試集: {len(test_df)} 樣本 ({test_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {test_df['date'].iloc[-1].strftime('%Y-%m-%d')})")

# ============================================
# 創建數據集
# ============================================
print("\n【創建數據集】")
print("-" * 50)

PATCH_LEN = best_config['patch_len']
STRIDE = best_config['stride']

train_dataset = TimeSeriesDataset(train_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
val_dataset = TimeSeriesDataset(val_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
test_dataset = TimeSeriesDataset(test_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)

print(f"✓ Patch 長度: {PATCH_LEN}")
print(f"✓ Stride: {STRIDE}")
print(f"✓ 訓練樣本: {len(train_dataset)}")
print(f"✓ 驗證樣本: {len(val_dataset)}")
print(f"✓ 測試樣本: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================
# 創建模型
# ============================================
print("\n【模型配置】")
print("-" * 50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"設備: {device}")

model_config = {
    'n_features': N_FEATURES,
    'seq_len': SEQ_LEN,
    'pred_len': PRED_LEN,
    'patch_len': PATCH_LEN,
    'stride': STRIDE,
    'd_model': best_config['d_model'],
    'n_heads': best_config['n_heads'],
    'n_layers': best_config['n_layers'],
    'd_ff': best_config['d_ff'],
    'dropout': best_config['dropout'],
    'head_type': 'regression',
    'use_revin': True
}

model = PatchTST(**model_config).to(device)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型參數量: {count_params(model):,}")

# ============================================
# 訓練
# ============================================
print("\n【開始訓練】")
print("-" * 50)

criterion = nn.HuberLoss(delta=0.1)
optimizer = optim.Adam(model.parameters(), lr=best_config['learning_rate'], weight_decay=1e-5)

epochs = 100
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

early_stop_patience = 15
best_val_acc = 0
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None
best_epoch = 0

train_losses = []
val_losses = []
val_accs = []

print(f"訓練輪數: {epochs}")
print(f"早停耐心: {early_stop_patience} epochs")
print()

for epoch in range(epochs):
    # 訓練
    model.train()
    train_loss = 0
    
    for batch in train_loader:
        x = batch['x'].to(device)
        y_return = batch['y_return'].to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y_return)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # 驗證
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(device)
            y_return = batch['y_return'].to(device)
            y_direction = batch['y_direction'].to(device)
            
            pred = model(x)
            loss = criterion(pred, y_return)
            val_loss += loss.item()
            
            pred_direction = (pred.squeeze() > 0).long()
            val_correct += (pred_direction == y_direction).sum().item()
            val_total += y_direction.size(0)
    
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    
    scheduler.step()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f} (Acc: {val_acc:.2%}) | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # 早停
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        best_epoch = epoch + 1
    else:
        patience_counter += 1
    
    if patience_counter >= early_stop_patience:
        print(f"\n✓ 早停觸發於 Epoch {epoch + 1}")
        break

# 載入最佳模型
model.load_state_dict(best_model_state)

print(f"\n訓練完成!")
print(f"  最佳驗證準確率: {best_val_acc:.2%} (Epoch {best_epoch})")
print(f"  最佳驗證損失: {best_val_loss:.6f}")

# ============================================
# 測試集評估
# ============================================
print("\n【測試集評估】")
print("-" * 50)

model.eval()
all_preds = []
all_targets = []
all_directions = []

with torch.no_grad():
    for batch in test_loader:
        x = batch['x'].to(device)
        y_return = batch['y_return'].to(device)
        y_direction = batch['y_direction'].to(device)
        
        pred = model(x)
        
        all_preds.extend(pred.squeeze().cpu().numpy())
        all_targets.extend(y_return.squeeze().cpu().numpy())
        all_directions.extend(y_direction.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_directions = np.array(all_directions)
pred_directions = (all_preds > 0).astype(int)

# 計算指標
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix

direction_accuracy = accuracy_score(all_directions, pred_directions)
mse = mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
rmse = np.sqrt(mse)

print(f"✓ Directional Accuracy: {direction_accuracy:.2%}")
print(f"✓ MSE: {mse:.6f}")
print(f"✓ MAE: {mae:.6f}")
print(f"✓ RMSE: {rmse:.6f}")

# 混淆矩陣
cm = confusion_matrix(all_directions, pred_directions)
print(f"\n混淆矩陣:")
print(f"              Predicted")
print(f"           Down   Up")
print(f"Actual Down  {cm[0,0]:3d}   {cm[0,1]:3d}")
print(f"       Up    {cm[1,0]:3d}   {cm[1,1]:3d}")

# ============================================
# 回測
# ============================================
print("\n【簡化回測】")
print("-" * 50)

returns = []
for i in range(len(all_preds)):
    if pred_directions[i] == 1:  # 預測上漲
        actual_return = all_targets[i]
    else:  # 預測下跌，不持有
        actual_return = 0
    returns.append(actual_return)

returns = np.array(returns)
cumulative_returns = np.cumsum(returns)

total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(52) if np.std(returns) > 0 else 0
max_drawdown = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns)) if len(cumulative_returns) > 0 else 0
win_rate = np.sum((returns > 0)) / len(returns) if len(returns) > 0 else 0

print(f"✓ 總收益率: {total_return:.2%}")
print(f"✓ Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"✓ Max Drawdown: {max_drawdown:.2%}")
print(f"✓ Win Rate: {win_rate:.2%}")

# ============================================
# 保存模型
# ============================================
print("\n【保存模型】")
print("-" * 50)

torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': model_config,
    'best_val_acc': best_val_acc,
    'best_val_loss': best_val_loss,
    'test_acc': direction_accuracy,
    'test_mse': mse,
    'hyperparameters': best_config,
    'training_info': {
        'best_epoch': best_epoch,
        'total_epochs': len(train_losses),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }
}, 'patchtst_final_model.pth')

print("✓ 最終模型已保存: patchtst_final_model.pth")

# ============================================
# 可視化
# ============================================
print("\n【生成可視化】")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 訓練歷史
ax1 = axes[0, 0]
ax1.plot(train_losses, label='Train Loss', alpha=0.8)
ax1.plot(val_losses, label='Val Loss', alpha=0.8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Huber Loss')
ax1.set_title('Training History (Optimized Model)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 驗證準確率
ax2 = axes[0, 1]
ax2.plot(val_accs, label='Val Direction Accuracy', color='green', alpha=0.8)
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
ax2.axhline(y=best_val_acc, color='blue', linestyle='--', alpha=0.5, label=f'Best: {best_val_acc:.1%}')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Direction Accuracy')
ax2.set_ylim(0.3, 0.7)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 預測 vs 實際
ax3 = axes[1, 0]
ax3.scatter(all_targets, all_preds, alpha=0.5, edgecolors='black', linewidth=0.5)
ax3.plot([-0.3, 0.3], [-0.3, 0.3], 'r--', label='Perfect')
ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax3.set_xlabel('Actual Return')
ax3.set_ylabel('Predicted Return')
ax3.set_title(f'Predicted vs Actual (Test Acc: {direction_accuracy:.1%})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 回測累計收益
ax4 = axes[1, 1]
ax4.plot(cumulative_returns * 100, label='Strategy', linewidth=2)
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.fill_between(range(len(cumulative_returns)), cumulative_returns * 100, 0, 
                  where=(cumulative_returns > 0), alpha=0.3, color='green')
ax4.fill_between(range(len(cumulative_returns)), cumulative_returns * 100, 0, 
                  where=(cumulative_returns < 0), alpha=0.3, color='red')
ax4.set_xlabel('Trade Number')
ax4.set_ylabel('Cumulative Return (%)')
ax4.set_title(f'Backtest (Sharpe: {sharpe_ratio:.2f})')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_model_results.png', dpi=150, bbox_inches='tight')
print("✓ 結果圖表已保存: final_model_results.png")

# ============================================
# 總結
# ============================================
print("\n" + "=" * 70)
print("【最終模型訓練完成】")
print("=" * 70)
print(f"\n模型配置:")
print(f"  d_model: {best_config['d_model']}")
print(f"  n_heads: {best_config['n_heads']}")
print(f"  n_layers: {best_config['n_layers']}")
print(f"  patch_len: {best_config['patch_len']}")
print(f"  dropout: {best_config['dropout']}")
print(f"\n性能指標:")
print(f"  驗證集準確率: {best_val_acc:.2%}")
print(f"  測試集準確率: {direction_accuracy:.2%}")
print(f"  測試集 MSE: {mse:.6f}")
print(f"  回測收益率: {total_return:.2%}")
print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"\n文件保存:")
print(f"  ✓ patchtst_final_model.pth")
print(f"  ✓ final_model_results.png")
print("=" * 70)
