import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from step3_data_preprocessing import TimeSeriesDataset, RevIN
from step4_patchtst_model import PatchTST

print("=" * 60)
print("【步驟 5/8】模型訓練 - 使用真實數據")
print("=" * 60)

print("\n訓練策略:")
print("- Huber Loss (魯棒回歸)")
print("- Cosine Annealing 學習率調度")
print("- 早停 (監控 Directional Accuracy)")
print("-" * 60)

# 載入配置
with open('data_config.pkl', 'rb') as f:
    data_config = pickle.load(f)

SEQ_LEN = data_config['seq_len']
PRED_LEN = data_config['pred_len']
PATCH_LEN = data_config['patch_len']
STRIDE = data_config['stride']
BATCH_SIZE = data_config['batch_size']
N_FEATURES = data_config['n_features']
feature_cols = data_config['feature_cols']

print(f"\n配置:")
print(f"  序列長度: {SEQ_LEN} 天")
print(f"  特徵數: {N_FEATURES}")

# 載入數據
df = pd.read_csv('../data/xiaomi_real.csv')
df['date'] = pd.to_datetime(df['date'])

# 重新計算特徵 (與step3一致)
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

df_clean = df[feature_cols + ['target_return_5d', 'target_direction', 'date']].dropna()

# 劃分數據
train_ratio = 0.7
val_ratio = 0.15
n_total = len(df_clean)
n_train = int(n_total * train_ratio)

train_df = df_clean.iloc[:n_train].copy()
val_df = df_clean.iloc[n_train:n_train+int(n_total*val_ratio)].copy()
test_df = df_clean.iloc[n_train+int(n_total*val_ratio):].copy()

# 創建數據集
train_dataset = TimeSeriesDataset(train_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
val_dataset = TimeSeriesDataset(val_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n數據集:")
print(f"  訓練樣本: {len(train_dataset)}")
print(f"  驗證樣本: {len(val_dataset)}")

# 訓練配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n設備: {device}")

model_config = {
    'n_features': N_FEATURES,
    'seq_len': SEQ_LEN,
    'pred_len': PRED_LEN,
    'patch_len': PATCH_LEN,
    'stride': STRIDE,
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 2,
    'd_ff': 128,
    'dropout': 0.1,
    'head_type': 'regression',
    'use_revin': True
}

model = PatchTST(**model_config).to(device)
criterion = nn.HuberLoss(delta=0.1)
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

epochs = 100
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

print(f"\n訓練配置:")
print(f"  訓練輪數: {epochs}")
print(f"  學習率: {learning_rate}")
print(f"  早停耐心: 15 epochs")

# 訓練循環
early_stop_patience = 15
best_val_acc = 0
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

train_losses = []
val_losses = []
val_accs = []

print(f"\n開始訓練...")
print("-" * 60)

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
    else:
        patience_counter += 1
    
    if patience_counter >= early_stop_patience:
        print(f"\n✓ 早停觸發 (best acc: {best_val_acc:.2%})")
        break

# 載入最佳模型
if best_model_state:
    model.load_state_dict(best_model_state)

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': model_config,
    'best_val_acc': best_val_acc,
    'best_val_loss': best_val_loss
}, 'patchtst_model.pth')

print(f"\n✓ 模型已保存: patchtst_model.pth")

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(train_losses, label='Train Loss', alpha=0.8)
ax1.plot(val_losses, label='Val Loss', alpha=0.8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Huber Loss')
ax1.set_title('Training History')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(val_accs, label='Val Direction Accuracy', color='green', alpha=0.8)
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
ax2.axhline(y=best_val_acc, color='blue', linestyle='--', alpha=0.5, label=f'Best: {best_val_acc:.2%}')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Direction Accuracy')
ax2.set_ylim(0.3, 0.7)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("✓ 訓練歷史已保存: training_history.png")

print(f"\n訓練摘要:")
print(f"  總輪數: {len(train_losses)}")
print(f"  最佳驗證準確率: {best_val_acc:.2%}")
print(f"  最佳驗證損失: {best_val_loss:.6f}")

print("\n" + "=" * 60)
print("【關鍵節點 5/8 完成】模型訓練完成")
print("=" * 60)
