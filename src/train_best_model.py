import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from step3_data_preprocessing import TimeSeriesDataset, RevIN
from step4_patchtst_model import PatchTST

print("=" * 70)
print("【最終版本】原配置 + 平衡採樣")
print("=" * 70)

# 使用原配置 (表現最好的)
ORIG_CONFIG = {
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 2,
    'dropout': 0.1,
    'patch_len': 5,
    'learning_rate': 1e-3
}

with open('data_config.pkl', 'rb') as f:
    data_config = pickle.load(f)

SEQ_LEN = data_config['seq_len']
PRED_LEN = data_config['pred_len']
N_FEATURES = data_config['n_features']
feature_cols = data_config['feature_cols']
BATCH_SIZE = 64

PATCH_LEN = ORIG_CONFIG['patch_len']
STRIDE = PATCH_LEN // 2  # 2

print(f"\n【配置】")
print(f"d_model: {ORIG_CONFIG['d_model']}")
print(f"n_heads: {ORIG_CONFIG['n_heads']}")
print(f"n_layers: {ORIG_CONFIG['n_layers']}")
print(f"patch_len: {PATCH_LEN}, stride: {STRIDE}")

# 載入數據
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
    df['volatility_20d'] = df['return_1d'].rolling(window=20).std()
    df['target_return_5d'] = df['close'].pct_change(5).shift(-5)
    df['target_direction'] = (df['target_return_5d'] > 0).astype(int)
    
    return df

df = compute_features(df)
df_clean = df[feature_cols + ['target_return_5d', 'target_direction', 'date']].dropna()

# 劃分
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
test_dataset = TimeSeriesDataset(test_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)

print(f"\n樣本數: 訓練={len(train_dataset)}, 驗證={len(val_dataset)}, 測試={len(test_dataset)}")

# ============================================
# 平衡採樣
# ============================================
train_directions = [train_dataset[i]['y_direction'].item() for i in range(len(train_dataset))]
up_count = sum(train_directions)
down_count = len(train_directions) - up_count

print(f"\n原始分布: 上漲={up_count}({up_count/len(train_dataset):.1%}), 下跌={down_count}({down_count/len(train_dataset):.1%})")

# 計算權重 - 讓少數類有更多採樣機會
class_counts = [down_count, up_count]
class_weights = [len(train_dataset) / (2 * c) for c in class_counts]  # 反向頻率
sample_weights = [class_weights[d] for d in train_directions]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"類別權重: 下跌={class_weights[0]:.3f}, 上漲={class_weights[1]:.3f}")
print(f"✓ 使用平衡採樣")

# ============================================
# 模型
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n設備: {device}")

model_config = {
    'n_features': N_FEATURES,
    'seq_len': SEQ_LEN,
    'pred_len': PRED_LEN,
    'patch_len': PATCH_LEN,
    'stride': STRIDE,
    'd_model': ORIG_CONFIG['d_model'],
    'n_heads': ORIG_CONFIG['n_heads'],
    'n_layers': ORIG_CONFIG['n_layers'],
    'd_ff': ORIG_CONFIG['d_model'] * 2,
    'dropout': ORIG_CONFIG['dropout'],
    'head_type': 'regression',
    'use_revin': True
}

model = PatchTST(**model_config).to(device)

print(f"參數量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================
# 訓練 - 只用 Huber Loss，但監控方向準確率
# ============================================
criterion = nn.HuberLoss(delta=0.1)
optimizer = optim.Adam(model.parameters(), lr=ORIG_CONFIG['learning_rate'], weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

epochs = 100
patience = 15
best_val_acc = 0
best_model_state = None
best_epoch = 0
patience_counter = 0

train_losses = []
val_losses = []
val_accs = []

print(f"\n訓練: {epochs} epochs, patience={patience}")
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
        print(f"Epoch [{epoch+1:3d}/{epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f} (Acc: {val_acc:.2%})")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        best_epoch = epoch + 1
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"\n✓ 早停於 Epoch {epoch + 1}")
        break

model.load_state_dict(best_model_state)

print(f"\n訓練完成! 最佳驗證準確率: {best_val_acc:.2%} (Epoch {best_epoch})")

# ============================================
# 評估
# ============================================
print("\n【評估】")
print("-" * 50)

def evaluate(loader, name):
    model.eval()
    all_preds = []
    all_dirs = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y_dir = batch['y_direction'].to(device)
            pred = model(x)
            all_preds.extend(pred.squeeze().cpu().numpy())
            all_dirs.extend(y_dir.cpu().numpy())
    
    all_preds = np.array(all_preds)
    pred_dirs = (all_preds > 0).astype(int)
    all_dirs = np.array(all_dirs)
    
    from sklearn.metrics import accuracy_score, confusion_matrix
    acc = accuracy_score(all_dirs, pred_dirs)
    cm = confusion_matrix(all_dirs, pred_dirs)
    
    pred_up = np.sum(pred_dirs)
    pred_down = len(pred_dirs) - pred_up
    
    print(f"\n{name}:")
    print(f"  準確率: {acc:.2%}")
    print(f"  預測: 上漲 {pred_up} ({pred_up/len(pred_dirs):.1%}) / 下跌 {pred_down} ({pred_down/len(pred_dirs):.1%})")
    print(f"  混淆矩陣: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    return acc, all_preds, all_dirs

train_acc, _, _ = evaluate(train_loader, "訓練集")
val_acc, _, _ = evaluate(val_loader, "驗證集")
test_acc, test_preds, test_dirs = evaluate(test_loader, "測試集")

# ============================================
# 保存
# ============================================
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': model_config,
    'val_acc': best_val_acc,
    'test_acc': test_acc,
    'training_info': {'best_epoch': best_epoch, 'balanced': True}
}, 'patchtst_best_model.pth')

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.plot(train_losses, label='Train')
ax1.plot(val_losses, label='Val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss History')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(val_accs, label='Val Acc', color='green', linewidth=2)
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
ax2.axhline(y=test_acc, color='blue', linestyle='--', alpha=0.7, label=f'Test: {test_acc:.1%}')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Direction Accuracy')
ax2.set_ylim(0.3, 0.8)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('best_model_results.png', dpi=150)
print("\n✓ 圖表: best_model_results.png")

print("\n" + "=" * 70)
print("【完成】")
print("=" * 70)
print(f"\n測試集準確率: {test_acc:.2%}")
print(f"模型: patchtst_best_model.pth")
print("=" * 70)
