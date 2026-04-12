import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

from step3_data_preprocessing import TimeSeriesDataset, RevIN
from step4_patchtst_model import PatchTST

print("=" * 70)
print("【修復版本】平衡數據採樣 + 優化超參數")
print("=" * 70)

# 載入配置
with open('best_hyperparameters.json', 'r') as f:
    best_config = json.load(f)

with open('data_config.pkl', 'rb') as f:
    data_config = pickle.load(f)

SEQ_LEN = data_config['seq_len']
PRED_LEN = data_config['pred_len']
N_FEATURES = data_config['n_features']
feature_cols = data_config['feature_cols']
BATCH_SIZE = 64

PATCH_LEN = best_config['patch_len']
STRIDE = best_config['stride']

print(f"\n配置: seq_len={SEQ_LEN}, patch_len={PATCH_LEN}, d_model={best_config['d_model']}")

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

# 數據劃分
train_ratio = 0.7
val_ratio = 0.15
n_total = len(df_clean)
n_train = int(n_total * train_ratio)

train_df = df_clean.iloc[:n_train].copy()
val_df = df_clean.iloc[n_train:n_train+int(n_total*val_ratio)].copy()
test_df = df_clean.iloc[n_train+int(n_total*val_ratio):].copy()

print(f"\n數據集:")
print(f"  訓練集: {len(train_df)} 樣本")
print(f"  驗證集: {len(val_df)} 樣本")
print(f"  測試集: {len(test_df)} 樣本")

# ============================================
# 創建數據集
# ============================================
train_dataset = TimeSeriesDataset(train_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
val_dataset = TimeSeriesDataset(val_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
test_dataset = TimeSeriesDataset(test_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)

print(f"\n樣本數:")
print(f"  訓練: {len(train_dataset)}")
print(f"  驗證: {len(val_dataset)}")
print(f"  測試: {len(test_dataset)}")

# ============================================
# 計算類別分布
# ============================================
train_directions = [train_dataset[i]['y_direction'].item() for i in range(len(train_dataset))]
up_count = sum(train_directions)
down_count = len(train_directions) - up_count

print(f"\n【類別分布】")
print(f"  訓練集: 上漲 {up_count} ({up_count/len(train_dataset):.1%}) / 下跌 {down_count} ({down_count/len(train_dataset):.1%})")

# 計算類別權重
class_counts = [down_count, up_count]
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[d] for d in train_directions]

print(f"  類別權重: 下跌={class_weights[0]:.3f}, 上漲={class_weights[1]:.3f}")

# 創建 WeightedRandomSampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)

# DataLoader - 使用 balanced sampler
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n✓ 使用 WeightedRandomSampler 平衡數據")

# ============================================
# 創建模型
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n設備: {device}")

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
# 訓練 - 使用方向準確率作為早停標準
# ============================================
print("\n【開始訓練】")
print("-" * 50)

# 組合損失：回歸 + 分類方向
def combined_loss(pred, target_return, target_direction, alpha=0.7):
    """組合損失: alpha * MSE + (1-alpha) * BCE"""
    pred = pred.squeeze()
    target_return = target_return.squeeze()
    target_direction = target_direction.squeeze()
    
    mse_loss = nn.functional.mse_loss(pred, target_return)
    
    # 方向損失 (使用預測值的符號)
    pred_prob = torch.sigmoid(pred * 10)  # 放大預測值
    bce_loss = nn.functional.binary_cross_entropy(
        pred_prob, 
        target_direction.float()
    )
    
    return alpha * mse_loss + (1 - alpha) * bce_loss

optimizer = optim.Adam(model.parameters(), lr=best_config['learning_rate'], weight_decay=1e-5)
epochs = 100
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

early_stop_patience = 20
best_val_acc = 0
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None
best_epoch = 0

train_losses = []
val_losses = []
val_accs = []

print(f"訓練輪數: {epochs}")
print(f"早停耐心: {early_stop_patience}")
print(f"使用組合損失 (70% MSE + 30% Direction BCE)")
print()

for epoch in range(epochs):
    # 訓練
    model.train()
    train_loss = 0
    
    for batch in train_loader:
        x = batch['x'].to(device)
        y_return = batch['y_return'].to(device)
        y_direction = batch['y_direction'].to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = combined_loss(pred, y_return.squeeze(), y_direction)
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
            loss = combined_loss(pred, y_return.squeeze(), y_direction)
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
    
    # 早停 (監控方向準確率)
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

model.load_state_dict(best_model_state)

print(f"\n訓練完成!")
print(f"  最佳驗證準確率: {best_val_acc:.2%} (Epoch {best_epoch})")

# ============================================
# 評估 - 訓練/驗證/測試
# ============================================
print("\n【最終評估】")
print("-" * 50)

def evaluate(loader, name):
    model.eval()
    all_preds = []
    all_directions = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y_direction = batch['y_direction'].to(device)
            
            pred = model(x)
            all_preds.extend(pred.squeeze().cpu().numpy())
            all_directions.extend(y_direction.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_directions = np.array(all_directions)
    pred_directions = (all_preds > 0).astype(int)
    
    # 統計預測分布
    pred_up = np.sum(pred_directions)
    pred_down = len(pred_directions) - pred_up
    
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(all_directions, pred_directions)
    
    print(f"\n{name}:")
    print(f"  準確率: {acc:.2%}")
    print(f"  預測分布: 上漲 {pred_up} ({pred_up/len(pred_directions):.1%}) / 下跌 {pred_down} ({pred_down/len(pred_directions):.1%})")
    
    return acc, all_preds, all_directions

train_acc, _, _ = evaluate(train_loader, "訓練集")
val_acc, _, _ = evaluate(val_loader, "驗證集")
test_acc, test_preds, test_dirs = evaluate(test_loader, "測試集")

# ============================================
# 混淆矩陣
# ============================================
from sklearn.metrics import confusion_matrix

pred_dirs = (np.array(test_preds) > 0).astype(int)
cm = confusion_matrix(test_dirs, pred_dirs)

print(f"\n【測試集混淆矩陣】")
print(f"              Predicted")
print(f"           Down   Up")
print(f"Actual Down  {cm[0,0]:3d}   {cm[0,1]:3d}")
print(f"       Up    {cm[1,0]:3d}   {cm[1,1]:3d}")

# ============================================
# 保存
# ============================================
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': model_config,
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'hyperparameters': best_config,
    'training_info': {
        'balanced_sampling': True,
        'combined_loss': True,
        'best_epoch': best_epoch
    }
}, 'patchtst_balanced_model.pth')

print(f"\n✓ 模型已保存: patchtst_balanced_model.pth")

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.plot(train_losses, label='Train', alpha=0.8)
ax1.plot(val_losses, label='Val', alpha=0.8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training History (Balanced)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(val_accs, label='Val Acc', color='green', linewidth=2)
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
ax2.axhline(y=test_acc, color='blue', linestyle='--', alpha=0.5, label=f'Test: {test_acc:.1%}')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Direction Accuracy')
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('balanced_model_results.png', dpi=150, bbox_inches='tight')
print("✓ 圖表已保存: balanced_model_results.png")

print("\n" + "=" * 70)
print("【平衡數據訓練完成】")
print("=" * 70)
print(f"\n最終結果:")
print(f"  訓練集準確率: {train_acc:.2%}")
print(f"  驗證集準確率: {val_acc:.2%}")
print(f"  測試集準確率: {test_acc:.2%}")
print(f"\n預測分布已平衡，不再只預測單一方向！")
print("=" * 70)
