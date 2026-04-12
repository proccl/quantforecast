import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

from step3_data_preprocessing import TimeSeriesDataset, RevIN
from step4_patchtst_model import PatchTST

print("=" * 70)
print("【貝葉斯優化配置 - 最終模型訓練】")
print("=" * 70)

# 貝葉斯優化找到的最佳配置
BEST_PARAMS = {
    'd_model': 32,
    'n_heads': 2,
    'n_layers': 3,
    'dropout': 0.15,
    'patch_len': 3,
    'stride': 1,  # patch_len // 2
    'learning_rate': 0.00020269658755951023,
    'batch_size': 32,
    'seq_len': 20,
    'pred_len': 5,
}

print("\n【使用配置】")
for k, v in BEST_PARAMS.items():
    print(f"  {k}: {v}")

# 設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n設備: {device}")

# 加載數據配置
with open('data_config.pkl', 'rb') as f:
    data_config = pickle.load(f)

feature_cols = data_config['feature_cols']

# 載入並處理數據
df = pd.read_csv('xiaomi_real.csv')
df['date'] = pd.to_datetime(df['date'])

# 特徵工程（完整版）
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
df['rsi_14'] = 100 - (100 / (1 + gain/loss))

df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma_20']

df['return_1d'] = df['close'].pct_change(1)
df['return_5d'] = df['close'].pct_change(5)
df['volatility_20d'] = df['return_1d'].rolling(window=20).std()
df['target_return_5d'] = df['close'].pct_change(5).shift(-5)
df['target_direction'] = (df['target_return_5d'] > 0).astype(int)

df_clean = df[feature_cols + ['target_return_5d', 'target_direction', 'date']].dropna()

# 劃分數據集
train_ratio = 0.7
val_ratio = 0.15
n_total = len(df_clean)
n_train = int(n_total * train_ratio)

train_df = df_clean.iloc[:n_train].copy()
val_df = df_clean.iloc[n_train:n_train+int(n_total*val_ratio)].copy()
test_df = df_clean.iloc[n_train+int(n_total*val_ratio):].copy()

print(f"\n數據集劃分:")
print(f"  訓練集: {len(train_df)} 樣本 ({train_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {train_df['date'].iloc[-1].strftime('%Y-%m-%d')})")
print(f"  驗證集: {len(val_df)} 樣本 ({val_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {val_df['date'].iloc[-1].strftime('%Y-%m-%d')})")
print(f"  測試集: {len(test_df)} 樣本 ({test_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {test_df['date'].iloc[-1].strftime('%Y-%m-%d')})")

# 創建數據集
train_dataset = TimeSeriesDataset(train_df, feature_cols, BEST_PARAMS['seq_len'], 
                                   BEST_PARAMS['pred_len'], BEST_PARAMS['patch_len'], 
                                   BEST_PARAMS['stride'])
val_dataset = TimeSeriesDataset(val_df, feature_cols, BEST_PARAMS['seq_len'], 
                                 BEST_PARAMS['pred_len'], BEST_PARAMS['patch_len'], 
                                 BEST_PARAMS['stride'])
test_dataset = TimeSeriesDataset(test_df, feature_cols, BEST_PARAMS['seq_len'], 
                                  BEST_PARAMS['pred_len'], BEST_PARAMS['patch_len'], 
                                  BEST_PARAMS['stride'])

print(f"\n數據集樣本數:")
print(f"  訓練集: {len(train_dataset)}")
print(f"  驗證集: {len(val_dataset)}")
print(f"  測試集: {len(test_dataset)}")

# 平衡採樣
train_directions = [train_dataset[i]['y_direction'].item() for i in range(len(train_dataset))]
class_counts = [len(train_directions) - sum(train_directions), sum(train_directions)]
class_weights = [len(train_directions) / (2 * max(c, 1)) for c in class_counts]
sample_weights = [class_weights[d] for d in train_directions]
sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BEST_PARAMS['batch_size'], sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=False)

# 創建模型
model_config = {
    'n_features': len(feature_cols),
    'seq_len': BEST_PARAMS['seq_len'],
    'pred_len': BEST_PARAMS['pred_len'],
    'patch_len': BEST_PARAMS['patch_len'],
    'stride': BEST_PARAMS['stride'],
    'd_model': BEST_PARAMS['d_model'],
    'n_heads': BEST_PARAMS['n_heads'],
    'n_layers': BEST_PARAMS['n_layers'],
    'd_ff': BEST_PARAMS['d_model'] * 2,
    'dropout': BEST_PARAMS['dropout'],
    'head_type': 'regression',
    'use_revin': True
}

model = PatchTST(**model_config).to(device)
print(f"\n模型參數量: {sum(p.numel() for p in model.parameters()):,}")

# 訓練設置
criterion = nn.HuberLoss(delta=0.1)
optimizer = optim.Adam(model.parameters(), lr=BEST_PARAMS['learning_rate'], weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 訓練循環
print("\n" + "=" * 70)
print("開始訓練...")
print("=" * 70)

best_val_loss = float('inf')
best_val_acc = 0
patience = 15
patience_counter = 0

train_losses = []
val_losses = []
val_accs = []

for epoch in range(100):
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
    train_losses.append(train_loss)
    
    # 驗證
    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(device)
            y_return = batch['y_return'].to(device)
            y_direction = batch['y_direction']
            
            pred = model(x)
            loss = criterion(pred, y_return)
            val_loss += loss.item()
            
            val_preds.extend(pred.squeeze().cpu().numpy())
            val_targets.extend(y_direction.numpy())
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    pred_directions = (val_preds > 0).astype(int)
    val_acc = accuracy_score(val_targets, pred_directions)
    val_accs.append(val_acc)
    
    # 早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epoch': epoch
        }, 'patchtst_bayesian_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Val Acc={val_acc:.2%}")
    
    if patience_counter >= patience:
        print(f"\n早停於第 {epoch+1} 輪")
        break

print(f"\n最佳驗證損失: {best_val_loss:.6f}")
print(f"最佳驗證準確率: {best_val_acc:.2%}")

# ============================================
# 測試集評估
# ============================================
print("\n" + "=" * 70)
print("【測試集評估】")
print("=" * 70)

# 載入最佳模型
checkpoint = torch.load('patchtst_bayesian_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 測試集預測
test_preds = []
test_targets_return = []
test_targets_direction = []
test_dates = []
test_prices = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        x = batch['x'].to(device)
        y_return = batch['y_return']
        y_direction = batch['y_direction']
        
        pred = model(x)
        
        test_preds.extend(pred.squeeze().cpu().numpy())
        test_targets_return.extend(y_return.numpy())
        test_targets_direction.extend(y_direction.numpy())
        
        # 記錄日期和價格
        for j in range(len(y_direction)):
            idx = i * BEST_PARAMS['batch_size'] + j + BEST_PARAMS['seq_len'] - 1
            if idx < len(test_df):
                test_dates.append(test_df.iloc[idx]['date'])
                test_prices.append(test_df.iloc[idx]['close'])

test_preds = np.array(test_preds[:len(test_dates)])
test_targets_return = np.array(test_targets_return[:len(test_dates)])
test_targets_direction = np.array(test_targets_direction[:len(test_dates)])
test_prices = np.array(test_prices[:len(test_dates)])

pred_directions = (test_preds > 0).astype(int)

# 計算指標
from sklearn.metrics import mean_squared_error, mean_absolute_error
test_mse = mean_squared_error(test_targets_return, test_preds)
test_mae = mean_absolute_error(test_targets_return, test_preds)
test_acc = accuracy_score(test_targets_direction, pred_directions)
cm = confusion_matrix(test_targets_direction, pred_directions)

# 預測分佈
pred_up_ratio = np.mean(pred_directions)

print(f"\n測試集樣本數: {len(test_preds)}")
print(f"\n【回歸指標】")
print(f"  MSE: {test_mse:.6f}")
print(f"  MAE: {test_mae:.6f}")
print(f"  RMSE: {np.sqrt(test_mse):.6f}")

print(f"\n【方向預測】")
print(f"  準確率: {test_acc:.2%}")
print(f"  預測上漲比例: {pred_up_ratio:.2%}")
print(f"  預測下跌比例: {1-pred_up_ratio:.2%}")

print(f"\n【混淆矩陣】")
print(f"              預測")
print(f"           下跌   上漲")
print(f"實際 下跌  {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"     上漲  {cm[1,0]:4d}   {cm[1,1]:4d}")

# 簡化回測
print(f"\n【簡化回測】")
initial = 100000
capital = initial
strategy_returns = []
buyhold_returns = []

for i in range(len(test_preds)):
    if pred_directions[i] == 1:
        daily_ret = test_targets_return[i]
    else:
        daily_ret = 0
    capital *= (1 + daily_ret)
    strategy_returns.append(capital)

# 買入持有
buyhold_capital = initial * (test_prices / test_prices[0])

total_return = (capital / initial - 1)
buyhold_return = (buyhold_capital[-1] / initial - 1) if len(buyhold_capital) > 0 else 0

print(f"  初始資金: {initial:,.0f} HKD")
print(f"  策略終值: {capital:,.0f} HKD ({total_return:+.2%})")
print(f"  買入持有: {buyhold_return:+.2%}")
print(f"  超額收益: {total_return - buyhold_return:+.2%}")

# ============================================
# 可視化
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 訓練曲線
ax1 = axes[0, 0]
ax1.plot(train_losses, label='Train Loss', color='blue')
ax1.plot(val_losses, label='Val Loss', color='orange')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 驗證準確率
ax2 = axes[0, 1]
ax2.plot(val_accs, color='green')
ax2.axhline(y=0.5, color='red', linestyle='--', label='Random')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Accuracy')
ax2.set_title(f'Validation Direction Accuracy (Best: {best_val_acc:.2%})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 測試集散點圖
ax3 = axes[1, 0]
colors = ['green' if d == 1 else 'red' for d in test_targets_direction]
ax3.scatter(test_preds * 100, test_targets_return * 100, c=colors, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.plot([-20, 20], [-20, 20], 'b--', label='Perfect')
ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax3.set_xlabel('Predicted Return (%)')
ax3.set_ylabel('Actual Return (%)')
ax3.set_title(f'Test Set: Predicted vs Actual (Acc: {test_acc:.1%})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 混淆矩陣
ax4 = axes[1, 1]
im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
ax4.figure.colorbar(im, ax=ax4)
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Down', 'Up'])
ax4.set_yticklabels(['Down', 'Up'])
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
ax4.set_title(f'Confusion Matrix (Up: {pred_up_ratio:.1%})')

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax4.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('bayesian_final_results.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 結果圖表已保存: bayesian_final_results.png")

# ============================================
# 保存結果
# ============================================
import json

result = {
    'best_params': BEST_PARAMS,
    'val_loss': float(best_val_loss),
    'val_accuracy': float(best_val_acc),
    'test_mse': float(test_mse),
    'test_mae': float(test_mae),
    'test_accuracy': float(test_acc),
    'test_confusion_matrix': cm.tolist(),
    'pred_up_ratio': float(pred_up_ratio),
    'strategy_return': float(total_return),
    'buyhold_return': float(buyhold_return),
    'excess_return': float(total_return - buyhold_return)
}

with open('bayesian_final_results.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"✓ 結果已保存: bayesian_final_results.json")

print("\n" + "=" * 70)
print("【訓練完成】")
print("=" * 70)
