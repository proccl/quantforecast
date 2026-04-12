import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix
import pickle

from step3_data_preprocessing import TimeSeriesDataset, RevIN
from step4_patchtst_model import PatchTST

print("=" * 60)
print("【步驟 6/8】模型評估與回測 - 使用真實數據")
print("=" * 60)

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

print(f"\n配置: 序列長度 {SEQ_LEN} 天")

# 載入模型
checkpoint = torch.load('patchtst_model.pth', map_location='cpu')
model_config = checkpoint['model_config']

model = PatchTST(**model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ 模型已載入")

# 載入並處理數據
df = pd.read_csv('xiaomi_real.csv')
df['date'] = pd.to_datetime(df['date'])

# 特徵工程
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

# 劃分
train_ratio = 0.7
val_ratio = 0.15
n_total = len(df_clean)
n_train = int(n_total * train_ratio)

test_df = df_clean.iloc[n_train+int(n_total*val_ratio):].copy()

print(f"✓ 測試集: {len(test_df)} 樣本 ({test_df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {test_df['date'].iloc[-1].strftime('%Y-%m-%d')})")

# 創建測試數據集
from torch.utils.data import DataLoader
test_dataset = TimeSeriesDataset(test_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 預測
all_preds = []
all_targets = []
all_directions = []

with torch.no_grad():
    for batch in test_loader:
        x = batch['x']
        y_return = batch['y_return']
        y_direction = batch['y_direction']
        
        pred = model(x)
        
        all_preds.extend(pred.squeeze().numpy())
        all_targets.extend(y_return.squeeze().numpy())
        all_directions.extend(y_direction.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_directions = np.array(all_directions)
pred_directions = (all_preds > 0).astype(int)

print(f"\n【評估指標】")
print("-" * 40)

# 方向準確率
direction_accuracy = accuracy_score(all_directions, pred_directions)
print(f"✓ Directional Accuracy: {direction_accuracy:.2%}")

# 回歸指標
mse = mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
rmse = np.sqrt(mse)

print(f"✓ MSE:  {mse:.6f}")
print(f"✓ MAE:  {mae:.6f}")
print(f"✓ RMSE: {rmse:.6f}")

# 混淆矩陣
cm = confusion_matrix(all_directions, pred_directions)
print(f"\n混淆矩陣:")
print(f"              預測")
print(f"           下跌   上漲")
print(f"實際 下跌  {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"     上漲  {cm[1,0]:4d}   {cm[1,1]:4d}")

# 簡化回測
print(f"\n【簡化回測】")
print("-" * 40)

returns = []
for i in range(len(all_preds)):
    if pred_directions[i] == 1:
        actual_return = all_targets[i]
    else:
        actual_return = 0
    returns.append(actual_return)

returns = np.array(returns)
cumulative_returns = np.cumsum(returns)

total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(52)  # 週化
max_drawdown = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns)) if len(cumulative_returns) > 0 else 0
win_rate = np.sum((returns > 0)) / len(returns) if len(returns) > 0 else 0

print(f"✓ 總收益率: {total_return:.2%}")
print(f"✓ Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"✓ Maximum Drawdown: {max_drawdown:.2%}")
print(f"✓ Win Rate: {win_rate:.2%}")

# 可視化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax1.scatter(all_targets, all_preds, alpha=0.5, edgecolors='black', linewidth=0.5)
ax1.plot([-0.3, 0.3], [-0.3, 0.3], 'r--', label='Perfect Prediction')
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax1.set_xlabel('Actual Return')
ax1.set_ylabel('Predicted Return')
ax1.set_title(f'Predicted vs Actual (Dir Acc: {direction_accuracy:.1%})')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
errors = all_preds - all_targets
ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', label='Zero Error')
ax2.axvline(x=np.mean(errors), color='blue', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
ax2.set_xlabel('Prediction Error')
ax2.set_ylabel('Frequency')
ax2.set_title('Prediction Error Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[1, 0]
ax3.plot(cumulative_returns * 100, label='Strategy', linewidth=2)
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax3.fill_between(range(len(cumulative_returns)), cumulative_returns * 100, 0, 
                  where=(cumulative_returns > 0), alpha=0.3, color='green')
ax3.fill_between(range(len(cumulative_returns)), cumulative_returns * 100, 0, 
                  where=(cumulative_returns < 0), alpha=0.3, color='red')
ax3.set_xlabel('Trade Number')
ax3.set_ylabel('Cumulative Return (%)')
ax3.set_title(f'Backtest (Sharpe: {sharpe_ratio:.2f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
im = ax4.imshow(cm, interpolation='nearest', cmap='Blues')
ax4.set_title('Confusion Matrix')
tick_marks = np.arange(2)
ax4.set_xticks(tick_marks)
ax4.set_yticks(tick_marks)
ax4.set_xticklabels(['Predict: Down', 'Predict: Up'])
ax4.set_yticklabels(['Actual: Down', 'Actual: Up'])

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax4.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
print("\n✓ 評估結果圖表已保存: evaluation_results.png")

print("\n" + "=" * 60)
print("【關鍵節點 6/8 完成】模型評估與回測完成")
print("=" * 60)
