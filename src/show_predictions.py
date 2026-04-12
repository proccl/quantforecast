import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix
import pickle

from step3_data_preprocessing import TimeSeriesDataset, RevIN
from step4_patchtst_model import PatchTST

print("=" * 70)
print("【預測結果可視化】")
print("=" * 70)

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

print(f"\n配置: seq_len={SEQ_LEN}, patch_len={PATCH_LEN}, stride={STRIDE}")

# 載入模型
checkpoint = torch.load('patchtst_best_model.pth', map_location='cpu')
model_config = checkpoint['model_config']

model = PatchTST(**model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ 模型已載入 (test_acc: {checkpoint['test_acc']:.2%})")

# 載入並處理數據
df = pd.read_csv('../data/xiaomi_real.csv')
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

df_clean = df[feature_cols + ['target_return_5d', 'target_direction', 'date', 'close']].dropna()

# 劃分測試集
train_ratio = 0.7
val_ratio = 0.15
n_total = len(df_clean)
test_df = df_clean.iloc[int(n_total*(train_ratio+val_ratio)):].copy().reset_index(drop=True)

print(f"✓ 測試集: {len(test_df)} 樣本 ({test_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {test_df['date'].iloc[-1].strftime('%Y-%m-%d')})")

# 創建測試數據集
from torch.utils.data import DataLoader
test_dataset = TimeSeriesDataset(test_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✓ 可預測樣本: {len(test_dataset)}")

# 預測
all_preds = []
all_targets = []
all_directions = []
all_dates = []
all_prices = []

with torch.no_grad():
    for batch in test_loader:
        x = batch['x']
        y_return = batch['y_return']
        y_direction = batch['y_direction']
        
        pred = model(x)
        
        all_preds.extend(pred.squeeze().numpy())
        all_targets.extend(y_return.squeeze().numpy())
        all_directions.extend(y_direction.numpy())

# 截齊數據
min_len = min(len(all_preds), len(test_dataset))
all_preds = np.array(all_preds[:min_len])
all_targets = np.array(all_targets[:min_len])
all_directions = np.array(all_directions[:min_len])

# 獲取對應的日期和價格
for i in range(min_len):
    idx = i + SEQ_LEN - 1
    if idx < len(test_df):
        all_dates.append(test_df.iloc[idx]['date'])
        all_prices.append(test_df.iloc[idx]['close'])

all_dates = all_dates[:min_len]
all_prices = np.array(all_prices[:min_len])

pred_directions = (all_preds > 0).astype(int)

# 計算指標
accuracy = accuracy_score(all_directions, pred_directions)
cm = confusion_matrix(all_directions, pred_directions)
mse = mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)

print(f"\n【測試結果】")
print(f"  樣本數: {len(all_preds)}")
print(f"  方向準確率: {accuracy:.2%}")
print(f"  MSE: {mse:.6f}")
print(f"  MAE: {mae:.6f}")
print(f"  混淆矩陣: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

# ============================================
# 創建可視化
# ============================================
fig = plt.figure(figsize=(16, 12))

# 1. 預測vs實際收益率散點圖
ax1 = plt.subplot(3, 2, 1)
colors = ['green' if d == 1 else 'red' for d in all_directions]
ax1.scatter(all_preds * 100, all_targets * 100, c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax1.plot([-15, 15], [-15, 15], 'b--', linewidth=2, label='Perfect')
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax1.set_xlabel('Predicted Return (%)')
ax1.set_ylabel('Actual Return (%)')
ax1.set_title('Predicted vs Actual Returns (5-Day)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-15, 15)
ax1.set_ylim(-15, 15)

# 2. 預測收益率時間序列
ax2 = plt.subplot(3, 2, 2)
ax2.plot(all_dates, all_preds * 100, linewidth=1.5, color='blue', label='Predicted')
ax2.fill_between(all_dates, all_preds * 100, 0, where=(all_preds > 0), alpha=0.3, color='green')
ax2.fill_between(all_dates, all_preds * 100, 0, where=(all_preds <= 0), alpha=0.3, color='red')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_ylabel('Predicted Return (%)')
ax2.set_title('Predicted Returns Over Time')
ax2.grid(True, alpha=0.3)

# 3. 實際收益率時間序列
ax3 = plt.subplot(3, 2, 3)
ax3.plot(all_dates, all_targets * 100, linewidth=1.5, color='orange', label='Actual')
ax3.fill_between(all_dates, all_targets * 100, 0, where=(all_targets > 0), alpha=0.3, color='green')
ax3.fill_between(all_dates, all_targets * 100, 0, where=(all_targets <= 0), alpha=0.3, color='red')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_ylabel('Actual Return (%)')
ax3.set_title('Actual Returns Over Time')
ax3.grid(True, alpha=0.3)

# 4. 預測方向 vs 實際方向
ax4 = plt.subplot(3, 2, 4)
show_n = min(40, len(all_dates))
show_idx = np.linspace(0, len(all_dates)-1, show_n, dtype=int)
x_pos = np.arange(show_n)
width = 0.35

ax4.bar(x_pos - width/2, all_directions[show_idx], width, label='Actual', color='orange', alpha=0.7)
ax4.bar(x_pos + width/2, pred_directions[show_idx], width, label='Predicted', color='blue', alpha=0.7)
ax4.set_xlabel('Sample')
ax4.set_ylabel('Direction (0=Down, 1=Up)')
ax4.set_title('Actual vs Predicted Direction (Sample)')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. 混淆矩陣熱力圖
ax5 = plt.subplot(3, 2, 5)
im = ax5.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax5.figure.colorbar(im, ax=ax5)
ax5.set_xticks([0, 1])
ax5.set_yticks([0, 1])
ax5.set_xticklabels(['Down', 'Up'])
ax5.set_yticklabels(['Down', 'Up'])
ax5.set_xlabel('Predicted')
ax5.set_ylabel('Actual')
ax5.set_title(f'Confusion Matrix (Acc: {accuracy:.1%})')

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax5.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16, fontweight='bold')

# 6. 預測誤差分布
ax6 = plt.subplot(3, 2, 6)
errors = (all_preds - all_targets) * 100
ax6.hist(errors, bins=25, alpha=0.7, color='steelblue', edgecolor='black')
ax6.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax6.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}%')
ax6.set_xlabel('Prediction Error (%)')
ax6.set_ylabel('Frequency')
ax6.set_title('Prediction Error Distribution')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
print("\n✓ 預測結果圖: prediction_results.png")

# ============================================
# 回測可視化
# ============================================
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# 策略回測
initial = 100000
capital = initial
equity_curve = []
trade_returns = []

for i in range(len(all_preds)):
    if pred_directions[i] == 1:
        daily_ret = all_targets[i]
    else:
        daily_ret = 0
    capital *= (1 + daily_ret)
    equity_curve.append(capital)
    trade_returns.append(daily_ret)

# 買入持有
buy_hold_curve = initial * (all_prices / all_prices[0])

# 1. 資金曲線
ax1 = axes[0, 0]
ax1.plot(all_dates, equity_curve, linewidth=2, color='blue', label='Strategy')
ax1.plot(all_dates, buy_hold_curve, linewidth=2, color='gray', linestyle='--', label='Buy & Hold')
ax1.axhline(y=initial, color='red', linestyle=':', alpha=0.5)
ax1.set_ylabel('Capital (HKD)')
ax1.set_title('Equity Curve Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 累計收益
cum_ret_strategy = (np.array(equity_curve) / initial - 1) * 100
cum_ret_buyhold = (buy_hold_curve / initial - 1) * 100

ax2 = axes[0, 1]
ax2.plot(all_dates, cum_ret_strategy, linewidth=2, color='blue', label='Strategy')
ax2.plot(all_dates, cum_ret_buyhold, linewidth=2, color='gray', linestyle='--', label='Buy & Hold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.fill_between(all_dates, cum_ret_strategy, 0, alpha=0.3, color='blue')
ax2.set_ylabel('Cumulative Return (%)')
ax2.set_title('Cumulative Return Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 日收益分布
ax3 = axes[1, 0]
returns_pct = np.array(trade_returns) * 100
ax3.hist(returns_pct, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.axvline(x=np.mean(returns_pct), color='green', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(returns_pct):.3f}%')
ax3.set_xlabel('Daily Return (%)')
ax3.set_ylabel('Frequency')
ax3.set_title('Strategy Return Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. 預測信號時間軸
ax4 = axes[1, 1]
colors_signal = ['green' if d == 1 else 'red' for d in pred_directions]
ax4.scatter(all_dates, pred_directions, c=colors_signal, alpha=0.7, s=30)
ax4.set_ylabel('Signal (1=Long, 0=Cash)')
ax4.set_title('Trading Signals')
ax4.set_ylim(-0.2, 1.2)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_comparison.png', dpi=150, bbox_inches='tight')
print("✓ 回測對比圖: backtest_comparison.png")

# 輸出統計
final_capital = equity_curve[-1]
strategy_return = (final_capital / initial - 1)
buyhold_return = (buy_hold_curve[-1] / initial - 1)

print(f"\n【回測統計】")
print(f"  初始資金: {initial:,.0f} HKD")
print(f"  策略終值: {final_capital:,.0f} HKD ({strategy_return:+.2%})")
print(f"  買入持有: {buyhold_return:+.2%}")
print(f"  超額收益: {strategy_return - buyhold_return:+.2%}")

# 計算額外指標
positive_trades = sum(1 for r in trade_returns if r > 0)
win_rate = positive_trades / len(trade_returns) if trade_returns else 0
max_drawdown = min((np.array(equity_curve) / np.maximum.accumulate(equity_curve) - 1)) if equity_curve else 0

print(f"  勝率: {win_rate:.2%}")
print(f"  最大回撤: {max_drawdown:.2%}")

print("\n" + "=" * 70)
