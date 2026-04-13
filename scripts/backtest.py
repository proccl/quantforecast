#!/usr/bin/env python3
"""
完整回測分析 - 與 main 分支一致
"""

import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from scipy import stats
import json
from pathlib import Path
import sys
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.models.patchtst import PatchTST

print("=" * 70)
print("【完整回測分析】")
print("=" * 70)

# 設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = get_config('config/config.yaml')

# 加載數據
loader = DataLoader(config.data)
df = loader.load()

engineer = FeatureEngineer()
df_features = engineer.create_features(df)
feature_cols = engineer.get_feature_columns()
df_clean = engineer.clean(df_features, feature_cols)

print(f"\n【配置】")
print(f"特徵數量: {len(feature_cols)}")

# 加載測試數據
test_df = df_clean.copy()
test_df['date'] = pd.to_datetime(test_df['date'])

# 只取最近1年數據進行回測
latest_date = test_df['date'].iloc[-1]
three_months_ago = latest_date - pd.Timedelta(days=90)
test_df = test_df[test_df['date'] >= three_months_ago].reset_index(drop=True)

print(f"\n【測試數據】")
print(f"樣本數: {len(test_df)}")
print(f"時間範圍: {test_df['date'].iloc[0]} ~ {test_df['date'].iloc[-1]}")

# RevIN 標準化類
class RevIN:
    def __init__(self, num_features):
        self.num_features = num_features
        self.eps = 1e-5
    
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True)
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + self.eps)
            x = (x - self.mean) / self.stdev
        elif mode == 'denorm':
            x = x * self.stdev + self.mean
        return x

# 查找最新模型
model_dir = Path(config.paths.model_dir)
model_files = sorted(model_dir.glob("patchtst_model_*.pth"), reverse=True)

if model_files:
    model_path = model_files[0]
    print(f"✓ 加載最新模型: {model_path.name}")
elif (model_dir / "patchtst_bayesian_best.pth").exists():
    model_path = model_dir / "patchtst_bayesian_best.pth"
    print(f"✓ 加載貝葉斯模型: {model_path.name}")
else:
    print("✗ 未找到模型")
    sys.exit(1)

checkpoint = torch.load(model_path, map_location=device, weights_only=False)

# 獲取模型配置
if 'model_config' in checkpoint:
    loaded_config = checkpoint['model_config']
elif 'config' in checkpoint and hasattr(checkpoint['config'], 'model'):
    cfg = checkpoint['config']
    loaded_config = {
        'seq_len': cfg.data.seq_len,
        'pred_len': cfg.data.pred_len,
        'd_model': cfg.model.d_model,
        'n_heads': cfg.model.n_heads,
        'n_layers': cfg.model.n_layers,
        'dropout': cfg.model.dropout,
        'patch_len': 5,
        'stride': 2,
        'd_ff': cfg.model.d_ff
    }
else:
    loaded_config = {
        'seq_len': config.data.seq_len,
        'pred_len': config.data.pred_len,
        'd_model': config.model.d_model,
        'n_heads': config.model.n_heads,
        'n_layers': config.model.n_layers,
        'dropout': config.model.dropout,
        'patch_len': 5,
        'stride': 2,
        'd_ff': config.model.d_ff
    }

seq_len = loaded_config['seq_len']
pred_len = loaded_config['pred_len']

model = PatchTST(
    n_features=len(feature_cols),
    seq_len=seq_len,
    pred_len=pred_len,
    d_model=loaded_config['d_model'],
    n_heads=loaded_config['n_heads'],
    n_layers=loaded_config['n_layers'],
    dropout=loaded_config['dropout'],
    patch_len=loaded_config['patch_len'],
    stride=loaded_config['stride'],
    d_ff=loaded_config.get('d_ff', loaded_config['d_model'] * 2),
    head_type='regression',
    use_revin=True
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\n✓ 模型已載入")
print(f"  序列長度: {seq_len}")
print(f"  預測長度: {pred_len}")
print(f"  d_model: {loaded_config['d_model']}")
print(f"  n_heads: {loaded_config['n_heads']}")
print(f"  n_layers: {loaded_config['n_layers']}")

# 預測
predictions = []
actuals = []
test_dates = []
test_dates_T5 = []
test_close = []
test_close_T5 = []
test_directions = []

with torch.no_grad():
    for i in range(len(test_df) - seq_len - pred_len + 1):
        seq_data = test_df[feature_cols].iloc[i:i+seq_len].values
        close_T = test_df['close'].iloc[i+seq_len]
        close_T5 = test_df['close'].iloc[i+seq_len:i+seq_len+pred_len].values[-1]
        target_return = close_T5 / close_T - 1
        
        x = torch.FloatTensor(seq_data).unsqueeze(0).to(device)
        pred = model(x)
        
        pred_return = pred.squeeze().cpu().numpy()
        if len(pred_return.shape) > 0:
            pred_return = pred_return[0] if pred_return.shape[0] > 0 else 0
        
        predictions.append(float(pred_return))
        actuals.append(float(target_return))
        test_dates.append(test_df['date'].iloc[i+seq_len])
        test_dates_T5.append(test_df['date'].iloc[i+seq_len+pred_len-1])
        test_close.append(close_T)
        test_close_T5.append(close_T5)
        test_directions.append(1 if target_return > 0 else 0)

# 轉為numpy
test_directions = np.array(test_directions)

# 創建對齊的DataFrame
aligned_df = pd.DataFrame({
    'date': test_dates,
    'date_T5': test_dates_T5,
    'close': test_close,
    'close_T5': test_close_T5,
    'pred_return': predictions,
    'actual_return': actuals,
    'pred_direction': [1 if p > 0 else 0 for p in predictions],
    'actual_direction': test_directions
})

print(f"\n對齊後樣本數: {len(aligned_df)}")

# ============================================
# 回測
# ============================================
initial_capital = 100000
capital = initial_capital
position = 0
trades = []
equity_curve = []

for i in range(len(aligned_df)):
    current_price = aligned_df['close'].iloc[i]
    pred_direction = aligned_df['pred_direction'].iloc[i]
    
    if pred_direction == 1 and position == 0:
        position = capital / current_price
        capital = 0
        trades.append({
            'date': aligned_df['date'].iloc[i],
            'action': 'BUY',
            'price': current_price,
            'shares': position
        })
    elif pred_direction == 0 and position > 0:
        capital = position * current_price
        trades.append({
            'date': aligned_df['date'].iloc[i],
            'action': 'SELL',
            'price': current_price,
            'proceeds': capital
        })
        position = 0
    
    current_equity = capital if position == 0 else position * current_price
    equity_curve.append(current_equity)

equity_curve = np.array(equity_curve)
final_capital = capital if position == 0 else position * aligned_df['close'].iloc[-1]
total_return = (final_capital / initial_capital - 1) * 100

# 買入持有基準
buy_hold_shares = initial_capital / aligned_df['close'].iloc[0]
buy_hold_final = buy_hold_shares * aligned_df['close'].iloc[-1]
buy_hold_return = (buy_hold_final / initial_capital - 1) * 100

# 計算風險指標
daily_returns = np.diff(equity_curve) / equity_curve[:-1]
volatility = np.std(daily_returns) * np.sqrt(252) * 100

running_max = np.maximum.accumulate(equity_curve)
drawdown = (equity_curve - running_max) / running_max
max_drawdown = np.min(drawdown) * 100

sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
downside_returns = daily_returns[daily_returns < 0]
downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
sortino_ratio = np.mean(daily_returns) / downside_std * np.sqrt(252)
calmar_ratio = (total_return / 100) / (abs(max_drawdown) / 100 + 1e-8)

print(f"\n【回測結果】")
print(f"初始資金: {initial_capital:,.0f} HKD")
print(f"最終資金: {final_capital:,.0f} HKD")
print(f"策略總收益: {total_return:.2f}%")
print(f"買入持有: {buy_hold_return:.2f}%")
print(f"超額收益: {total_return - buy_hold_return:+.2f}%")

print(f"\n【風險指標】")
print(f"年化波動率: {volatility:.2f}%")
print(f"最大回撤: {max_drawdown:.2f}%")
print(f"夏普比率: {sharpe_ratio:.4f}")
print(f"索提諾比率: {sortino_ratio:.4f}")
print(f"卡爾瑪比率: {calmar_ratio:.4f}")

# 交易統計
trades_df = pd.DataFrame(trades)
buy_trades = trades_df[trades_df['action'].str.contains('BUY')]
sell_trades = trades_df[trades_df['action'].str.contains('SELL')]

win_count = 0
loss_count = 0
for i in range(len(sell_trades)):
    if i < len(buy_trades):
        buy_price = buy_trades.iloc[i]['price']
        sell_price = sell_trades.iloc[i]['price']
        if sell_price > buy_price:
            win_count += 1
        else:
            loss_count += 1

win_rate = win_count / max(win_count + loss_count, 1) * 100

print(f"\n【交易統計】")
print(f"總交易次數: {len(buy_trades)} 次")
print(f"平均持倉天數: {len(aligned_df) / max(len(buy_trades), 1):.1f} 天")
print(f"盈利交易: {win_count} 次")
print(f"虧損交易: {loss_count} 次")
print(f"勝率: {win_rate:.2f}%")

strategy_returns = np.diff(equity_curve) / equity_curve[:-1]

# ============================================
# 未來5天預測
# ============================================
print("\n" + "=" * 70)
print("【未來5天預測】")
print("=" * 70)

# 獲取最新數據
latest_data = df_clean[feature_cols].iloc[-seq_len:].values
latest_close = df_clean['close'].iloc[-1]

# 預測未來5天收益
with torch.no_grad():
    x = torch.FloatTensor(latest_data).unsqueeze(0).to(device)
    pred = model(x)
    pred_values = pred.squeeze().cpu().numpy()

# 計算未來交易日
future_dates = []
current_date = latest_date
while len(future_dates) < pred_len:
    current_date = current_date + pd.Timedelta(days=1)
    if current_date.weekday() < 5:
        future_dates.append(current_date)

# 處理 pred_values
if isinstance(pred_values, np.ndarray):
    pred_scalar = float(pred_values.item()) if pred_values.size == 1 else float(pred_values[0])
else:
    pred_scalar = float(pred_values)

# 計算每一天的預測價格
future_prices = []
for i in range(pred_len):
    progress = (i + 1) / pred_len
    price = latest_close * (1 + pred_scalar * progress)
    future_prices.append(price)

print(f"\n最新收盤價 ({latest_date.strftime('%Y-%m-%d')}): {latest_close:.2f} HKD")
print(f"預測未來5天收益: {pred_scalar*100:+.2f}%")
print(f"預測5天後價格: {future_prices[-1]:.2f} HKD")
print(f"\n預測日期範圍: {future_dates[0].strftime('%Y-%m-%d')} ~ {future_dates[-1].strftime('%Y-%m-%d')}")

# ============================================
# 可視化 (3行布局 - 深色模式)
# ============================================
fig = plt.figure(figsize=(16, 14), facecolor='#1a1a1a')

# 設置深色主題
plt.rcParams['axes.facecolor'] = '#2d2d2d'
plt.rcParams['axes.edgecolor'] = '#666666'
plt.rcParams['axes.labelcolor'] = '#cccccc'
plt.rcParams['text.color'] = '#cccccc'
plt.rcParams['xtick.color'] = '#cccccc'
plt.rcParams['ytick.color'] = '#cccccc'
plt.rcParams['grid.color'] = '#444444'
plt.rcParams['figure.facecolor'] = '#1a1a1a'

# 第1行: 2x2
# 1. 混淆矩陣
ax1 = plt.subplot(3, 2, 1)
cm = confusion_matrix(test_directions[:len(aligned_df)], aligned_df['pred_direction'])
im = ax1.imshow(cm, interpolation='nearest', cmap='YlOrRd')
ax1.figure.colorbar(im, ax=ax1, label='Count')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Down', 'Up'], color='#cccccc')
ax1.set_yticklabels(['Down', 'Up'], color='#cccccc')
ax1.set_xlabel('Predicted', color='#cccccc')
ax1.set_ylabel('Actual', color='#cccccc')
ax1.set_title(f'Confusion Matrix (Acc: {accuracy_score(test_directions[:len(aligned_df)], aligned_df["pred_direction"]):.1%})', color='#ffffff')

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax1.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16, fontweight='bold')

# 2. 每日收益分布
ax2 = plt.subplot(3, 2, 2)
returns_pct = strategy_returns * 100
mu, sigma = np.mean(returns_pct), np.std(returns_pct)
n, bins, patches = ax2.hist(returns_pct, bins=30, alpha=0.6, color='#4a9eff', 
                            edgecolor='#2d2d2d', density=True, label='Histogram')
x = np.linspace(bins[0], bins[-1], 100)
ax2.plot(x, stats.norm.pdf(x, mu, sigma), '--', color='#ffd43b', linewidth=2, 
         label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})')
ax2.axvline(x=0, color='#ff6b6b', linestyle='-', linewidth=1.5, alpha=0.7, label='Zero')
ax2.axvline(x=mu, color='#51cf66', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Mean: {mu:.3f}%')
ax2.set_xlabel('Daily Return (%)', color='#cccccc')
ax2.set_ylabel('Density', color='#cccccc')
ax2.set_title('Daily Return Distribution', color='#ffffff')
ax2.legend(facecolor='none', edgecolor='none', labelcolor='#cccccc', loc='upper left')
ax2.grid(True, alpha=0.3, axis='y')

# 第2行: 2x2
# 3. 預測 vs 實際散點圖
ax3 = plt.subplot(3, 2, 3)
aligned_directions = test_directions[:len(aligned_df)]
up_indices = np.where(aligned_directions == 1)[0]
down_indices = np.where(aligned_directions == 0)[0]
pred_returns_arr = aligned_df['pred_return'].values * 100
actual_returns_arr = aligned_df['actual_return'].values * 100

ax3.scatter(pred_returns_arr[up_indices], actual_returns_arr[up_indices],
            c='#51cf66', alpha=0.6, edgecolors='#2d2d2d', linewidth=0.5,
            label='Actual Up', s=30)
ax3.scatter(pred_returns_arr[down_indices], actual_returns_arr[down_indices],
            c='#ff6b6b', alpha=0.6, edgecolors='#2d2d2d', linewidth=0.5,
            label='Actual Down', s=30)
ax3.plot([-20, 20], [-20, 20], '#4a9eff', linestyle='--', linewidth=1.5, label='Perfect Prediction')
ax3.axhline(y=0, color='#666666', linestyle='-', alpha=0.5)
ax3.axvline(x=0, color='#666666', linestyle='-', alpha=0.5)
ax3.set_xlabel('Predicted Return (%)', color='#cccccc')
ax3.set_ylabel('Actual Return (%)', color='#cccccc')
ax3.set_title('Prediction vs Actual', color='#ffffff')
ax3.legend(facecolor='none', edgecolor='none', labelcolor='#cccccc', loc='upper left')
ax3.grid(True, alpha=0.3)

# 4. 累計收益對比
ax4 = plt.subplot(3, 2, 4)
buyhold_curve = initial_capital * (aligned_df['close'] / aligned_df['close'].iloc[0])
strat_cumret = (equity_curve / initial_capital - 1) * 100
buyhold_cumret = (buyhold_curve / initial_capital - 1) * 100

line1, = ax4.plot(aligned_df['date'], strat_cumret, linewidth=2.5, color='#4a9eff', label='Strategy')
line2, = ax4.plot(aligned_df['date'], buyhold_cumret, linewidth=2, color='#888888', linestyle='--', label='Buy & Hold')
ax4.axhline(y=0, color='#666666', linestyle='-', linewidth=1)
ax4.fill_between(aligned_df['date'], strat_cumret, 0, where=(strat_cumret > 0), alpha=0.3, color='#51cf66')
ax4.fill_between(aligned_df['date'], strat_cumret, 0, where=(strat_cumret <= 0), alpha=0.3, color='#ff6b6b')
ax4.set_ylabel('Cumulative Return (%)', color='#cccccc')
ax4.set_xlabel('Date', color='#cccccc')
ax4.set_title(f'Cumulative Return (Long: {np.mean(aligned_df["pred_direction"])*100:.1f}%)', color='#ffffff')
ax4.grid(True, alpha=0.3)

# 交易信號疊加
ax4b = ax4.twinx()
signals = aligned_df['pred_direction'].values
y_min, y_max = ax4.get_ylim()
line_y = y_min + (y_max - y_min) * 0.01

dates_num = mdates.date2num(aligned_df['date'].values)
points = np.array([dates_num, np.full(len(signals), line_y)]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
colors = ['#51cf66' if signals[i+1] == 1 else '#ff6b6b' for i in range(len(signals)-1)]

lc = LineCollection(segments, colors=colors, linewidths=2, capstyle='butt')
ax4b.add_collection(lc)
ax4b.set_ylim(y_min, y_max)
ax4b.set_yticks([])

legend_elements = [
    line1,
    line2,
    Line2D([0], [0], color='#51cf66', linewidth=2, label='Long Position'),
    Line2D([0], [0], color='#ff6b6b', linewidth=2, label='Cash Position')
]
ax4.legend(handles=legend_elements, loc='lower left', facecolor='none',
           edgecolor='none', labelcolor='#cccccc')

# 第3行: 長圖 (跨越兩列)
# 5. 預測股價 vs 實際股價 + 未來5天預測
ax5 = plt.subplot(3, 1, 3)

# 藍線: 完整實際價格
df_plot = df_clean.copy()
df_plot['date'] = pd.to_datetime(df_plot['date'])
df_plot = df_plot[df_plot['date'] >= three_months_ago].reset_index(drop=True)

ax5.plot(df_plot['date'], df_plot['close'], 
         linewidth=2, color='#4a9eff', alpha=0.7, label='Actual Price', zorder=1)

# 黃線: T+5 時刻的預測價格
aligned_df['pred_price_T5'] = aligned_df['close'] * (1 + aligned_df['pred_return'])

# 預測區間開始日期
future_trading_dates = future_dates
prediction_start = future_trading_dates[0]

# 歷史預測（黃色虛線）
historical_mask = aligned_df['date_T5'] < prediction_start
ax5.plot(aligned_df.loc[historical_mask, 'date_T5'], 
         aligned_df.loc[historical_mask, 'pred_price_T5'], 
         linewidth=2.5, color='#ffd43b', linestyle='--', alpha=0.9, label='Predicted Price @T+5', zorder=3)

# 未來5天預測（綠色/紅色）
future_color = '#51cf66' if pred_scalar > 0 else '#ff6b6b'

# 黃線最後一個點
last_yellow_date = aligned_df.loc[historical_mask, 'date_T5'].iloc[-1] if historical_mask.sum() > 0 else aligned_df['date_T5'].iloc[-1]
last_yellow_price = aligned_df.loc[historical_mask, 'pred_price_T5'].iloc[-1] if historical_mask.sum() > 0 else aligned_df['pred_price_T5'].iloc[-1]

# 繪製未來預測線
if future_trading_dates:
    ax5.plot([last_yellow_date, future_trading_dates[0]], 
             [last_yellow_price, future_prices[0]], 
             linewidth=2.5, color='#ffd43b', linestyle='--', alpha=0.9, zorder=3)
    
    ax5.plot(future_trading_dates, future_prices, 
             linewidth=3, color=future_color, linestyle='-', alpha=0.9, marker='o', markersize=8,
             label=f'Future 5-Day ({pred_scalar*100:+.2f}%)', zorder=4)
    
    ax5.axvspan(future_trading_dates[0], future_trading_dates[-1], alpha=0.1, color=future_color)

ax5.set_xlabel('Date', color='#cccccc')
ax5.set_ylabel('Price (HKD)', color='#cccccc')
ax5.set_title(f'Stock Price: Actual vs Predicted (T+5) | Future 5-Day Forecast: {pred_scalar*100:+.2f}%', color='#ffffff')
ax5.legend(facecolor='none', edgecolor='none', labelcolor='#cccccc', loc='upper left')
ax5.grid(True, alpha=0.3)

plt.tight_layout()

Path(config.paths.results_dir).mkdir(parents=True, exist_ok=True)
output_path = f"{config.paths.results_dir}/complete_backtest_results.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
print(f"\n✓ 回測圖表已保存: {output_path}")

# ============================================
# 保存結果
# ============================================
result = {
    'initial_capital': initial_capital,
    'final_capital': float(final_capital),
    'total_return_pct': float(total_return),
    'buyhold_return_pct': float(buy_hold_return),
    'excess_return_pct': float(total_return - buy_hold_return),
    'max_drawdown_pct': float(max_drawdown),
    'volatility_annual_pct': float(volatility),
    'sharpe_ratio': float(sharpe_ratio),
    'sortino_ratio': float(sortino_ratio),
    'calmar_ratio': float(calmar_ratio),
    'total_trades': len(buy_trades),
    'win_count': win_count,
    'loss_count': loss_count,
    'win_rate_pct': float(win_rate),
    'test_accuracy': float(accuracy_score(test_directions[:len(aligned_df)], aligned_df['pred_direction'])),
    'test_samples': len(aligned_df),
    'date_range': f"{aligned_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {aligned_df['date'].iloc[-1].strftime('%Y-%m-%d')}"
}

json_path = f"{config.paths.results_dir}/complete_backtest_results.json"
with open(json_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"✓ 回測數據已保存: {json_path}")

# 保存預測結果
prediction_result = {
    'latest_date': latest_date.strftime('%Y-%m-%d'),
    'latest_close': float(latest_close),
    'future_return_pct': float(pred_scalar * 100),
    'future_price_day5': float(future_prices[-1]),
    'future_prices_all': [float(p) for p in future_prices],
    'prediction_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
    'pred_direction': 'UP' if pred_scalar > 0 else 'DOWN'
}

pred_json_path = f"{config.paths.results_dir}/future_prediction.json"
with open(pred_json_path, 'w') as f:
    json.dump(prediction_result, f, indent=2)

print(f"✓ 預測結果已保存: {pred_json_path}")

print("\n" + "=" * 70)
print("【回測完成】")
print("=" * 70)
