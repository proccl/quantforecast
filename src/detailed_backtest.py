import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

from step3_data_preprocessing import TimeSeriesDataset, RevIN
from step4_patchtst_model import PatchTST

print("=" * 70)
print("【詳細回測】PatchTST 策略 vs 買入持有")
print("=" * 70)

# 載入模型
checkpoint = torch.load('patchtst_best_model.pth', map_location='cpu')
model_config = checkpoint['model_config']
model = PatchTST(**model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ 模型已載入")

# 載入配置
with open('data_config.pkl', 'rb') as f:
    data_config = pickle.load(f)

SEQ_LEN = data_config['seq_len']
PRED_LEN = data_config['pred_len']
N_FEATURES = data_config['n_features']
feature_cols = data_config['feature_cols']
PATCH_LEN = model_config['patch_len']
STRIDE = model_config['stride']

print(f"模型期望特徵數: {model_config['n_features']}")
print(f"配置特徵數: {N_FEATURES}")
print(f"實際feature_cols數: {len(feature_cols)}")

# 載入數據
df = pd.read_csv('xiaomi_real.csv')
df['date'] = pd.to_datetime(df['date'])

# 使用與訓練時相同的特徵
def compute_features(df, feature_cols):
    """計算特徵，只返回需要的列"""
    # 趨勢特徵
    for span in [5, 10, 20]:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        df[f'ema_{span}_ratio'] = df['close'] / df[f'ema_{span}'] - 1
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    df['atr_ratio'] = df['atr_14'] / df['close']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 成交量
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # 收益率
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
    df['volatility_20d'] = df['return_1d'].rolling(window=20).std()
    
    # 目標
    df['target_return_5d'] = df['close'].pct_change(5).shift(-5)
    df['target_direction'] = (df['target_return_5d'] > 0).astype(int)
    
    return df

df = compute_features(df, feature_cols)
df_clean = df[feature_cols + ['target_return_5d', 'target_direction', 'date', 'close', 'open', 'high', 'low']].dropna()

# 獲取測試集
train_ratio = 0.7
val_ratio = 0.15
n_total = len(df_clean)
test_df = df_clean.iloc[int(n_total * (train_ratio + val_ratio)):].copy()

print(f"\n【測試集信息】")
print("-" * 50)
print(f"時間範圍: {test_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {test_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"交易日數: {len(test_df)}")
start_price = test_df['close'].iloc[0]
if isinstance(start_price, pd.Series):
    start_price = start_price.iloc[0]
start_price = float(start_price)

end_price = test_df['close'].iloc[-1]
if isinstance(end_price, pd.Series):
    end_price = end_price.iloc[0]
end_price = float(end_price)

print(f"起始價格: {start_price:.2f} HKD")
print(f"結束價格: {end_price:.2f} HKD")

# 創建測試數據集
test_dataset = TimeSeriesDataset(test_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)

print(f"可交易信號數: {len(test_dataset)}")

# ============================================
# 生成預測
# ============================================
print("\n【生成預測】")
print("-" * 50)

predictions = []
dates = []
actual_returns = []
actual_directions = []

with torch.no_grad():
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        x = sample['x'].unsqueeze(0)  # [1, seq_len, features]
        
        pred = model(x)
        pred_return = pred.squeeze().item()
        pred_direction = 1 if pred_return > 0 else 0
        
        # 獲取對應日期和實際收益
        # 預測是在序列最後一天做出的
        seq_end_idx = i + SEQ_LEN - 1
        pred_date = test_df.iloc[seq_end_idx]['date']
        actual_ret = sample['y_return'].item()
        actual_dir = sample['y_direction'].item()
        
        predictions.append({
            'date': pred_date,
            'pred_return': pred_return,
            'pred_direction': pred_direction,
            'actual_return_5d': actual_ret,
            'actual_direction': actual_dir,
            'close_price': test_df.iloc[seq_end_idx]['close']
        })

pred_df = pd.DataFrame(predictions)
pred_df['date'] = pd.to_datetime(pred_df['date'])
pred_df = pred_df.sort_values('date').reset_index(drop=True)

print(f"✓ 生成 {len(pred_df)} 個預測")

# 統計預測分布
print(f"\n預測分布:")
print(f"  看漲: {sum(pred_df['pred_direction'])} ({sum(pred_df['pred_direction'])/len(pred_df):.1%})")
print(f"  看跌: {len(pred_df) - sum(pred_df['pred_direction'])} ({(len(pred_df) - sum(pred_df['pred_direction']))/len(pred_df):.1%})")

# ============================================
# 策略回測
# ============================================
print("\n" + "=" * 70)
print("【策略回測】")
print("=" * 70)

# 參數
initial_capital = 100000  # 初始資金 10萬 HKD
position_size = 1.0       # 全倉交易
commission = 0.001        # 手續費 0.1%

# 策略：根據預測方向交易
# 看漲 -> 買入持有5天
# 看跌 -> 空倉（或做空，這裡簡化為空倉）

capital = initial_capital
position = 0  # 0 = 空倉, 1 = 多倉
trades = []
equity_curve = []

for i, row in pred_df.iterrows():
    date = row['date']
    price = row['close_price']
    pred_dir = row['pred_direction']
    actual_ret = row['actual_return_5d']
    
    # 簡化策略：看漲時持有，看跌時空倉
    if pred_dir == 1:  # 看漲
        strategy_return = actual_ret  # 獲得實際收益
    else:  # 看跌
        strategy_return = 0  # 空倉，無收益
    
    # 扣除手續費（只有在換倉時）
    # 簡化：每次預測都當作一次交易決策
    if i > 0:
        prev_dir = pred_df.iloc[i-1]['pred_direction']
        if pred_dir != prev_dir:  # 換倉
            strategy_return -= commission * 2  # 買入+賣出手續費
    
    capital *= (1 + strategy_return)
    
    trades.append({
        'date': date,
        'price': price,
        'pred_direction': pred_dir,
        'actual_return': actual_ret,
        'strategy_return': strategy_return,
        'capital': capital
    })

trades_df = pd.DataFrame(trades)

# ============================================
# 基準對比（買入持有）
# ============================================
start_price = pred_df['close_price'].iloc[0]
end_price = pred_df['close_price'].iloc[-1]
buy_hold_return = (end_price / start_price) - 1
buy_hold_capital = initial_capital * (1 + buy_hold_return)

print(f"\n【買入持有基準】")
print("-" * 50)
print(f"起始價格: {start_price:.2f} HKD")
print(f"結束價格: {end_price:.2f} HKD")
print(f"總收益率: {buy_hold_return:.2%}")
print(f"最終資金: {buy_hold_capital:,.0f} HKD")

# ============================================
# 策略表現
# ============================================
final_capital = trades_df['capital'].iloc[-1]
total_return = (final_capital / initial_capital) - 1
trading_days = len(trades_df)

print(f"\n【策略表現】")
print("-" * 50)
print(f"初始資金: {initial_capital:,.0f} HKD")
print(f"最終資金: {final_capital:,.0f} HKD")
print(f"總收益率: {total_return:.2%}")
print(f"超額收益: {total_return - buy_hold_return:.2%}")
print(f"交易天數: {trading_days}")

# 計算風險指標
daily_returns = trades_df['strategy_return'].values

# 年化收益 (按252交易日)
annual_return = (1 + total_return) ** (252 / trading_days) - 1

# 波動率
volatility = np.std(daily_returns) * np.sqrt(252)

# Sharpe Ratio (假設無風險利率0%)
sharpe_ratio = annual_return / volatility if volatility > 0 else 0

# Max Drawdown
cumulative = np.cumprod(1 + daily_returns)
running_max = np.maximum.accumulate(cumulative)
drawdown = (cumulative - running_max) / running_max
max_drawdown = np.min(drawdown)

# Calmar Ratio
calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

# Win Rate
positive_trades = np.sum(daily_returns > 0)
win_rate = positive_trades / len(daily_returns)

# Profit Factor
gross_profit = np.sum(daily_returns[daily_returns > 0])
gross_loss = abs(np.sum(daily_returns[daily_returns < 0]))
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

# 最大連續虧損
max_consecutive_losses = 0
current_losses = 0
for ret in daily_returns:
    if ret < 0:
        current_losses += 1
        max_consecutive_losses = max(max_consecutive_losses, current_losses)
    else:
        current_losses = 0

print(f"\n【風險指標】")
print("-" * 50)
print(f"年化收益: {annual_return:.2%}")
print(f"年化波動: {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Calmar Ratio: {calmar_ratio:.4f}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Profit Factor: {profit_factor:.4f}")
print(f"Max Consecutive Losses: {max_consecutive_losses}")

# ============================================
# 月度統計
# ============================================
print(f"\n【月度收益統計】")
print("-" * 50)

trades_df['month'] = trades_df['date'].dt.to_period('M')
monthly_stats = trades_df.groupby('month').agg({
    'strategy_return': 'sum',
    'date': 'count'
}).rename(columns={'date': 'trades', 'strategy_return': 'monthly_return'})

monthly_stats['cumulative'] = (1 + monthly_stats['monthly_return']).cumprod() - 1

print(monthly_stats.to_string())

# ============================================
# 可視化
# ============================================
print(f"\n【生成可視化】")
print("-" * 50)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# 1. 資金曲線
ax1 = axes[0, 0]
buy_hold_curve = initial_capital * (1 + np.cumsum(pred_df['actual_return_5d']))
ax1.plot(trades_df['date'], trades_df['capital'], label='Strategy', linewidth=2, color='blue')
ax1.plot(trades_df['date'], buy_hold_curve, label='Buy & Hold', linewidth=2, color='gray', linestyle='--')
ax1.axhline(y=initial_capital, color='red', linestyle=':', alpha=0.5)
ax1.set_ylabel('Capital (HKD)')
ax1.set_title('Equity Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 回撤曲線
ax2 = axes[0, 1]
cumulative_pct = (trades_df['capital'] / initial_capital - 1) * 100
dd_pct = drawdown * 100
ax2.fill_between(trades_df['date'], dd_pct, 0, alpha=0.5, color='red')
ax2.plot(trades_df['date'], dd_pct, color='darkred', linewidth=1)
ax2.set_ylabel('Drawdown (%)')
ax2.set_title(f'Max Drawdown: {max_drawdown:.2%}')
ax2.grid(True, alpha=0.3)

# 3. 日收益分布
ax3 = axes[1, 0]
ax3.hist(daily_returns * 100, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.axvline(x=np.mean(daily_returns) * 100, color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(daily_returns)*100:.2f}%')
ax3.set_xlabel('Daily Return (%)')
ax3.set_ylabel('Frequency')
ax3.set_title('Return Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 月度收益
ax4 = axes[1, 1]
monthly_returns_pct = monthly_stats['monthly_return'].values * 100
colors = ['green' if r > 0 else 'red' for r in monthly_returns_pct]
ax4.bar(range(len(monthly_returns_pct)), monthly_returns_pct, color=colors, alpha=0.7, edgecolor='black')
ax4.axhline(y=0, color='black', linewidth=1)
ax4.set_xlabel('Month')
ax4.set_ylabel('Monthly Return (%)')
ax4.set_title('Monthly Returns')
ax4.set_xticks(range(0, len(monthly_returns_pct), 2))
ax4.grid(True, alpha=0.3, axis='y')

# 5. 預測 vs 實際散點圖
ax5 = axes[2, 0]
ax5.scatter(pred_df['pred_return'] * 100, pred_df['actual_return_5d'] * 100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax5.plot([-20, 20], [-20, 20], 'r--', label='Perfect')
ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax5.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax5.set_xlabel('Predicted Return (%)')
ax5.set_ylabel('Actual Return (%)')
ax5.set_title('Predicted vs Actual Returns')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. 交易信號時間軸
ax6 = axes[2, 1]
colors_signal = ['green' if d == 1 else 'red' for d in pred_df['pred_direction']]
ax6.scatter(pred_df['date'], pred_df['pred_direction'], c=colors_signal, alpha=0.6, s=50)
ax6.set_ylabel('Signal (1=Long, 0=Cash)')
ax6.set_title('Trading Signals')
ax6.set_ylim(-0.2, 1.2)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('detailed_backtest.png', dpi=150, bbox_inches='tight')
print("✓ 詳細回測圖表: detailed_backtest.png")

# ============================================
# 總結
# ============================================
print("\n" + "=" * 70)
print("【回測總結】")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│ 策略表現                                                    │
├─────────────────────────────────────────────────────────────┤
│ 總收益率:     {total_return:>8.2%}  (買入持有: {buy_hold_return:>8.2%})          │
│ 超額收益:     {total_return - buy_hold_return:>8.2%}                              │
│ 年化收益:     {annual_return:>8.2%}                              │
│ 年化波動:     {volatility:>8.2%}                              │
│ Sharpe:       {sharpe_ratio:>8.4f}                              │
│ Max DD:       {max_drawdown:>8.2%}                              │
│ Win Rate:     {win_rate:>8.2%}                              │
│ Profit Factor:{profit_factor:>8.4f}                              │
└─────────────────────────────────────────────────────────────┘
""")

# 保存回測結果
backtest_results = {
    'initial_capital': initial_capital,
    'final_capital': final_capital,
    'total_return': total_return,
    'buy_hold_return': buy_hold_return,
    'excess_return': total_return - buy_hold_return,
    'annual_return': annual_return,
    'volatility': volatility,
    'sharpe_ratio': sharpe_ratio,
    'max_drawdown': max_drawdown,
    'calmar_ratio': calmar_ratio,
    'win_rate': win_rate,
    'profit_factor': profit_factor,
    'trades': len(trades_df),
    'test_period': {
        'start': test_df['date'].iloc[0].strftime('%Y-%m-%d'),
        'end': test_df['date'].iloc[-1].strftime('%Y-%m-%d')
    }
}

import json
with open('backtest_results.json', 'w') as f:
    json.dump(backtest_results, f, indent=2)

print("✓ 回測結果已保存: backtest_results.json")
print("=" * 70)
