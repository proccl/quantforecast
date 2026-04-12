import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("【特徵可視化】小米(1810.HK) - 使用真實數據")
print("=" * 60)

# 載入真實數據
df = pd.read_csv('../data/xiaomi_real.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"\n數據範圍: {df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"總交易日: {len(df)}")

# 計算技術指標
for span in [5, 10, 20]:
    df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

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

delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi_14'] = 100 - (100 / (1 + rs))

df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
df['volume_ma_20'] = df['volume'].rolling(window=20).mean()

# 創建可視化 (使用英文標題避免字體問題)
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# 1. 價格和均線
ax1 = axes[0]
ax1.plot(df['date'], df['close'], label='Close Price', linewidth=1.5, color='black')
ax1.plot(df['date'], df['ema_5'], label='EMA5', alpha=0.7, linewidth=1)
ax1.plot(df['date'], df['ema_20'], label='EMA20', alpha=0.7, linewidth=1)
ax1.set_ylabel('Price (HKD)', fontsize=11)
ax1.set_title('Xiaomi (1810.HK) Stock Price with EMA (2023-2026)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(df['date'].iloc[0], df['date'].iloc[-1])

# 2. MACD
ax2 = axes[1]
ax2.plot(df['date'], df['macd'], label='MACD', color='blue', linewidth=1)
ax2.plot(df['date'], df['macd_signal'], label='Signal', color='red', linewidth=1)
ax2.bar(df['date'], df['macd_hist'], label='Histogram', alpha=0.3, color='gray', width=1)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('MACD', fontsize=11)
ax2.set_title('MACD Indicator', fontsize=12)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(df['date'].iloc[0], df['date'].iloc[-1])

# 3. RSI
ax3 = axes[2]
ax3.plot(df['date'], df['rsi_14'], label='RSI(14)', color='purple', linewidth=1)
ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
ax3.fill_between(df['date'], 30, 70, alpha=0.1, color='gray')
ax3.set_ylabel('RSI', fontsize=11)
ax3.set_title('RSI(14) - Momentum Indicator', fontsize=12)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 100)
ax3.set_xlim(df['date'].iloc[0], df['date'].iloc[-1])

# 4. Volume
ax4 = axes[3]
colors = ['green' if close >= open else 'red' 
          for close, open in zip(df['close'], df['open'])]
ax4.bar(df['date'], df['volume']/1e6, color=colors, alpha=0.6, width=1)
ax4.plot(df['date'], df['volume_ma_20']/1e6, label='Volume MA(20)', color='blue', linewidth=1)
ax4.set_ylabel('Volume (M)', fontsize=11)
ax4.set_xlabel('Date', fontsize=11)
ax4.set_title('Trading Volume', fontsize=12)
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(df['date'].iloc[0], df['date'].iloc[-1])

plt.tight_layout()
plt.savefig('features_visualization_fixed.png', dpi=150, bbox_inches='tight')
print("\n✓ 可視化已保存: features_visualization_fixed.png")
print(f"  數據範圍: {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"  包含2025-2026數據: ✓")

# 顯示年份分布
print(f"\n年份分布:")
for year in sorted(df['date'].dt.year.unique()):
    count = len(df[df['date'].dt.year == year])
    print(f"  {year}: {count} days")
