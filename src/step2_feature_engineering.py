import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("【步驟 2/8】特徵工程 - 根據最佳實踐文檔")
print("=" * 60)

# 載入數據
df = pd.read_csv('xiaomi_raw.csv')
df['date'] = pd.to_datetime(df['date'])

print("\n【2.1】技術指標特徵計算")
print("-" * 40)

# 1. 趨勢特徵 (根據文檔: EMA, MACD 重要性 ★★★★★)
for span in [5, 10, 20, 60]:
    df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    df[f'ema_{span}_ratio'] = df['close'] / df[f'ema_{span}'] - 1

# MACD
df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

print("✓ 趨勢特徵: EMA(5,10,20,60), MACD")

# 2. 波動率特徵 (ATR, Bollinger Bands 重要性 ★★★★★)
df['tr1'] = df['high'] - df['low']
df['tr2'] = abs(df['high'] - df['close'].shift(1))
df['tr3'] = abs(df['low'] - df['close'].shift(1))
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr_14'] = df['tr'].rolling(window=14).mean()
df['atr_ratio'] = df['atr_14'] / df['close']

# Bollinger Bands
df['bb_middle'] = df['close'].rolling(window=20).mean()
df['bb_std'] = df['close'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

print("✓ 波動率特徵: ATR(14), Bollinger Bands")

# 3. 動量特徵 (RSI 重要性 ★★★★☆)
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi_14'] = 100 - (100 / (1 + rs))

# Stochastic
low_min = df['low'].rolling(window=14).min()
high_max = df['high'].rolling(window=14).max()
df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

print("✓ 動量特徵: RSI(14), Stochastic")

# 4. 成交量特徵 (OBV 重要性 ★★★★☆)
df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma_20']

print("✓ 成交量特徵: OBV, Volume Ratio")

# 5. 多時間尺度收益率
df['return_1d'] = df['close'].pct_change(1)
df['return_5d'] = df['close'].pct_change(5)
df['return_10d'] = df['close'].pct_change(10)
df['return_20d'] = df['close'].pct_change(20)
df['volatility_20d'] = df['return_1d'].rolling(window=20).std()

print("✓ 多時間尺度收益率: 1d, 5d, 10d, 20d")

# 6. 目標變量 (預測未來5天收益率)
df['target_return_5d'] = df['close'].pct_change(5).shift(-5)
df['target_direction'] = (df['target_return_5d'] > 0).astype(int)

print("\n【2.2】特徵選擇")
print("-" * 40)

# 選擇特徵列 (根據最佳實踐文檔推薦)
feature_cols = [
    # 價格相關
    'open', 'high', 'low', 'close', 'volume',
    # 趨勢
    'ema_5_ratio', 'ema_10_ratio', 'ema_20_ratio', 'ema_60_ratio',
    'macd', 'macd_hist',
    # 波動率
    'atr_ratio', 'bb_position',
    # 動量
    'rsi_14', 'stoch_k',
    # 成交量
    'volume_ratio', 'obv',
    # 收益率
    'return_1d', 'return_5d', 'volatility_20d'
]

# 去除缺失值
df_clean = df[feature_cols + ['target_return_5d', 'target_direction', 'date']].dropna()

print(f"✓ 選定特徵數: {len(feature_cols)}")
print(f"✓ 特徵列表: {feature_cols}")
print(f"✓ 清洗後樣本數: {len(df_clean)} (原{len(df)}, 去除缺失值)")

# 保存特徵數據
df_clean.to_csv('xiaomi_features.csv', index=False)
print("\n✓ 特徵數據已保存: xiaomi_features.csv")

# 可視化特徵
print("\n【2.3】特徵可視化")
print("-" * 40)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# 價格和趨勢
ax1 = axes[0, 0]
ax1.plot(df['date'], df['close'], label='Close', alpha=0.8)
ax1.plot(df['date'], df['ema_20'], label='EMA20', alpha=0.7)
ax1.fill_between(df['date'], df['bb_upper'], df['bb_lower'], alpha=0.2, label='Bollinger Bands')
ax1.set_title('小米股價與趨勢指標')
ax1.set_ylabel('Price (HKD)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# MACD
ax2 = axes[0, 1]
ax2.plot(df['date'], df['macd'], label='MACD', color='blue')
ax2.plot(df['date'], df['macd_signal'], label='Signal', color='red')
ax2.bar(df['date'], df['macd_hist'], label='Histogram', alpha=0.5, color='gray')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_title('MACD 指標')
ax2.legend()
ax2.grid(True, alpha=0.3)

# RSI
ax3 = axes[1, 0]
ax3.plot(df['date'], df['rsi_14'], label='RSI(14)', color='purple')
ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought(70)')
ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold(30)')
ax3.fill_between(df['date'], 30, 70, alpha=0.1)
ax3.set_title('RSI 動量指標')
ax3.set_ylabel('RSI')
ax3.set_ylim(0, 100)
ax3.legend()
ax3.grid(True, alpha=0.3)

# ATR
ax4 = axes[1, 1]
ax4.plot(df['date'], df['atr_ratio'] * 100, label='ATR Ratio(%)', color='orange')
ax4.set_title('ATR 波動率指標')
ax4.set_ylabel('ATR (%)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 成交量
ax5 = axes[2, 0]
colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
          for i in range(len(df))]
ax5.bar(df['date'], df['volume']/1e6, color=colors, alpha=0.6, width=0.8)
ax5.plot(df['date'], df['volume_ma_20']/1e6, color='blue', label='Volume MA20')
ax5.set_title('成交量')
ax5.set_ylabel('Volume (M)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 目標變量分布
ax6 = axes[2, 1]
ax6.hist(df_clean['target_return_5d'] * 100, bins=50, alpha=0.7, edgecolor='black')
ax6.axvline(x=0, color='red', linestyle='--', label='Zero Line')
ax6.set_title('5日收益率分布 (目標變量)')
ax6.set_xlabel('Return (%)')
ax6.set_ylabel('Frequency')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('features_visualization.png', dpi=150, bbox_inches='tight')
print("✓ 特徵可視化已保存: features_visualization.png")

# 特徵相關性分析
correlation = df_clean[feature_cols + ['target_return_5d']].corr()['target_return_5d'].sort_values(ascending=False)
print(f"\n【2.4】特徵與目標的相關性 (Top 10):")
print("-" * 40)
for feat, corr in correlation.head(11).items():
    if feat != 'target_return_5d':
        print(f"  {feat:20s}: {corr:+.4f}")

print("\n" + "=" * 60)
print("【關鍵節點 2/8 完成】特徵工程完成")
print("請檢查特徵是否合理，確認後繼續下一步")
print("=" * 60)
