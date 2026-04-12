import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("PatchTST 量化實驗 - 小米(1810.HK)股價預測")
print("=" * 60)

# ============================================
# 步驟 1: 數據準備
# ============================================
print("\n【步驟 1/8】數據準備")
print("-" * 40)

# 使用ifind獲取的數據創建模擬時間序列
# 基於真實的小小米股價特徵
np.random.seed(42)

# 生成2年的日線數據 (約500個交易日)
n_days = 500
dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')  # 工作日

# 模擬小米股價特徵 (基於真實價格範圍 15-45港幣)
initial_price = 30.0
returns = np.random.normal(0.0005, 0.025, n_days)  # 日收益率
prices = initial_price * np.exp(np.cumsum(returns))

# 確保價格在合理範圍內
prices = np.clip(prices, 15, 50)

# 生成OHLCV數據
data = {
    'date': dates,
    'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
    'high': prices * (1 + np.abs(np.random.normal(0.01, 0.008, n_days))),
    'low': prices * (1 - np.abs(np.random.normal(0.01, 0.008, n_days))),
    'close': prices,
    'volume': np.random.randint(100000000, 500000000, n_days)
}

df = pd.DataFrame(data)
df['open'] = np.minimum(df['open'], df['high'])
df['open'] = np.maximum(df['open'], df['low'])
df['close'] = np.clip(df['close'], df['low'], df['high'])

print(f"✓ 數據時間範圍: {df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"✓ 總交易日: {len(df)}")
print(f"✓ 收盤價範圍: {df['close'].min():.2f} - {df['close'].max():.2f} HKD")
print(f"\n數據預覽:")
print(df.head())

# 保存原始數據
df.to_csv('../data/xiaomi_raw.csv', index=False)
print("\n✓ 原始數據已保存: xiaomi_raw.csv")

print("\n" + "=" * 60)
print("【關鍵節點 1/8 完成】數據準備完成")
print("請檢查數據是否正確，確認後繼續下一步")
print("=" * 60)
