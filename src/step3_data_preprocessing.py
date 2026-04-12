import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("【步驟 3/8】數據預處理 - 使用真實數據 + RevIN")
print("=" * 60)

# 讀取真實數據
df = pd.read_csv('../data/xiaomi_real.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"\n【3.0】載入真實數據")
print("-" * 40)
print(f"✓ 數據來源: 小米(1810.HK)真實股價")
print(f"✓ 時間範圍: {df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"✓ 總交易日: {len(df)}")

# ============================================
# 特徵工程
# ============================================
print("\n【3.1】特徵工程")
print("-" * 40)

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

# 波動率
df['tr1'] = df['high'] - df['low']
df['tr2'] = abs(df['high'] - df['close'].shift(1))
df['tr3'] = abs(df['low'] - df['close'].shift(1))
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr_14'] = df['tr'].rolling(window=14).mean()
df['atr_ratio'] = df['atr_14'] / df['close']

# 動量
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

# 目標變量
df['target_return_5d'] = df['close'].pct_change(5).shift(-5)
df['target_direction'] = (df['target_return_5d'] > 0).astype(int)

print("✓ 技術指標計算完成")

# 選擇特徵
feature_cols = ['open', 'high', 'low', 'close', 'volume',
                'ema_5_ratio', 'ema_10_ratio', 'ema_20_ratio',
                'macd', 'macd_hist', 'atr_ratio', 'rsi_14',
                'volume_ratio', 'obv', 'return_1d', 'return_5d', 'volatility_20d']

# 清洗
df_clean = df[feature_cols + ['target_return_5d', 'target_direction', 'date']].dropna()
print(f"✓ 清洗後樣本: {len(df_clean)}")

# ============================================
# 數據劃分
# ============================================
print("\n【3.2】時間序列數據集劃分")
print("-" * 40)

train_ratio = 0.7
val_ratio = 0.15
n_total = len(df_clean)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

train_df = df_clean.iloc[:n_train].copy()
val_df = df_clean.iloc[n_train:n_train+n_val].copy()
test_df = df_clean.iloc[n_train+n_val:].copy()

print(f"✓ 總樣本數: {n_total}")
print(f"✓ 訓練集: {len(train_df)} (70%) - {train_df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {train_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"✓ 驗證集: {len(val_df)} (15%) - {val_df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {val_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"✓ 測試集: {len(test_df)} (15%) - {test_df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {test_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"✓ 特徵維度: {len(feature_cols)}")

# ============================================
# RevIN 歸一化
# ============================================
print("\n【3.3】RevIN (Reversible Instance Normalization)")
print("-" * 40)

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x
    
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
    
    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x
    
    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps*0)
        x = x * self.stdev
        x = x + self.mean
        return x

print("✓ RevIN 模塊定義完成")

# ============================================
# 創建數據集 - 使用 20 天序列
# ============================================
print("\n【3.4】創建時間序列數據集")
print("-" * 40)

# 使用搜索得到的最優序列長度: 20天
SEQ_LEN = 20       # 20個交易日 (~1個月)
PRED_LEN = 5       # 預測5天收益率
PATCH_LEN = 5      # patch 長度 (調整為適合短序列)
STRIDE = 2         # patch 步長
BATCH_SIZE = 64

print(f"✓ 序列長度 (seq_len): {SEQ_LEN} 天 (~1個月)")
print(f"✓ 預測長度 (pred_len): {PRED_LEN} 天")
print(f"✓ Patch 長度: {PATCH_LEN}")
print(f"✓ Patch 步長: {STRIDE}")

class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len, pred_len, patch_len, stride):
        self.data = df[feature_cols].values
        self.target = df['target_return_5d'].values
        self.direction = df['target_direction'].values
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_samples = len(df) - seq_len - pred_len + 1
        
    def __len__(self):
        return max(0, self.n_samples)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y_return = self.target[idx+self.seq_len-1]
        y_direction = self.direction[idx+self.seq_len-1]
        return {
            'x': torch.FloatTensor(x),
            'y_return': torch.FloatTensor([y_return]),
            'y_direction': torch.LongTensor([y_direction])
        }

# 創建數據集
train_dataset = TimeSeriesDataset(train_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
val_dataset = TimeSeriesDataset(val_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
test_dataset = TimeSeriesDataset(test_df, feature_cols, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)

print(f"\n✓ 訓練樣本數: {len(train_dataset)}")
print(f"✓ 驗證樣本數: {len(val_dataset)}")
print(f"✓ 測試樣本數: {len(test_dataset)}")
print(f"✓ Batch Size: {BATCH_SIZE}")

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 測試 RevIN
sample_batch = next(iter(train_loader))
x_sample = sample_batch['x']
print(f"\n輸入形狀: {x_sample.shape} [batch, seq_len, features]")

revin = RevIN(num_features=len(feature_cols))
normalized = revin(x_sample, mode='norm')
denormalized = revin(normalized, mode='denorm')

reconstruction_error = torch.abs(x_sample - denormalized).mean()
print(f"✓ RevIN 可逆性誤差: {reconstruction_error.item():.8f}")

# 保存配置
data_info = {
    'feature_cols': feature_cols,
    'seq_len': SEQ_LEN,
    'pred_len': PRED_LEN,
    'patch_len': PATCH_LEN,
    'stride': STRIDE,
    'batch_size': BATCH_SIZE,
    'n_features': len(feature_cols),
    'train_dates': (train_df['date'].iloc[0].strftime('%Y-%m-%d'), train_df['date'].iloc[-1].strftime('%Y-%m-%d')),
    'val_dates': (val_df['date'].iloc[0].strftime('%Y-%m-%d'), val_df['date'].iloc[-1].strftime('%Y-%m-%d')),
    'test_dates': (test_df['date'].iloc[0].strftime('%Y-%m-%d'), test_df['date'].iloc[-1].strftime('%Y-%m-%d'))
}

with open('data_config.pkl', 'wb') as f:
    pickle.dump(data_info, f)

print("\n✓ 數據配置已保存: data_config.pkl")

print("\n" + "=" * 60)
print("【關鍵節點 3/8 完成】數據預處理完成")
print(f"✓ 使用真實數據，序列長度: {SEQ_LEN}天")
print("=" * 60)
