import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

print("=" * 70)
print("【真實數據序列長度搜索】小米(1810.HK)")
print("=" * 70)

# 讀取真實數據
df = pd.read_csv('../data/xiaomi_real.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"\n數據概覽:")
print(f"  總交易日: {len(df)} (~{len(df)/250:.1f}年)")
print(f"  價格範圍: {df['close'].min():.2f} - {df['close'].max():.2f} HKD")

# ============================================
# 特徵工程 (簡化版)
# ============================================
print("\n【1】特徵工程")
print("-" * 50)

# 技術指標
df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
df['ema_60'] = df['close'].ewm(span=60, adjust=False).mean()

df['ema_5_ratio'] = df['close'] / df['ema_5'] - 1
df['ema_10_ratio'] = df['close'] / df['ema_10'] - 1
df['ema_20_ratio'] = df['close'] / df['ema_20'] - 1
df['ema_60_ratio'] = df['close'] / df['ema_60'] - 1

# MACD
df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_hist'] = df['macd'] - df['macd_signal']

# RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi_14'] = 100 - (100 / (1 + rs))

# ATR
df['tr1'] = df['high'] - df['low']
df['tr2'] = abs(df['high'] - df['close'].shift(1))
df['tr3'] = abs(df['low'] - df['close'].shift(1))
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['atr_14'] = df['tr'].rolling(window=14).mean()
df['atr_ratio'] = df['atr_14'] / df['close']

# 成交量
df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma_20']

# 收益率
df['return_1d'] = df['close'].pct_change(1)
df['return_5d'] = df['close'].pct_change(5)
df['target_return_5d'] = df['close'].pct_change(5).shift(-5)
df['target_direction'] = (df['target_return_5d'] > 0).astype(int)

# 選擇特徵
feature_cols = ['open', 'high', 'low', 'close', 'volume',
                'ema_5_ratio', 'ema_10_ratio', 'ema_20_ratio', 'ema_60_ratio',
                'macd', 'macd_hist', 'atr_ratio', 'rsi_14', 'volume_ratio', 
                'return_1d', 'return_5d']

df_clean = df[feature_cols + ['target_return_5d', 'target_direction', 'date']].dropna()
print(f"✓ 清洗後樣本: {len(df_clean)}")

# ============================================
# 序列長度搜索
# ============================================
print("\n【2】序列長度搜索")
print("-" * 50)

# 數據劃分
train_ratio = 0.7
val_ratio = 0.15
n_total = len(df_clean)
n_train = int(n_total * train_ratio)

train_df = df_clean.iloc[:n_train].copy()
val_df = df_clean.iloc[n_train:].copy()

print(f"訓練集: {len(train_df)} ({train_df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {train_df['date'].iloc[-1].strftime('%Y-%m-%d')})")
print(f"驗證集: {len(val_df)} ({val_df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {val_df['date'].iloc[-1].strftime('%Y-%m-%d')})")

# 測試序列長度 (根據800天數據調整)
seq_lengths = [20, 40, 60, 96, 126, 192, 252]

print(f"\n測試長度: {seq_lengths}")
print(f"目標: 預測5日收益率方向")

# 設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 簡化版PatchTST
class SimplePatchTST(nn.Module):
    def __init__(self, n_features, seq_len, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 簡化: 直接投影
        self.input_proj = nn.Linear(n_features, d_model)
        
        # LSTM代替Transformer (更快)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True, dropout=0.1)
        
        # 輸出層
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # 取最後一個時間步
        return self.fc(x)

# 數據集
class SimpleDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len):
        self.data = df[feature_cols].values
        self.target = df['target_return_5d'].values
        self.direction = df['target_direction'].values
        self.seq_len = seq_len
        self.n_samples = len(df) - seq_len
    
    def __len__(self):
        return max(0, self.n_samples)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.target[idx+self.seq_len-1]
        d = self.direction[idx+self.seq_len-1]
        return {
            'x': torch.FloatTensor(x),
            'y': torch.FloatTensor([y]),
            'd': torch.LongTensor([d])
        }

results = []

for seq_len in seq_lengths:
    print(f"\n{'='*50}")
    print(f"測試序列長度: {seq_len} (~{seq_len//20}個月)")
    print(f"{'='*50}")
    
    # 檢查樣本數
    train_dataset = SimpleDataset(train_df, feature_cols, seq_len)
    val_dataset = SimpleDataset(val_df, feature_cols, seq_len)
    
    if len(train_dataset) < 50 or len(val_dataset) < 20:
        print(f"⚠ 樣本不足 (train: {len(train_dataset)}, val: {len(val_dataset)})")
        continue
    
    print(f"訓練樣本: {len(train_dataset)}, 驗證樣本: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 模型
    model = SimplePatchTST(len(feature_cols), seq_len).to(device)
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 訓練
    best_val_acc = 0
    epochs = 20
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        
        # 驗證
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                d = batch['d'].to(device)
                pred = model(x)
                pred_d = (pred.squeeze() > 0).long()
                val_correct += (pred_d == d).sum().item()
                val_total += d.size(0)
        
        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    print(f"✓ 最佳驗證方向準確率: {best_val_acc:.2%}")
    
    results.append({
        'seq_len': seq_len,
        'months': seq_len // 20,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'best_val_acc': best_val_acc
    })

# 結果
print("\n" + "=" * 70)
print("【搜索結果】")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('best_val_acc', ascending=False)
print("\n按驗證準確率排序:")
print(results_df.to_string(index=False))

# 可視化
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

colors = ['green' if acc > 0.52 else 'orange' if acc > 0.50 else 'red' 
          for acc in results_df['best_val_acc']]
bars = ax.bar(range(len(results_df)), results_df['best_val_acc'] * 100, 
              color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Random (50%)')
ax.axhline(y=52, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Usable (52%)')
ax.axhline(y=55, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Good (55%)')

ax.set_xticks(range(len(results_df)))
ax.set_xticklabels([f"{int(r['seq_len'])}\n(~{int(r['months'])}M)" for _, r in results_df.iterrows()], fontsize=11)
ax.set_ylabel('Validation Directional Accuracy (%)', fontsize=12)
ax.set_xlabel('Sequence Length (Trading Days)', fontsize=12)
ax.set_title('Xiaomi (1810.HK) - Optimal Lookback Window Search\n(Real Data: 2023-2026)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(45, max(results_df['best_val_acc'] * 100) + 5)

# 添加數值標籤
for i, (bar, acc) in enumerate(zip(bars, results_df['best_val_acc'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('seq_len_search_real_data.png', dpi=150, bbox_inches='tight')
print("\n✓ 結果圖表已保存: seq_len_search_real_data.png")

# 推薦
best = results_df.iloc[0]
print(f"\n{'='*70}")
print("【推薦配置】")
print(f"{'='*70}")
print(f"\n🎯 最優序列長度: {int(best['seq_len'])} 天 (~{int(best['months'])}個月)")
print(f"   驗證方向準確率: {best['best_val_acc']:.2%}")

if best['best_val_acc'] >= 0.55:
    print(f"\n✅ 表現優秀！該配置可用於實盤")
elif best['best_val_acc'] >= 0.52:
    print(f"\n⚠️  表現一般，可用於實盤但需謹慎")
else:
    print(f"\n❌ 表現不佳，不建議用於實盤")
    print(f"   建議: 調整特徵工程或嘗試其他模型")

print(f"\n{'='*70}")
