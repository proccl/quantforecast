import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import itertools
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from step3_data_preprocessing import TimeSeriesDataset, RevIN
from step4_patchtst_model import PatchTST

print("=" * 70)
print("【超參數優化搜索】PatchTST for Xiaomi (1810.HK)")
print("=" * 70)

# 載入數據配置
import pickle
with open('data_config.pkl', 'rb') as f:
    data_config = pickle.load(f)

SEQ_LEN = data_config['seq_len']  # 20
PRED_LEN = data_config['pred_len']  # 5
N_FEATURES = data_config['n_features']  # 17
feature_cols = data_config['feature_cols']
BATCH_SIZE = 64

print(f"\n固定配置:")
print(f"  序列長度: {SEQ_LEN} 天")
print(f"  預測長度: {PRED_LEN} 天")
print(f"  特徵數: {N_FEATURES}")

# 載入數據
df = pd.read_csv('xiaomi_real.csv')
df['date'] = pd.to_datetime(df['date'])

# 特徵工程
def compute_features(df):
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
    
    return df

df = compute_features(df)
df_clean = df[feature_cols + ['target_return_5d', 'target_direction', 'date']].dropna()

# 數據劃分
train_ratio = 0.7
val_ratio = 0.15
n_total = len(df_clean)
n_train = int(n_total * train_ratio)

train_df = df_clean.iloc[:n_train].copy()
val_df = df_clean.iloc[n_train:n_train+int(n_total*val_ratio)].copy()

print(f"\n數據集:")
print(f"  訓練集: {len(train_df)} 樣本")
print(f"  驗證集: {len(val_df)} 樣本")

# ============================================
# 超參數搜索空間
# ============================================
print("\n" + "=" * 70)
print("【超參數搜索空間】")
print("=" * 70)

param_grid = {
    'd_model': [32, 64, 128],           # 模型維度
    'n_heads': [2, 4, 8],                # 注意力頭數
    'n_layers': [1, 2, 3],               # Transformer層數
    'dropout': [0.1, 0.2],               # Dropout率
    'patch_len': [4, 5, 10],             # Patch長度
    'learning_rate': [1e-3, 5e-4],       # 學習率
}

print(f"搜索參數:")
for k, v in param_grid.items():
    print(f"  {k}: {v}")

# 生成所有組合 (限制總數)
all_combinations = list(itertools.product(*param_grid.values()))
print(f"\n總組合數: {len(all_combinations)}")

# 為了效率，隨機採樣或限制數量
np.random.seed(42)
max_trials = 20  # 最多測試20組
if len(all_combinations) > max_trials:
    selected_indices = np.random.choice(len(all_combinations), max_trials, replace=False)
    all_combinations = [all_combinations[i] for i in selected_indices]
    print(f"隨機採樣: {max_trials} 組")

# ============================================
# 訓練函數
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n設備: {device}")

def train_and_evaluate(config, train_df, val_df, max_epochs=30, patience=10):
    """訓練單個配置並返回驗證指標"""
    
    patch_len = config['patch_len']
    stride = max(1, patch_len // 2)  # stride = patch_len / 2
    
    # 確保patch配置有效
    if patch_len > SEQ_LEN:
        return None
    
    n_patches = (SEQ_LEN - patch_len) // stride + 1
    if n_patches < 2:
        return None
    
    # 創建數據集
    train_dataset = TimeSeriesDataset(train_df, feature_cols, SEQ_LEN, PRED_LEN, patch_len, stride)
    val_dataset = TimeSeriesDataset(val_df, feature_cols, SEQ_LEN, PRED_LEN, patch_len, stride)
    
    if len(train_dataset) < 50 or len(val_dataset) < 20:
        return None
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 模型配置
    model_config = {
        'n_features': N_FEATURES,
        'seq_len': SEQ_LEN,
        'pred_len': PRED_LEN,
        'patch_len': patch_len,
        'stride': stride,
        'd_model': config['d_model'],
        'n_heads': config['n_heads'],
        'n_layers': config['n_layers'],
        'd_ff': config['d_model'] * 2,
        'dropout': config['dropout'],
        'head_type': 'regression',
        'use_revin': True
    }
    
    model = PatchTST(**model_config).to(device)
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # 訓練
        model.train()
        for batch in train_loader:
            x = batch['x'].to(device)
            y_return = batch['y_return'].to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y_return)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        # 驗證
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y_return = batch['y_return'].to(device)
                y_direction = batch['y_direction'].to(device)
                
                pred = model(x)
                loss = criterion(pred, y_return)
                val_loss += loss.item()
                
                pred_direction = (pred.squeeze() > 0).long()
                val_correct += (pred_direction == y_direction).sum().item()
                val_total += y_direction.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return {
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'model_config': model_config
    }

# ============================================
# 執行搜索
# ============================================
print("\n" + "=" * 70)
print("【開始搜索】")
print("=" * 70)

results = []
param_names = list(param_grid.keys())

for trial_idx, param_values in enumerate(all_combinations):
    config = dict(zip(param_names, param_values))
    
    print(f"\n【試驗 {trial_idx + 1}/{len(all_combinations)}】")
    print(f"配置: d_model={config['d_model']}, heads={config['n_heads']}, "
          f"layers={config['n_layers']}, dropout={config['dropout']}, "
          f"patch_len={config['patch_len']}, lr={config['learning_rate']}")
    
    try:
        result = train_and_evaluate(config, train_df, val_df)
        
        if result is None:
            print("  ⚠ 無效配置，跳過")
            continue
        
        print(f"  ✓ Val Acc: {result['best_val_acc']:.2%}, Loss: {result['best_val_loss']:.6f}, Epochs: {result['epochs_trained']}")
        
        results.append({
            **config,
            **result
        })
        
    except Exception as e:
        print(f"  ✗ 錯誤: {str(e)[:50]}")
        continue

# ============================================
# 結果分析
# ============================================
print("\n" + "=" * 70)
print("【搜索結果】")
print("=" * 70)

if len(results) == 0:
    print("沒有有效的搜索結果")
else:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('best_val_acc', ascending=False)
    
    print(f"\n完成試驗數: {len(results)}")
    print(f"\n【Top 5 配置】")
    print("-" * 70)
    
    display_cols = ['d_model', 'n_heads', 'n_layers', 'dropout', 'patch_len', 
                    'learning_rate', 'best_val_acc', 'best_val_loss']
    
    for idx, row in results_df.head(5).iterrows():
        print(f"\n排名 {results_df.index.get_loc(idx) + 1}:")
        print(f"  d_model={int(row['d_model'])}, n_heads={int(row['n_heads'])}, "
              f"n_layers={int(row['n_layers'])}, dropout={row['dropout']}")
        print(f"  patch_len={int(row['patch_len'])}, lr={row['learning_rate']}")
        print(f"  → Val Acc: {row['best_val_acc']:.2%}, Loss: {row['best_val_loss']:.6f}")
    
    # 最佳配置
    best = results_df.iloc[0]
    print(f"\n" + "=" * 70)
    print("【🏆 最佳配置】")
    print("=" * 70)
    print(f"d_model: {int(best['d_model'])}")
    print(f"n_heads: {int(best['n_heads'])}")
    print(f"n_layers: {int(best['n_layers'])}")
    print(f"dropout: {best['dropout']}")
    print(f"patch_len: {int(best['patch_len'])}")
    print(f"stride: {int(best['patch_len'] // 2)}")
    print(f"learning_rate: {best['learning_rate']}")
    print(f"\n驗證準確率: {best['best_val_acc']:.2%}")
    print(f"驗證損失: {best['best_val_loss']:.6f}")
    
    # 保存結果
    results_df.to_csv('hyperparameter_search_results.csv', index=False)
    
    # 保存最佳配置
    best_config = {
        'd_model': int(best['d_model']),
        'n_heads': int(best['n_heads']),
        'n_layers': int(best['n_layers']),
        'dropout': best['dropout'],
        'patch_len': int(best['patch_len']),
        'stride': int(best['patch_len'] // 2),
        'd_ff': int(best['d_model'] * 2),
        'learning_rate': best['learning_rate'],
        'best_val_acc': best['best_val_acc'],
        'best_val_loss': best['best_val_loss']
    }
    
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\n✓ 結果已保存: hyperparameter_search_results.csv")
    print(f"✓ 最佳配置已保存: best_hyperparameters.json")

print("\n" + "=" * 70)
print("【超參數優化完成】")
print("=" * 70)
