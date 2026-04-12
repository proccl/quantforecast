import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from step3_data_preprocessing import TimeSeriesDataset, RevIN
from step4_patchtst_model import PatchTST
from sklearn.metrics import accuracy_score

print("=" * 70)
print("【貝葉斯超參數優化】使用 Optuna")
print("=" * 70)

# 固定種子確保可復現
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# 載入數據配置
with open('data_config.pkl', 'rb') as f:
    data_config = pickle.load(f)

feature_cols = data_config['feature_cols']
SEQ_LEN = data_config['seq_len']
PRED_LEN = data_config['pred_len']

# 載入並處理數據
df = pd.read_csv('../data/xiaomi_real.csv')
df['date'] = pd.to_datetime(df['date'])

# 特徵工程（完整版）
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
df['rsi_14'] = 100 - (100 / (1 + gain/loss))
df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
df['volume_ratio'] = df['volume'] / df['volume_ma_20']
df['return_1d'] = df['close'].pct_change(1)
df['return_5d'] = df['close'].pct_change(5)
df['volatility_20d'] = df['return_1d'].rolling(window=20).std()
df['target_return_5d'] = df['close'].pct_change(5).shift(-5)
df['target_direction'] = (df['target_return_5d'] > 0).astype(int)

df_clean = df[feature_cols + ['target_return_5d', 'target_direction', 'date']].dropna()

# 劃分數據集
train_ratio = 0.7
val_ratio = 0.15
n_total = len(df_clean)
n_train = int(n_total * train_ratio)

train_df = df_clean.iloc[:n_train].copy()
val_df = df_clean.iloc[n_train:n_train+int(n_total*val_ratio)].copy()

print(f"數據集: 訓練={len(train_df)}, 驗證={len(val_df)}")

# 設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"設備: {device}\n")

# ============================================
# Optuna 目標函數
# ============================================
def objective(trial):
    """
    Optuna 目標函數 - 同時優化驗證損失和方向準確率
    """
    # 定義超參數搜索空間
    params = {
        'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
        'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
        'n_layers': trial.suggest_int('n_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.05, 0.3, step=0.05),
        'patch_len': trial.suggest_categorical('patch_len', [3, 5, 10]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
    }
    
    # 確保 d_model 能被 n_heads 整除
    if params['d_model'] % params['n_heads'] != 0:
        # 調整 d_model 到最接近的可整除值
        params['d_model'] = (params['d_model'] // params['n_heads']) * params['n_heads']
        if params['d_model'] < 32:
            params['d_model'] = 32
    
    stride = params['patch_len'] // 2
    
    # 創建數據集
    train_dataset = TimeSeriesDataset(train_df, feature_cols, SEQ_LEN, PRED_LEN, 
                                      params['patch_len'], stride)
    val_dataset = TimeSeriesDataset(val_df, feature_cols, SEQ_LEN, PRED_LEN, 
                                    params['patch_len'], stride)
    
    if len(train_dataset) < 50 or len(val_dataset) < 20:
        return float('inf'), 0.0  # 數據太少，跳過
    
    # 平衡採樣
    train_directions = [train_dataset[i]['y_direction'].item() for i in range(len(train_dataset))]
    class_counts = [len(train_directions) - sum(train_directions), sum(train_directions)]
    class_weights = [len(train_directions) / (2 * max(c, 1)) for c in class_counts]
    sample_weights = [class_weights[d] for d in train_directions]
    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # 創建模型
    model_config = {
        'n_features': len(feature_cols),
        'seq_len': SEQ_LEN,
        'pred_len': PRED_LEN,
        'patch_len': params['patch_len'],
        'stride': stride,
        'd_model': params['d_model'],
        'n_heads': params['n_heads'],
        'n_layers': params['n_layers'],
        'd_ff': params['d_model'] * 2,
        'dropout': params['dropout'],
        'head_type': 'regression',
        'use_revin': True
    }
    
    try:
        model = PatchTST(**model_config).to(device)
    except Exception as e:
        return float('inf'), 0.0
    
    # 訓練
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):  # 最多50輪
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
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y_return = batch['y_return'].to(device)
                y_direction = batch['y_direction']
                
                pred = model(x)
                loss = criterion(pred, y_return)
                val_loss += loss.item()
                
                val_preds.extend(pred.squeeze().cpu().numpy())
                val_targets.extend(y_direction.numpy())
        
        val_loss /= len(val_loader)
        
        # 計算方向準確率
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        pred_directions = (val_preds > 0).astype(int)
        val_acc = accuracy_score(val_targets, pred_directions)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # 檢查預測分佈（避免偏向單一方向）
    pred_up_ratio = np.mean(pred_directions)
    if pred_up_ratio < 0.2 or pred_up_ratio > 0.8:
        # 預測嚴重偏向，給予懲罰
        best_val_loss *= 2
    
    return best_val_loss, val_acc

# ============================================
# 創建多目標優化研究
# ============================================
print("開始貝葉斯優化...")
print("優化目標：1) 最小化驗證損失  2) 最大化方向準確率")
print()

# 創建研究（多目標）- 注意：多目標不支持剪枝
study = optuna.create_study(
    directions=['minimize', 'maximize'],  # 雙目標
    sampler=optuna.samplers.TPESampler(seed=42)  # 貝葉斯采樣
)

# 運行優化
study.optimize(objective, n_trials=30, show_progress_bar=True)

# ============================================
# 輸出結果
# ============================================
print("\n" + "=" * 70)
print("【優化完成】")
print("=" * 70)

# 找到帕累托前沿（多目標最優解）
print(f"\n完成試驗數: {len(study.trials)}")
print(f"帕累托最優解數量: {len(study.best_trials)}")

print("\n【Top 5 配置（按方向準確率排序）】")
print("-" * 70)

# 按方向準確率排序
trials_df = study.trials_dataframe()
trials_df = trials_df.sort_values('values_1', ascending=False).head(5)

for idx, row in trials_df.iterrows():
    trial = study.trials[idx]
    print(f"\n配置 {idx}:")
    print(f"  d_model={int(row['params_d_model'])}, "
          f"n_heads={int(row['params_n_heads'])}, "
          f"n_layers={int(row['params_n_layers'])}")
    print(f"  dropout={row['params_dropout']:.2f}, "
          f"patch_len={int(row['params_patch_len'])}, "
          f"lr={row['params_learning_rate']:.2e}")
    print(f"  驗證損失: {row['values_0']:.6f}")
    print(f"  方向準確率: {row['values_1']:.2%}")

# 找到驗證損失最低且準確率 > 50% 的配置
valid_trials = [t for t in study.trials 
                if t.values and t.values[1] > 0.5 and t.values[0] != float('inf')]

if valid_trials:
    best_trial = min(valid_trials, key=lambda t: t.values[0])
    
    print("\n" + "=" * 70)
    print("【推薦配置】驗證損失最低且方向準確率 > 50%")
    print("=" * 70)
    
    p = best_trial.params
    print(f"""
d_model: {p['d_model']}
n_heads: {p['n_heads']}
n_layers: {p['n_layers']}
dropout: {p['dropout']}
patch_len: {p['patch_len']}
stride: {p['patch_len'] // 2}
learning_rate: {p['learning_rate']:.2e}
batch_size: {p['batch_size']}

驗證損失: {best_trial.values[0]:.6f}
方向準確率: {best_trial.values[1]:.2%}
""")
    
    # 保存結果
    import json
    result = {
        'best_params': p,
        'val_loss': best_trial.values[0],
        'val_accuracy': best_trial.values[1],
        'all_trials': len(study.trials)
    }
    
    with open('optuna_best_params.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("✓ 結果已保存: optuna_best_params.json")

print("\n" + "=" * 70)
