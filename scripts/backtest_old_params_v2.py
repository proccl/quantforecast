#!/usr/bin/env python3
"""
使用舊版參數模型進行回測 - 內容與 complete_backtest_results 完全一致
"""

import torch
import torch.nn as nn
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
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.patchtst import PatchTST
from src.data.loader import DataLoader as DataLoaderClass
from src.data.features import FeatureEngineer

print("=" * 70)
print("【使用舊版參數模型進行回測】")
print("=" * 70)

# 設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "models/patchtst_old_params_20260414_235003.pth"

# 加載數據
print("\n【1/5】加載數據...")
df = pd.read_csv("data/xiaomi_real.csv")
df['date'] = pd.to_datetime(df['date'])
print(f"  原始數據: {len(df)} 條")

# 特徵工程
print("\n【2/5】特徵工程...")
feature_engineer = FeatureEngineer()
df_features = feature_engineer.create_features(df)
df_clean = df_features.dropna()
print(f"  清洗後: {len(df_clean)} 條")

feature_cols = [c for c in df_clean.columns if c not in ['date', 'target_return_5d', 'target_direction']]
print(f"  特徵數: {len(feature_cols)}")

# 只取最近3個月數據進行回測
latest_date = df_clean['date'].iloc[-1]
three_months_ago = latest_date - pd.Timedelta(days=90)
test_df = df_clean[df_clean['date'] >= three_months_ago].reset_index(drop=True)
test_df['date'] = pd.to_datetime(test_df['date'])

print(f"\n【測試數據】")
print(f"  樣本數: {len(test_df)}")
print(f"  時間範圍: {test_df['date'].iloc[0]} ~ {test_df['date'].iloc[-1]}")

# 加載模型
print("\n【3/5】加載模型...")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model_config = checkpoint.get('model_config', {})
seq_len = model_config.get('seq_len', 20)
pred_len = model_config.get('pred_len', 5)

model = PatchTST(
    n_features=len(feature_cols),
    seq_len=seq_len,
    pred_len=pred_len,
    patch_len=model_config.get('patch_len', 5),
    stride=model_config.get('stride', 2),
    d_model=model_config.get('d_model', 64),
    n_heads=model_config.get('n_heads', 8),
    n_layers=model_config.get('n_layers', 3),
    d_ff=model_config.get('d_ff', 128),
    dropout=model_config.get('dropout', 0.2),
    head_type='classification',
    use_revin=True
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"  ✓ 模型已載入")
print(f"    序列長度: {seq_len}")
print(f"    預測長度: {pred_len}")
print(f"    d_model: {model_config.get('d_model', 64)}")
print(f"    n_heads: {model_config.get('n_heads', 8)}")
print(f"    n_layers: {model_config.get('n_layers', 3)}")

# ============================================
# 預測 - 與 backtest.py 邏輯一致
# ============================================
print("\n【4/5】執行預測與回測...")

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
        target_direction = 1 if target_return > 0 else 0
        
        x = torch.FloatTensor(seq_data).unsqueeze(0).to(device)
        pred = model(x)
        
        # 分類模型：獲取預測類別和置信度
        pred_probs = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred, dim=1).item()
        pred_confidence = pred_probs[0][pred_class].item()
        
        # 將置信度轉換為