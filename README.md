# QuantForecast

基於 **PatchTST (Channel-Time Patch Time-Series Transformer)** 的量化預測系統，專注於港股（小米 1810.HK）的價格預測與回測。

## 📊 項目概述

本項目使用 Transformer 架構處理時間序列數據，通過 Patch-based 方法提取局部時序特徵，結合多維技術指標進行股票價格預測。

### 核心特性

- 🔮 **PatchTST 模型**: 使用 Channel-Time Patch 架構處理多變量時間序列
- 📈 **多維特徵**: 整合價格、技術指標、波動率等 17+ 維特徵
- 🎯 **方向預測**: 預測 T+5 日收益率方向（漲/跌）
- 📊 **完整回測**: 支持多時間段回測（1個月/3個月/6個月/1年）
- 🎨 **視覺化**: 暗色主題專業圖表，清晰展示預測結果
- ⚖️ **風控機制**: 動態倉位管理，基於預測置信度調整頭寸

## 🏗️ 項目結構

```
quantforecast/
├── src/                    # 源代碼
│   ├── complete_backtest.py        # 完整回測主程序
│   ├── train_final_model.py        # 模型訓練
│   ├── step2_feature_engineering.py # 特徵工程
│   ├── step5_training.py           # 訓練流程
│   ├── step6_evaluation.py         # 模型評估
│   ├── hyperparameter_search.py    # 超參數搜索
│   └── ...
├── data/                   # 數據文件
│   ├── xiaomi_real.csv             # 真實股價數據
│   ├── xiaomi_features.csv         # 特徵數據
│   └── xiaomi_raw.csv              # 原始數據
├── results/                # 結果輸出
│   ├── *.png                       # 圖表
│   ├── *.json                      # 結果數據
│   └── complete_backtest_results.json
├── models/                 # 模型文件（運行時生成）
├── notebooks/              # Jupyter 筆記本（可選）
├── config/                 # 配置文件
└── docs/                   # 文檔
```

## 🚀 快速開始

### 環境要求

```bash
pip install torch pandas numpy matplotlib scikit-learn optuna
```

### 運行回測

```bash
cd quantforecast/src
python complete_backtest.py
```

### 訓練新模型

```bash
python train_final_model.py
```

## 📈 模型架構

### PatchTST 配置

```python
seq_len = 20        # 輸入序列長度（20個交易日）
pred_len = 5        # 預測長度（5個交易日）
d_model = 32        # 嵌入維度
n_heads = 2         # 注意力頭數
n_layers = 3        # Transformer 層數
patch_len = 3       # Patch 長度
stride = 1          # Patch 步長
dropout = 0.2       # Dropout 率
```

### 特徵工程

- **價格特徵**: Open, High, Low, Close, Volume
- **技術指標**: 
  - 趨勢: EMA5, EMA20, EMA60, MACD
  - 動量: RSI, CCI, Williams %R
  - 波動率: ATR, Bollinger Bands
  - 成交量: OBV, Volume MA

## 🎯 回測結果示例

### 三個月回測（2026-01-13 ~ 2026-04-10）

| 指標 | 數值 |
|------|------|
| 策略總收益 | -9.73% |
| Buy & Hold | -10.42% |
| 超額收益 | +0.69% |
| 測試準確率 | 48.6% |
| 最大回撤 | -12.73% |
| 夏普比率 | -2.41 |
| 勝率 | 50.00% |

### 未來預測（2026-04-13 ~ 2026-04-17）

- **最新收盤價**: 30.90 HKD
- **預測收益**: +3.78%
- **目標價格**: 32.07 HKD

## 📝 關鍵文件說明

| 文件 | 說明 |
|------|------|
| `complete_backtest.py` | 主回測程序，包含可視化 |
| `train_final_model.py` | 模型訓練腳本 |
| `hyperparameter_search.py` | Optuna 超參數優化 |
| `step2_feature_engineering.py` | 特徵工程處理 |
| `xiaomi_real.csv` | 小米股價數據 |

## 🖼️ 圖表說明

回測圖表包含三行佈局：

1. **第一行**: Confusion Matrix | Daily Return Distribution
2. **第二行**: Prediction vs Actual | Cumulative Return
3. **第三行**: Stock Price（長圖）
   - 🔵 藍線: 實際收盤價
   - 🟡 黃虛線: 歷史預測價格（T+5）
   - 🟢 綠線: 未來5天預測（僅在預測區間顯示）

## ⚠️ 免責聲明

本項目僅供研究學習使用，不構成任何投資建議。股市有風險，投資需謹慎。

## 📄 License

MIT License

## 🙏 致謝

- PatchTST: 基於 Nie et al. (2023) 的時間序列預測架構
- Optuna: 超參數優化框架
- PyTorch: 深度學習框架
