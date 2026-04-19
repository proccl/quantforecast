# QuantForecast

[![Version](https://img.shields.io/badge/version-1.1.1-blue.svg)](https://github.com/proccl/quantforecast)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

基於 **PatchTST (Channel-Time Patch Time-Series Transformer)** 的量化預測系統，專注於港股（小米 1810.HK）的價格預測與回測。

**版本**: v1.1.1 (2026-04-19) - 分類模型 + 滾動預測版本

## 📊 項目概述

本項目使用 Transformer 架構處理時間序列數據，通過 Patch-based 方法提取局部時序特徵，結合多維技術指標進行股票價格方向預測（分類任務）。

### 核心特性

- 🔮 **PatchTST 分類模型**: 預測 T+5 日價格方向（漲/跌），輸出上漲概率 prob_up
- 📈 **特徵工程**: 原始價格 + 技術指標（EMA比率/MACD/RSI/ATR比率/OBV/成交量比率/收益率）
- 🎯 **方向預測**: 二分類問題（漲/跌），預測收益率轉換: `(prob_up - 0.5) * 0.1`
- 📊 **完整回測**: 支持多時間段回測、Walk-forward CV 參數優化、風險指標計算
- 🎨 **數據可視化**: 暗色主題專業圖表（含混淆矩陣、價格散點圖、持倉狀態條）
- ⚖️ **Walk-forward CV**: Optuna + 時間序列交叉驗證，避免過擬合
- 🔄 **滾動預測**: 紅線使用歷史5天數據分別預測未來5天（非插值）

## 🏗️ 項目結構

```
quantforecast/
├── src/                          # 源代碼
│   ├── backtest/                 # 回測引擎
│   │   ├── engine.py
│   │   └── reporting.py
│   ├── data/                     # 數據處理
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── features.py
│   ├── models/                   # 模型定義
│   │   ├── patchtst.py           # PatchTST 分類/回歸模型
│   │   └── revin.py              # 可逆實例歸一化
│   ├── training/                 # 訓練與優化
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── optimizers/
│   │       └── optuna_optimizer.py
│   ├── utils/                    # 工具函數
│   └── config.py                 # 配置管理
├── scripts/                      # 執行腳本
│   ├── backtest.py               # 完整回測（含圖表輸出）
│   ├── backtest_old_params.py    # 舊參數回測可視化
│   ├── visualize_backtest.py     # 回測可視化模塊
│   ├── train.py                  # 模型訓練
│   ├── optimize.py               # Optuna 模型超參優化
│   ├── strategy_optimize.py      # 交易策略參數優化
│   ├── plot_optimization.py        # 優化結果可視化
│   ├── plot_params_comparison.py # 參數對比可視化
│   ├── update_realtime.py        # 實時數據更新
│   ├── daily_pipeline.py         # 每日數據管道
│   └── scheduled_backtest.py     # 定時回測任務
├── data/                         # 數據文件
│   └── xiaomi_real.csv           # 小米股價數據 (2023-01 ~ 2026-04)
├── results/                      # 結果輸出
│   ├── complete_backtest_results.png   # 回測圖表
│   ├── complete_backtest_results.json  # 回測數據
│   ├── future_prediction.json          # 未來5天預測
│   └── optuna_best_params_*.json       # 最佳參數
├── models/                       # 模型文件
│   └── patchtst_classification_fixed_20260416_121241.pth  # 當前模型
├── logs/                         # 日誌文件
└── README.md                     # 本文件
```

## 🚀 快速開始

### 環境要求

```bash
pip install -r requirements.txt
```

### 更新數據（獲取最新股價）

```bash
cd quantforecast
python3 scripts/update_realtime.py
```

數據源：
- **akshare**: 歷史日線數據（延遲但完整）
- **新浪/騰訊實時**: 當日收盤價（實時更新）

### 運行回測

```bash
cd quantforecast
# 使用優化後的參數運行回測
python3 scripts/backtest.py --optimized --months 3

# 或使用默認參數
python3 scripts/backtest.py
```

回測結果將輸出至：
- `results/complete_backtest_results.png` - 回測圖表（含回測時間戳與 T+1 建議）
- `results/complete_backtest_results.json` - 詳細回測數據
- `results/future_prediction.json` - 未來5天預測

### 訓練新模型

```bash
# Walk-forward CV 優化（推薦）
python3 scripts/optimize.py

# 常規訓練
python3 scripts/train.py
```

### 策略參數優化

```bash
# 優化交易策略參數（概率閾值/持有天數/止損比例）
python3 scripts/strategy_optimize.py
```

## 📈 當前模型

### 模型信息

**模型文件**: `patchtst_classification_fixed_20260416_121241.pth`

**模型類型**: 分類模型（Classification）

**輸出**: 
- `prob_up`: 上漲概率 (0~1)
- 預測收益率: `(prob_up - 0.5) * 0.1`

**最佳超參數** (Walk-forward CV + Optuna):

```python
seq_len = 20        # 輸入序列長度（20個交易日）
pred_len = 5        # 預測長度（5個交易日）
d_model = 64        # 嵌入維度
n_heads = 8         # 注意力頭數
n_layers = 3        # Transformer 層數
patch_len = 5       # Patch 長度
stride = 2          # Patch 步長
dropout = 0.2       # Dropout 率
learning_rate = 2.05e-4  # 學習率
batch_size = 32     # 批次大小
```

**策略參數** (優化後):
```python
prob_threshold = 0.3792    # 概率閾值
holding_days = 4           # 持有天數
stop_loss = 0.06           # 止損比例 (6%)
```

### 特徵工程

| 特徵類型 | 具體特徵 | 處理方式 |
|----------|----------|----------|
| 價格 | open, high, low, close | 原始值 |
| 成交量 | volume, volume_ratio | 原始值 / 20日均值比率 |
| 資金流 | OBV | 累積和: `sign(價格變化) * volume 的累積` |
| 趨勢 | EMA5/10/20_ratio | 比率: `close/ema - 1` |
| 動量 | MACD, MACD_hist | 原始值 |
| 波動率 | ATR_ratio (ATR/close), RSI_14, volatility_20d | 比率/原始值/標準差 |
| 收益率 | return_1d, return_5d | 百分比變化 |

## 🎯 最新回測結果

### 回測概覽 (2026-01-19 ~ 2026-04-17)

| 指標 | 數值 |
|------|------|
| **策略總收益** | **+10.22%** |
| Buy & Hold | -16.37% |
| **超額收益** | **+26.59%** |
| 最大回撤 | -4.63% |
| 年化波動率 | 27.00% |
| 夏普比率 | 2.73 |
| 索提諾比率 | 3.92 |
| 卡爾瑪比率 | 2.21 |
| 總交易次數 | 4次 |
| 勝率 | 100% (4勝0負) |
| 測試準確率 | **97.1%** |

### T+1 操作建議

- **回測時間**: 2026-04-19 18:39
- **最新日期**: 2026-04-17
- **操作**: 🔴 **SELL**
- **當前價格**: 32.00 HKD
- **目標價格**: 31.60 HKD
- **預期收益 (T+5)**: -1.25%
- **預測區間**: 2026-04-20 ~ 2026-04-24

### 滾動預測詳情

紅線使用**滾動窗口預測**（非線性插值），基於過去5天數據分別預測對應的未來日期：
- T-4 數據 → 預測 T+1
- T-3 數據 → 預測 T+2
- ...
- T 數據 → 預測 T+5

每個預測點都是獨立的模型推理，預測價格計算：
```
pred_price = hist_close * (1 + (prob_up - 0.5) * 0.1)
```

| 歷史日期 | 目標日期 | 預測價格 | prob_up |
|----------|----------|----------|---------|
| 2026-04-13 | 2026-04-20 | 30.29 HKD | 0.3780 |
| 2026-04-14 | 2026-04-21 | 30.51 HKD | 0.3789 |
| 2026-04-15 | 2026-04-22 | 30.51 HKD | 0.3744 |
| 2026-04-16 | 2026-04-23 | 31.66 HKD | 0.3737 |
| 2026-04-17 | 2026-04-24 | 31.60 HKD | 0.3751 |

## 🖼️ 圖表說明

![Backtest Results](results/complete_backtest_results.png)

`complete_backtest_results.png` 包含6個子圖，配合回測數據解讀：

### 1. Confusion Matrix（混淆矩陣）

預測方向 vs 實際方向，**對齊黃線邏輯**（T vs T-5 比較）：

|  | 預測跌 | 預測漲 |
|--|--------|--------|
| **實際跌** | TN=26 | FP=0 |
| **實際漲** | FN=1 | TP=8 |

- **準確率**: 97.1%
- **Precision (漲)**: 100%（預測漲的全對）
- **Recall (漲)**: 88.9%（漏了1個漲的）

### 2. Daily Return Distribution（日收益分布）

策略每日收益的直方圖與核密度估計，展示收益分布形態。

### 3. Predicted Price vs Actual Price（價格散點圖）

**X軸**: 預測價格（T-5 預測的 T 時刻價格）  
**Y軸**: 實際價格（T 時刻真實價格）  
**藍虛線**: 完美預測對角線

- 綠點 = 預測漲（prob_up > 0.5）
- 紅點 = 預測跌（prob_up ≤ 0.5）
- 點越靠近對角線 = 預測越準確

### 4. Cumulative Return（累計收益）

- **藍線**: 策略累計收益 (+10.22%)
- **灰虛線**: Buy & Hold 基準 (-16.37%)
- **綠/紅區域**: 策略正/負收益時段
- **底部顏色條**: 持倉狀態（綠=持倉，紅=空倉）

### 5. Stock Price（股價與預測）

- **藍線**: 實際股價
- **黃線**: 歷史 T+5 預測（T-5 預測的 T 時刻價格，用於回測驗證）
- **紅線**: 未來5天滾動預測（T-4→T+1, T-3→T+2...）
- **紅色陰影區**: 未來預測區間

### 數據可視化特性

- **黃線**: 用 T-5 數據預測的 T 時刻價格（與藍線同時間軸對比）
- **紅線**: 用過去5天數據分別預測未來5天的滾動預測（非插值）
- **持倉狀態條**: 綠色 = 持倉中，紅色 = 空倉

## 🔧 Walk-forward CV 優化

為解決過擬合問題，使用 Walk-forward 交叉驗證：

```python
# 時間序列交叉驗證（避免未來數據洩露）
for train_idx, val_idx in TimeSeriesSplit(n_splits=5):
    # 只在歷史數據上訓練
    # 在之後的數據上驗證
```

**優化結果**:
- **Trial**: 100次
- **CV Score**: 最佳驗證分數
- **測試集準確率**: 約 51-60%
- **模型穩定性**: Walk-forward CV 顯著提升

### 樣本加權（指數衰減）

訓練時可選擇對近期數據賦予更高權重（指數衰減）：

```python
# 指數衰減權重：w_t = exp(λ * (t - T))
# λ 越大，對近期數據權重越高
weights = np.exp(decay_lambda * (time_indices - T))
```

用途：讓模型更關注近期市場模式，適應市場結構變化。

## 🎯 策略優化

### 方法：網格搜索 + 多目標評估

策略優化使用**網格搜索**（Grid Search）測試所有參數組合：

**優化參數範圍**:
```python
prob_threshold = [0.35, 0.37, 0.40, 0.45, 0.50]  # 概率閾值
holding_days = [1, 3, 5, 7, 10]                    # 持有天數
stop_loss = [0.05, 0.08, 0.10, 0.15]              # 止損比例
```

**評估指標**:
- 總回報（Total Return）
- Sharpe Ratio（風險調整後收益）
- 最大回撤（Max Drawdown）
- 交易次數（Trade Count）

**優化邏輯**:
1. 對每個參數組合運行完整回測
2. 計算上述評估指標
3. 選擇 Sharpe Ratio 最高的參數組合作為最佳策略

## 📝 關鍵文件說明

| 文件 | 說明 |
|------|------|
| `scripts/backtest.py` | 主回測程序，輸出含大標題的圖表與 T+1 建議 |
| `scripts/visualize_backtest.py` | 回測可視化模塊（暗色主題、價格對齊） |
| `scripts/optimize.py` | Walk-forward CV + Optuna 模型超參優化 |
| `scripts/strategy_optimize.py` | 交易策略參數優化（概率閾值/持有天數/止損比例） |
| `scripts/train.py` | 模型訓練（支持分類/回歸） |
| `scripts/update_realtime.py` | 實時數據更新（akshare + 新浪/騰訊） |
| `scripts/daily_pipeline.py` | 每日數據管道（收市後自動運行） |
| `src/models/patchtst.py` | PatchTST 分類/回歸模型架構 |
| `src/data/loader.py` | 數據加載與驗證 |
| `data/xiaomi_real.csv` | 小米真實股價數據 |

## 📜 版本歷史

### v1.1.1 (2026-04-19)
- 分類模型支持（輸出 prob_up 概率）
- 滾動預測：紅線使用歷史5天數據分別預測未來5天
- 價格對齊：散點圖顯示預測價格 vs 實際價格
- 策略優化：獨立腳本優化交易參數
- 清理多餘模型和腳本

### v1.1.0 (2026-04-14)
- Walk-forward CV 優化流程
- 特徵工程標準化
- 基礎回測框架

### v1.0.0 (2026-04-10)
- 初始版本
- PatchTST 基礎模型
- 簡單回測功能

## ⚠️ 免責聲明

本項目僅供研究學習使用，不構成任何投資建議。股市有風險，投資需謹慎。

模型預測存在風險與局限性：
- 近期回測準確率約 97%，但歷史表現不代表未來收益
- 預測收益波動較大，數值穩定性待改進
- 模型基於歷史數據訓練，可能無法預測黑天鵝事件或市場結構突變
- 回測期間交易次數較少（4次），統計意義有限

## 📄 License

MIT License

## 🙏 致謝

- PatchTST: 基於 Nie et al. (2023) 的時間序列預測架構
- Optuna: 超參數優化框架
- PyTorch: 深度學習框架
- akshare: 財經數據接口
