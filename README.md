# QuantForecast

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/proccl/quantforecast)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

基於 **PatchTST (Channel-Time Patch Time-Series Transformer)** 的量化預測系統，專注於港股（小米 1810.HK）的價格預測與回測。

**版本**: v1.0.0 (2026-04-13) - 首次穩定版本

## 📊 項目概述

本項目使用 Transformer 架構處理時間序列數據，通過 Patch-based 方法提取局部時序特徵，結合多維技術指標進行股票價格預測。

### 核心特性

- 🔮 **PatchTST 模型**: 使用 Channel-Time Patch 架構處理多變量時間序列
- 📈 **特徵縮放**: 價格歸一化 + 成交量對數變換，解決量級差異問題
- 🎯 **方向預測**: 預測 T+5 日收益率方向（漲/跌）
- 📊 **完整回測**: 支持多時間段回測與風險指標計算
- 🎨 **視覺化**: 暗色主題專業圖表
- ⚖️ **貝葉斯優化**: Optuna 自動超參數搜索

## 🏗️ 項目結構

```
quantforecast/
├── src/                          # 源代碼
│   ├── complete_backtest.py      # 完整回測主程序
│   ├── train_bayesian_final.py   # 模型訓練（Optuna優化）
│   ├── step3_data_preprocessing.py # 數據預處理 + RevIN
│   ├── step4_patchtst_model.py   # PatchTST 模型定義
│   ├── step1-8*.py               # 其他輔助模塊
│   └── data_config.pkl           # 數據配置
├── data/                         # 數據文件
│   └── xiaomi_real.csv           # 小米股價數據 (2023-01 ~ 2026-04)
├── results/                      # 結果輸出
│   ├── backtest_report.md        # 回測報告
│   ├── backtest_analysis_dark.png # 回測分析圖（深色）
│   ├── future_prediction_dark.png # 未來預測圖（深色）
│   ├── trades_history.csv        # 交易記錄
│   └── complete_backtest_results.json # T+1操作建議
├── models/                       # 模型文件
│   └── patchtst_bayesian_best.pth # 最佳模型
└── README.md                     # 本文件
```

## 🚀 快速開始

### 環境要求

```bash
pip install -r requirements.txt
```

### 運行回測

```bash
cd quantforecast/src
python complete_backtest.py
```

回測結果將輸出至 `results/complete_backtest_results.json`：
```json
{
  "date": "2026-04-13",
  "current_price": 30.9,
  "signal": "BUY",
  "t1_target_price": 31.10,
  "t1_expected_return_pct": 0.66,
  "confidence": "HIGH"
}
```

### 訓練新模型

```bash
python train_bayesian_final.py
```

使用 Optuna 進行貝葉斯超參數優化，自動搜索最佳配置。

## 📈 模型架構

### 最佳超參數 (Optuna優化)

```python
seq_len = 24        # 輸入序列長度（24個交易日）
pred_len = 5        # 預測長度（5個交易日）
d_model = 32        # 嵌入維度
n_heads = 4         # 注意力頭數
n_layers = 3        # Transformer 層數
patch_len = 5       # Patch 長度
stride = 2          # Patch 步長
dropout = 0.113     # Dropout 率
learning_rate = 1.5e-4  # 學習率
batch_size = 32     # 批次大小
```

### 特徵工程

| 特徵類型 | 具體特徵 | 處理方式 |
|----------|----------|----------|
| 價格 | open, high, low, close | 歸一化: `col/close - 1` |
| 成交量 | volume | 對數變換: `log1p(volume)` |
| 趨勢 | EMA5/10/20 | 比率: `close/ema - 1` |
| 動量 | MACD, MACD hist | 原始值 |
| 波動率 | ATR, RSI, 20日波動率 | 標準化 |
| 資金流 | OBV | 對數變換: `log1p(\|obv\|) * sign` |

## 🎯 回測結果

### 最新回測 (2026-02-09 ~ 2026-04-01)

| 指標 | 數值 |
|------|------|
| 策略總收益 | -15.60% |
| Buy & Hold | -9.03% |
| 超額收益 | -6.56% |
| 最大回撤 | -19.76% |
| 夏普比率 | -3.58 |
| 勝率 | 12.5% (1勝7負) |
| 測試準確率 | 45.71% |

### T+1 操作建議

- **日期**: 2026-04-13
- **操作**: 🔴 **買入**
- **當前價格**: 30.90 HKD
- **目標價格**: 31.10 HKD
- **預期收益**: +0.66%
- **信心指數**: HIGH

## 📝 關鍵文件說明

| 文件 | 說明 |
|------|------|
| `complete_backtest.py` | 主回測程序，輸出 T+1 操作建議 |
| `train_bayesian_final.py` | 模型訓練 + Optuna 超參優化 |
| `step4_patchtst_model.py` | PatchTST 模型架構定義 |
| `step3_data_preprocessing.py` | 數據預處理 + RevIN 歸一化 |
| `xiaomi_real.csv` | 小米真實股價數據 |

## 🔧 特徵縮放修復

**問題**: 原始特徵量級差異導致模型輸出異常（固定在 3706%）

**解決方案**:
```python
# 價格歸一化
for col in ['open', 'high', 'low', 'close']:
    df[f'{col}_norm'] = df[col] / df['close'] - 1

# 成交量對數變換
df['volume_log'] = np.log1p(df['volume'])
df['obv_log'] = np.log1p(np.abs(df['obv'])) * np.sign(df['obv'])
```

修復後預測值範圍: -3.79% ~ +6.60%（正常）

## 🖼️ 輸出圖表

- `backtest_analysis_dark.png`: 交易收益分佈、累計收益曲線、勝率圓餅圖、持倉天數
- `future_prediction_dark.png`: 未來5天價格預測走勢

## ⚠️ 免責聲明

本項目僅供研究學習使用，不構成任何投資建議。股市有風險，投資需謹慎。

模型存在過擬合風險（驗證集 65.91% vs 測試集 45.71%），實際交易請謹慎評估。

## 📄 License

MIT License

## 🙏 致謝

- PatchTST: 基於 Nie et al. (2023) 的時間序列預測架構
- Optuna: 超參數優化框架
- PyTorch: 深度學習框架
