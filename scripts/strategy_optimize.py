#!/usr/bin/env python3
"""
策略參數優化 - 測試交易閾值、持有週期和止損比例

目標：找到最佳的交易閾值和持有週期參數，使策略回報最大化
模型：models/patchtst_classification_fixed_20260416_121241.pth

參數組合：
1. 概率閾值: [0.35, 0.37, 0.40, 0.45, 0.50]
2. 持有週期: [1, 3, 5, 7, 10] 天
3. 止損比例: [0.05, 0.08, 0.10, 0.15]

評估指標：總回報、Sharpe Ratio、最大回撤、交易次數
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
from itertools import product
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.patchtst import PatchTST
from src.data.loader import DataLoader as DataLoaderClass
from src.data.features import FeatureEngineer

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class StrategyParams:
    """策略參數"""
    prob_threshold: float  # 概率閾值
    prob_percentile: int   # 百分位數
    holding_period: int    # 持有週期（天）
    stop_loss: float       # 止損比例


@dataclass
class StrategyResult:
    """策略回測結果"""
    params: StrategyParams
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_count: int
    loss_count: int
    win_rate_pct: float
    final_capital: float
    equity_curve: np.ndarray
    dates: pd.DatetimeIndex
    trades: List[Dict]


def load_data_and_model(model_path: str):
    """加載數據和模型 - 使用與訓練時相同的特徵"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加載數據
    df = pd.read_csv("data/xiaomi_real.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # 特徵工程 - 使用與訓練時相同的11個特徵
    df['returns'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['volatility'] = df['returns'].rolling(20).std()
    
    # 目標變量 (5天後是否上漲超過0.5%)
    df['target_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target_direction'] = (df['target_return'] > 0.005).astype(int)
    
    df_clean = df.dropna()
    
    # 使用與訓練時相同的特徵列
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                   'returns', 'sma_5', 'sma_20', 'rsi', 'macd', 'volatility']
    
    # 只取最近3個月數據進行回測
    latest_date = df_clean['date'].iloc[-1]
    three_months_ago = latest_date - pd.Timedelta(days=90)
    test_df = df_clean[df_clean['date'] >= three_months_ago].reset_index(drop=True)
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # 加載模型
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
    
    return device, test_df, feature_cols, model, seq_len, pred_len


def generate_predictions(model, device, test_df, feature_cols, seq_len, pred_len):
    """生成預測結果"""
    predictions = []
    pred_probs_list = []
    pred_classes = []
    actuals = []
    test_dates = []
    test_close = []
    test_close_T5 = []
    
    with torch.no_grad():
        for i in range(len(test_df) - seq_len - pred_len + 1):
            seq_data = test_df[feature_cols].iloc[i:i+seq_len].values
            close_T = test_df['close'].iloc[i+seq_len]
            close_T5 = test_df['close'].iloc[i+seq_len:i+seq_len+pred_len].values[-1]
            target_return = close_T5 / close_T - 1
            
            x = torch.FloatTensor(seq_data).unsqueeze(0).to(device)
            pred = model(x)
            
            # 分類模型：獲取預測類別和概率
            pred_probs = torch.softmax(pred, dim=1)
            pred_class = torch.argmax(pred, dim=1).item()
            up_prob = pred_probs[0][1].item()  # 上漲概率
            
            predictions.append(up_prob)
            pred_probs_list.append(pred_probs[0].cpu().numpy())
            pred_classes.append(pred_class)
            actuals.append(target_return)
            test_dates.append(test_df['date'].iloc[i+seq_len])
            test_close.append(close_T)
            test_close_T5.append(close_T5)
    
    return pd.DataFrame({
        'date': test_dates,
        'close': test_close,
        'close_T5': test_close_T5,
        'up_probability': predictions,
        'pred_class': pred_classes,
        'actual_return': actuals
    })


def run_strategy_backtest(
    pred_df: pd.DataFrame,
    params: StrategyParams,
    initial_capital: float = 100000
) -> StrategyResult:
    """
    執行策略回測
    
    策略邏輯：
    1. 當上漲概率 > threshold 時開倉（買入）
    2. 持有 holding_period 天後平倉
    3. 若期間跌幅達到 stop_loss 則止損平倉
    """
    capital = initial_capital
    position = 0.0  # 持倉數量
    entry_price = 0.0
    entry_date = None
    holding_days = 0
    
    trades = []
    equity_curve = [initial_capital]
    
    for i in range(len(pred_df)):
        current_price = pred_df['close'].iloc[i]
        up_prob = pred_df['up_probability'].iloc[i]
        current_date = pred_df['date'].iloc[i]
        
        # 檢查是否觸發止損
        if position > 0:
            loss_pct = (entry_price - current_price) / entry_price
            if loss_pct >= params.stop_loss:
                # 止損平倉
                capital = position * current_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'action': 'STOP_LOSS',
                    'return_pct': -params.stop_loss * 100,
                    'holding_days': holding_days
                })
                position = 0
                entry_price = 0
                holding_days = 0
            else:
                holding_days += 1
                # 檢查是否達到持有週期
                if holding_days >= params.holding_period:
                    capital = position * current_price
                    trade_return = (current_price - entry_price) / entry_price
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': current_date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'action': 'HOLD_EXPIRE',
                        'return_pct': trade_return * 100,
                        'holding_days': holding_days
                    })
                    position = 0
                    entry_price = 0
                    holding_days = 0
        
        # 檢查是否開倉（無持倉且上漲概率超過閾值）
        if position == 0 and up_prob >= params.prob_threshold:
            position = capital / current_price
            capital = 0
            entry_price = current_price
            entry_date = current_date
            holding_days = 0
            trades.append({
                'entry_date': current_date,
                'entry_price': current_price,
                'action': 'OPEN',
                'up_probability': up_prob
            })
        
        # 計算當前權益
        current_equity = capital if position == 0 else position * current_price
        equity_curve.append(current_equity)
    
    # 最後一日平倉（如有持倉）
    if position > 0:
        final_price = pred_df['close'].iloc[-1]
        capital = position * final_price
        trade_return = (final_price - entry_price) / entry_price
        trades.append({
            'entry_date': entry_date,
            'exit_date': pred_df['date'].iloc[-1],
            'entry_price': entry_price,
            'exit_price': final_price,
            'action': 'FINAL_EXIT',
            'return_pct': trade_return * 100,
            'holding_days': holding_days
        })
    
    equity_curve = np.array(equity_curve)
    final_capital = equity_curve[-1]
    total_return_pct = (final_capital / initial_capital - 1) * 100
    
    # 計算風險指標
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = np.min(drawdown) * 100
    
    # 統計交易
    completed_trades = [t for t in trades if 'exit_date' in t]
    total_trades = len(completed_trades)
    
    win_count = sum(1 for t in completed_trades if t['return_pct'] > 0)
    loss_count = sum(1 for t in completed_trades if t['return_pct'] <= 0)
    win_rate_pct = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    return StrategyResult(
        params=params,
        total_return_pct=total_return_pct,
        sharpe_ratio=sharpe_ratio,
        max_drawdown_pct=max_drawdown_pct,
        total_trades=total_trades,
        win_count=win_count,
        loss_count=loss_count,
        win_rate_pct=win_rate_pct,
        final_capital=final_capital,
        equity_curve=equity_curve,
        dates=pred_df['date'],
        trades=trades
    )


def calculate_score(result: StrategyResult) -> float:
    """
    計算綜合評分
    綜合考慮：總回報、Sharpe、最大回撤、交易次數
    """
    # 標準化各項指標
    return_score = result.total_return_pct  # 直接使用百分比
    sharpe_score = result.sharpe_ratio * 20  # Sharpe 乘以20放大
    
    # 回撤懲罰（回撤越大扣分越多）
    drawdown_penalty = abs(result.max_drawdown_pct) * 0.5
    
    # 交易次數獎勵（有一定交易次數才可靠）
    trade_bonus = min(result.total_trades * 0.5, 5)  # 最多加5分
    
    # 勝率獎勵
    winrate_bonus = (result.win_rate_pct - 50) * 0.1 if result.win_rate_pct > 50 else 0
    
    final_score = return_score + sharpe_score - drawdown_penalty + trade_bonus + winrate_bonus
    
    return final_score


def create_results_table(results: List[StrategyResult]) -> pd.DataFrame:
    """創建優化結果表格"""
    data = []
    for r in results:
        score = calculate_score(r)
        data.append({
            '百分位': f"P{r.params.prob_percentile}",
            '閾值': f"{r.params.prob_threshold:.3f}",
            '持有': r.params.holding_period,
            '止損': f"{r.params.stop_loss:.1%}",
            '回報(%)': f"{r.total_return_pct:+.2f}",
            'Sharpe': f"{r.sharpe_ratio:.3f}",
            '回撤(%)': f"{r.max_drawdown_pct:.2f}",
            '交易': r.total_trades,
            '勝率(%)': f"{r.win_rate_pct:.1f}",
            '評分': f"{score:+.2f}"
        })
    
    return pd.DataFrame(data)


def plot_best_result(result: StrategyResult, output_path: str):
    """繪製最佳參數的回測結果圖（深色風格，與 complete_backtest_results 一致）"""
    
    # 深色主題設置
    plt.rcParams['axes.facecolor'] = '#2d2d2d'
    plt.rcParams['axes.edgecolor'] = '#666666'
    plt.rcParams['axes.labelcolor'] = '#cccccc'
    plt.rcParams['text.color'] = '#cccccc'
    plt.rcParams['xtick.color'] = '#cccccc'
    plt.rcParams['ytick.color'] = '#cccccc'
    plt.rcParams['grid.color'] = '#444444'
    plt.rcParams['figure.facecolor'] = '#1a1a1a'
    
    fig = plt.figure(figsize=(16, 14), facecolor='#1a1a1a')
    
    # 主標題
    backtest_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.suptitle(
        f'Best Strategy Backtest | P{result.params.prob_percentile} Threshold={result.params.prob_threshold:.3f} | '
        f'Hold={result.params.holding_period}d | Stop={result.params.stop_loss:.1%}\n'
        f'Total Return: {result.total_return_pct:+.2f}% | Sharpe: {result.sharpe_ratio:.3f} | '
        f'Max DD: {result.max_drawdown_pct:.2f}%\n'
        f'Backtest Time: {backtest_time} | T+1 Trading Enabled',
        fontsize=14, color='white', y=0.98
    )
    
    equity_curve = result.equity_curve
    initial_capital = 100000
    dates = result.dates
    
    # 1. 資金曲線
    ax1 = plt.subplot(3, 2, 1)
    strat_cumret = (equity_curve / initial_capital - 1) * 100
    ax1.plot(dates, strat_cumret[1:], linewidth=2.5, color='#4a9eff', label='Strategy')
    ax1.axhline(y=0, color='#666666', linestyle='-', linewidth=1)
    ax1.fill_between(dates, strat_cumret[1:], 0, where=(strat_cumret[1:] > 0), alpha=0.3, color='#51cf66')
    ax1.fill_between(dates, strat_cumret[1:], 0, where=(strat_cumret[1:] <= 0), alpha=0.3, color='#ff6b6b')
    ax1.set_ylabel('Cumulative Return (%)', color='#cccccc')
    ax1.set_title(f'Equity Curve | Trades: {result.total_trades} | Win Rate: {result.win_rate_pct:.1f}%')
    ax1.grid(True, alpha=0.3)
    ax1.legend(facecolor='none', edgecolor='none', labelcolor='#cccccc')
    
    # 2. 每日收益分布
    ax2 = plt.subplot(3, 2, 2)
    daily_returns = np.diff(equity_curve) / equity_curve[:-1] * 100
    mu, sigma = np.mean(daily_returns), np.std(daily_returns)
    n, bins, patches = ax2.hist(daily_returns, bins=30, alpha=0.6, color='#4a9eff',
                                edgecolor='#2d2d2d', density=True, label='Histogram')
    x = np.linspace(bins[0], bins[-1], 100)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), '--', color='#ffd43b', linewidth=2,
             label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
    ax2.axvline(x=0, color='#ff6b6b', linestyle='-', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Daily Return (%)', color='#cccccc')
    ax2.set_ylabel('Density', color='#cccccc')
    ax2.set_title('Daily Return Distribution')
    ax2.legend(facecolor='none', edgecolor='none', labelcolor='#cccccc', loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 回撤曲線
    ax3 = plt.subplot(3, 2, 3)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max * 100
    ax3.fill_between(dates, drawdown[1:], 0, alpha=0.5, color='#ff6b6b')
    ax3.plot(dates, drawdown[1:], linewidth=1.5, color='#ff6b6b', label='Drawdown')
    ax3.axhline(y=result.max_drawdown_pct, color='#ff6b6b', linestyle='--', 
                linewidth=2, label=f'Max DD: {result.max_drawdown_pct:.2f}%')
    ax3.set_ylabel('Drawdown (%)', color='#cccccc')
    ax3.set_title('Drawdown Curve')
    ax3.legend(facecolor='none', edgecolor='none', labelcolor='#cccccc')
    ax3.grid(True, alpha=0.3)
    
    # 4. 交易收益分布
    ax4 = plt.subplot(3, 2, 4)
    completed_trades = [t for t in result.trades if 'return_pct' in t]
    if completed_trades:
        trade_returns = [t['return_pct'] for t in completed_trades]
        colors = ['#51cf66' if r > 0 else '#ff6b6b' for r in trade_returns]
        ax4.bar(range(len(trade_returns)), trade_returns, color=colors, alpha=0.7, edgecolor='#2d2d2d')
        ax4.axhline(y=0, color='#666666', linestyle='-', linewidth=1)
        ax4.set_xlabel('Trade Number', color='#cccccc')
        ax4.set_ylabel('Return (%)', color='#cccccc')
        ax4.set_title(f'Individual Trade Returns (W:{result.win_count}/L:{result.loss_count})')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No completed trades', ha='center', va='center', 
                transform=ax4.transAxes, color='#cccccc')
    
    # 5. 持有週期分析
    ax5 = plt.subplot(3, 2, 5)
    completed_trades = [t for t in result.trades if 'return_pct' in t]
    if completed_trades:
        holding_days = [t['holding_days'] for t in completed_trades]
        trade_rets = [t['return_pct'] for t in completed_trades]
        colors = ['#51cf66' if r > 0 else '#ff6b6b' for r in trade_rets]
        ax5.scatter(holding_days, trade_rets, c=colors, alpha=0.7, s=50, edgecolors='#2d2d2d')
        ax5.axhline(y=0, color='#666666', linestyle='-', linewidth=1)
        ax5.set_xlabel('Holding Days', color='#cccccc')
        ax5.set_ylabel('Return (%)', color='#cccccc')
        ax5.set_title('Return vs Holding Period')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No completed trades', ha='center', va='center',
                transform=ax5.transAxes, color='#cccccc')
    
    # 6. 參數與統計摘要
    ax6 = plt.subplot(3, 2, 6)
    ax6.set_facecolor('#2d2d2d')
    ax6.axis('off')
    
    summary_text = f"""
【最佳策略參數】

百分位數: P{result.params.prob_percentile}
概率閾值: {result.params.prob_threshold:.4f}
持有週期: {result.params.holding_period} 天
止損比例: {result.params.stop_loss:.1%}

【回測結果】
總回報: {result.total_return_pct:+.2f}%
Sharpe: {result.sharpe_ratio:.3f}
最大回撤: {result.max_drawdown_pct:.2f}%
最終資金: {result.final_capital:,.0f}

【交易統計】
總交易次數: {result.total_trades}
勝率: {result.win_rate_pct:.1f}%
盈利次數: {result.win_count}
虧損次數: {result.loss_count}

【建議】
{'✓ 策略表現良好' if result.total_return_pct > 0 and result.sharpe_ratio > 0.5 else '⚠ 策略需要改進'}
{'✓ 風險控制有效' if abs(result.max_drawdown_pct) < 15 else '⚠ 注意回撤風險'}
{'✓ 交易頻率適中' if 5 <= result.total_trades <= 30 else '⚠ 交易頻率需調整'}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=11, color='#cccccc', verticalalignment='top',
             fontfamily='monospace', linespacing=1.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close(fig)
    print(f"✓ 最佳策略圖表已保存: {output_path}")


def save_optimization_results(results: List[StrategyResult], best_result: StrategyResult, 
                               output_json: str, timestamp: str):
    """保存優化結果到 JSON"""
    
    # 準備所有結果數據
    all_results_data = []
    for r in results:
        score = calculate_score(r)
        all_results_data.append({
            'prob_percentile': int(r.params.prob_percentile),
            'prob_threshold': float(r.params.prob_threshold),
            'holding_period': int(r.params.holding_period),
            'stop_loss': float(r.params.stop_loss),
            'total_return_pct': float(r.total_return_pct),
            'sharpe_ratio': float(r.sharpe_ratio),
            'max_drawdown_pct': float(r.max_drawdown_pct),
            'total_trades': int(r.total_trades),
            'win_count': int(r.win_count),
            'loss_count': int(r.loss_count),
            'win_rate_pct': float(r.win_rate_pct),
            'final_capital': float(r.final_capital),
            'score': float(score)
        })
    
    # 按評分排序
    all_results_data.sort(key=lambda x: x['score'], reverse=True)
    
    output_data = {
        'timestamp': str(timestamp),
        'model_used': 'patchtst_classification_fixed_20260416_121241.pth',
        'best_params': {
            'prob_percentile': int(best_result.params.prob_percentile),
            'prob_threshold': float(best_result.params.prob_threshold),
            'holding_period': int(best_result.params.holding_period),
            'stop_loss': float(best_result.params.stop_loss)
        },
        'best_metrics': {
            'total_return_pct': float(best_result.total_return_pct),
            'sharpe_ratio': float(best_result.sharpe_ratio),
            'max_drawdown_pct': float(best_result.max_drawdown_pct),
            'total_trades': int(best_result.total_trades),
            'win_count': int(best_result.win_count),
            'loss_count': int(best_result.loss_count),
            'win_rate_pct': float(best_result.win_rate_pct),
            'final_capital': float(best_result.final_capital),
            'score': float(calculate_score(best_result))
        },
        'all_results': all_results_data,
        'parameter_ranges': {
            'prob_percentiles': sorted([int(x) for x in set(r.params.prob_percentile for r in results)]),
            'prob_thresholds': sorted([float(x) for x in set(r.params.prob_threshold for r in results)]),
            'holding_period': sorted([int(x) for x in set(r.params.holding_period for r in results)]),
            'stop_loss': sorted([float(x) for x in set(r.params.stop_loss for r in results)])
        }
    }
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ 優化結果已保存: {output_json}")


def main():
    print("=" * 80)
    print("【策略參數優化】")
    print("=" * 80)
    
    MODEL_PATH = "models/patchtst_classification_fixed_20260416_121241.pth"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 加載數據和模型
    print("\n【1/4】加載數據和模型...")
    device, test_df, feature_cols, model, seq_len, pred_len = load_data_and_model(MODEL_PATH)
    print(f"  ✓ 測試數據: {len(test_df)} 條")
    print(f"  ✓ 特徵數: {len(feature_cols)}")
    print(f"  ✓ 序列長度: {seq_len}")
    print(f"  ✓ 預測長度: {pred_len}")
    
    # 生成預測
    print("\n【2/4】生成預測...")
    pred_df = generate_predictions(model, device, test_df, feature_cols, seq_len, pred_len)
    print(f"  ✓ 預測樣本: {len(pred_df)}")
    print(f"  ✓ 時間範圍: {pred_df['date'].iloc[0]} ~ {pred_df['date'].iloc[-1]}")
    
    # 顯示概率分布統計
    probs = pred_df['up_probability'].values
    print(f"\n  【概率分布統計】")
    print(f"    最小值: {probs.min():.4f}")
    print(f"    最大值: {probs.max():.4f}")
    print(f"    平均值: {probs.mean():.4f}")
    print(f"    中位數: {np.median(probs):.4f}")
    print(f"    P35: {np.percentile(probs, 35):.4f}")
    print(f"    P40: {np.percentile(probs, 40):.4f}")
    print(f"    P50: {np.percentile(probs, 50):.4f}")
    
    # 使用百分位數作為閾值（基於實際概率分布）
    prob_percentiles = [30, 35, 40, 45, 50, 55, 60, 65, 70]  # 百分位數
    prob_thresholds = [np.percentile(probs, p) for p in prob_percentiles]
    
    print(f"\n  【自適應閾值（基於百分位數）】")
    for p, thresh in zip(prob_percentiles, prob_thresholds):
        count = sum(probs >= thresh)
        print(f"    P{p:2d}: {thresh:.4f} -> {count:2d} 個信號 ({count/len(probs)*100:.1f}%)")
    
    # 擴展參數範圍，使用隨機採樣達到100次試錯
    prob_percentiles_full = list(range(25, 76, 5))  # 25, 30, 35, ..., 70, 75
    prob_thresholds_full = [np.percentile(probs, p) for p in prob_percentiles_full]
    holding_periods_full = list(range(1, 16))  # 1-15天
    stop_losses_full = [0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    
    # 隨機採樣100組參數
    n_trials = 100
    np.random.seed(42)  # 固定隨機種子以便重現
    
    sampled_params = []
    for _ in range(n_trials):
        prob_pct = np.random.choice(prob_percentiles_full)
        prob_thresh = np.percentile(probs, prob_pct)
        hold = np.random.choice(holding_periods_full)
        stop = np.random.choice(stop_losses_full)
        sampled_params.append((prob_pct, prob_thresh, hold, stop))
    
    # 同時保留原始的網格搜索組合作為基準
    holding_periods = [1, 3, 5, 7, 10]
    stop_losses = [0.05, 0.08, 0.10, 0.15]
    
    total_combinations = n_trials
    
    print(f"\n模型: {MODEL_PATH}")
    print(f"參數組合數: {total_combinations}")
    
    # 測試所有參數組合
    print(f"\n【3/4】測試參數組合 ({total_combinations} 組隨機採樣)...")
    print("-" * 80)
    
    results = []
    best_score = -float('inf')
    best_result = None
    
    for i, (prob_pct, prob_thresh, hold, stop) in enumerate(sampled_params, 1):
        params = StrategyParams(
            prob_threshold=prob_thresh,
            prob_percentile=prob_pct,
            holding_period=hold,
            stop_loss=stop
        )
        result = run_strategy_backtest(pred_df, params)
        score = calculate_score(result)
        
        results.append(result)
        
        if score > best_score:
            best_score = score
            best_result = result
        
        # 打印進度
        print(f"  [{i:3d}/{total_combinations}] "
              f"P{prob_pct:2d}={prob_thresh:.3f} H={hold:2d} S={stop:.1%} | "
              f"Return={result.total_return_pct:+.2f}% "
              f"Sharpe={result.sharpe_ratio:.3f} "
              f"DD={result.max_drawdown_pct:.2f}% "
              f"Trades={result.total_trades:2d} "
              f"Score={score:+.2f}")
    
    print("-" * 80)
    
    # 創建結果表格
    print("\n【優化結果表格】")
    results_table = create_results_table(results)
    print(results_table.to_string(index=False))
    
    # 顯示最佳參數
    print("\n" + "=" * 80)
    print("【最佳參數組合】")
    print("=" * 80)
    print(f"百分位數: P{best_result.params.prob_percentile}")
    print(f"概率閾值: {best_result.params.prob_threshold:.4f}")
    print(f"持有週期: {best_result.params.holding_period} 天")
    print(f"止損比例: {best_result.params.stop_loss:.1%}")
    print(f"\n【最佳策略表現】")
    print(f"總回報: {best_result.total_return_pct:+.2f}%")
    print(f"Sharpe Ratio: {best_result.sharpe_ratio:.3f}")
    print(f"最大回撤: {best_result.max_drawdown_pct:.2f}%")
    print(f"交易次數: {best_result.total_trades}")
    print(f"勝率: {best_result.win_rate_pct:.1f}%")
    print(f"綜合評分: {best_score:.2f}")
    
    # 保存結果
    print("\n【4/4】保存結果...")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    json_path = results_dir / f"strategy_optimization_{timestamp}.json"
    png_path = results_dir / f"strategy_optimization_{timestamp}.png"
    
    save_optimization_results(results, best_result, str(json_path), timestamp)
    plot_best_result(best_result, str(png_path))
    
    print("\n" + "=" * 80)
    print("【優化完成】")
    print("=" * 80)
    print(f"結果文件:")
    print(f"  - JSON: {json_path}")
    print(f"  - PNG:  {png_path}")


if __name__ == '__main__':
    main()
