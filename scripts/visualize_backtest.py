#!/usr/bin/env python3
"""
回測可視化工具 - 通用版本
用於生成統一格式的回測結果圖表

使用方法:
    python scripts/visualize_backtest.py --results results/xxx.json --output results/xxx.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from scipy import stats
import json
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix

# 深色主題配色方案
THEME = {
    'bg': '#1a1a1a',
    'axes_bg': '#2d2d2d',
    'grid': '#444444',
    'edge': '#666666',
    'text': '#cccccc',
    'title': '#ffffff',
    'blue': '#4a9eff',
    'green': '#51cf66',
    'red': '#ff6b6b',
    'yellow': '#ffd43b',
    'gray': '#888888',
}


def setup_theme():
    """設置深色主題"""
    plt.rcParams['axes.facecolor'] = THEME['axes_bg']
    plt.rcParams['axes.edgecolor'] = THEME['edge']
    plt.rcParams['axes.labelcolor'] = THEME['text']
    plt.rcParams['text.color'] = THEME['text']
    plt.rcParams['xtick.color'] = THEME['text']
    plt.rcParams['ytick.color'] = THEME['text']
    plt.rcParams['grid.color'] = THEME['grid']
    plt.rcParams['figure.facecolor'] = THEME['bg']


def load_backtest_data(json_path):
    """加載回測結果數據"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def plot_confusion_matrix(ax, y_true, y_pred, accuracy):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, interpolation='nearest', cmap='YlOrRd')
    ax.figure.colorbar(im, ax=ax, label='Count')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Down', 'Up'], color=THEME['text'])
    ax.set_yticklabels(['Down', 'Up'], color=THEME['text'])
    ax.set_xlabel('Predicted', color=THEME['text'])
    ax.set_ylabel('Actual', color=THEME['text'])
    ax.set_title(f'Confusion Matrix (Acc: {accuracy:.1%})', color=THEME['title'])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')


def plot_return_distribution(ax, returns, label='Daily Return'):
    """繪製收益分布直方圖"""
    returns_pct = returns * 100
    mu, sigma = np.mean(returns_pct), np.std(returns_pct)
    
    ax.hist(returns_pct, bins=30, alpha=0.6, color=THEME['blue'], 
            edgecolor=THEME['axes_bg'], density=True, label='Histogram')
    
    x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), '--', color=THEME['yellow'], 
            linewidth=2, label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})')
    ax.axvline(x=0, color=THEME['red'], linestyle='-', linewidth=1.5, 
               alpha=0.7, label='Zero')
    ax.axvline(x=mu, color=THEME['green'], linestyle='-', linewidth=1.5, 
               alpha=0.7, label=f'Mean: {mu:.3f}%')
    
    ax.set_xlabel(f'{label} (%)', color=THEME['text'])
    ax.set_ylabel('Density', color=THEME['text'])
    ax.set_title(f'{label} Distribution', color=THEME['title'])
    ax.legend(facecolor='none', edgecolor='none', labelcolor=THEME['text'], loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')


def plot_prediction_vs_actual(ax, pred_prices, actual_prices, pred_directions=None):
    """繪製預測價格 vs 實際價格散點圖"""
    pred_prices_arr = np.array(pred_prices)
    actual_prices_arr = np.array(actual_prices)
    
    # 計算完美預測線的範圍
    min_price = min(pred_prices_arr.min(), actual_prices_arr.min())
    max_price = max(pred_prices_arr.max(), actual_prices_arr.max())
    
    # 如果有方向信息，用不同顏色
    if pred_directions is not None:
        up_indices = np.where(np.array(pred_directions) == 1)[0]
        down_indices = np.where(np.array(pred_directions) == 0)[0]
        
        ax.scatter(pred_prices_arr[up_indices], actual_prices_arr[up_indices],
                   c=THEME['green'], alpha=0.6, edgecolors=THEME['axes_bg'], 
                   linewidth=0.5, label='Pred Up', s=30)
        ax.scatter(pred_prices_arr[down_indices], actual_prices_arr[down_indices],
                   c=THEME['red'], alpha=0.6, edgecolors=THEME['axes_bg'], 
                   linewidth=0.5, label='Pred Down', s=30)
    else:
        ax.scatter(pred_prices_arr, actual_prices_arr,
                   c=THEME['blue'], alpha=0.6, edgecolors=THEME['axes_bg'], 
                   linewidth=0.5, s=30)
    
    # 完美預測對角線
    ax.plot([min_price, max_price], [min_price, max_price], 
            THEME['blue'], linestyle='--', linewidth=1.5, label='Perfect Prediction')
    
    ax.set_xlabel('Predicted Price (HKD)', color=THEME['text'])
    ax.set_ylabel('Actual Price (HKD)', color=THEME['text'])
    ax.set_title('Predicted Price vs Actual Price', color=THEME['title'])
    ax.legend(facecolor='none', edgecolor='none', labelcolor=THEME['text'], loc='upper left')
    ax.grid(True, alpha=0.3)


def plot_cumulative_return(ax, dates, equity_curve, buyhold_curve, 
                           signals=None, trades=None, initial_capital=100000):
    """繪製累計收益對比，底部顯示實際持倉狀態"""
    strat_cumret = (np.array(equity_curve) / initial_capital - 1) * 100
    buyhold_cumret = (np.array(buyhold_curve) / initial_capital - 1) * 100
    
    dates = pd.to_datetime(dates)
    
    # 確保數據長度一致
    min_len = min(len(dates), len(strat_cumret), len(buyhold_cumret))
    dates = dates[:min_len]
    strat_cumret = strat_cumret[:min_len]
    buyhold_cumret = buyhold_cumret[:min_len]
    
    line1, = ax.plot(dates, strat_cumret, linewidth=2.5, color=THEME['blue'], label='Strategy', zorder=1)
    line2, = ax.plot(dates, buyhold_cumret, linewidth=2, color=THEME['gray'], 
                     linestyle='--', label='Buy & Hold', zorder=1)
    ax.axhline(y=0, color=THEME['edge'], linestyle='-', linewidth=1)
    
    # 填充收益區域
    ax.fill_between(dates, strat_cumret, 0, where=(np.array(strat_cumret) > 0), 
                    alpha=0.3, color=THEME['green'])
    ax.fill_between(dates, strat_cumret, 0, where=(np.array(strat_cumret) <= 0), 
                    alpha=0.3, color=THEME['red'])
    
    ax.set_ylabel('Cumulative Return (%)', color=THEME['text'])
    ax.set_xlabel('Date', color=THEME['text'])
    
    # 根據交易記錄計算實際持倉狀態
    position_status = np.zeros(len(dates))  # 0 = 空倉, 1 = 持倉
    
    if trades is not None and len(trades) > 0:
        # 構建買賣事件時間線
        events = []
        for trade in trades:
            if isinstance(trade, dict):
                action = trade.get('action', '')
                
                # 買入事件 - BUY 交易用 'date' 字段
                entry_date = trade.get('entry_date') or trade.get('date')
                if entry_date and action in ['BUY']:
                    events.append({'date': pd.to_datetime(entry_date), 'type': 'BUY'})
                
                # 賣出事件 - SELL 交易用 'exit_date' 字段
                exit_date = trade.get('exit_date')
                if exit_date and action in ['SELL', 'SELL_HOLDING', 'STOP_LOSS']:
                    events.append({'date': pd.to_datetime(exit_date), 'type': 'SELL'})
        
        # 按日期排序事件
        events.sort(key=lambda x: x['date'])
        
        # 計算每日持倉狀態
        in_position = False
        event_idx = 0
        for i, d in enumerate(dates):
            # 處理當天發生的所有事件
            while event_idx < len(events) and events[event_idx]['date'] <= d:
                if events[event_idx]['type'] == 'BUY':
                    in_position = True
                else:  # SELL
                    in_position = False
                event_idx += 1
            position_status[i] = 1 if in_position else 0
    
    # 計算持倉時間百分比
    long_pct = np.mean(position_status) * 100
    trade_count = len([t for t in (trades or []) if isinstance(t, dict) and t.get('action') == 'BUY'])
    ax.set_title(f'Cumulative Return (Trades: {trade_count} | Long: {long_pct:.1f}%)', 
                 color=THEME['title'])
    ax.grid(True, alpha=0.3)
    
    # 添加持倉狀態顏色條（基於實際交易記錄）
    ax2 = ax.twinx()
    y_min, y_max = ax.get_ylim()
    line_y = y_min + (y_max - y_min) * 0.01
    
    dates_num = mdates.date2num(dates.values if hasattr(dates, 'values') else dates)
    points = np.array([dates_num, np.full(len(dates), line_y)]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 使用 position_status 確定顏色
    colors = [THEME['green'] if position_status[i+1] == 1 else THEME['red'] 
              for i in range(len(position_status)-1)]
    
    lc = LineCollection(segments, colors=colors, linewidths=3, capstyle='butt')
    ax2.add_collection(lc)
    ax2.set_ylim(y_min, y_max)
    ax2.set_yticks([])
    
    # 圖例
    legend_elements = [
        line1,
        line2,
        Line2D([0], [0], color=THEME['green'], linewidth=3, label='Long Position'),
        Line2D([0], [0], color=THEME['red'], linewidth=3, label='Cash Position')
    ]
    
    ax.legend(handles=legend_elements, loc='lower left', facecolor='none',
              edgecolor='none', labelcolor=THEME['text'])


def plot_price_forecast(ax, df_plot, aligned_df, future_dates, future_prices, 
                        pred_return, three_months_ago=None):
    """繪製股價對比與未來預測"""
    df_plot = df_plot.copy()
    df_plot['date'] = pd.to_datetime(df_plot['date'])
    
    if three_months_ago:
        df_plot = df_plot[df_plot['date'] >= three_months_ago].reset_index(drop=True)
    
    # 實際價格
    ax.plot(df_plot['date'], df_plot['close'], 
            linewidth=2, color=THEME['blue'], alpha=0.7, label='Actual Price', zorder=1)
    
    # T+5預測價格 - 正確對齊：用T-5預測的T時刻價格
    if 'pred_return' in aligned_df.columns:
        aligned_df = aligned_df.copy()
        aligned_df['date'] = pd.to_datetime(aligned_df['date'])
        
        # 計算預測的T+5價格
        pred_len = 5
        aligned_df['pred_price_future'] = aligned_df['close'] * (1 + aligned_df['pred_return'])
        
        # 將預測價格向前平移5天：用T時刻的預測對應T+5時刻的點
        # 這樣黃線@T = 用T-5預測的T時刻價格
        aligned_df['pred_price_aligned'] = aligned_df['pred_price_future'].shift(pred_len)
        
        future_trading_dates = pd.to_datetime(future_dates)
        prediction_start = future_trading_dates[0]
        
        # 只繪製有對齊預測值的點（前5個點沒有T-5預測，所以為空）
        historical_mask = (aligned_df['date'] < prediction_start) & aligned_df['pred_price_aligned'].notna()
        
        if historical_mask.sum() > 0:
            ax.plot(aligned_df.loc[historical_mask, 'date'], 
                    aligned_df.loc[historical_mask, 'pred_price_aligned'], 
                    linewidth=2.5, color=THEME['yellow'], linestyle='--', 
                    alpha=0.9, label='Predicted Price (T-5 → T)', zorder=3)
        
        # 未來預測 - 使用最後一個對齊的預測價格連接到未來預測
        valid_mask = aligned_df['pred_price_aligned'].notna()
        last_historical = aligned_df[valid_mask].iloc[-1] if valid_mask.sum() > 0 else aligned_df.iloc[-1]
        
        future_color = THEME['green'] if pred_return > 0 else THEME['red']
        
        if len(future_trading_dates) > 0:
            # 使用最後一個有效預測價格或當天收盤價
            last_pred_price = last_historical['pred_price_aligned'] if pd.notna(last_historical['pred_price_aligned']) else last_historical['close']
            ax.plot([last_historical['date'], future_trading_dates[0]], 
                    [last_pred_price, future_prices[0]], 
                    linewidth=2.5, color=THEME['yellow'], linestyle='--', alpha=0.9, zorder=3)
            
            ax.plot(future_trading_dates, future_prices, 
                    linewidth=3, color=future_color, linestyle='-', alpha=0.9, 
                    marker='o', markersize=8,
                    label=f'Future 5-Day ({pred_return*100:+.2f}%)', zorder=4)
            
            ax.axvspan(future_trading_dates[0], future_trading_dates[-1], 
                      alpha=0.1, color=future_color)
    
    ax.set_xlabel('Date', color=THEME['text'])
    ax.set_ylabel('Price (HKD)', color=THEME['text'])
    ax.set_title(f'Stock Price: Actual vs Predicted (T+5) | Future 5-Day Forecast: {pred_return*100:+.2f}%', 
                 color=THEME['title'])
    ax.legend(facecolor='none', edgecolor='none', labelcolor=THEME['text'], loc='upper left')
    ax.grid(True, alpha=0.3)


def create_backtest_figure(backtest_data, prediction_data, df_full, 
                           model_label='Classification P65/5D/15SL'):
    """
    創建回測可視化圖表
    
    Parameters:
        backtest_data: dict with backtest results
        prediction_data: dict with prediction results
        df_full: full DataFrame with price data
        model_label: model label for title
    """
    setup_theme()
    
    # 提取數據
    dates = pd.to_datetime(backtest_data['dates'])
    equity_curve = backtest_data['equity_curve']
    buyhold_curve = backtest_data['buyhold_curve']
    predictions = backtest_data['predictions']
    actuals = backtest_data['actuals']
    # 使用對齊的價格（用於散點圖）
    pred_prices_aligned = backtest_data.get('pred_prices_aligned', [])
    actual_prices_aligned = backtest_data.get('actual_prices_aligned', [])
    pred_directions = backtest_data.get('pred_directions', [1 if p > 0 else 0 for p in predictions])
    actual_directions = backtest_data.get('actual_directions', [1 if a > 0 else 0 for a in actuals])
    trades = backtest_data.get('trades', None)  # 獲取交易記錄
    
    # 計算每日收益
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # 計算準確率
    accuracy = np.mean(np.array(pred_directions) == np.array(actual_directions))
    
    # 創建圖表
    fig = plt.figure(figsize=(16, 14), facecolor=THEME['bg'])
    
    # 標題
    backtest_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    pred_return = prediction_data.get('future_return_pct', 0) / 100
    t1_signal = "BUY" if pred_return > 0 else "SELL"
    latest_close = prediction_data.get('latest_close', 0)
    target_price = prediction_data.get('future_price_day5', latest_close)
    target_change = (target_price - latest_close) / latest_close * 100 if latest_close > 0 else 0
    
    fig.suptitle(
        f"Backtest Time: {backtest_time} | T+1 Signal: {t1_signal} ({model_label}) | Target: {target_price:.2f} HKD ({target_change:+.2f}%)",
        color=THEME['title'], fontsize=16, fontweight="bold", y=0.995
    )
    
    # 創建子圖
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.25)
    
    # 1. 混淆矩陣
    ax1 = fig.add_subplot(gs[0, 0])
    plot_confusion_matrix(ax1, actual_directions, pred_directions, accuracy)
    
    # 2. 收益分布
    ax2 = fig.add_subplot(gs[0, 1])
    plot_return_distribution(ax2, daily_returns)
    
    # 3. 預測 vs 實際（使用對齊的價格）
    ax3 = fig.add_subplot(gs[1, 0])
    plot_prediction_vs_actual(ax3, pred_prices_aligned, actual_prices_aligned, pred_directions)
    
    # 4. 累計收益（帶買賣標記）
    ax4 = fig.add_subplot(gs[1, 1])
    plot_cumulative_return(ax4, dates, equity_curve, buyhold_curve, 
                           signals=pred_directions, trades=trades)
    
    # 5. 價格預測
    ax5 = fig.add_subplot(gs[2, :])
    
    # 構建對齊的DataFrame - 使用日期匹配而不是位置索引
    df_full_copy = df_full.copy()
    df_full_copy['date'] = pd.to_datetime(df_full_copy['date'])
    
    aligned_close = []
    for d in dates:
        close_val = df_full_copy[df_full_copy['date'] == d]['close'].values
        if len(close_val) > 0:
            aligned_close.append(close_val[0])
        else:
            aligned_close.append(None)
    
    aligned_df = pd.DataFrame({
        'date': dates,
        'close': aligned_close,
        'pred_return': predictions,
    })
    
    future_dates = prediction_data.get('prediction_dates', [])
    future_prices = prediction_data.get('future_prices_all', [])
    
    plot_price_forecast(ax5, df_full, aligned_df, future_dates, future_prices, 
                        pred_return, three_months_ago=dates[0] if len(dates) > 0 else None)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='生成回測結果可視化圖表')
    parser.add_argument('--backtest', required=True, help='回測結果JSON文件路徑')
    parser.add_argument('--prediction', help='預測結果JSON文件路徑')
    parser.add_argument('--price-data', help='價格數據CSV文件路徑')
    parser.add_argument('--output', required=True, help='輸出圖表路徑')
    parser.add_argument('--model-label', default='PatchTST', help='模型標籤')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("【回測可視化】")
    print("=" * 70)
    
    # 加載數據
    print(f"\n加載回測數據: {args.backtest}")
    backtest_data = load_backtest_data(args.backtest)
    
    prediction_data = {}
    if args.prediction:
        print(f"加載預測數據: {args.prediction}")
        prediction_data = load_backtest_data(args.prediction)
    
    df_full = None
    if args.price_data:
        print(f"加載價格數據: {args.price_data}")
        df_full = pd.read_csv(args.price_data)
        df_full['date'] = pd.to_datetime(df_full['date'])
    
    # 生成圖表
    print(f"\n生成可視化圖表...")
    fig = create_backtest_figure(backtest_data, prediction_data, df_full, 
                                  model_label=args.model_label)
    
    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight', facecolor=THEME['bg'])
    plt.close(fig)
    
    print(f"\n✓ 圖表已保存: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
