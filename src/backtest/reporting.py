"""
報告生成模塊
處理回測結果可視化和文件保存
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Dict
import logging

from src.backtest.engine import BacktestResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """回測報告生成器"""
    
    def __init__(self, dark_mode: bool = True):
        self.dark_mode = dark_mode
        
    def generate(self, result: BacktestResult, output_path: str) -> None:
        """
        生成完整回測報告（圖表 + JSON）
        
        Args:
            result: 回測結果
            output_path: PNG 輸出路徑（JSON 會輸出到同目錄）
        """
        logger.info("生成回測報告...")
        
        # 生成圖表
        self._plot_results(result, output_path)
        
        # 保存 JSON 結果
        json_path = output_path.replace('.png', '.json')
        self._save_json(result, json_path)
        
        logger.info(f"✓ 報告生成完成: {output_path}, {json_path}")
    
    def generate_future_prediction_report(
        self,
        prediction: Dict,
        output_path: str
    ) -> None:
        """保存未來預測結果"""
        with open(output_path, 'w') as f:
            json.dump(prediction, f, indent=2)
        logger.info(f"✓ 預測結果已保存: {output_path}")
    
    def _plot_results(self, result: BacktestResult, output_path: str) -> None:
        """繪製回測結果圖表"""
        if self.dark_mode:
            self._setup_dark_theme()
        
        fig = plt.figure(figsize=(16, 14), facecolor='#1a1a1a' if self.dark_mode else 'white')
        
        aligned_df = result.aligned_df
        equity_curve = result.equity_curve
        initial_capital = result.initial_capital
        test_directions = aligned_df['actual_direction'].values
        pred_directions = aligned_df['pred_direction'].values
        strategy_returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # 1. 混淆矩陣
        ax1 = plt.subplot(3, 2, 1)
        self._plot_confusion_matrix(ax1, test_directions, pred_directions)
        
        # 2. 每日收益分布
        ax2 = plt.subplot(3, 2, 2)
        self._plot_return_distribution(ax2, strategy_returns)
        
        # 3. 預測 vs 實際
        ax3 = plt.subplot(3, 2, 3)
        self._plot_prediction_scatter(ax3, aligned_df)
        
        # 4. 累計收益
        ax4 = plt.subplot(3, 2, 4)
        self._plot_cumulative_return(ax4, aligned_df, equity_curve, initial_capital)
        
        # 5. 股價預測（底部大圖）
        ax5 = plt.subplot(3, 1, 3)
        self._plot_price_forecast(ax5, aligned_df)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='#1a1a1a' if self.dark_mode else 'white')
        plt.close(fig)
    
    def _setup_dark_theme(self) -> None:
        """設置深色主題"""
        plt.rcParams['axes.facecolor'] = '#2d2d2d'
        plt.rcParams['axes.edgecolor'] = '#666666'
        plt.rcParams['axes.labelcolor'] = '#cccccc'
        plt.rcParams['text.color'] = '#cccccc'
        plt.rcParams['xtick.color'] = '#cccccc'
        plt.rcParams['ytick.color'] = '#cccccc'
        plt.rcParams['grid.color'] = '#444444'
        plt.rcParams['figure.facecolor'] = '#1a1a1a'
    
    def _plot_confusion_matrix(self, ax, y_true, y_pred) -> None:
        """繪製混淆矩陣"""
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, interpolation='nearest', cmap='YlOrRd')
        ax.figure.colorbar(im, ax=ax, label='Count')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Down', 'Up'])
        ax.set_yticklabels(['Down', 'Up'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix (Acc: {accuracy_score(y_true, y_pred):.1%})')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=16, fontweight='bold')
    
    def _plot_return_distribution(self, ax, returns) -> None:
        """繪製收益分布"""
        returns_pct = returns * 100
        mu, sigma = np.mean(returns_pct), np.std(returns_pct)
        n, bins, patches = ax.hist(returns_pct, bins=30, alpha=0.6, color='#4a9eff',
                                    edgecolor='#2d2d2d', density=True, label='Histogram')
        x = np.linspace(bins[0], bins[-1], 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), '--', color='#ffd43b', linewidth=2,
               label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})')
        ax.axvline(x=0, color='#ff6b6b', linestyle='-', linewidth=1.5, alpha=0.7, label='Zero')
        ax.axvline(x=mu, color='#51cf66', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Mean: {mu:.3f}%')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Density')
        ax.set_title('Daily Return Distribution')
        ax.legend(facecolor='none', edgecolor='none', labelcolor='#cccccc', loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_prediction_scatter(self, ax, aligned_df) -> None:
        """繪製預測 vs 實際散點圖"""
        up_indices = np.where(aligned_df['actual_direction'].values == 1)[0]
        down_indices = np.where(aligned_df['actual_direction'].values == 0)[0]
        pred_returns = aligned_df['pred_return'].values * 100
        actual_returns = aligned_df['actual_return'].values * 100
        
        ax.scatter(pred_returns[up_indices], actual_returns[up_indices],
                  c='#51cf66', alpha=0.6, edgecolors='#2d2d2d', linewidth=0.5,
                  label='Actual Up', s=30)
        ax.scatter(pred_returns[down_indices], actual_returns[down_indices],
                  c='#ff6b6b', alpha=0.6, edgecolors='#2d2d2d', linewidth=0.5,
                  label='Actual Down', s=30)
        ax.plot([-20, 20], [-20, 20], '#4a9eff', linestyle='--', linewidth=1.5, label='Perfect Prediction')
        ax.axhline(y=0, color='#666666', linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='#666666', linestyle='-', alpha=0.5)
        ax.set_xlabel('Predicted Return (%)')
        ax.set_ylabel('Actual Return (%)')
        ax.set_title('Prediction vs Actual')
        ax.legend(facecolor='none', edgecolor='none', labelcolor='#cccccc', loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_cumulative_return(self, ax, aligned_df, equity_curve, initial_capital) -> None:
        """繪製累計收益對比"""
        buyhold_curve = initial_capital * (aligned_df['close'] / aligned_df['close'].iloc[0])
        strat_cumret = (equity_curve / initial_capital - 1) * 100
        buyhold_cumret = (buyhold_curve / initial_capital - 1) * 100
        
        line1, = ax.plot(aligned_df['date'], strat_cumret, linewidth=2.5, color='#4a9eff', label='Strategy')
        line2, = ax.plot(aligned_df['date'], buyhold_cumret, linewidth=2, color='#888888',
                        linestyle='--', label='Buy & Hold')
        ax.axhline(y=0, color='#666666', linestyle='-', linewidth=1)
        ax.fill_between(aligned_df['date'], strat_cumret, 0, where=(strat_cumret > 0), alpha=0.3, color='#51cf66')
        ax.fill_between(aligned_df['date'], strat_cumret, 0, where=(strat_cumret <= 0), alpha=0.3, color='#ff6b6b')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_xlabel('Date')
        ax.set_title(f'Cumulative Return (Long: {np.mean(aligned_df["pred_direction"])*100:.1f}%)')
        ax.grid(True, alpha=0.3)
        
        # 交易信號疊加
        ax4b = ax.twinx()
        signals = aligned_df['pred_direction'].values
        y_min, y_max = ax.get_ylim()
        line_y = y_min + (y_max - y_min) * 0.01
        
        dates_num = mdates.date2num(aligned_df['date'].values)
        points = np.array([dates_num, np.full(len(signals), line_y)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = ['#51cf66' if signals[i+1] == 1 else '#ff6b6b' for i in range(len(signals)-1)]
        
        lc = LineCollection(segments, colors=colors, linewidths=2, capstyle='butt')
        ax4b.add_collection(lc)
        ax4b.set_ylim(y_min, y_max)
        ax4b.set_yticks([])
        
        legend_elements = [
            line1,
            line2,
            Line2D([0], [0], color='#51cf66', linewidth=2, label='Long Position'),
            Line2D([0], [0], color='#ff6b6b', linewidth=2, label='Cash Position')
        ]
        ax.legend(handles=legend_elements, loc='lower left', facecolor='none',
                 edgecolor='none', labelcolor='#cccccc')
    
    def _plot_price_forecast(self, ax, aligned_df) -> None:
        """繪製股價預測圖"""
        # 簡化版本，只繪製歷史預測價格
        aligned_df['pred_price_T5'] = aligned_df['close'] * (1 + aligned_df['pred_return'])
        
        ax.plot(aligned_df['date'], aligned_df['close'],
               linewidth=2, color='#4a9eff', alpha=0.7, label='Actual Price', zorder=1)
        ax.plot(aligned_df['date_T5'], aligned_df['pred_price_T5'],
               linewidth=2.5, color='#ffd43b', linestyle='--', alpha=0.9,
               label='Predicted Price @T+5', zorder=3)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (HKD)')
        ax.set_title('Stock Price: Actual vs Predicted (T+5)')
        ax.legend(facecolor='none', edgecolor='none', labelcolor='#cccccc', loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _save_json(self, result: BacktestResult, output_path: str) -> None:
        """保存 JSON 結果"""
        data = {
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_return_pct': result.total_return_pct,
            'buyhold_return_pct': result.buyhold_return_pct,
            'excess_return_pct': result.excess_return_pct,
            'max_drawdown_pct': result.max_drawdown_pct,
            'volatility_annual_pct': result.volatility_annual_pct,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'calmar_ratio': result.calmar_ratio,
            'total_trades': result.total_trades,
            'win_count': result.win_count,
            'loss_count': result.loss_count,
            'win_rate_pct': result.win_rate_pct,
            'test_accuracy': result.test_accuracy,
            'test_samples': result.test_samples,
            'date_range': result.date_range
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
