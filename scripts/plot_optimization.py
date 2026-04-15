#!/usr/bin/env python3
"""
優化結果可視化 - 與 complete_backtest_results 風格一致
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from scipy import stats
import json
from pathlib import Path
from datetime import datetime

def plot_optimization_results():
    """繪製優化結果圖表"""
    
    # 加載優化歷史
    history_files = sorted(Path('results').glob('optuna_history_*.json'))
    if not history_files:
        print("未找到優化歷史文件")
        return
    
    # 使用最新的歷史文件
    latest_history = history_files[-1]
    print(f"使用歷史文件: {latest_history.name}")
    with open(latest_history) as f:
        history = json.load(f)
    
    # 加載最佳參數
    best_params_files = sorted(Path('results').glob('optuna_best_params_*.json'))
    if best_params_files:
        with open(best_params_files[-1]) as f:
            best_info = json.load(f)
        best_params = best_info.get('best_params', {})
        best_score = best_info.get('best_cv_score', 0)
    else:
        best_params = {}
        best_score = 0
    
    # 解析歷史數據
    trials = [h['number'] for h in history]
    scores = [h['value'] if h['value'] is not None else 0 for h in history]
    
    # 計算累計最佳
    best_so_far = []
    current_best = 0
    for s in scores:
        if s > current_best:
            current_best = s
        best_so_far.append(current_best)
    
    # 提取參數變化
    d_models = [h['params'].get('d_model', 64) for h in history]
    n_layers = [h['params'].get('n_layers', 3) for h in history]
    dropouts = [h['params'].get('dropout', 0.2) for h in history]
    lrs = [h['params'].get('lr', 0.001) for h in history]
    
    # 創建圖表 - 與 complete_backtest_results 一致的 3x2 佈局
    fig = plt.figure(figsize=(16, 14), facecolor='#1a1a1a')
    
    # 大標題
    backtest_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"Optimization Results: {backtest_time} | Best CV Score: {best_score:.4f} | Trials: {len(trials)}",
        color="#ffffff", fontsize=16, fontweight="bold", y=0.995
    )
    
    plt.rcParams['axes.facecolor'] = '#2d2d2d'
    plt.rcParams['axes.edgecolor'] = '#666666'
    plt.rcParams['axes.labelcolor'] = '#cccccc'
    plt.rcParams['text.color'] = '#cccccc'
    plt.rcParams['xtick.color'] = '#cccccc'
    plt.rcParams['ytick.color'] = '#cccccc'
    plt.rcParams['grid.color'] = '#444444'
    plt.rcParams['figure.facecolor'] = '#1a1a1a'
    
    # 1. 優化分數曲線 (左上)
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(trials, scores, 'o-', color='#4a9eff', alpha=0.6, linewidth=1, markersize=4, label='Trial Score')
    ax1.plot(trials, best_so_far, '-', color='#51cf66', linewidth=2.5, label='Best So Far')
    ax1.axhline(y=best_score, color='#ffd43b', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Best: {best_score:.4f}')
    ax1.set_xlabel('Trial', color='#cccccc')
    ax1.set_ylabel('CV Score', color='#cccccc')
    ax1.set_title('Optimization Progress', color='#ffffff', fontweight='bold')
    ax1.legend(facecolor='none', edgecolor='none', labelcolor='#cccccc', loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 分數分布 (右上)
    ax2 = plt.subplot(3, 2, 2)
    valid_scores = [s for s in scores if s > 0]
    if len(valid_scores) > 5:
        mu, sigma = np.mean(valid_scores), np.std(valid_scores)
        n, bins, patches = ax2.hist(valid_scores, bins=min(20, len(valid_scores)//2), 
                                    alpha=0.6, color='#4a9eff', edgecolor='#2d2d2d', 
                                    density=True, label='Score Distribution')
        x = np.linspace(min(valid_scores), max(valid_scores), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), '--', color='#ffd43b', linewidth=2,
                label=f'Normal Fit (μ={mu:.3f}, σ={sigma:.3f})')
    ax2.axvline(x=best_score, color='#51cf66', linestyle='-', linewidth=2, 
               label=f'Best: {best_score:.4f}')
    ax2.set_xlabel('CV Score', color='#cccccc')
    ax2.set_ylabel('Density', color='#cccccc')
    ax2.set_title('Score Distribution', color='#ffffff', fontweight='bold')
    ax2.legend(facecolor='none', edgecolor='none', labelcolor='#cccccc', loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. d_model 參數探索 (中左)
    ax3 = plt.subplot(3, 2, 3)
    scatter = ax3.scatter(trials, d_models, c=scores, cmap='RdYlGn', 
                         s=80, alpha=0.7, edgecolors='#2d2d2d', linewidth=0.5)
    if best_params.get('d_model'):
        ax3.axhline(y=best_params['d_model'], color='#51cf66', linestyle='--', 
                   linewidth=2, label=f"Best: {best_params['d_model']}")
    ax3.set_xlabel('Trial', color='#cccccc')
    ax3.set_ylabel('d_model', color='#cccccc')
    ax3.set_title('d_model Exploration', color='#ffffff', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3, label='Score')
    cbar.ax.yaxis.set_tick_params(color='#cccccc')
    cbar.set_label('CV Score', color='#cccccc')
    ax3.grid(True, alpha=0.3)
    
    # 4. n_layers 參數探索 (中右)
    ax4 = plt.subplot(3, 2, 4)
    # 為每個 n_layers 計算平均得分
    unique_layers = sorted(set(n_layers))
    avg_scores = []
    for l in unique_layers:
        layer_scores = [scores[i] for i in range(len(scores)) if n_layers[i] == l and scores[i] > 0]
        avg_scores.append(np.mean(layer_scores) if layer_scores else 0)
    
    bars = ax4.bar([str(l) for l in unique_layers], avg_scores, 
                   color='#4a9eff', alpha=0.7, edgecolor='#666666')
    # 標記最佳
    if best_params.get('n_layers') in unique_layers:
        best_idx = unique_layers.index(best_params['n_layers'])
        bars[best_idx].set_color('#51cf66')
        bars[best_idx].set_alpha(0.9)
    ax4.set_xlabel('n_layers', color='#cccccc')
    ax4.set_ylabel('Average CV Score', color='#cccccc')
    ax4.set_title('n_layers Performance', color='#ffffff', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. dropout vs lr 熱力圖 (底部橫跨兩列)
    ax5 = plt.subplot(3, 1, 3)
    
    # 創建散點圖代替熱力圖（更適合離散數據）
    scatter = ax5.scatter(dropouts, lrs, c=scores, cmap='RdYlGn', 
                         s=150, alpha=0.7, edgecolors='#2d2d2d', linewidth=1)
    
    # 標記最佳參數點
    if best_params.get('dropout') and best_params.get('lr'):
        ax5.scatter(best_params['dropout'], best_params['lr'], 
                   s=400, c='#ffd43b', marker='*', edgecolors='white', linewidth=2,
                   label=f"Best: dropout={best_params['dropout']:.3f}, lr={best_params['lr']:.5f}",
                   zorder=5)
    
    ax5.set_xlabel('Dropout Rate', color='#cccccc')
    ax5.set_ylabel('Learning Rate', color='#cccccc')
    ax5.set_title('Dropout vs Learning Rate Exploration', color='#ffffff', fontweight='bold')
    ax5.set_yscale('log')
    ax5.set_ylim(min(lrs) * 0.5, max(lrs) * 2)
    cbar2 = plt.colorbar(scatter, ax=ax5, label='CV Score')
    cbar2.ax.yaxis.set_tick_params(color='#cccccc')
    cbar2.set_label('CV Score', color='#cccccc')
    ax5.legend(facecolor='none', edgecolor='none', labelcolor='#cccccc', loc='lower left')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = 'results/optimization_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"✓ 優化結果圖表已保存: {output_path}")
    
    # 輸出最佳參數摘要
    print("\n" + "=" * 70)
    print("【優化完成】最佳參數:")
    print("=" * 70)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\n  最佳 CV 分數: {best_score:.4f}")

if __name__ == '__main__':
    plot_optimization_results()
