#!/usr/bin/env python3
"""
QuantForecast 優化監控腳本
運行優化 + 定期報告進度 + 生成結果圖表
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import threading
import queue

PROJECT_DIR = "/root/.openclaw/workspace/quantforecast"
RESULTS_DIR = f"{PROJECT_DIR}/results"
MODELS_DIR = f"{PROJECT_DIR}/models"

def monitor_optimization():
    """
    監控優化進程並定期報告
    """
    print("=" * 70)
    print("【QuantForecast Walk-forward CV 優化監控】")
    print("=" * 70)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"預計耗時: 30-40 分鐘 (20 trials)")
    print("=" * 70)
    
    os.chdir(PROJECT_DIR)
    
    # 啟動優化進程
    process = subprocess.Popen(
        ["python3", "scripts/optimize.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # 讀取輸出
    output_lines = []
    last_report_time = time.time()
    report_interval = 60  # 每60秒報告一次
    
    best_score_so_far = None
    trials_completed = 0
    
    try:
        for line in process.stdout:
            line = line.strip()
            output_lines.append(line)
            
            # 解析關鍵信息
            if "Trial" in line and "Score:" in line:
                # 提取 trial 信息
                try:
                    parts = line.split("|")
                    for part in parts:
                        if "Best:" in part:
                            best_score_so_far = float(part.split(":")[1].strip().split()[0])
                        if "Trial" in part and "/20" in part:
                            trial_str = part.split("/20")[0].split()[-1]
                            trials_completed = int(trial_str)
                except:
                    pass
            
            # 打印輸出
            print(line)
            
            # 定期報告
            current_time = time.time()
            if current_time - last_report_time >= report_interval:
                elapsed = current_time - start_time if 'start_time' in locals() else 0
                if best_score_so_far:
                    print(f"\n📊 [進度報告] Trials: {trials_completed}/20 | Best Score: {best_score_so_far:.4f} | Elapsed: {elapsed/60:.1f}min\n")
                last_report_time = current_time
            
            # 記錄開始時間
            if 'start_time' not in locals():
                start_time = time.time()
        
        process.wait()
        
        if process.returncode == 0:
            print("\n" + "=" * 70)
            print("【優化完成】")
            print("=" * 70)
            return True
        else:
            print(f"\n❌ 優化進程異常退出 (code: {process.returncode})")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ 用戶中斷")
        process.terminate()
        return False

def generate_optimization_chart():
    """
    生成優化結果圖表
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n" + "=" * 70)
    print("【生成優化結果圖表】")
    print("=" * 70)
    
    # 查找最新的優化歷史文件
    history_files = sorted(Path(RESULTS_DIR).glob("optuna_history_*.json"))
    best_files = sorted(Path(RESULTS_DIR).glob("optuna_best_params_*.json"))
    
    if not history_files:
        print("❌ 未找到優化歷史文件")
        return False
    
    latest_history = history_files[-1]
    latest_best = best_files[-1] if best_files else None
    
    print(f"✓ 使用歷史文件: {latest_history.name}")
    
    # 讀取數據
    with open(latest_history, 'r') as f:
        trials = json.load(f)
    
    best_params = None
    if latest_best:
        with open(latest_best, 'r') as f:
            best_params = json.load(f)
    
    # 提取數據
    trial_numbers = [t['number'] for t in trials]
    trial_values = [t['value'] for t in trials]
    
    # 計算最佳值累積
    best_values = []
    best_so_far = float('-inf')
    for v in trial_values:
        if v > best_so_far:
            best_so_far = v
        best_values.append(best_so_far)
    
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#1a1a1a')
    plt.rcParams['axes.facecolor'] = '#2d2d2d'
    plt.rcParams['axes.edgecolor'] = '#666666'
    plt.rcParams['axes.labelcolor'] = '#cccccc'
    plt.rcParams['text.color'] = '#cccccc'
    plt.rcParams['xtick.color'] = '#cccccc'
    plt.rcParams['ytick.color'] = '#cccccc'
    plt.rcParams['grid.color'] = '#444444'
    plt.rcParams['figure.facecolor'] = '#1a1a1a'
    
    # 標題信息
    timestamp = latest_history.stem.split('_')[-2] + '_' + latest_history.stem.split('_')[-1] if latest_best else 'unknown'
    fig.suptitle(
        f"Walk-forward CV Optimization Results | {timestamp} | Trials: {len(trials)}",
        color="#ffffff", fontsize=14, fontweight="bold"
    )
    
    # 1. Trial 分數散點圖 + 最佳值曲線
    ax1 = axes[0, 0]
    ax1.scatter(trial_numbers, trial_values, c='#4a9eff', alpha=0.6, s=50, label='Trial Score')
    ax1.plot(trial_numbers, best_values, color='#51cf66', linewidth=2, marker='o', markersize=4, label='Best So Far')
    ax1.axhline(y=np.mean(trial_values), color='#ffd43b', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(trial_values):.4f}')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('CV Score')
    ax1.set_title(f'Optimization Progress (Best: {best_so_far:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 參數分布 - d_model
    ax2 = axes[0, 1]
    d_models = [t['params'].get('d_model', 32) for t in trials]
    ax2.scatter(d_models, trial_values, c=trial_values, cmap='viridis', s=60, alpha=0.7)
    ax2.set_xlabel('d_model')
    ax2.set_ylabel('CV Score')
    ax2.set_title('Score vs d_model')
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Score')
    
    # 3. 參數分布 - n_layers
    ax3 = axes[1, 0]
    n_layers = [t['params'].get('n_layers', 2) for t in trials]
    ax3.scatter(n_layers, trial_values, c=trial_values, cmap='plasma', s=60, alpha=0.7)
    ax3.set_xlabel('n_layers')
    ax3.set_ylabel('CV Score')
    ax3.set_title('Score vs n_layers')
    ax3.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar2.set_label('Score')
    
    # 4. 最佳參數表格
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if best_params:
        params_text = "Best Parameters:\n\n"
        bp = best_params.get('best_params', {})
        for k, v in bp.items():
            params_text += f"{k}: {v}\n"
        params_text += f"\nBest CV Score: {best_params.get('best_cv_score', 'N/A'):.4f}\n"
        params_text += f"Final Test Acc: {best_params.get('final_test_accuracy', 'N/A'):.2%}\n"
        params_text += f"Total Trials: {best_params.get('n_trials', len(trials))}"
    else:
        params_text = "No best params found"
    
    ax4.text(0.1, 0.9, params_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', color='white',
             family='monospace', bbox=dict(boxstyle='round', facecolor='#2d2d2d', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存圖表
    output_path = f"{RESULTS_DIR}/optimization_results_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"✓ 優化結果圖表已保存: {output_path}")
    
    return output_path

def main():
    """主函數"""
    start_time = time.time()
    
    # 運行優化
    success = monitor_optimization()
    
    if success:
        # 生成圖表
        chart_path = generate_optimization_chart()
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("【全部完成】")
        print("=" * 70)
        print(f"總耗時: {elapsed/60:.1f} 分鐘")
        if chart_path:
            print(f"結果圖表: {chart_path}")
        print("\n可以運行回測了:")
        print("  python scripts/backtest.py")
        return 0
    else:
        return 1

if __name__ == '__main__':
    sys.exit(main())
