#!/usr/bin/env python3
"""
QuantForecast 每日回測 + 微信推送腳本
流程: 更新數據 → 運行回測 → 讀取結果 → 發送微信
"""

import subprocess
import sys
import os
from datetime import datetime
import json

# 配置
PROJECT_DIR = "/root/.openclaw/workspace/quantforecast"
RESULTS_DIR = f"{PROJECT_DIR}/results"
CHART_PATH = f"{RESULTS_DIR}/complete_backtest_results.png"
BACKTEST_JSON = f"{RESULTS_DIR}/complete_backtest_results.json"
PREDICTION_JSON = f"{RESULTS_DIR}/future_prediction.json"

def run_backtest():
    """運行回測"""
    print("\n" + "=" * 60)
    print("【QuantForecast 每日回測】")
    print("=" * 60)
    
    os.chdir(PROJECT_DIR)
    
    # 更新數據
    print("\n📊 步驟 1/3: 更新數據...")
    result = subprocess.run(
        ["python3", "scripts/update_realtime.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ 數據更新失敗: {result.stderr}")
        return False
    print("✓ 數據更新完成")
    
    # 運行回測
    print("\n📈 步驟 2/3: 運行回測...")
    result = subprocess.run(
        ["python3", "scripts/backtest.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ 回測失敗: {result.stderr}")
        return False
    print("✓ 回測完成")
    
    return True

def generate_report():
    """生成報告文字"""
    try:
        with open(BACKTEST_JSON, 'r') as f:
            backtest = json.load(f)
        with open(PREDICTION_JSON, 'r') as f:
            pred = json.load(f)
        
        backtest_time = backtest.get('backtest_timestamp', datetime.now().strftime('%Y-%m-%d %H:%M'))
        signal = "看漲 📈" if pred.get('pred_direction') == 'UP' else "看跌 📉"
        current_price = pred.get('latest_close', 0)
        future_return = pred.get('future_return_pct', 0)
        t1_target = current_price * (1 + future_return / 5 / 100)
        
        report = f"""📊 QuantForecast 每日回測報告

📅 回測時間: {backtest_time}
📈 T+1 信號: {signal}
💰 當前價格: {current_price:.2f} HKD
🎯 目標價格: {t1_target:.2f} HKD ({future_return/5:+.2f}%)

📉 策略收益: {backtest.get('total_return_pct', 0):+.2f}%
📊 Buy&Hold: {backtest.get('buyhold_return_pct', 0):+.2f}%
📈 超額收益: {backtest.get('excess_return_pct', 0):+.2f}%

🎯 測試準確率: {backtest.get('test_accuracy', 0)*100:.1f}%
📊 夏普比率: {backtest.get('sharpe_ratio', 0):.2f}
📉 最大回撤: {backtest.get('max_drawdown_pct', 0):.2f}%
💼 總交易次數: {backtest.get('total_trades', 0)}次

⚠️ 免責聲明：本報告僅供研究學習，不構成投資建議"""
        
        return report
    except Exception as e:
        return f"生成報告失敗: {e}"

def main():
    """主函數"""
    start_time = datetime.now()
    print(f"開始時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 運行回測
    if not run_backtest():
        print("\n❌ 回測失敗，退出")
        return 1
    
    # 生成報告
    print("\n📝 步驟 3/3: 生成報告...")
    report = generate_report()
    print(report)
    
    # 保存報告供 OpenClaw 讀取
    report_file = f"{RESULTS_DIR}/daily_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # 輸出關鍵信息供 cron job 解析
    print("\n" + "=" * 60)
    print("【輸出信息】")
    print("=" * 60)
    print(f"CHART_PATH: {CHART_PATH}")
    print(f"REPORT_PATH: {report_file}")
    print(f"BACKTEST_JSON: {BACKTEST_JSON}")
    print(f"PREDICTION_JSON: {PREDICTION_JSON}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n✅ 完成，耗時: {duration:.1f} 秒")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
