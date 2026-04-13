#!/usr/bin/env python3
"""
每日自動回測腳本
流程: 更新數據 → 運行回測 → 輸出結果
"""

import subprocess
import sys
import os
from datetime import datetime
import json

def run_command(cmd, description):
    """運行命令並顯示結果"""
    print("\n" + "=" * 70)
    print(f"【{description}】")
    print("=" * 70)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def main():
    """主函數"""
    start_time = datetime.now()
    
    print("\n" + "🔥" * 35)
    print(f"  QuantForecast 每日自動回測")
    print(f"  開始時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔥" * 35)
    
    # 切換到項目目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    results = {}
    
    # 步驟 1: 更新數據
    print("\n" + "📊" * 35)
    success = run_command("python3 update_data.py", "步驟 1/3: 更新港股數據")
    results['data_update'] = '成功' if success else '失敗'
    
    if not success:
        print("\n❌ 數據更新失敗，停止後續流程")
        return 1
    
    # 步驟 2: 運行回測
    print("\n" + "📈" * 35)
    success = run_command("python3 complete_backtest.py", "步驟 2/3: 運行回測")
    results['backtest'] = '成功' if success else '失敗'
    
    if not success:
        print("\n⚠️ 回測執行失敗")
    
    # 步驟 3: 讀取並顯示結果
    print("\n" + "📋" * 35)
    print("【步驟 3/3: 回測結果】")
    print("=" * 70)
    
    # 讀取未來預測結果
    pred_file = '../results/future_prediction.json'
    backtest_file = '../results/complete_backtest_results.json'
    
    try:
        # 讀取預測結果
        if os.path.exists(pred_file):
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
            
            print(f"\n📅 最新數據日期: {pred_data.get('latest_date', 'N/A')}")
            print(f"💰 當前價格: {pred_data.get('latest_close', 'N/A')} HKD")
            
            direction = pred_data.get('pred_direction', 'N/A')
            signal_emoji = {
                'UP': '🟢 看漲',
                'DOWN': '🔴 看跌',
                'FLAT': '🟡 盤整'
            }.get(direction, direction)
            print(f"📍 預測方向: {signal_emoji}")
            
            if 'future_return_pct' in pred_data:
                ret = pred_data['future_return_pct']
                ret_str = f"+{ret:.2f}%" if ret >= 0 else f"{ret:.2f}%"
                print(f"📈 T+5 預期收益: {ret_str}")
            if 'future_price_day5' in pred_data:
                print(f"🎯 T+5 目標價: {pred_data['future_price_day5']:.2f} HKD")
            if pred_data.get('future_prices_all'):
                prices = pred_data['future_prices_all']
                dates = pred_data.get('prediction_dates', [])
                print(f"\n📊 未來5天預測:")
                for i, (d, p) in enumerate(zip(dates, prices)):
                    print(f"   {d}: {p:.2f} HKD")
        
        # 讀取回測統計
        if os.path.exists(backtest_file):
            with open(backtest_file, 'r') as f:
                backtest_data = json.load(f)
            
            print(f"\n📈 回測統計:")
            print(f"   總收益: {backtest_data.get('total_return_pct', 0):.2f}%")
            print(f"   最大回撤: {backtest_data.get('max_drawdown_pct', 0):.2f}%")
            print(f"   夏普比率: {backtest_data.get('sharpe_ratio', 0):.2f}")
            print(f"   勝率: {backtest_data.get('win_rate_pct', 0):.1f}%")
        
        results['result_display'] = '成功'
    except Exception as e:
        print(f"讀取結果失敗: {e}")
        results['result_display'] = '失敗'
    
    # 總結
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "✅" * 35)
    print(f"  執行完成")
    print(f"  結束時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  總耗時: {duration:.1f} 秒")
    print("\n  執行摘要:")
    for step, status in results.items():
        icon = "✓" if status == "成功" else "✗"
        print(f"    {icon} {step}: {status}")
    print("✅" * 35)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
