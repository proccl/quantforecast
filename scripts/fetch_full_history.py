#!/usr/bin/env python3
"""
獲取小米(1810.HK)完整歷史數據
從 2018年7月9日 上市至今
"""

import pandas as pd
import akshare as ak
from datetime import datetime
import os
import sys

def fetch_full_history(symbol='01810', data_file='data/xiaomi_real.csv'):
    """
    獲取完整歷史數據並保存
    """
    print("=" * 70)
    print("【獲取小米完整歷史數據】")
    print("=" * 70)
    
    # 獲取全部歷史數據
    print(f"\n📊 從 akshare 獲取 {symbol} 全部歷史數據...")
    try:
        df = ak.stock_hk_daily(symbol=symbol, adjust='qfq')
        print(f"✓ 獲取成功: {len(df)} 條數據")
        print(f"  日期範圍: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    except Exception as e:
        print(f"✗ 獲取失敗: {e}")
        return False
    
    # 數據處理
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # 確保數據類型正確
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(int)
    
    # 格式化日期
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # 保存數據
    df.to_csv(data_file, index=False)
    
    print(f"\n✓ 數據已保存: {data_file}")
    print(f"  總共: {len(df)} 個交易日")
    print(f"  約 {len(df)/252:.1f} 年數據")
    
    # 計算IPO日期到現在
    first_date = pd.to_datetime(df['date'].iloc[0])
    last_date = pd.to_datetime(df['date'].iloc[-1])
    days_span = (last_date - first_date).days
    
    print(f"\n📈 數據統計:")
    print(f"  IPO日期: {first_date.strftime('%Y-%m-%d')} ({first_date.strftime('%A')})")
    print(f"  最新日期: {last_date.strftime('%Y-%m-%d')}")
    print(f"  總跨度: {days_span} 天 ({days_span/365:.1f} 年)")
    print(f"  價格範圍: {df['close'].min():.2f} ~ {df['close'].max():.2f} HKD")
    print(f"  平均成交量: {df['volume'].mean()/10000:.0f} 萬股")
    
    # 按年份統計
    df['year'] = pd.to_datetime(df['date']).dt.year
    yearly = df.groupby('year').size()
    print(f"\n📅 每年交易日數:")
    for year, count in yearly.items():
        print(f"  {year}: {count} 天")
    
    return True

if __name__ == '__main__':
    # 切換到項目目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 備份原數據
    data_file = '../data/xiaomi_real.csv'
    if os.path.exists(data_file):
        backup_file = f"../data/xiaomi_real_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"\n💾 備份原數據到: {backup_file}")
        os.rename(data_file, backup_file)
    
    # 獲取完整數據
    success = fetch_full_history(data_file=data_file)
    
    sys.exit(0 if success else 1)
