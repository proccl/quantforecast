#!/usr/bin/env python3
"""
港股數據更新腳本 - 小米(1810.HK)
每天收市後運行，獲取最新數據並追加到數據集

數據源優先級:
1. 收市後: 使用 akshare stock_hk_daily（歷史日線，有延遲但完整）
2. 交易時段: 使用新浪/騰訊實時行情（即時但需手動合成日線）
"""

import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import os
import sys
import requests
import json

def get_hk_realtime_sina(symbol='01810'):
    """新浪港股實時行情"""
    url = f'https://hq.sinajs.cn/list=rt_hk{symbol}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://finance.sina.com.cn'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        text = resp.text
        
        if 'hq_str_rt_hk' not in text or 'var hq_str_rt_hk' in text and '"";' in text:
            return None
        
        data_part = text.split('=\"')[1].split('\";')[0]
        fields = data_part.split(',')
        
        return {
            'name': fields[1],
            'open': float(fields[2]),
            'pre_close': float(fields[3]),
            'high': float(fields[4]),
            'low': float(fields[5]),
            'close': float(fields[6]),  # 當前價作為收盤價
            'change': float(fields[7]),
            'change_pct': float(fields[8]),
            'volume': int(float(fields[12])) if len(fields) > 12 and fields[12] else 0,
            'amount': float(fields[11]) if len(fields) > 11 and fields[11] else 0,
            'date': fields[17] if len(fields) > 17 else datetime.now().strftime('%Y-%m-%d'),
            'time': fields[18] if len(fields) > 18 else '',
        }
    except Exception as e:
        print(f"新浪數據獲取失敗: {e}")
        return None

def get_hk_realtime_tencent(symbol='01810'):
    """騰訊港股實時行情（備用）"""
    url = f'https://qt.gtimg.cn/q=r_hk{symbol.zfill(5)}'
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        text = resp.text
        
        if 'v_r_hk' not in text:
            return None
        
        data_part = text.split('=\"')[1].split('\";')[0]
        fields = data_part.split('~')
        
        # 騰訊返回的成交量是「股」數，需要轉換為「手」（1手=100股）
        volume_raw = float(fields[6]) if fields[6] else 0
        
        return {
            'name': fields[1],
            'open': float(fields[5]),
            'high': float(fields[33]),
            'low': float(fields[34]),
            'close': float(fields[3]),
            'pre_close': float(fields[4]),
            'change_pct': float(fields[32]),
            'volume': int(volume_raw / 100),  # 轉換為手
            'amount': float(fields[37]) * 10000 if len(fields) > 37 and fields[37] else 0,
            'datetime': fields[30] if len(fields) > 30 else datetime.now().strftime('%Y-%m-%d'),
        }
    except Exception as e:
        print(f"騰訊數據獲取失敗: {e}")
        return None

def get_today_data_from_realtime(symbol='01810'):
    """從實時行情獲取今日數據（收市後使用）"""
    # 優先使用新浪
    data = get_hk_realtime_sina(symbol)
    if data:
        return data
    
    # 備用騰訊
    data = get_hk_realtime_tencent(symbol)
    if data:
        return data
    
    return None

def update_hk_stock_data(symbol='01810', data_file='../data/xiaomi_real.csv'):
    """
    更新港股數據
    symbol: 港股代碼 (01810 = 小米)
    data_file: 本地數據文件路徑
    """
    print("=" * 60)
    print(f"【數據更新】小米(1810.HK) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 讀取現有數據
    if os.path.exists(data_file):
        existing_df = pd.read_csv(data_file)
        existing_df['date'] = pd.to_datetime(existing_df['date'])
        last_date = existing_df['date'].iloc[-1]
        print(f"\n現有數據最後日期: {last_date.strftime('%Y-%m-%d')}")
        print(f"現有數據條數: {len(existing_df)}")
    else:
        existing_df = None
        last_date = None
        print("\n現有數據文件不存在，將創建新文件")
    
    # 方法1: 嘗試從 akshare 獲取歷史數據
    print(f"\n【方法1】從 akshare 獲取 {symbol} 數據...")
    try:
        new_df = ak.stock_hk_daily(symbol=symbol, adjust='qfq')
        new_df['date'] = pd.to_datetime(new_df['date'])
        print(f"✓ 獲取到 {len(new_df)} 條歷史數據")
        print(f"  akshare 最新日期: {new_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
        
        # 檢查 akshare 是否有今天數據
        today_str = datetime.now().strftime('%Y-%m-%d')
        today_data_ak = new_df[new_df['date'] == today_str]
        
        if len(today_data_ak) == 0 and last_date and last_date.strftime('%Y-%m-%d') < today_str:
            print(f"\n⚠ akshare 尚未更新今天({today_str})數據，嘗試實時數據源...")
            
            # 方法2: 從實時行情獲取今天數據
            print(f"\n【方法2】從新浪/騰訊實時行情獲取今日數據...")
            realtime_data = get_today_data_from_realtime(symbol)
            
            if realtime_data:
                print(f"✓ 獲取到實時數據:")
                print(f"  日期: {realtime_data['date']}")
                print(f"  開盤: {realtime_data['open']}")
                print(f"  最高: {realtime_data['high']}")
                print(f"  最低: {realtime_data['low']}")
                print(f"  收盤: {realtime_data['close']}")
                print(f"  成交量: {realtime_data['volume']:,}")
                
                # 創建今天的數據記錄
                today_record = pd.DataFrame([{
                    'date': realtime_data['date'],
                    'open': realtime_data['open'],
                    'high': realtime_data['high'],
                    'low': realtime_data['low'],
                    'close': realtime_data['close'],
                    'volume': realtime_data['volume']
                }])
                
                # 合併數據
                new_df['date'] = new_df['date'].dt.strftime('%Y-%m-%d')
                today_record['date'] = pd.to_datetime(today_record['date']).dt.strftime('%Y-%m-%d')
                
                # 檢查是否已經有今天的數據
                if today_record['date'].iloc[0] not in new_df['date'].values:
                    new_df = pd.concat([new_df, today_record], ignore_index=True)
                    print(f"\n✓ 已將實時數據合併到數據集")
            else:
                print(f"✗ 實時數據獲取失敗")
        
    except Exception as e:
        print(f"✗ akshare 獲取失敗: {e}")
        print(f"\n嘗試從實時數據源獲取...")
        
        realtime_data = get_today_data_from_realtime(symbol)
        if realtime_data:
            new_df = pd.DataFrame([{
                'date': realtime_data['date'],
                'open': realtime_data['open'],
                'high': realtime_data['high'],
                'low': realtime_data['low'],
                'close': realtime_data['close'],
                'volume': realtime_data['volume']
            }])
            new_df['date'] = pd.to_datetime(new_df['date'])
        else:
            print("✗ 所有數據源都失敗")
            return False
    
    # 只保留需要的列
    new_df = new_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    if existing_df is not None:
        # 合併數據
        existing_df['date'] = existing_df['date'].dt.strftime('%Y-%m-%d')
        
        # 找到需要添加的新數據
        last_date_str = last_date.strftime('%Y-%m-%d') if last_date else None
        if last_date_str:
            new_records = new_df[new_df['date'] > last_date_str].copy()
        else:
            new_records = new_df.copy()
        
        if len(new_records) == 0:
            print(f"\n✓ 數據已是最新，無需更新")
            return True
        
        print(f"\n發現 {len(new_records)} 條新數據:")
        for _, row in new_records.iterrows():
            print(f"  {row['date']}: 收盤 {float(row['close']):.2f}, 成交量 {int(row['volume']):,}")
        
        # 合併數據
        combined_df = pd.concat([existing_df, new_records], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
        
        print(f"\n更新後數據條數: {len(combined_df)}")
    else:
        combined_df = new_df
        print(f"\n新創建數據文件，共 {len(combined_df)} 條記錄")
    
    # 保存數據
    combined_df['date'] = pd.to_datetime(combined_df['date']).dt.strftime('%Y-%m-%d')
    combined_df['open'] = combined_df['open'].astype(float)
    combined_df['high'] = combined_df['high'].astype(float)
    combined_df['low'] = combined_df['low'].astype(float)
    combined_df['close'] = combined_df['close'].astype(float)
    combined_df['volume'] = combined_df['volume'].astype(int)
    
    combined_df.to_csv(data_file, index=False)
    print(f"\n✓ 數據已保存到: {data_file}")
    print(f"  時間範圍: {combined_df['date'].iloc[0]} ~ {combined_df['date'].iloc[-1]}")
    
    return True

if __name__ == '__main__':
    # 切換到腳本所在目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = update_hk_stock_data()
    sys.exit(0 if success else 1)
