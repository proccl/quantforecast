#!/usr/bin/env python3
"""
回測腳本 - 支持命令行參數配置
整合所有回測功能到單一腳本
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from datetime import datetime
import json
import subprocess
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.patchtst import PatchTST
from src.config import get_config
from src.data.features import FeatureEngineer
from sklearn.metrics import accuracy_score


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='回測分析')
    parser.add_argument('--optimized', action='store_true', 
                       help='使用優化後的參數 (P75/4D/6SL)')
    parser.add_argument('--prob-threshold', type=float, default=0.5,
                       help='概率閾值 (默認: 0.5)')
    parser.add_argument('--holding-days', type=int, default=None,
                       help='持有天數 (默認: None, 每日調倉)')
    parser.add_argument('--stop-loss', type=float, default=0.15,
                       help='止損比例 (默認: 0.15)')
    parser.add_argument('--full-history', action='store_true',
                       help='使用全部歷史數據')
    parser.add_argument('--months', type=int, default=3,
                       help='回測月數 (默認: 3)')
    parser.add_argument('--model', type=str, default=None,
                       help='模型文件名')
    parser.add_argument('--output-prefix', type=str, default='complete_backtest',
                       help='輸出文件名前綴')
    return parser.parse_args()


def get_strategy_params(args):
    """根據命令行參數獲取策略配置"""
    if args.optimized:
        # 優化後的參數
        params = {
            'prob_threshold': 0.3792,  # P75
            'holding_days': 4,
            'stop_loss': 0.06,
            'prob_percentile': 75,
            'source': 'optimized'
        }
        print("\n【使用優化後的參數】")
        print(f"  概率閾值: {params['prob_threshold']:.4f}")
        print(f"  持有天數: {params['holding_days']}")
        print(f"  止損比例: {params['stop_loss']*100:.1f}%")
    else:
        # 命令行指定的參數
        params = {
            'prob_threshold': args.prob_threshold,
            'holding_days': args.holding_days,
            'stop_loss': args.stop_loss,
            'source': 'cli'
        }
        print("\n【使用命令行參數】")
        print(f"  概率閾值: {params['prob_threshold']}")
        if params['holding_days']:
            print(f"  持有天數: {params['holding_days']}")
            print(f"  止損比例: {params['stop_loss']*100:.1f}%")
        else:
            print("  模式: 每日調倉")
    
    return params


def run_backtest_with_params(aligned_df, params, initial_capital=100000):
    """
    使用指定參數運行回測
    
    params: {
        'prob_threshold': 0.3792,
        'holding_days': 4,  # None表示每日調倉
        'stop_loss': 0.06
    }
    """
    prob_threshold = params.get('prob_threshold', 0.5)
    holding_days = params.get('holding_days', None)
    stop_loss = params.get('stop_loss', 0.15)
    
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_date = None
    holding_counter = 0
    
    equity_curve = [initial_capital]
    trades = []
    
    for i in range(len(aligned_df)):
        current_date = aligned_df['date'].iloc[i]
        current_price = aligned_df['close'].iloc[i]
        prob_up = aligned_df['prob_up'].iloc[i]
        
        # 止損檢查（只在持有固定天數模式下）
        if holding_days is not None and position > 0:
            current_return = (current_price - entry_price) / entry_price
            holding_counter += 1
            
            # 止損
            if current_return < -stop_loss:
                capital = position * current_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'action': 'STOP_LOSS',
                    'return_pct': current_return * 100,
                    'holding_days': holding_counter
                })
                position = 0
                holding_counter = 0
            # 持有期滿
            elif holding_counter >= holding_days:
                capital = position * current_price
                trade_return = (current_price - entry_price) / entry_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'action': 'SELL_HOLDING',
                    'return_pct': trade_return * 100,
                    'holding_days': holding_counter
                })
                position = 0
                holding_counter = 0
        
        # 買入信號
        elif position == 0 and prob_up > prob_threshold:
            shares = capital / current_price
            position = shares
            entry_price = current_price
            entry_date = current_date
            holding_counter = 0
            trades.append({
                'date': current_date,
                'action': 'BUY',
                'price': current_price,
                'shares': shares,
                'confidence': prob_up
            })
        # 賣出信號 (只在每日調倉模式下)
        elif holding_days is None and position > 0 and prob_up < (1 - prob_threshold):
            capital = position * current_price
            trade_return = (current_price - entry_price) / entry_price
            trades.append({
                'entry_date': entry_date,
                'exit_date': current_date,
                'action': 'SELL',
                'price': current_price,
                'proceeds': capital,
                'return_pct': trade_return * 100,
                'holding_days': holding_counter
            })
            position = 0
            holding_counter = 0
        
        current_equity = capital if position == 0 else position * current_price
        equity_curve.append(current_equity)
    
    # 平倉
    final_capital = capital if position == 0 else position * aligned_df['close'].iloc[-1]
    
    return np.array(equity_curve), final_capital, trades


def main():
    args = parse_args()
    
    print("=" * 70)
    print("【完整回測分析】")
    print("=" * 70)
    
    # 獲取策略參數
    params = get_strategy_params(args)
    
    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = get_config('config/config.yaml')
    
    # 加載數據
    print("\n【數據加載】")
    df = pd.read_csv("data/xiaomi_real.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # 特徵工程（與訓練一致）
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df)
    feature_cols = engineer.get_feature_columns()
    df_clean = engineer.clean(df_features, feature_cols)
    
    print(f"\n【配置】")
    print(f"特徵數量: {len(feature_cols)}")
    
    # 選擇回測數據範圍
    test_df = df_clean.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    if args.full_history:
        print("\n【使用全部歷史數據】")
    else:
        latest_date = test_df['date'].iloc[-1]
        months_ago = latest_date - pd.Timedelta(days=30*args.months)
        test_df = test_df[test_df['date'] >= months_ago].reset_index(drop=True)
        print(f"\n【使用最近{args.months}個月數據】")
    
    print(f"樣本數: {len(test_df)}")
    print(f"時間範圍: {test_df['date'].iloc[0]} ~ {test_df['date'].iloc[-1]}")
    
    # 加載模型
    model_dir = Path(config.paths.model_dir)
    
    if args.model:
        model_path = model_dir / args.model
    else:
        CLASSIFICATION_MODEL = "patchtst_model_20260420_155712.pth"
        model_path = model_dir / CLASSIFICATION_MODEL
    
    if not model_path.exists():
        print(f"✗ 模型不存在: {model_path}")
        sys.exit(1)
    
    print(f"\n✓ 加載模型: {model_path.name}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_config' in checkpoint:
        loaded_config = checkpoint['model_config']
    else:
        loaded_config = {
            'seq_len': config.data.seq_len,
            'pred_len': config.data.pred_len,
            'd_model': config.model.d_model,
            'n_heads': config.model.n_heads,
            'n_layers': config.model.n_layers,
            'dropout': config.model.dropout,
            'patch_len': 5,
            'stride': 2,
            'd_ff': config.model.d_ff
        }
    
    model_type = loaded_config.get('head_type', 'regression')
    if 'head.0.weight' in checkpoint.get('model_state_dict', {}):
        model_type = 'classification'
        print(f"  檢測到: 分類模型")
    else:
        print(f"  檢測到: 回歸模型")
    
    seq_len = loaded_config['seq_len']
    pred_len = loaded_config['pred_len']
    
    model = PatchTST(
        n_features=len(feature_cols),
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=loaded_config['d_model'],
        n_heads=loaded_config['n_heads'],
        n_layers=loaded_config['n_layers'],
        dropout=loaded_config['dropout'],
        patch_len=loaded_config['patch_len'],
        stride=loaded_config['stride'],
        d_ff=loaded_config.get('d_ff', loaded_config['d_model'] * 2),
        head_type=model_type,
        use_revin=True
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 預測
    predictions = []
    actuals = []
    test_dates = []
    test_close = []
    test_directions = []
    
    with torch.no_grad():
        for i in range(len(test_df) - seq_len - pred_len + 1):
            seq_data = test_df[feature_cols].iloc[i:i+seq_len].values
            close_T = test_df['close'].iloc[i+seq_len]
            close_T5 = test_df['close'].iloc[i+seq_len:i+seq_len+pred_len].values[-1]
            target_return = close_T5 / close_T - 1
            
            x = torch.FloatTensor(seq_data).unsqueeze(0).to(device)
            pred = model(x)
            
            if model_type == 'classification':
                pred_probs = torch.softmax(pred, dim=1)
                prob_up = pred_probs[0][1].item()
                
                predictions.append({
                    'type': 'classification',
                    'prob_up': prob_up,
                    'prob_down': pred_probs[0][0].item(),
                    'pred_class': torch.argmax(pred, dim=1).item(),
                    'confidence': pred_probs[0][torch.argmax(pred, dim=1).item()].item()
                })
            else:
                pred_return = pred.squeeze().cpu().numpy()
                if len(pred_return.shape) > 0:
                    pred_return = pred_return[0] if pred_return.shape[0] > 0 else 0
                predictions.append(float(pred_return))
            
            actuals.append(float(target_return))
            test_dates.append(test_df['date'].iloc[i+seq_len])
            test_close.append(close_T)
            # 實際方向：T時刻價格 vs T-5時刻價格（對齊黃線顯示邏輯）
            if i >= pred_len:
                price_t_minus_5 = test_df['close'].iloc[i+seq_len-pred_len]
                actual_direction = 1 if close_T > price_t_minus_5 else 0
            else:
                # 前5個點沒有T-5數據，用 T+5 vs T 代替
                actual_direction = 1 if target_return > 0 else 0
            test_directions.append(actual_direction)
    
    test_directions = np.array(test_directions)
    
    if model_type == 'classification':
        prob_ups = [p['prob_up'] for p in predictions]
        # 轉換為預測收益率 (prob_up - 0.5) * 0.1
        pred_returns_for_display = [(p['prob_up'] - 0.5) * 0.1 for p in predictions]
    else:
        prob_ups = predictions
        pred_returns_for_display = predictions
    
    # 計算正確的預測方向：比較預測T+5價格 vs T-5實際價格
    pred_directions = []
    for i in range(len(test_close)):
        # 預測的 T+5 價格
        pred_price_t5 = test_close[i] * (1 + pred_returns_for_display[i])
        # T-5 的實際價格（如果存在的話）
        if i >= pred_len:
            price_t_minus_5 = test_close[i - pred_len]
            # 比較預測T+5 vs T-5
            pred_directions.append(1 if pred_price_t5 > price_t_minus_5 else 0)
        else:
            # 前5個點沒有T-5數據，使用當天比較
            pred_directions.append(1 if pred_returns_for_display[i] > 0 else 0)
    
    aligned_df = pd.DataFrame({
        'date': test_dates,
        'close': test_close,
        'pred_return': pred_returns_for_display,
        'actual_return': actuals,
        'prob_up': prob_ups,
        'pred_direction': pred_directions,
        'actual_direction': test_directions
    })
    
    print(f"\n對齊後樣本數: {len(aligned_df)}")
    print(f"預測看漲: {sum(pred_directions)} 次")
    print(f"預測看跌: {len(pred_directions) - sum(pred_directions)} 次")
    if model_type == 'classification':
        print(f"平均上漲概率: {np.mean(prob_ups):.3f}")
    
    # 運行回測
    print("\n【運行回測】")
    initial_capital = 100000
    equity_curve, final_capital, trades = run_backtest_with_params(
        aligned_df, params, initial_capital
    )
    
    total_return = (final_capital / initial_capital - 1) * 100
    
    # 買入持有基準
    buy_hold_shares = initial_capital / aligned_df['close'].iloc[0]
    buy_hold_final = buy_hold_shares * aligned_df['close'].iloc[-1]
    buy_hold_return = (buy_hold_final / initial_capital - 1) * 100
    
    # 計算風險指標
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    volatility = np.std(daily_returns) * np.sqrt(252) * 100
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100
    
    sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
    sortino_ratio = np.mean(daily_returns) / downside_std * np.sqrt(252)
    calmar_ratio = (total_return / 100) / (abs(max_drawdown) / 100 + 1e-8)
    
    # 交易統計
    buy_trades = [t for t in trades if t.get('action') == 'BUY']
    completed_trades = [t for t in trades if t.get('action') in ['SELL', 'STOP_LOSS', 'SELL_HOLDING']]
    
    win_count = sum(1 for t in completed_trades if t.get('return_pct', 0) > 0)
    loss_count = len(completed_trades) - win_count
    win_rate = win_count / max(len(completed_trades), 1) * 100
    
    print(f"\n【回測結果】")
    print(f"初始資金: {initial_capital:,.0f} HKD")
    print(f"最終資金: {final_capital:,.0f} HKD")
    print(f"策略總收益: {total_return:.2f}%")
    print(f"買入持有: {buy_hold_return:.2f}%")
    print(f"超額收益: {total_return - buy_hold_return:+.2f}%")
    
    print(f"\n【風險指標】")
    print(f"年化波動率: {volatility:.2f}%")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"夏普比率: {sharpe_ratio:.4f}")
    print(f"索提諾比率: {sortino_ratio:.4f}")
    print(f"卡爾瑪比率: {calmar_ratio:.4f}")
    
    print(f"\n【交易統計】")
    print(f"總交易次數: {len(buy_trades)} 次")
    print(f"盈利交易: {win_count} 次")
    print(f"虧損交易: {loss_count} 次")
    print(f"勝率: {win_rate:.2f}%")
    
    # 保存結果
    backtest_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 處理 predictions 用於保存
    if model_type == 'classification':
        predictions_for_save = [(p['prob_up'] - 0.5) * 0.1 for p in predictions]
        pred_probs_for_save = [p['prob_up'] for p in predictions]
    else:
        predictions_for_save = predictions
        pred_probs_for_save = []
    
    # 計算對齊的預測和實際收益率（用於散點圖）：T vs T-5
    pred_returns_aligned = []
    actual_returns_aligned = []
    # 計算對齊的預測價格和實際價格（用於散點圖）
    pred_prices_aligned = []
    actual_prices_aligned = []
    
    for i in range(len(test_close)):
        if i >= pred_len:
            # T-5 的價格
            price_t_minus_5 = test_close[i - pred_len]
            # T 的實際價格
            price_t = test_close[i]
            # 預測的 T 價格（用 T-5 的數據預測）
            pred_price_t = price_t_minus_5 * (1 + pred_returns_for_display[i])
            
            # 對齊的預測收益率：(預測T價格 / T-5價格) - 1
            pred_ret_aligned = (pred_price_t / price_t_minus_5) - 1
            # 對齊的實際收益率：(T價格 / T-5價格) - 1
            actual_ret_aligned = (price_t / price_t_minus_5) - 1
            
            pred_returns_aligned.append(pred_ret_aligned)
            actual_returns_aligned.append(actual_ret_aligned)
            pred_prices_aligned.append(pred_price_t)
            actual_prices_aligned.append(price_t)
        else:
            # 前5個點沒有T-5數據，用原始值
            pred_returns_aligned.append(pred_returns_for_display[i])
            actual_returns_aligned.append(actuals[i])
            pred_prices_aligned.append(test_close[i] * (1 + pred_returns_for_display[i]))
            actual_prices_aligned.append(test_close[i])
    
    backtest_data = {
        'dates': [d.strftime('%Y-%m-%d') for d in test_dates],
        'equity_curve': [float(v) for v in equity_curve],
        'buyhold_curve': [float(v) for v in (initial_capital * aligned_df['close'] / aligned_df['close'].iloc[0]).values],
        'predictions': [float(v) for v in predictions_for_save],
        'pred_probs': [float(v) for v in pred_probs_for_save] if pred_probs_for_save else [],
        'actuals': [float(v) for v in actuals],
        'pred_directions': [int(v) for v in aligned_df['pred_direction']],
        'actual_directions': [int(v) for v in aligned_df['actual_direction']],
        # 新增：對齊的收益率（用於散點圖）
        'pred_returns_aligned': [float(v) for v in pred_returns_aligned],
        'actual_returns_aligned': [float(v) for v in actual_returns_aligned],
        # 新增：對齊的價格（用於散點圖）
        'pred_prices_aligned': [float(v) for v in pred_prices_aligned],
        'actual_prices_aligned': [float(v) for v in actual_prices_aligned],
        'metrics': {
            'total_return_pct': float(total_return),
            'buyhold_return_pct': float(buy_hold_return),
            'excess_return_pct': float(total_return - buy_hold_return),
            'max_drawdown_pct': float(max_drawdown),
            'volatility_annual_pct': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'total_trades': len(buy_trades),
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate_pct': float(win_rate),
            'test_accuracy': float(accuracy_score(test_directions[:len(aligned_df)], aligned_df['pred_direction'])),
            'test_samples': len(aligned_df),
            'date_range': f"{test_dates[0].strftime('%Y-%m-%d')} ~ {test_dates[-1].strftime('%Y-%m-%d')}"
        },
        'strategy_params': params,
        'trades': [
            {
                k: (v.strftime('%Y-%m-%d %H:%M:%S') if hasattr(v, 'strftime') else 
                    (v.isoformat() if hasattr(v, 'isoformat') else v))
                for k, v in t.items()
            }
            for t in trades
        ]
    }
    
    results_dir = Path(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = results_dir / f"{args.output_prefix}_results.json"
    with open(json_path, 'w') as f:
        json.dump(backtest_data, f, indent=2)
    print(f"\n✓ 回測數據已保存: {json_path}")
    
    # 未來5天預測 - 用過去5天的數據分別做T+5預測
    print("\n" + "=" * 70)
    print("【未來5天預測】")
    print("=" * 70)
    
    latest_close = df_clean['close'].iloc[-1]
    latest_date = df_clean['date'].iloc[-1]
    
    # 生成未來5個交易日日期
    future_dates = []
    current_date = latest_date
    while len(future_dates) < pred_len:
        current_date = current_date + pd.Timedelta(days=1)
        if current_date.weekday() < 5:
            future_dates.append(current_date)
    
    # 用過去5天的數據分別預測對應的未來日期
    # 例如：用 T-4 預測 T+1, 用 T-3 預測 T+2, ..., 用 T 預測 T+5
    historical_predictions = []
    
    with torch.no_grad():
        for i in range(pred_len):
            # 獲取歷史數據起點（從最新往前推 i+1 天）
            hist_start_idx = len(df_clean) - seq_len - (pred_len - 1 - i)
            
            if hist_start_idx >= 0:
                hist_data = df_clean[feature_cols].iloc[hist_start_idx:hist_start_idx+seq_len].values
                hist_close = df_clean['close'].iloc[hist_start_idx+seq_len-1]  # 該段數據的最後收盤價
                
                x = torch.FloatTensor(hist_data).unsqueeze(0).to(device)
                pred = model(x)
                
                if model_type == 'classification':
                    pred_probs = torch.softmax(pred, dim=1)
                    prob_up = pred_probs[0][1].item()
                    # 轉換為收益率
                    pred_return = (prob_up - 0.5) * 0.1
                    if prob_up < 0.5:
                        pred_return = -abs(pred_return)
                else:
                    pred_values = pred.squeeze().cpu().numpy()
                    pred_return = float(pred_values[0]) if pred_values.size > 0 else 0
                    prob_up = None
                
                # 計算預測價格（基於該段歷史數據的最後收盤價）
                pred_price = hist_close * (1 + pred_return)
                
                hist_date = df_clean['date'].iloc[hist_start_idx+seq_len-1]
                target_date = future_dates[i]
                
                historical_predictions.append({
                    'hist_date': hist_date.strftime('%Y-%m-%d'),
                    'target_date': target_date.strftime('%Y-%m-%d'),
                    'hist_close': float(hist_close),
                    'pred_return': float(pred_return),
                    'pred_price': float(pred_price),
                    'prob_up': float(prob_up) if prob_up is not None else None
                })
    
    # 提取預測價格序列
    future_prices = [p['pred_price'] for p in historical_predictions]
    
    # 計算整體預測收益（從最新收盤價到最遠預測價格）
    total_pred_return = (future_prices[-1] - latest_close) / latest_close if latest_close > 0 else 0
    
    print(f"\n最新收盤價 ({latest_date.strftime('%Y-%m-%d')}): {latest_close:.2f} HKD")
    print(f"\n基於過去5天數據的滾動T+5預測:")
    for p in historical_predictions:
        if p['prob_up'] is not None:
            print(f"  用 {p['hist_date']} 數據預測 {p['target_date']}: {p['pred_price']:.2f} HKD (prob_up={p['prob_up']:.4f})")
        else:
            print(f"  用 {p['hist_date']} 數據預測 {p['target_date']}: {p['pred_price']:.2f} HKD")
    
    print(f"\n預測總收益: {total_pred_return*100:+.2f}%")
    print(f"預測日期範圍: {future_dates[0].strftime('%Y-%m-%d')} ~ {future_dates[-1].strftime('%Y-%m-%d')}")
    
    prediction_data = {
        'latest_date': latest_date.strftime('%Y-%m-%d'),
        'latest_close': float(latest_close),
        'future_return_pct': float(total_pred_return * 100),
        'future_price_day5': float(future_prices[-1]),
        'future_prices_all': [float(p) for p in future_prices],
        'prediction_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
        'historical_predictions': historical_predictions,
        'pred_direction': 'UP' if total_pred_return > 0 else 'DOWN'
    }
    
    pred_json_path = results_dir / "future_prediction.json"
    with open(pred_json_path, 'w') as f:
        json.dump(prediction_data, f, indent=2)
    print(f"✓ 預測數據已保存: {pred_json_path}")
    
    # 價格數據
    price_data_path = results_dir / "price_data_for_viz.csv"
    df_clean.to_csv(price_data_path, index=False)
    print(f"✓ 價格數據已保存: {price_data_path}")
    
    # 生成可視化
    print("\n【生成可視化圖表】")
    print("-" * 70)
    
    viz_script = str(Path(__file__).parent / "visualize_backtest.py")
    cmd = [
        sys.executable, viz_script,
        "--backtest", str(json_path),
        "--prediction", str(pred_json_path),
        "--price-data", str(price_data_path),
        "--output", str(results_dir / f"{args.output_prefix}_results.png"),
        "--model-label", f"Backtest ({model_type})"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("警告:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"可視化生成失敗: {e}")
        print(f"錯誤輸出: {e.stderr}")
    except FileNotFoundError:
        print(f"找不到可視化腳本: {viz_script}")
    
    print("\n" + "=" * 70)
    print("【回測完成】")
    print("=" * 70)


if __name__ == '__main__':
    main()
