import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

print("=" * 60)
print("【步驟 7/8】風險管理分析")
print("=" * 60)

print("\n根據最佳實踐文檔風險管理:")
print("- 過擬合檢測")
print("- 凱利公式倉位管理")
print("- 止損策略")
print("-" * 60)

# 載入訓練歷史
try:
    from step5_training import train_losses, val_losses, val_direction_accs
    has_history = True
except:
    has_history = False
    print("⚠ 無法載入訓練歷史，跳過過擬合檢測")

print("\n【7.1】過擬合檢測")
print("-" * 40)

if has_history:
    # 計算訓練/驗證損失差距
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    loss_gap = (final_val_loss - final_train_loss) / final_train_loss * 100
    
    print(f"✓ 最終訓練損失: {final_train_loss:.6f}")
    print(f"✓ 最終驗證損失: {final_val_loss:.6f}")
    print(f"✓ 損失差距: {loss_gap:.1f}%")
    
    if loss_gap > 20:
        print(f"⚠ 警告: 可能存在過擬合 (差距 > 20%)")
        print(f"  建議: 增加正則化或簡化模型")
    else:
        print(f"✓ 無明顯過擬合跡象 (差距 < 20%)")
    
    # 繪製過擬合分析圖
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(train_losses, label='Train Loss', alpha=0.8)
    ax.plot(val_losses, label='Val Loss', alpha=0.8)
    ax.fill_between(range(len(train_losses)), train_losses, val_losses, 
                     alpha=0.3, color='red', label=f'Gap: {loss_gap:.1f}%')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Overfitting Analysis: Train vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('overfitting_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ 過擬合分析圖已保存: overfitting_analysis.png")

print("\n【7.2】凱利公式倉位管理")
print("-" * 40)

def kelly_criterion(win_prob, avg_win, avg_loss):
    """
    凱利公式計算最優倉位比例
    f* = (p * b - q) / b
    其中: p = 獲勝概率, q = 失敗概率, b = 平均盈利/平均虧損
    """
    q = 1 - win_prob
    if avg_loss == 0:
        return 0
    b = avg_win / avg_loss
    kelly = (win_prob * b - q) / b
    return max(0, min(kelly, 0.5))  # 限制最大倉位50%

# 模擬歷史交易數據
np.random.seed(42)
n_trades = 100

# 假設策略的勝率和盈虧比
win_prob = 0.55  # 55%勝率 (基於我們的方向準確率)
avg_win = 0.05   # 平均盈利5%
avg_loss = 0.03  # 平均虧損3%

# 生成模擬交易
wins = np.random.binomial(1, win_prob, n_trades)
trade_returns = np.where(wins == 1, 
                         np.random.normal(avg_win, 0.02, n_trades),
                         np.random.normal(-avg_loss, 0.015, n_trades))

# 計算凱利倉位
kelly_position = kelly_criterion(win_prob, avg_win, avg_loss)
print(f"✓ 策略勝率: {win_prob:.1%}")
print(f"✓ 平均盈利: {avg_win:.2%}")
print(f"✓ 平均虧損: {avg_loss:.2%}")
print(f"✓ 盈虧比: {avg_win/avg_loss:.2f}")
print(f"✓ 凱利公式最優倉位: {kelly_position:.1%}")

# 不同倉位下的收益模擬
position_sizes = [0.1, 0.2, 0.3, kelly_position, 0.5, 0.7, 1.0]
results = []

for pos in position_sizes:
    final_returns = np.cumsum(trade_returns * pos)
    total_return = final_returns[-1]
    volatility = np.std(trade_returns * pos)
    sharpe = np.mean(trade_returns * pos) / (volatility + 1e-8) * np.sqrt(252)
    max_dd = np.min(final_returns - np.maximum.accumulate(final_returns))
    
    results.append({
        'Position': f'{pos:.0%}',
        'Total Return': f'{total_return:.1%}',
        'Volatility': f'{volatility:.2%}',
        'Sharpe': f'{sharpe:.2f}',
        'Max DD': f'{max_dd:.1%}'
    })

results_df = pd.DataFrame(results)
print(f"\n不同倉位策略表現:")
print(results_df.to_string(index=False))

print("\n【7.3】止損策略")
print("-" * 40)

# 計算 ATR 止損
try:
    df = pd.read_csv('../data/xiaomi_raw.csv')
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr_14 = df['tr'].rolling(window=14).mean().iloc[-1]
    current_price = df['close'].iloc[-1]
    
    # ATR 止損位
    atr_stop_2x = current_price - 2 * atr_14
    atr_stop_3x = current_price - 3 * atr_14
    
    # 固定比例止損
    stop_2pct = current_price * 0.98
    stop_5pct = current_price * 0.95
    
    print(f"✓ 當前價格: {current_price:.2f} HKD")
    print(f"✓ ATR(14): {atr_14:.2f} HKD ({atr_14/current_price*100:.2f}%)")
    print(f"\n止損位建議:")
    print(f"  • ATR 2x 止損: {atr_stop_2x:.2f} HKD (風險: {2*atr_14/current_price*100:.1f}%)")
    print(f"  • ATR 3x 止損: {atr_stop_3x:.2f} HKD (風險: {3*atr_14/current_price*100:.1f}%)")
    print(f"  • 固定 2% 止損: {stop_2pct:.2f} HKD")
    print(f"  • 固定 5% 止損: {stop_5pct:.2f} HKD")
    
except Exception as e:
    print(f"⚠ 無法計算止損位: {e}")

print("\n【7.4】風險指標彙總")
print("-" * 40)

risk_summary = f"""
風險管理配置建議:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 倉位管理
   • 建議倉位: {kelly_position:.0%} (凱利公式)
   • 最大倉位: 50% (風險控制)
   • 保守倉位: {kelly_position/2:.0%} (半凱利)

2. 止損策略
   • 推薦: ATR 2x 或 固定 2%
   • 止損觸發後全倉離場

3. 風險監控
   • 單筆最大虧損: ≤2%
   • 單月最大回撤: ≤10%
   • 連續虧損次數: ≥5次時減倉

4. 模型風險
   • 定期重新訓練 (每月)
   • 監控 Directional Accuracy 衰減
   • 準確率 < 52% 時暫停交易
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

print(risk_summary)

# 保存風險配置
risk_config = {
    'kelly_position': kelly_position,
    'max_position': 0.5,
    'stop_loss_method': 'ATR_2x',
    'max_loss_per_trade': 0.02,
    'max_drawdown_monthly': 0.10,
    'retrain_frequency': 'monthly',
    'min_accuracy': 0.52
}

with open('risk_config.pkl', 'wb') as f:
    pickle.dump(risk_config, f)

print("✓ 風險配置已保存: risk_config.pkl")

print("\n" + "=" * 60)
print("【關鍵節點 7/8 完成】風險管理分析完成")
print("請檢查風險配置建議，確認後繼續下一步")
print("=" * 60)
