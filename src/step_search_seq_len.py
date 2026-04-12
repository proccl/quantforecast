import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pickle
import itertools

# 導入之前的模塊
from step3_data_preprocessing import TimeSeriesDataset, RevIN
from step4_patchtst_model import PatchTST

print("=" * 70)
print("【序列長度搜索實驗】尋找最優 lookback window")
print("=" * 70)

print("\n根據 PatchTST 論文和量化實踐:")
print("- 序列長度直接影響模型能看到多少歷史信息")
print("- 過短: 無法捕捉趨勢")
print("- 過長: 計算成本高，可能包含無關的過時信息")
print("=" * 70)

# 載入數據
df = pd.read_csv('xiaomi_features.csv')
df['date'] = pd.to_datetime(df['date'])

feature_cols = [col for col in df.columns if col not in ['target_return_5d', 'target_direction', 'date']]

# 時間序列劃分 - 為長序列調整比例
train_ratio = 0.8
val_ratio = 0.2
n_total = len(df)
n_train = int(n_total * train_ratio)

train_df = df.iloc[:n_train].copy()
val_df = df.iloc[n_train:].copy()

print(f"\n數據總長度: {n_total} 天")
print(f"訓練集: {len(train_df)} 天 ({train_df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {train_df['date'].iloc[-1].strftime('%Y-%m-%d')})")
print(f"驗證集: {len(val_df)} 天 ({val_df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {val_df['date'].iloc[-1].strftime('%Y-%m-%d')})")

# 測試不同的序列長度
# 根據數據量475天，調整測試範圍
seq_lengths = [20, 30, 40, 48, 60, 72]
patch_len = 16
stride = 8
batch_size = 32

print(f"\n【測試配置】")
print(f"序列長度候選: {seq_lengths}")
print(f"Patch 長度: {patch_len}")
print(f"預測目標: 5日收益率")
print(f"評估指標: Validation Directional Accuracy")

# 模型配置 (固定其他參數)
model_config = {
    'n_features': len(feature_cols),
    'pred_len': 5,
    'patch_len': patch_len,
    'stride': stride,
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 3,
    'd_ff': 256,
    'dropout': 0.1,
    'head_type': 'regression',
    'use_revin': True
}

# 訓練配置 (輕量級快速評估)
epochs = 30  # 減少輪數做快速對比
patience = 10
learning_rate = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"設備: {device}")

results = []

print("\n" + "=" * 70)
print("開始搜索...")
print("=" * 70)

for seq_len in seq_lengths:
    print(f"\n{'='*50}")
    print(f"測試序列長度: {seq_len} (~{seq_len//20}個月)")
    print(f"{'='*50}")
    
    # 檢查是否有足夠數據
    if seq_len >= len(train_df) * 0.8:
        print(f"⚠ 跳過: 序列長度 {seq_len} 接近訓練集長度 {len(train_df)}，樣本不足")
        continue
    
    # 創建數據集
    try:
        train_dataset = TimeSeriesDataset(train_df, feature_cols, seq_len, 5, patch_len, stride)
        val_dataset = TimeSeriesDataset(val_df, feature_cols, seq_len, 5, patch_len, stride)
        
        if len(train_dataset) < 50 or len(val_dataset) < 20:
            print(f"⚠ 跳過: 樣本數不足 (train: {len(train_dataset)}, val: {len(val_dataset)})")
            continue
        
        print(f"訓練樣本: {len(train_dataset)}, 驗證樣本: {len(val_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 創建模型
        config = model_config.copy()
        config['seq_len'] = seq_len
        
        model = PatchTST(**config).to(device)
        criterion = nn.HuberLoss(delta=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # 訓練
        best_val_acc = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                x = batch['x'].to(device)
                y_return = batch['y_return'].to(device)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y_return)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['x'].to(device)
                    y_return = batch['y_return'].to(device)
                    y_direction = batch['y_direction'].to(device)
                    
                    pred = model(x)
                    loss = criterion(pred, y_return)
                    val_loss += loss.item()
                    
                    pred_direction = (pred.squeeze() > 0).long()
                    val_correct += (pred_direction == y_direction).sum().item()
                    val_total += y_direction.size(0)
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            scheduler.step()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # 記錄結果
        result = {
            'seq_len': seq_len,
            'months_approx': seq_len // 20,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
        results.append(result)
        
        print(f"✓ 最佳驗證準確率: {best_val_acc:.2%}")
        print(f"✓ 訓練輪數: {epoch + 1}")
        
    except Exception as e:
        print(f"✗ 錯誤: {e}")
        continue

# 結果分析
print("\n" + "=" * 70)
print("【搜索結果】")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('best_val_acc', ascending=False)

print("\n按驗證準確率排序:")
print(results_df.to_string(index=False))

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 準確率對比
ax1 = axes[0]
colors = ['green' if acc > 0.52 else 'red' for acc in results_df['best_val_acc']]
bars = ax1.bar(range(len(results_df)), results_df['best_val_acc'] * 100, color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
ax1.axhline(y=52, color='orange', linestyle='--', alpha=0.5, label='Minimum (52%)')
ax1.set_xticks(range(len(results_df)))
ax1.set_xticklabels([f"{r['seq_len']}\n(~{r['months_approx']}M)" for _, r in results_df.iterrows()])
ax1.set_ylabel('Validation Directional Accuracy (%)')
ax1.set_xlabel('Sequence Length (Trading Days)')
ax1.set_title('Sequence Length vs Directional Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 添加數值標籤
for i, (bar, acc) in enumerate(zip(bars, results_df['best_val_acc'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{acc:.1%}', ha='center', va='bottom', fontsize=10)

# 樣本數對比
ax2 = axes[1]
ax2_twin = ax2.twinx()

line1 = ax2.plot(range(len(results_df)), results_df['train_samples'], 'b-o', label='Train Samples')
line2 = ax2.plot(range(len(results_df)), results_df['val_samples'], 'g-s', label='Val Samples')
ax2.set_xticks(range(len(results_df)))
ax2.set_xticklabels([f"{r['seq_len']}\n(~{r['months_approx']}M)" for _, r in results_df.iterrows()])
ax2.set_ylabel('Number of Samples')
ax2.set_xlabel('Sequence Length (Trading Days)')
ax2.set_title('Sequence Length vs Available Samples')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('seq_len_search_results.png', dpi=150, bbox_inches='tight')
print("\n✓ 搜索結果圖表已保存: seq_len_search_results.png")

# 推薦
print("\n" + "=" * 70)
print("【推薦配置】")
print("=" * 70)

best_result = results_df.iloc[0]
print(f"\n🎯 最優序列長度: {best_result['seq_len']} 天 (~{best_result['months_approx']}個月)")
print(f"   驗證準確率: {best_result['best_val_acc']:.2%}")

# 分析建議
print(f"\n📊 分析建議:")
if best_result['seq_len'] <= 96:
    print("   • 短期配置適合日內/短線交易")
    print("   • 對市場變化反應靈敏")
elif best_result['seq_len'] <= 192:
    print("   • 中期配置適合波段操作")
    print("   • 平衡靈敏度和穩定性")
else:
    print("   • 長期配置適合趨勢跟蹤")
    print("   • 能捕捉完整市場週期")

# 效率考量
efficient_results = results_df[results_df['best_val_acc'] > 0.52].copy()
if len(efficient_results) > 0:
    efficient_results['efficiency'] = efficient_results['best_val_acc'] / efficient_results['seq_len']
    most_efficient = efficient_results.loc[efficient_results['efficiency'].idxmax()]
    print(f"\n⚡ 性價比最高: {most_efficient['seq_len']} 天")
    print(f"   準確率: {most_efficient['best_val_acc']:.2%}")
    print(f"   用最短的歷史數據達到可接受的準確率")

# 保存結果
results_df.to_csv('seq_len_search_results.csv', index=False)
print(f"\n✓ 詳細結果已保存: seq_len_search_results.csv")

print("\n" + "=" * 70)
