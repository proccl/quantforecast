#!/usr/bin/env python3
"""
統一超參數優化入口 - Walk-forward CV 版本
"""

import torch
import numpy as np
import pandas as pd
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.data.preprocessor import Preprocessor, TimeSeriesDataset
from src.models.patchtst import PatchTST
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.training.optimizers.optuna_optimizer import OptunaOptimizer
from src.training.optimizers.search_spaces import get_search_space
from src.utils.logger import setup_logger
from torch.utils.data import DataLoader as TorchDataLoader


def create_dataloaders_from_df(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    batch_size: int = 32
) -> Tuple[TorchDataLoader, int]:
    """從 DataFrame 創建 DataLoader"""
    valid_df = df.dropna(subset=['target_return_5d', 'target_direction'])
    
    data = valid_df[feature_cols].values
    target_return = valid_df['target_return_5d'].values
    target_direction = valid_df['target_direction'].values
    
    dataset = TimeSeriesDataset(
        data=data,
        target_return=target_return,
        target_direction=target_direction,
        seq_len=seq_len
    )
    
    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    ) if len(dataset) > 0 else None
    
    return loader, len(dataset)


def walk_forward_cv_objective(
    params: dict,
    df_clean: pd.DataFrame,
    feature_cols: List[str],
    config,
    device: torch.device,
    n_splits: int = 3
) -> float:
    """
    Walk-forward CV 目標函數
    
    返回: 平均驗證準確率 (減去過擬合懲罰)
    """
    seq_len = config.data.seq_len
    pred_len = config.data.pred_len
    
    # Walk-forward 分割參數
    total_len = len(df_clean)
    min_train_ratio = 0.5  # 第一個 fold 至少用 50% 數據訓練
    val_ratio_per_fold = 0.15
    
    fold_scores = []
    fold_train_scores = []
    
    for fold in range(n_splits):
        # 計算這個 fold 的分割點
        train_end_ratio = min_train_ratio + fold * 0.1  # 逐步增加訓練數據
        train_end = int(total_len * train_end_ratio)
        val_start = train_end
        val_end = min(val_start + int(total_len * val_ratio_per_fold), total_len)
        
        if val_end - val_start < 20:  # 驗證集太小，跳過
            continue
        
        train_df = df_clean.iloc[:train_end]
        val_df = df_clean.iloc[val_start:val_end]
        
        # 創建數據集
        batch_size = params.get('batch_size', config.training.batch_size)
        train_loader, train_size = create_dataloaders_from_df(
            train_df, feature_cols, seq_len, batch_size
        )
        val_loader, val_size = create_dataloaders_from_df(
            val_df, feature_cols, seq_len, batch_size
        )
        
        if train_loader is None or val_loader is None or train_size < 50:
            continue
        
        # 創建模型
        model = PatchTST(
            n_features=len(feature_cols),
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=params.get('patch_len', 5),
            stride=params.get('patch_len', 5) // 2,
            d_model=params['d_model'],
            n_heads=params['n_heads'],
            n_layers=params['n_layers'],
            d_ff=params['d_model'] * 2,
            dropout=params['dropout'],
            head_type='classification',
            use_revin=True
        )
        
        # 訓練配置
        training_cfg = config.training
        training_cfg.lr = params['lr']
        training_cfg.epochs = 25  # CV 時減少 epoch 加速
        training_cfg.early_stopping_patience = 8
        
        trainer = Trainer(model, training_cfg, device)
        history = trainer.train(train_loader, val_loader)
        
        val_acc = trainer.best_val_metric
        
        # 計算訓練集準確率（用於過擬合檢測）
        # 在訓練集最後一個 epoch 的 loss 轉換為近似準確率
        train_acc = 0.5  # 默認值，實際應該在 trainer 中記錄
        
        fold_scores.append(val_acc)
        
    if len(fold_scores) == 0:
        return 0.0
    
    # 計算平均分和標準差
    mean_val_acc = np.mean(fold_scores)
    std_val_acc = np.std(fold_scores)
    
    # 過擬合懲罰：標準差太大會扣分
    stability_penalty = max(0, std_val_acc - 0.05) * 0.5  # std > 5% 開始扣分
    
    # 最終得分
    final_score = mean_val_acc - stability_penalty
    
    return final_score


def main():
    config = get_config('config/config.yaml')
    
    logger = setup_logger(
        'optimize',
        level=config.logging.level,
        log_to_file=config.logging.log_to_file,
        log_to_console=config.logging.log_to_console,
        log_file=f"{config.paths.logs_dir}/optimize.log"
    )
    
    print("=" * 70)
    print("【超參數優化入口】Optuna + Walk-forward CV")
    print("=" * 70)
    
    # 1. 加載數據
    print("\n[1/3] 加載數據...")
    loader = DataLoader(config.data)
    df = loader.load()
    
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df)
    feature_cols = engineer.get_feature_columns()
    df_clean = engineer.clean(df_features, feature_cols)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  ✓ 設備: {device}")
    print(f"  ✓ 特徵數: {len(feature_cols)}")
    print(f"  ✓ 總樣本數: {len(df_clean)}")
    print(f"  ✓ Walk-forward splits: 3")
    
    # 2. 定義目標函數
    print(f"\n[2/3] 開始優化 (n_trials={config.optimization.n_trials})...")
    print(f"  優化目標: Walk-forward CV 平均準確率 (減穩定性懲罰)")
    print("-" * 70)
    
    def objective(params):
        score = walk_forward_cv_objective(
            params, df_clean, feature_cols, config, device, n_splits=3
        )
        return score
    
    # 3. 運行優化
    search_space = get_search_space(config.optimization.search_space)
    optimizer = OptunaOptimizer(
        study_name="walk_forward_cv",
        direction="maximize"
    )
    
    trial_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 中間結果文件路徑
    history_path = results_dir / f"optuna_history_{timestamp}.json"
    best_params_path = results_dir / f"optuna_best_params_{timestamp}.json"
    
    def progress_callback(study, trial):
        # 記錄當前 trial
        trial_data = {
            'number': trial.number,
            'params': trial.params,
            'value': trial.value,
            'datetime': datetime.now().isoformat()
        }
        trial_results.append(trial_data)
        
        # 即時保存歷史
        with open(history_path, 'w') as f:
            json.dump(trial_results, f, indent=2)
        
        # 即時保存當前最佳
        best = study.best_trial
        with open(best_params_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'best_params': best.params,
                'best_cv_score': best.value,
                'best_trial_number': best.number,
                'n_trials_completed': len(trial_results),
                'study_name': 'walk_forward_cv',
                'search_space': config.optimization.search_space
            }, f, indent=2)
        
        print(f"  Trial {trial.number:3d} | Score: {trial.value:.4f} | Best: {best.value:.4f} | Saved")
    
    optimizer.optimize(
        objective,
        search_space,
        n_trials=20,
        timeout=config.optimization.timeout,
        show_progress=True,
        callbacks=[progress_callback]
    )
    
    best_params = optimizer.get_best_params()
    best_score = optimizer.get_best_score()
    
    print("-" * 70)
    print(f"\n[3/3] 優化完成！")
    print(f"  最佳 CV 分數: {best_score:.4f}")
    print(f"  最佳參數:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    
    # 4. 用最佳參數訓練最終模型
    print("\n" + "=" * 70)
    print("【最終訓練】使用最佳參數訓練完整模型")
    print("=" * 70)
    
    preprocessor = Preprocessor(config.data)
    train_df, val_df, test_df = preprocessor.split(df_clean)
    train_ds, val_ds, test_ds = preprocessor.create_datasets(train_df, val_df, test_df, feature_cols)
    
    final_model = PatchTST(
        n_features=len(feature_cols),
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        patch_len=best_params.get('patch_len', 5),
        stride=best_params.get('patch_len', 5) // 2,
        d_model=best_params['d_model'],
        n_heads=best_params['n_heads'],
        n_layers=best_params['n_layers'],
        d_ff=best_params['d_model'] * 2,
        dropout=best_params['dropout'],
        head_type='classification',
        use_revin=True
    )
    
    batch_size = best_params.get('batch_size', config.training.batch_size)
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=batch_size
    )
    
    final_cfg = config.training
    final_cfg.lr = best_params['lr']
    final_cfg.epochs = 100  # 最終訓練用更多 epoch
    final_cfg.early_stopping_patience = 15
    
    final_trainer = Trainer(final_model, final_cfg, device)
    final_history = final_trainer.train(train_loader, val_loader)
    
    # 測試集評估
    evaluator = Evaluator(device, head_type='classification')
    test_results = evaluator.evaluate(final_model, test_loader)
    
    print(f"\n最終測試集準確率: {test_results['directional_accuracy']:.2%}")
    
    # 保存結果
    Path(config.paths.results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 更新最終結果（已完成即時保存，這裡只打印最終結果）
    final_best_params = optimizer.get_best_params()
    final_best_score = optimizer.get_best_score()
    
    # 更新最佳參數文件（添加完成標記）
    with open(best_params_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'best_params': final_best_params,
            'best_cv_score': final_best_score,
            'best_trial_number': optimizer.study.best_trial.number,
            'n_trials_completed': len(trial_results),
            'study_name': 'walk_forward_cv',
            'search_space': config.optimization.search_space,
            'completed': True
        }, f, indent=2)
    
    print(f"\n[3/3] 優化完成！")
    print(f"  最佳 CV 分數: {final_best_score:.4f}")
    print(f"  結果已即時保存於:")
    print(f"    - {history_path}")
    print(f"    - {best_params_path}")
    
    # 保存模型
    model_config = {
        'seq_len': config.data.seq_len,
        'pred_len': config.data.pred_len,
        'd_model': best_params['d_model'],
        'n_heads': best_params['n_heads'],
        'n_layers': best_params['n_layers'],
        'dropout': best_params['dropout'],
        'patch_len': best_params.get('patch_len', 5),
        'stride': best_params.get('patch_len', 5) // 2,
        'd_ff': best_params['d_model'] * 2
    }
    
    model_filename = f"patchtst_model_{timestamp}.pth"
    model_path = f"{config.paths.model_dir}/{model_filename}"
    Path(config.paths.model_dir).mkdir(parents=True, exist_ok=True)
    final_trainer.save_checkpoint(model_path, model_config)
    
    print(f"  ✓ 模型已保存: {model_path}")
    
    print("=" * 70)
    print("\n優化和訓練完成！可以運行 backtest 了：")
    print(f"  python scripts/backtest.py")


if __name__ == '__main__':
    main()
