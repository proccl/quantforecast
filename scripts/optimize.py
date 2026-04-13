#!/usr/bin/env python3
"""
統一超參數優化入口
"""

import torch
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.data.preprocessor import Preprocessor
from src.models.patchtst import PatchTST
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.training.optimizers.optuna_optimizer import OptunaOptimizer
from src.training.optimizers.search_spaces import get_search_space
from src.utils.logger import setup_logger


def main():
    config = get_config('config/config.yaml')
    
    logger = setup_logger(
        'optimize',
        level=config.logging.level,
        log_to_file=config.logging.log_to_file,
        log_to_console=config.logging.log_to_console,
        log_file=f"{config.paths.logs_dir}/optimize.log"
    )
    
    print("=" * 60)
    print("【超參數優化入口】Optuna")
    print("=" * 60)
    
    # 1. 加載數據
    print("\n[1/3] 加載數據...")
    loader = DataLoader(config.data)
    df = loader.load()
    
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df)
    feature_cols = engineer.get_feature_columns()
    df_clean = engineer.clean(df_features, feature_cols)
    
    preprocessor = Preprocessor(config.data)
    train_df, val_df, test_df = preprocessor.split(df_clean)
    train_ds, val_ds, test_ds = preprocessor.create_datasets(train_df, val_df, test_df, feature_cols)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  ✓ 設備: {device}")
    print(f"  ✓ 特徵數: {len(feature_cols)}")
    print(f"  ✓ 訓練樣本: {len(train_ds)}")
    
    # 2. 定義目標函數
    print(f"\n[2/3] 開始優化 (n_trials={config.optimization.n_trials})...")
    print("-" * 60)
    
    def objective(params):
        model = PatchTST(
            n_features=len(feature_cols),
            seq_len=config.data.seq_len,
            pred_len=config.data.pred_len,
            patch_len=params.get('patch_len', 5),
            stride=params.get('patch_len', 5) // 2,
            d_model=params['d_model'],
            n_heads=params['n_heads'],
            n_layers=params['n_layers'],
            d_ff=params['d_model'] * 2,
            dropout=params['dropout'],
            head_type='regression',
            use_revin=True
        )
        
        batch_size = params.get('batch_size', config.training.batch_size)
        train_loader, val_loader, _ = preprocessor.create_dataloaders(
            train_ds, val_ds, test_ds, batch_size=batch_size
        )
        
        training_cfg = config.training
        training_cfg.lr = params['lr']
        training_cfg.epochs = 50
        
        trainer = Trainer(model, training_cfg, device)
        trainer.train(train_loader, val_loader)
        
        return trainer.best_val_metric
    
    # 3. 運行優化
    search_space = get_search_space(config.optimization.search_space)
    optimizer = OptunaOptimizer(
        study_name=config.optimization.objective,
        direction=config.optimization.direction
    )
    
    # 自定義回調：每個 trial 結束後打印進度
    trial_results = []
    def progress_callback(study, trial):
        trial_results.append({
            'number': trial.number,
            'params': trial.params,
            'value': trial.value
        })
        best = study.best_trial
        print(f"  Trial {trial.number:3d}/{config.optimization.n_trials} | "
              f"Score: {trial.value:.4f} | "
              f"Best: {best.value:.4f} | "
              f"Params: d_model={trial.params.get('d_model', '-')}, "
              f"n_layers={trial.params.get('n_layers', '-')}, "
              f"lr={trial.params.get('lr', '-'):.4f}")
    
    optimizer.optimize(
        objective,
        search_space,
        n_trials=config.optimization.n_trials,
        timeout=config.optimization.timeout,
        show_progress=True,
        callbacks=[progress_callback]
    )
    
    best_params = optimizer.get_best_params()
    best_score = optimizer.get_best_score()
    
    print("-" * 60)
    print(f"\n[3/3] 優化完成！")
    print(f"  最佳驗證準確率: {best_score:.4f}")
    print(f"  最佳參數:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    
    # 保存結果
    Path(config.paths.results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 最佳參數
    best_path = f"{config.paths.results_dir}/optuna_best_params_{timestamp}.json"
    with open(best_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(trial_results),
            'study_name': config.optimization.objective,
            'search_space': config.optimization.search_space
        }, f, indent=2)
    print(f"\n  ✓ 最佳參數已保存: {best_path}")
    
    # 所有 trial 歷史
    history_path = f"{config.paths.results_dir}/optuna_history_{timestamp}.json"
    with open(history_path, 'w') as f:
        json.dump(trial_results, f, indent=2)
    print(f"  ✓ 優化歷史已保存: {history_path}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
