#!/usr/bin/env python3
"""
統一超參數優化入口
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

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
    
    logger.info("=" * 60)
    logger.info("【超參數優化入口】Optuna")
    logger.info("=" * 60)
    
    # 1. 加載數據
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
    
    # 2. 定義目標函數
    def objective(params):
        # 創建模型
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
        
        # 訓練
        training_cfg = config.training
        training_cfg.lr = params['lr']
        training_cfg.epochs = 50  # 優化時減少輪數
        
        trainer = Trainer(model, training_cfg, device)
        trainer.train(train_loader, val_loader)
        
        return trainer.best_val_metric
    
    # 3. 運行優化
    search_space = get_search_space(config.optimization.search_space)
    optimizer = OptunaOptimizer(
        study_name=config.optimization.objective,
        direction=config.optimization.direction
    )
    
    optimizer.optimize(
        objective,
        search_space,
        n_trials=config.optimization.n_trials,
        timeout=config.optimization.timeout
    )
    
    best_params = optimizer.get_best_params()
    logger.info(f"最佳參數: {best_params}")
    
    # 保存結果
    result_path = f"{config.paths.results_dir}/optuna_best_params.json"
    Path(config.paths.results_dir).mkdir(parents=True, exist_ok=True)
    import json
    with open(result_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': optimizer.get_best_score()
        }, f, indent=2)


if __name__ == '__main__':
    main()
