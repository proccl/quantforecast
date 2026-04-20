#!/usr/bin/env python3
"""
統一訓練入口
"""

import torch
import sys
from pathlib import Path
from datetime import datetime

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.data.preprocessor import Preprocessor
from src.models.patchtst import PatchTST
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.logger import setup_logger


def main():
    # 加載配置
    config = get_config('config/config.yaml')
    
    # 設置日誌
    logger = setup_logger(
        'train',
        level=config.logging.level,
        log_to_file=config.logging.log_to_file,
        log_to_console=config.logging.log_to_console,
        log_file=f"{config.paths.logs_dir}/train.log"
    )
    
    logger.info("=" * 60)
    logger.info("【訓練入口】PatchTST 模型訓練")
    logger.info("=" * 60)
    
    # 1. 加載數據
    logger.info("【1/4】加載數據")
    loader = DataLoader(config.data)
    df = loader.load()
    
    # 2. 特徵工程
    logger.info("【2/4】特徵工程")
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df)
    feature_cols = engineer.get_feature_columns()
    df_clean = engineer.clean(df_features, feature_cols)
    
    # 3. 預處理（帶指數衰減權重）
    logger.info("【3/4】數據預處理")
    preprocessor = Preprocessor(config.data)
    train_df, val_df, test_df = preprocessor.split(df_clean)
    
    # 訓練集使用指數衰減權重
    decay_lambda = 0.01
    train_ds = preprocessor.create_weighted_dataset(train_df, feature_cols, decay_lambda=decay_lambda)
    val_ds = preprocessor._create_single_dataset(val_df, feature_cols)
    test_ds = preprocessor._create_single_dataset(test_df, feature_cols)
    
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=config.training.batch_size
    )
    
    logger.info(f"  ✓ 訓練集指數衰減 λ={decay_lambda} (近期數據權重更高)")
    
    # 4. 訓練模型
    logger.info("【4/4】模型訓練")
    device = torch.device('cuda' if torch.cuda.is_available() and config.training.device != 'cpu' else 'cpu')
    
    model = PatchTST(
        n_features=len(feature_cols),
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        patch_len=5,
        stride=2,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        head_type='classification',
        use_revin=True
    )
    
    trainer = Trainer(model, config.training, device)
    history = trainer.train(train_loader, val_loader)
    
    # 保存模型（文件名包含訓練結束時間）
    end_time = datetime.now()
    timestamp = end_time.strftime("%Y%m%d_%H%M%S")
    model_filename = f"patchtst_model_{timestamp}.pth"
    model_path = f"{config.paths.model_dir}/{model_filename}"
    Path(config.paths.model_dir).mkdir(parents=True, exist_ok=True)
    
    # 構建模型配置供回測使用
    model_config = {
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
    trainer.save_checkpoint(model_path, model_config)
    
    # 測試集評估
    evaluator = Evaluator(device, head_type='classification')
    results = evaluator.evaluate(model, test_loader)
    
    logger.info("=" * 60)
    logger.info("【訓練完成】")
    logger.info(f"訓練結束時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"模型保存: {model_filename}")
    logger.info(f"最佳驗證準確率: {trainer.best_val_metric:.2%}")
    logger.info(f"測試集準確率: {results['directional_accuracy']:.2%}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
