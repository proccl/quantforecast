#!/usr/bin/env python3
"""
統一回測入口
"""

import torch
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config, BacktestConfig
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.models.patchtst import PatchTST
from src.backtest.engine import BacktestEngine
from src.backtest.reporting import ReportGenerator
from src.utils.logger import setup_logger


def main():
    config = get_config('config/config.yaml')
    
    logger = setup_logger(
        'backtest',
        level=config.logging.level,
        log_to_file=config.logging.log_to_file,
        log_to_console=config.logging.log_to_console,
        log_file=f"{config.paths.logs_dir}/backtest.log"
    )
    
    logger.info("=" * 60)
    logger.info("【回測入口】完整回測分析")
    logger.info("=" * 60)
    
    # 1. 加載數據
    loader = DataLoader(config.data)
    df = loader.load()
    
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df)
    feature_cols = engineer.get_feature_columns()
    df_clean = engineer.clean(df_features, feature_cols)
    
    # 2. 加載模型（優先加載最新的 timestamp 模型，再 fallback 到 bayesian）
    model_dir = Path(config.paths.model_dir)
    
    # 查找最新的 patchtst_model_YYYYMMDD_HHMMSS.pth
    model_files = sorted(model_dir.glob("patchtst_model_*.pth"), reverse=True)
    if model_files:
        model_path = model_files[0]
    elif (model_dir / "patchtst_bayesian_best.pth").exists():
        model_path = model_dir / "patchtst_bayesian_best.pth"
    else:
        model_path = model_dir / "patchtst_model.pth"
    
    logger.info(f"加載模型: {model_path.name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 兼容舊模型格式（有 model_config）和新格式（從 config.yaml 讀取）
    if 'model_config' in checkpoint:
        loaded_config = checkpoint['model_config']
    elif 'config' in checkpoint and hasattr(checkpoint['config'], 'model'):
        # 新格式：config 包含 data 和 model
        cfg = checkpoint['config']
        loaded_config = {
            'seq_len': cfg.data.seq_len,
            'pred_len': cfg.data.pred_len,
            'd_model': cfg.model.d_model,
            'n_heads': cfg.model.n_heads,
            'n_layers': cfg.model.n_layers,
            'dropout': cfg.model.dropout,
            'patch_len': 5,  # 默認值
            'stride': 2,     # 默認值
            'd_ff': cfg.model.d_ff
        }
    else:
        # 兜底：使用當前 config.yaml
        logger.warning("模型配置缺失，使用當前 config.yaml")
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
    
    model = PatchTST(
        n_features=len(feature_cols),
        seq_len=loaded_config['seq_len'],
        pred_len=loaded_config['pred_len'],
        d_model=loaded_config['d_model'],
        n_heads=loaded_config['n_heads'],
        n_layers=loaded_config['n_layers'],
        dropout=loaded_config['dropout'],
        patch_len=loaded_config['patch_len'],
        stride=loaded_config['stride'],
        d_ff=loaded_config.get('d_ff', loaded_config['d_model'] * 2),
        head_type='regression',
        use_revin=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. 回測（最近90天）
    latest_date = df_clean['date'].iloc[-1]
    three_months_ago = latest_date - pd.Timedelta(days=90)
    test_df = df_clean[df_clean['date'] >= three_months_ago].reset_index(drop=True)
    
    backtest_config = BacktestConfig()
    engine = BacktestEngine(model, device, backtest_config)
    result = engine.run(test_df, feature_cols, loaded_config['seq_len'], loaded_config['pred_len'])
    
    # 4. 未來預測
    latest_data = df_clean[feature_cols].iloc[-loaded_config['seq_len']:].values
    latest_close = df_clean['close'].iloc[-1]
    prediction = engine.predict_future(latest_data, latest_close, latest_date, loaded_config['pred_len'])
    
    # 5. 生成報告
    Path(config.paths.results_dir).mkdir(parents=True, exist_ok=True)
    reporter = ReportGenerator()
    reporter.generate(result, f"{config.paths.results_dir}/complete_backtest_results.png")
    reporter.generate_future_prediction_report(prediction, f"{config.paths.results_dir}/future_prediction.json")
    
    logger.info(f"\n策略總收益: {result.total_return_pct:.2f}%")
    logger.info(f"買入持有: {result.buyhold_return_pct:.2f}%")
    logger.info(f"測試準確率: {result.test_accuracy:.2%}")
    logger.info(f"未來5天預測: {prediction['pred_direction']} ({prediction['future_return_pct']:+.2f}%)")


if __name__ == '__main__':
    main()
