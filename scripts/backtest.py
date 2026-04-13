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
    
    # 2. 加載模型
    model_path = f"{config.paths.model_dir}/patchtst_bayesian_best.pth"
    if not Path(model_path).exists():
        model_path = f"{config.paths.model_dir}/patchtst_model.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    loaded_config = checkpoint['model_config']
    
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
