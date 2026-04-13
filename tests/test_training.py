"""
訓練模塊單元測試
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DataConfig, TrainingConfig
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.data.preprocessor import Preprocessor
from src.models.patchtst import PatchTST
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


def get_default_model_kwargs(**overrides):
    """獲取默認模型參數"""
    defaults = {
        'n_features': 5,
        'seq_len': 20,
        'pred_len': 5,
        'patch_len': 5,
        'stride': 2,
        'd_model': 32,
        'n_heads': 2,
        'n_layers': 2,
        'd_ff': 64,
        'dropout': 0.1,
        'head_type': 'regression',
        'use_revin': True
    }
    defaults.update(overrides)
    return defaults


class TestTrainer:
    """測試訓練器"""
    
    def test_train_epoch(self):
        config = DataConfig(data_file='data/xiaomi_real.csv')
        df = DataLoader(config).load()
        
        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)
        cols = engineer.get_feature_columns()
        df_clean = engineer.clean(df_features, cols)
        
        preprocessor = Preprocessor(config)
        train_df, val_df, test_df = preprocessor.split(df_clean)
        train_ds, val_ds, _ = preprocessor.create_datasets(train_df, val_df, test_df, cols)
        train_loader, val_loader, _ = preprocessor.create_dataloaders(train_ds, val_ds, val_ds, batch_size=32)
        
        kwargs = get_default_model_kwargs(
            n_features=len(cols),
            seq_len=config.seq_len,
            d_model=16,
            n_heads=2,
            n_layers=1
        )
        model = PatchTST(**kwargs)
        
        training_cfg = TrainingConfig(epochs=1, early_stopping=False)
        device = torch.device('cpu')
        trainer = Trainer(model, training_cfg, device)
        history = trainer.train(train_loader, val_loader)
        
        assert len(history['train_loss']) == 1
        assert len(history['val_loss']) == 1
    
    def test_checkpoint_save_load(self, tmp_path):
        kwargs = get_default_model_kwargs(n_features=3, seq_len=10, d_model=16, n_heads=2, n_layers=1)
        model = PatchTST(**kwargs)
        trainer = Trainer(model, TrainingConfig(), torch.device('cpu'))
        
        checkpoint_path = tmp_path / 'test_checkpoint.pth'
        trainer.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        trainer.load_checkpoint(str(checkpoint_path))
        assert trainer.best_model_state is None  # 載入時不會改變 best_model_state


class TestEvaluator:
    """測試評估器"""
    
    def test_evaluate(self):
        config = DataConfig(data_file='data/xiaomi_real.csv')
        df = DataLoader(config).load()
        
        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)
        cols = engineer.get_feature_columns()
        df_clean = engineer.clean(df_features, cols)
        
        preprocessor = Preprocessor(config)
        train_df, val_df, test_df = preprocessor.split(df_clean)
        _, _, test_ds = preprocessor.create_datasets(train_df, val_df, test_df, cols)
        _, _, test_loader = preprocessor.create_dataloaders(test_ds, test_ds, test_ds, batch_size=32)
        
        kwargs = get_default_model_kwargs(
            n_features=len(cols),
            seq_len=config.seq_len,
            d_model=16,
            n_heads=2,
            n_layers=1
        )
        model = PatchTST(**kwargs)
        
        evaluator = Evaluator(torch.device('cpu'))
        results = evaluator.evaluate(model, test_loader)
        
        assert 'directional_accuracy' in results
        assert 'mse' in results
        assert 'mae' in results
        assert 0.0 <= results['directional_accuracy'] <= 1.0
