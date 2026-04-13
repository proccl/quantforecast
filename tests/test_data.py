"""
數據模塊單元測試
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.config import DataConfig
from src.data.loader import DataLoader
from src.data.features import FeatureEngineer
from src.data.preprocessor import Preprocessor


class TestDataLoader:
    """測試數據加載器"""
    
    def test_load_existing_file(self):
        config = DataConfig(data_file='../../quantforecast/data/xiaomi_real.csv')
        loader = DataLoader(config)
        df = loader.load()
        
        assert len(df) > 0
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    def test_validate_price_data(self):
        config = DataConfig(data_file='../../quantforecast/data/xiaomi_real.csv')
        loader = DataLoader(config)
        df = loader.load()
        
        info = loader.get_data_info(df)
        assert info['symbol'] == '01810'
        assert info['n_samples'] > 0


class TestFeatureEngineer:
    """測試特徵工程"""
    
    def test_create_features(self):
        config = DataConfig(data_file='../../quantforecast/data/xiaomi_real.csv')
        df = DataLoader(config).load()
        
        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)
        
        assert 'macd' in df_features.columns
        assert 'rsi_14' in df_features.columns
        assert 'target_return_5d' in df_features.columns
    
    def test_feature_columns(self):
        cols = FeatureEngineer().get_feature_columns()
        assert len(cols) == 17
        assert 'close' in cols
        assert 'volume' in cols


class TestPreprocessor:
    """測試預處理器"""
    
    def test_create_dataset(self):
        config = DataConfig(data_file='../../quantforecast/data/xiaomi_real.csv')
        df = DataLoader(config).load()
        
        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)
        cols = engineer.get_feature_columns()
        df_clean = engineer.clean(df_features, cols)
        
        preprocessor = Preprocessor(config)
        train_df, val_df, test_df = preprocessor.split(df_clean)
        train_ds, val_ds, test_ds = preprocessor.create_datasets(train_df, val_df, test_df, cols)
        
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) >= 0
        
        sample = train_ds[0]
        assert sample['x'].shape == (config.seq_len, len(cols))
