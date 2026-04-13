"""
數據加載模塊
統一處理 CSV 數據加載和驗證
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

from src.config import DataConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """統一數據加載器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.data_path = Path(config.data_file)
        
    def load(self) -> pd.DataFrame:
        """
        加載 CSV 數據
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"數據文件不存在: {self.data_path}")
        
        logger.info(f"加載數據: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # 標準化列名
        df = self._standardize_columns(df)
        
        # 日期處理
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 數據驗證
        self._validate(df)
        
        logger.info(f"✓ 數據加載完成: {len(df)} 行, {df['date'].iloc[0]} 至 {df['date'].iloc[-1]}")
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """標準化列名（小寫）"""
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # 確保必要列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")
        
        return df
    
    def _validate(self, df: pd.DataFrame) -> None:
        """驗證數據完整性"""
        # 檢查空值
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            logger.warning(f"數據包含空值:\n{null_counts[null_counts > 0]}")
        
        # 檢查價格邏輯
        invalid_ohlc = (
            (df['low'] > df['high']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low'])
        )
        
        if invalid_ohlc.any():
            n_invalid = invalid_ohlc.sum()
            logger.warning(f"發現 {n_invalid} 行無效的 OHLC 數據")
            # 自動修復
            df.loc[invalid_ohlc, 'low'] = df.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].min(axis=1)
            df.loc[invalid_ohlc, 'high'] = df.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].max(axis=1)
        
        # 檢查成交量
        if (df['volume'] < 0).any():
            raise ValueError("成交量不能為負數")
        
        logger.info("✓ 數據驗證通過")
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """獲取數據基本信息"""
        return {
            'symbol': self.config.symbol,
            'n_samples': len(df),
            'date_range': (df['date'].iloc[0], df['date'].iloc[-1]),
            'price_range': (df['close'].min(), df['close'].max()),
            'columns': list(df.columns)
        }
