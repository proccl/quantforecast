"""
特徵工程模塊
計算技術指標和目標變量
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特徵工程：計算技術指標和目標變量"""
    
    def __init__(self):
        self.feature_cols: List[str] = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算完整特徵集
        
        Returns:
            DataFrame 包含原始列 + 技術指標 + 目標變量
        """
        df = df.copy()
        
        # 1. 趨勢特徵
        df = self._add_trend_features(df)
        
        # 2. 波動率特徵
        df = self._add_volatility_features(df)
        
        # 3. 動量特徵
        df = self._add_momentum_features(df)
        
        # 4. 成交量特徵
        df = self._add_volume_features(df)
        
        # 5. 收益率特徵
        df = self._add_return_features(df)
        
        # 6. 目標變量
        df = self._add_targets(df)
        
        logger.info("✓ 特徵計算完成")
        return df
    
    def get_feature_columns(self) -> List[str]:
        """返回標準特徵列列表"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'ema_5_ratio', 'ema_10_ratio', 'ema_20_ratio',
            'macd', 'macd_hist',
            'atr_ratio', 'rsi_14',
            'volume_ratio', 'obv',
            'return_1d', 'return_5d', 'volatility_20d'
        ]
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """趨勢特徵：EMA, MACD"""
        # EMA 及比率
        for span in [5, 10, 20]:
            ema_col = f'ema_{span}'
            df[ema_col] = df['close'].ewm(span=span, adjust=False).mean()
            df[f'ema_{span}_ratio'] = df['close'] / df[ema_col] - 1
        
        # MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波動率特徵：ATR, Bollinger Bands"""
        # ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr_14'] = df['tr'].rolling(window=14).mean()
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """動量特徵：RSI, Stochastic"""
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Stochastic
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量特徵：OBV, Volume Ratio"""
        # OBV
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        
        # Volume Ratio
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        return df
    
    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """收益率特徵"""
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['volatility_20d'] = df['return_1d'].rolling(window=20).std()
        
        return df
    
    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """目標變量"""
        df['target_return_5d'] = df['close'].pct_change(5).shift(-5)
        df['target_direction'] = (df['target_return_5d'] > 0).astype(int)
        
        return df
    
    def clean(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        清洗數據：去除特徵缺失值（保留目標為 NaN 的最新數據用於預測）
        
        Returns:
            清洗後的 DataFrame
        """
        if feature_cols is None:
            feature_cols = self.get_feature_columns()
        
        # 只檢查特徵列和必需列（不包括 target 列）
        required_cols = feature_cols + ['date']
        available_cols = [col for col in required_cols if col in df.columns]
        
        # 只 drop 特徵列為 NaN 的行
        df_clean = df[available_cols].dropna()
        
        # 將目標列加回來（可能包含 NaN）
        for col in ['target_return_5d', 'target_direction']:
            if col in df.columns:
                df_clean[col] = df.loc[df_clean.index, col]
        
        n_dropped = len(df) - len(df_clean)
        
        if n_dropped > 0:
            logger.info(f"去除 {n_dropped} 行缺失值，剩餘 {len(df_clean)} 行")
        
        return df_clean
