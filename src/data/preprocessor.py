"""
數據預處理模塊
創建時間序列數據集和 DataLoader
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import logging

from src.config import DataConfig

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """時間序列數據集"""
    
    def __init__(
        self,
        data: np.ndarray,
        target_return: np.ndarray,
        target_direction: np.ndarray,
        seq_len: int,
        sample_weights: Optional[np.ndarray] = None
    ):
        """
        Args:
            data: 特徵數據 [n_samples, n_features]
            target_return: 回歸目標 [n_samples]
            target_direction: 分類目標 [n_samples]
            seq_len: 輸入序列長度
            sample_weights: 樣本權重 [n_samples]，用於指數衰減等加權策略
        """
        self.data = data
        self.target_return = target_return
        self.target_direction = target_direction
        self.seq_len = seq_len
        self.n_samples = len(data) - seq_len
        
        # 樣本權重：如果沒有提供，默認均勻權重
        if sample_weights is not None:
            # 確保權重長度與有效樣本數匹配
            self.sample_weights = sample_weights[seq_len:seq_len + self.n_samples]
        else:
            self.sample_weights = np.ones(self.n_samples, dtype=np.float32)
        
    def __len__(self) -> int:
        return max(0, self.n_samples)
    
    def __getitem__(self, idx: int) -> dict:
        x = self.data[idx:idx + self.seq_len]
        y_return = self.target_return[idx + self.seq_len - 1]
        y_direction = self.target_direction[idx + self.seq_len - 1]
        weight = self.sample_weights[idx]
        
        return {
            'x': torch.FloatTensor(x),
            'y_return': torch.FloatTensor([y_return]),
            'y_direction': torch.LongTensor([y_direction]),
            'weight': torch.FloatTensor([weight])
        }


class Preprocessor:
    """數據預處理器：分割數據集並創建 DataLoader"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.batch_size = 32  # 默認，可從 training config 覆蓋
        
    def split(
        self,
        df: pd.DataFrame,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按時間順序分割數據集
        
        Returns:
            (train_df, val_df, test_df)
        """
        train_ratio = train_ratio or self.config.train_ratio
        val_ratio = val_ratio or self.config.val_ratio
        
        n_total = len(df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()
        
        logger.info(
            f"數據分割: 訓練集 {len(train_df)} | 驗證集 {len(val_df)} | 測試集 {len(test_df)}"
        )
        
        return train_df, val_df, test_df
    
    def create_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
        """
        創建時間序列數據集
        
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        train_dataset = self._create_single_dataset(train_df, feature_cols)
        val_dataset = self._create_single_dataset(val_df, feature_cols)
        test_dataset = self._create_single_dataset(test_df, feature_cols)
        
        logger.info(
            f"數據集創建: 訓練 {len(train_dataset)} | 驗證 {len(val_dataset)} | 測試 {len(test_dataset)}"
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        train_dataset: TimeSeriesDataset,
        val_dataset: TimeSeriesDataset,
        test_dataset: TimeSeriesDataset,
        batch_size: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        創建 DataLoader
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        batch_size = batch_size or self.batch_size
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        logger.info(f"DataLoader 創建完成，batch_size={batch_size}")
        return train_loader, val_loader, test_loader
    
    def _create_single_dataset(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        sample_weights: Optional[np.ndarray] = None
    ) -> TimeSeriesDataset:
        """創建單個數據集（過濾掉目標為 NaN 的樣本）"""
        # 只保留目標有效的行（用於訓練/驗證/測試）
        valid_df = df.dropna(subset=['target_return_5d', 'target_direction'])
        
        data = valid_df[feature_cols].values
        target_return = valid_df['target_return_5d'].values
        target_direction = valid_df['target_direction'].values
        
        return TimeSeriesDataset(
            data=data,
            target_return=target_return,
            target_direction=target_direction,
            seq_len=self.seq_len,
            sample_weights=sample_weights
        )
    
    def create_weighted_dataset(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        decay_lambda: float = 0.01
    ) -> TimeSeriesDataset:
        """
        創建帶指數衰減權重的數據集
        
        Args:
            df: 數據框
            feature_cols: 特徵列
            decay_lambda: 衰減係數，越大對近期數據權重越高
                         w_t = exp(λ * (t - T))
        """
        valid_df = df.dropna(subset=['target_return_5d', 'target_direction'])
        n_samples = len(valid_df)
        
        # 生成時間索引（0到n_samples-1）
        time_indices = np.arange(n_samples)
        
        # 指數衰減權重：w_t = exp(λ * (t - T))
        # t 是當前索引，T 是最新索引（n_samples - 1）
        T = n_samples - 1
        weights = np.exp(decay_lambda * (time_indices - T))
        
        # 歸一化權重（可選，但通常有助於穩定訓練）
        weights = weights / weights.sum() * n_samples
        
        logger.info(f"指數衰減權重統計: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
        
        data = valid_df[feature_cols].values
        target_return = valid_df['target_return_5d'].values
        target_direction = valid_df['target_direction'].values
        
        return TimeSeriesDataset(
            data=data,
            target_return=target_return,
            target_direction=target_direction,
            seq_len=self.seq_len,
            sample_weights=weights
        )
    
    def get_sample_shape(
        self,
        dataset: TimeSeriesDataset,
        n_features: int
    ) -> dict:
        """獲取樣本形狀信息"""
        if len(dataset) == 0:
            return {}
        
        sample = dataset[0]
        return {
            'x_shape': tuple(sample['x'].shape),  # [seq_len, n_features]
            'n_features': n_features,
            'seq_len': self.seq_len
        }
