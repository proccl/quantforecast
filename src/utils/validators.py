"""
數據驗證模塊
驗證數據完整性和模型輸出
"""

import pandas as pd
import torch
import numpy as np
from typing import List


class ValidationError(ValueError):
    """驗證錯誤"""
    pass


def validate_price_data(df: pd.DataFrame, required_cols: List[str] = None) -> None:
    """
    驗證價格數據完整性
    
    Args:
        df: 價格數據 DataFrame
        required_cols: 必需的列列表
    
    Raises:
        ValidationError: 數據驗證失敗
    """
    if required_cols is None:
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    # 檢查列存在
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValidationError(f"缺少必要列: {missing}")
    
    # 檢查空值
    if df.empty:
        raise ValidationError("數據為空")
    
    # 檢查成交量為負
    if (df['volume'] < 0).any():
        raise ValidationError("成交量不能為負數")
    
    # 檢查價格邏輯
    invalid = (
        (df['low'] > df['high']) |
        (df['open'] < df['low']) |
        (df['open'] > df['high']) |
        (df['close'] < df['low']) |
        (df['close'] > df['high'])
    )
    
    if invalid.any():
        n_invalid = invalid.sum()
        raise ValidationError(f"發現 {n_invalid} 行無效的 OHLC 數據")


def validate_model_output(pred: torch.Tensor, expected_shape: tuple = None) -> None:
    """
    驗證模型輸出
    
    Args:
        pred: 模型預測輸出
        expected_shape: 期望的輸出形狀
    
    Raises:
        ValidationError: 輸出驗證失敗
    """
    if pred is None:
        raise ValidationError("模型輸出為 None")
    
    if torch.isnan(pred).any():
        raise ValidationError("模型輸出包含 NaN")
    
    if torch.isinf(pred).any():
        raise ValidationError("模型輸出包含 Inf")
    
    if expected_shape is not None and pred.shape != expected_shape:
        raise ValidationError(f"輸出形狀不匹配: 期望 {expected_shape}, 實際 {pred.shape}")


def validate_config(config_dict: dict, required_keys: List[str]) -> None:
    """
    驗證配置字典
    
    Args:
        config_dict: 配置字典
        required_keys: 必需的鍵列表
    
    Raises:
        ValidationError: 配置驗證失敗
    """
    missing = [key for key in required_keys if key not in config_dict]
    if missing:
        raise ValidationError(f"配置缺少必要鍵: {missing}")
