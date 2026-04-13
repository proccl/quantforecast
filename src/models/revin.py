"""
RevIN (Reversible Instance Normalization) 模塊
處理時間序列數據的分佈偏移問題
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    可逆實例歸一化
    
    Reference:
        Kim et al., "Reversible Instance Normalization for Accurate 
        Time-Series Forecasting against Distribution Shift", ICLR 2022
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Args:
            x: 輸入張量 [batch, seq_len, features] 或 [batch, features]
            mode: 'norm' 或 'denorm'
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise ValueError(f"未知模式: {mode}")
        return x
    
    def _get_statistics(self, x: torch.Tensor) -> None:
        """計算均值和標準差"""
        # 沿時間維度計算 (除 batch 和 feature 維度外的所有維度)
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """歸一化"""
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """反歸一化"""
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * 0)
        x = x * self.stdev
        x = x + self.mean
        return x
