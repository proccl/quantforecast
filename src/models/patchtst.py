"""
PatchTST 模型實現
Channel-Independent Time Series Forecasting with Patching

Reference:
    Nie et al., "A Time Series is Worth 64 Words", NeurIPS 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from src.models.revin import RevIN


class PatchEmbedding(nn.Module):
    """Patch 嵌入層：將時間序列分割成 patches 並嵌入"""
    
    def __init__(self, d_model: int, patch_len: int, stride: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [batch, seq_len, n_features]
        Returns:
            patches: [batch * n_features, n_patches, d_model]
            n_features: 特徵數量
        """
        # [batch, seq_len, n_features] -> [batch, n_features, seq_len]
        x = x.permute(0, 2, 1)
        
        # unfold: [batch, n_features, n_patches, patch_len]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        batch_size, n_features, n_patches, _ = patches.shape
        
        # reshape: [batch * n_features, n_patches, patch_len]
        patches = patches.reshape(batch_size * n_features, n_patches, self.patch_len)
        
        # 線性嵌入
        patches = self.value_embedding(patches)
        patches = self.dropout(patches)
        
        return patches, n_features


class PositionalEncoding(nn.Module):
    """正弦位置編碼"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class PatchTST(nn.Module):
    """
    PatchTST 時間序列預測模型
    
    特點：
    1. Channel-Independent: 每個特徵獨立處理
    2. Patching: 將時間序列分割成 patches
    3. Transformer Encoder: 使用標準 Transformer 編碼器
    """
    
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        head_type: str = 'regression',
        use_revin: bool = True
    ):
        """
        Args:
            n_features: 輸入特徵數量
            seq_len: 輸入序列長度
            pred_len: 預測序列長度（未使用，保留兼容性）
            patch_len: patch 長度
            stride: patch 步長
            d_model: 嵌入維度
            n_heads: 注意力頭數
            n_layers: Transformer 層數
            d_ff: 前饋網絡維度
            dropout: Dropout 率
            head_type: 'regression' 或 'classification'
            use_revin: 是否使用 RevIN
        """
        super().__init__()
        
        self.n_features = n_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.head_type = head_type
        self.use_revin = use_revin
        
        # 計算 patch 數量
        self.n_patches = (seq_len - patch_len) // stride + 1
        
        # RevIN 歸一化
        if use_revin:
            self.revin = RevIN(n_features)
        
        # Patch 嵌入
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)
        
        # 位置編碼
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer 編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        # Flatten 層
        self.flatten = nn.Flatten(start_dim=-2)
        
        # 輸出頭
        if head_type == 'regression':
            # 回歸：預測收益率
            self.head = nn.Linear(d_model * self.n_patches, 1)
        else:
            # 分類：預測漲跌
            self.head = nn.Sequential(
                nn.Linear(d_model * self.n_patches, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 2)
            )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化權重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: [batch, seq_len, n_features]
        Returns:
            output: [batch, 1] 或 [batch, 2]
        """
        # RevIN 歸一化
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        # Patch 嵌入: [batch * n_features, n_patches, d_model]
        patches, n_features = self.patch_embedding(x)
        
        # 位置編碼
        patches = self.pos_encoder(patches)
        
        # Transformer 編碼
        encoded = self.transformer_encoder(patches)
        
        # Reshape: [batch, n_features, n_patches, d_model]
        batch_size = x.shape[0]
        encoded = encoded.reshape(batch_size, n_features, self.n_patches, self.d_model)
        
        # Flatten patches: [batch, n_features, n_patches * d_model]
        encoded = self.flatten(encoded)
        
        # Channel-independent: 平均所有特徵 [batch, n_patches * d_model]
        encoded = encoded.mean(dim=1)
        
        # 輸出頭
        output = self.head(encoded)
        
        return output
    
    def get_num_params(self) -> int:
        """獲取可訓練參數數量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
