"""
模型模塊單元測試
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.models.revin import RevIN
from src.models.patchtst import PatchTST


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


class TestRevIN:
    """測試 RevIN"""
    
    def test_reversible(self):
        revin = RevIN(num_features=5)
        x = torch.randn(2, 10, 5)
        
        normalized = revin(x, mode='norm')
        denormalized = revin(normalized, mode='denorm')
        
        error = torch.abs(x - denormalized).mean()
        assert error.item() < 1e-6
    
    def test_output_shape(self):
        revin = RevIN(num_features=3)
        x = torch.randn(4, 20, 3)
        out = revin(x, mode='norm')
        assert out.shape == x.shape


class TestPatchTST:
    """測試 PatchTST"""
    
    def test_forward_pass(self):
        kwargs = get_default_model_kwargs()
        model = PatchTST(**kwargs)
        
        x = torch.randn(4, 20, 5)
        out = model(x)
        
        assert out.shape == (4, 1)
    
    def test_classification_head(self):
        kwargs = get_default_model_kwargs(head_type='classification')
        model = PatchTST(**kwargs)
        
        x = torch.randn(2, 20, 5)
        out = model(x)
        
        assert out.shape == (2, 2)
    
    def test_count_parameters(self):
        kwargs = get_default_model_kwargs(
            n_features=3,
            seq_len=10,
            d_model=16,
            n_heads=2,
            n_layers=1
        )
        model = PatchTST(**kwargs)
        n_params = model.get_num_params()
        assert n_params > 0
