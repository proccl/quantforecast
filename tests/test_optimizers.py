"""
優化器模塊單元測試
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.optimizers.optuna_optimizer import OptunaOptimizer
from src.training.optimizers.search_spaces import get_search_space, PATCHTST_BASIC


class TestSearchSpaces:
    """測試搜索空間"""
    
    def test_get_basic_space(self):
        space = get_search_space('basic')
        assert 'd_model' in space
        assert 'n_heads' in space
        assert 'dropout' in space
    
    def test_get_advanced_space(self):
        space = get_search_space('advanced')
        assert 'patch_len' in space
        assert 'batch_size' in space
    
    def test_invalid_space_name(self):
        with pytest.raises(ValueError):
            get_search_space('invalid_name')


class TestOptunaOptimizer:
    """測試 Optuna 優化器"""
    
    def test_optimize_maximize(self):
        def objective(params):
            return params['d_model'] / 128.0 - params['dropout']
        
        optimizer = OptunaOptimizer('test_maximize', direction='maximize', seed=42)
        optimizer.optimize(objective, PATCHTST_BASIC, n_trials=2, show_progress=False)
        
        best = optimizer.get_best_params()
        assert 'd_model' in best
        assert 'dropout' in best
        assert optimizer.get_best_score() is not None
    
    def test_optimize_minimize(self):
        def objective(params):
            return params['dropout']
        
        optimizer = OptunaOptimizer('test_minimize', direction='minimize', seed=42)
        optimizer.optimize(objective, PATCHTST_BASIC, n_trials=2, show_progress=False)
        
        assert optimizer.get_best_score() is not None
    
    def test_get_all_trials(self):
        def objective(params):
            return 1.0
        
        optimizer = OptunaOptimizer('test_trials', seed=42)
        optimizer.optimize(objective, PATCHTST_BASIC, n_trials=2, show_progress=False)
        
        trials = optimizer.get_all_trials()
        assert len(trials) == 2
    
    def test_not_optimized_error(self):
        optimizer = OptunaOptimizer('test_not_optimized')
        
        with pytest.raises(RuntimeError):
            optimizer.get_best_params()
        
        with pytest.raises(RuntimeError):
            optimizer.get_best_score()
