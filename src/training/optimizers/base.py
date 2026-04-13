"""
超參數優化器基類
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class BaseHyperOptimizer(ABC):
    """超參數優化器基類"""
    
    def __init__(self, study_name: str, direction: str = "maximize"):
        """
        Args:
            study_name: 研究名稱
            direction: 優化方向，'maximize' 或 'minimize'
        """
        self.study_name = study_name
        self.direction = direction
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
    
    @abstractmethod
    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: Dict[str, Dict],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        show_progress: bool = True,
        callbacks: Optional[list] = None
    ) -> None:
        """
        執行優化
        
        Args:
            objective_fn: 目標函數，接收參數字典，返回評分
            search_space: 搜索空間定義
            n_trials: 試驗次數
        """
        pass
    
    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """返回最佳參數"""
        pass
    
    def suggest_from_space(self, trial, name: str, config: Dict) -> Any:
        """
        根據搜索空間配置建議參數值
        
        Args:
            trial: Optuna trial 對象
            name: 參數名稱
            config: 參數配置 {'type': 'int'/'float'/'categorical', ...}
        
        Returns:
            建議的參數值
        """
        param_type = config.get('type')
        
        if param_type == 'int':
            return trial.suggest_int(
                name,
                config['low'],
                config['high'],
                step=config.get('step', 1)
            )
        elif param_type == 'float':
            if config.get('log'):
                return trial.suggest_float(
                    name,
                    config['low'],
                    config['high'],
                    log=True
                )
            else:
                return trial.suggest_float(
                    name,
                    config['low'],
                    config['high'],
                    step=config.get('step')
                )
        elif param_type == 'categorical':
            return trial.suggest_categorical(name, config['choices'])
        else:
            raise ValueError(f"不支持的參數類型: {param_type}")
