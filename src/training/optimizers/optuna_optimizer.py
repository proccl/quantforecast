"""
Optuna 超參數優化實現
"""

import optuna
from typing import Dict, Any, Callable, Optional, List
import logging

from .base import BaseHyperOptimizer

logger = logging.getLogger(__name__)


class OptunaOptimizer(BaseHyperOptimizer):
    """Optuna 貝葉斯優化實現"""
    
    def __init__(
        self,
        study_name: str,
        direction: str = "maximize",
        seed: int = 42
    ):
        super().__init__(study_name, direction)
        self.seed = seed
        self.study: Optional[optuna.Study] = None
    
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
        執行 Optuna 優化
        
        Args:
            objective_fn: 目標函數，接收參數字典，返回評分
            search_space: 搜索空間定義
            n_trials: 試驗次數
            timeout: 超時時間（秒）
            show_progress: 是否顯示進度條
        """
        logger.info(f"開始 Optuna 優化: {self.study_name}")
        logger.info(f"試驗次數: {n_trials}, 優化方向: {self.direction}")
        
        def objective(trial):
            # 從搜索空間構建參數
            params = {}
            for name, config in search_space.items():
                params[name] = self.suggest_from_space(trial, name, config)
            
            # 執行目標函數
            try:
                score = objective_fn(params)
                return score
            except Exception as e:
                logger.warning(f"Trial 失敗: {e}")
                return float('-inf') if self.direction == 'maximize' else float('inf')
        
        # 創建研究
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        
        # 運行優化
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
            callbacks=callbacks or []
        )
        
        # 保存最佳結果
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"優化完成！最佳分數: {self.best_score:.6f}")
        logger.info(f"最佳參數: {self.best_params}")
    
    def get_best_params(self) -> Dict[str, Any]:
        """返回最佳參數"""
        if self.best_params is None:
            raise RuntimeError("優化尚未完成，請先調用 optimize()")
        return self.best_params
    
    def get_best_score(self) -> float:
        """返回最佳分數"""
        if self.best_score is None:
            raise RuntimeError("優化尚未完成，請先調用 optimize()")
        return self.best_score
    
    def get_all_trials(self) -> List[Dict]:
        """返回所有試驗結果"""
        if self.study is None:
            raise RuntimeError("優化尚未完成，請先調用 optimize()")
        
        trials = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials.append({
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value
                })
        return trials
    
    def plot_optimization_history(self, output_path: str) -> None:
        """生成優化歷史圖表"""
        if self.study is None:
            raise RuntimeError("優化尚未完成，請先調用 optimize()")
        
        try:
            import optuna.visualization as vis
            fig = vis.plot_optimization_history(self.study)
            fig.write_image(output_path)
            logger.info(f"優化歷史圖表已保存: {output_path}")
        except Exception as e:
            logger.warning(f"無法生成圖表: {e}")
    
    def plot_param_importances(self, output_path: str) -> None:
        """生成參數重要性圖表"""
        if self.study is None:
            raise RuntimeError("優化尚未完成，請先調用 optimize()")
        
        try:
            import optuna.visualization as vis
            fig = vis.plot_param_importances(self.study)
            fig.write_image(output_path)
            logger.info(f"參數重要性圖表已保存: {output_path}")
        except Exception as e:
            logger.warning(f"無法生成圖表: {e}")
