"""
配置管理模块
统一加载和管理配置文件
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class DataConfig:
    """数据配置"""
    symbol: str = "01810"
    name: str = "xiaomi"
    data_file: str = "data/xiaomi_real.csv"
    seq_len: int = 20
    pred_len: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "PatchTST"
    patch_size: int = 16
    stride: int = 8
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    output_dim: int = 1


@dataclass
class TrainingConfig:
    """训练配置"""
    optimizer: str = "Adam"
    lr: float = 0.001
    weight_decay: float = 1e-4
    scheduler: str = "ReduceLROnPlateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    epochs: int = 100
    batch_size: int = 32
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    device: str = "auto"


@dataclass
class OptimizationConfig:
    """超参数优化配置"""
    enabled: bool = False
    n_trials: int = 100
    timeout: int = 3600
    search_space: str = "basic"
    objective: str = "val_accuracy"
    direction: str = "maximize"


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    long_threshold: float = 0.01
    short_threshold: float = -0.01
    max_position: float = 1.0
    stop_loss: float = -0.05
    take_profit: float = 0.10


@dataclass
class PathsConfig:
    """路径配置"""
    data_dir: str = "data"
    model_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Config:
    """主配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            optimization=OptimizationConfig(**config_dict.get('optimization', {})),
            backtest=BacktestConfig(**config_dict.get('backtest', {})),
            paths=PathsConfig(**config_dict.get('paths', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def to_yaml(self, path: str) -> None:
        """保存配置到 YAML 文件"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'optimization': self.optimization.__dict__,
            'backtest': self.backtest.__dict__,
            'paths': self.paths.__dict__,
            'logging': self.logging.__dict__
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """更新配置（用于超参数优化）"""
        for key, value in updates.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.training, key):
                setattr(self.training, key, value)


# 全局配置实例（延迟加载）
_config: Optional[Config] = None


def get_config(path: str = "config/config.yaml") -> Config:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = Config.from_yaml(path)
    return _config


def reload_config(path: str = "config/config.yaml") -> Config:
    """重新加载配置"""
    global _config
    _config = Config.from_yaml(path)
    return _config
