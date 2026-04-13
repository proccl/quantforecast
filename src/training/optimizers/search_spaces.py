"""
超參數搜索空間定義
"""

# ============ 基礎搜索空間 ============
PATCHTST_BASIC = {
    'd_model': {
        'type': 'int',
        'low': 32,
        'high': 128,
        'step': 32
    },
    'n_heads': {
        'type': 'int',
        'low': 2,
        'high': 8,
        'step': 2
    },
    'n_layers': {
        'type': 'int',
        'low': 1,
        'high': 4
    },
    'dropout': {
        'type': 'float',
        'low': 0.0,
        'high': 0.3,
        'step': 0.05
    },
    'lr': {
        'type': 'float',
        'low': 1e-4,
        'high': 1e-2,
        'log': True
    }
}

# ============ 高級搜索空間 ============
PATCHTST_ADVANCED = {
    'd_model': {
        'type': 'int',
        'low': 32,
        'high': 256,
        'step': 32
    },
    'n_heads': {
        'type': 'int',
        'low': 2,
        'high': 8,
        'step': 2
    },
    'n_layers': {
        'type': 'int',
        'low': 1,
        'high': 6
    },
    'dropout': {
        'type': 'float',
        'low': 0.0,
        'high': 0.5,
        'step': 0.05
    },
    'patch_len': {
        'type': 'int',
        'low': 3,
        'high': 10,
        'step': 1
    },
    'lr': {
        'type': 'float',
        'low': 1e-4,
        'high': 1e-2,
        'log': True
    },
    'batch_size': {
        'type': 'categorical',
        'choices': [16, 32, 64, 128]
    }
}

# ============ 搜索空間註冊表 ============
SEARCH_SPACES = {
    'basic': PATCHTST_BASIC,
    'advanced': PATCHTST_ADVANCED,
    'patchtst_basic': PATCHTST_BASIC,
    'patchtst_advanced': PATCHTST_ADVANCED,
}


def get_search_space(name: str) -> dict:
    """
    根據名稱獲取搜索空間
    
    Args:
        name: 搜索空間名稱
    
    Returns:
        搜索空間字典
    """
    if name not in SEARCH_SPACES:
        available = ', '.join(SEARCH_SPACES.keys())
        raise ValueError(f"未知搜索空間 '{name}'，可用選項: {available}")
    
    return SEARCH_SPACES[name].copy()


def validate_search_space(search_space: dict) -> None:
    """驗證搜索空間配置是否合法"""
    for name, config in search_space.items():
        if 'type' not in config:
            raise ValueError(f"搜索空間參數 '{name}' 缺少 'type' 字段")
        
        param_type = config['type']
        if param_type not in ('int', 'float', 'categorical'):
            raise ValueError(f"搜索空間參數 '{name}' 類型不合法: {param_type}")
        
        if param_type == 'categorical' and 'choices' not in config:
            raise ValueError(f"搜索空間參數 '{name}' (categorical) 缺少 'choices' 字段")
        
        if param_type in ('int', 'float') and ('low' not in config or 'high' not in config):
            raise ValueError(f"搜索空間參數 '{name}' 缺少 'low' 或 'high' 字段")
