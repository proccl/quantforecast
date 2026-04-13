"""
日誌工具模塊
統一日誌配置
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_file: Optional[str] = None,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    設置統一日誌配置
    
    Args:
        name: 日誌器名稱
        level: 日誌級別 (DEBUG/INFO/WARNING/ERROR)
        log_to_file: 是否寫入文件
        log_to_console: 是否輸出到控制台
        log_file: 日誌文件路徑
        format_str: 日誌格式
    
    Returns:
        配置好的 Logger 實例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重複添加 handler
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter(format_str)
    
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_to_file and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
