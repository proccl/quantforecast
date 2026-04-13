#!/usr/bin/env python3
"""
統一數據更新入口
雙源數據獲取（akshare + 新浪/騰訊實時）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.config import get_config
from src.utils.logger import setup_logger


def main():
    config = get_config('config/config.yaml')
    
    logger = setup_logger(
        'update',
        level=config.logging.level,
        log_to_file=config.logging.log_to_file,
        log_to_console=config.logging.log_to_console,
        log_file=f"{config.paths.logs_dir}/update.log"
    )
    
    logger.info("=" * 60)
    logger.info("【數據更新入口】")
    logger.info("=" * 60)
    
    # TODO: 實現數據更新邏輯
    # 這裡應該調用 src/update_data.py 的邏輯
    logger.info("數據更新功能待實現（需要從原項目遷移）")
    logger.info(f"目標數據文件: {config.data.data_file}")


if __name__ == '__main__':
    main()
