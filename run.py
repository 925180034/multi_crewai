#!/usr/bin/env python
# run.py - 优化版本
import sys
import os
import warnings
import logging
from dotenv import load_dotenv
from pathlib import Path

# 配置日志级别以抑制特定警告
logger = logging.getLogger(__name__)
logging.getLogger('chromadb').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=ResourceWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 获取项目根目录
ROOT_DIR = Path(__file__).parent

# 加载.env文件
load_dotenv(os.path.join(ROOT_DIR, '.env'))

# 添加源代码目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# 获取Spider数据集路径，如果不存在则设为None
spider_dir = os.getenv('SPIDER_DATASET_PATH')
if spider_dir and os.path.exists(spider_dir):
    logger.info(f'已加载Spider数据集路径: {spider_dir}')
else:
    logger.warning('未找到Spider数据集路径或路径无效，将使用默认查询')
    spider_dir = None

if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='运行查询流程')
    parser.add_argument('--dataset-path', type=str, default=spider_dir, help='数据集路径')
    parser.add_argument('--dataset-type', type=str, default='spider', choices=['spider', 'custom', 'default'], help='数据集类型')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'], help='运行模式: 单个查询或批量测试')
    parser.add_argument('--limit', type=int, default=10, help='批量测试时的查询数量限制')
    parser.add_argument('--start-index', type=int, default=0, help='批量测试的起始索引')
    args = parser.parse_args()
    
    try:
        # 根据模式运行
        if args.mode == 'single':
            # 单个查询模式
            from multi.main import kickoff
            kickoff(dataset_path=args.dataset_path, dataset_type=args.dataset_type)
        else:
            # 批量测试模式
            from multi.main import batch_test_spider
            batch_test_spider(
                spider_dataset_path=args.dataset_path, 
                limit=args.limit, 
                start_index=args.start_index
            )
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(0)
    except Exception as e:
        logger.error(f"运行时发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)