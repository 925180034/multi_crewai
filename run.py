# run.py
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
    args = parser.parse_args()
    
    # 使用命令行参数运行
    from multi.main import kickoff
    kickoff(dataset_path=args.dataset_path, dataset_type=args.dataset_type)