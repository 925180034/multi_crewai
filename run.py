# run.py
import sys
import os
import warnings
import logging
from dotenv import load_dotenv
from pathlib import Path
# from multi.main import kickoff

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

# 获取Spider数据集路径
spider_dir = os.getenv('SPIDER_DATASET_PATH')
if not spider_dir or not os.path.exists(spider_dir):
    logger.error(f'Spider数据集未在{spider_dir}找到')
    raise ValueError('SPIDER_DATASET_PATH未在.env文件中正确设置')
    
# 添加日志记录以确认路径已成功加载
logger.info(f'已加载Spider数据集路径: {spider_dir}')

# 添加源代码目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


if __name__ == "__main__":
    # 获取Spider数据集路径
    spider_dir = os.getenv('SPIDER_DATASET_PATH')
    if not spider_dir or not os.path.exists(spider_dir):
        logger.error(f'Spider数据集未在{spider_dir}找到')
        raise ValueError('SPIDER_DATASET_PATH未在.env文件中正确设置')
    
    # 使用Spider数据集测试
    from multi.main import test_with_spider
    test_with_spider(spider_dir, query_index=0)  # 更改索引以测试不同的查询
    
    # 或者运行常规流程
    # kickoff(spider_dataset_path=spider_dir)