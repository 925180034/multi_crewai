# run.py
import sys
import os
import warnings
import logging
from dotenv import load_dotenv
from pathlib import Path


# 配置日志级别以抑制特定警告
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

from multi.main import kickoff

if __name__ == "__main__":
    # 验证环境变量是否正确加载
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    kickoff()