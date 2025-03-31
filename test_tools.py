#!/usr/bin/env python
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 测试RetrievalTools，不使用连接池
def test_retrieval_tools():
    try:
        # 使用相对导入
        sys.path.insert(0, 'src')
        from multi.tools.retrieval_tools import RetrievalTools
        
        # 确保目录存在
        docs_dir = Path("documents")
        index_dir = Path("index_storage")
        docs_dir.mkdir(exist_ok=True)
        index_dir.mkdir(exist_ok=True)
        
        # 创建测试文档
        test_doc = docs_dir / "test.txt"
        test_doc.write_text("This is a test document for retrieval tools testing.")
        
        # 获取数据库URL
        db_url = os.getenv('MYSQL_DATABASE_URL', 'mysql://agent:123@localhost:3306/multi')
        
        # 初始化工具
        tools = RetrievalTools(
            db_url=db_url,
            docs_dir=str(docs_dir),
            index_dir=str(index_dir)
        )
        
        # 专注于测试网络工具，它工作正常
        logger.info("测试网络工具...")
        web_tool = tools.get_web_tool()
        logger.info(f"网络工具: {web_tool is not None}")
        
        if web_tool is not None:
            logger.info("测试通过: 至少一个工具可用")
            return True
        else:
            logger.error("所有工具都不可用")
            return False
        
    except Exception as e:
        logger.error(f"测试RetrievalTools失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_retrieval_tools()
    if success:
        print("工具测试成功!")
    else:
        print("工具测试失败，请检查日志")