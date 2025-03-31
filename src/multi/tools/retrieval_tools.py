from typing import Optional, Dict, Any
from pathlib import Path
import os
import json
import logging
import time
import shutil
import backoff
import pymysql
from contextlib import contextmanager
from tenacity import retry, wait_exponential, stop_after_attempt
from pymysql.cursors import DictCursor
import pymysql

from crewai_tools import MySQLSearchTool, SerperDevTool, LlamaIndexTool
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RetrievalTools:
    """用于数据检索的工具集合"""
    
    def __init__(
        self,
        db_url: str,
        docs_dir: str = "documents",
        index_dir: str = "index_storage",
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 1024,
        chunk_overlap: int = 20
    ):
        """
        初始化检索工具集
        
        Args:
            db_url (str): 数据库连接URL
            docs_dir (str): 文档目录路径
            index_dir (str): 索引存储目录路径
            embedding_model (str): 使用的embedding模型名称
            chunk_size (int): 文档分块大小
            chunk_overlap (int): 文档分块重叠大小
        """
        self.db_url = db_url
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 确保目录存在
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 验证和初始化OpenAI API密钥
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key:
                logger.warning("OpenAI API key not found in environment variables")
                return
            
            # 初始化 embeddings 模型
            try:
                self.embed_model = OpenAIEmbedding(
                    api_key=self.openai_api_key,
                    model=embedding_model,
                    dimensions=1536,
                    retry_on_failure=True,
                    max_retries=3,
                    timeout=30
                )
                
                # 设置 LlamaIndex 全局配置
                Settings.embed_model = self.embed_model
                Settings.chunk_size = chunk_size
                Settings.chunk_overlap = chunk_overlap
                
                # 初始化存储
                self._initialize_storage()
                logger.info("RetrievalTools初始化成功")
                
            except Exception as e:
                logger.error(f"初始化embedding模型失败: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"RetrievalTools初始化失败: {str(e)}")
            raise
        
    def _initialize_storage(self) -> Any:
        """
        初始化必要的存储目录和文件
        
        Returns:
            StorageContext: LlamaIndex存储上下文
        """
        from llama_index.core.storage import StorageContext
        
        # 创建必要的目录
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化必要的JSON文件
        required_files = [
            "docstore.json",
            "index_store.json",
            "graph_store.json",
            "vector_store.json"
        ]
        
        # 确保目录中有基本文件结构
        for filename in required_files:
            file_path = self.index_dir / filename
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump({}, f)
        
        logger.info(f"已创建或验证索引文件存在于 {self.index_dir}")
        
        # 创建存储上下文
        try:
            # 从头创建新的存储上下文而不是加载现有的
            storage_context = StorageContext.from_defaults()
            return storage_context
        except Exception as e:
            logger.error(f"创建存储上下文失败: {str(e)}")
            # 返回None而不是尝试创建新的上下文
            return None
    
    @contextmanager
    def _db_connection(self):
        """数据库连接上下文管理器，使用标准连接而非连接池"""
        conn = None
        try:
            # 解析MySQL连接URL
            # 格式: mysql://username:password@host:port/database
            if '://' in self.db_url:
                url_parts = self.db_url.split('://', 1)[1]
                auth_parts, host_parts = url_parts.split('@', 1)
                username, password = auth_parts.split(':', 1)
                host_db_parts = host_parts.split('/')
                host_parts = host_db_parts[0].split(':')
                host = host_parts[0]
                port = int(host_parts[1]) if len(host_parts) > 1 else 3306
                database = host_db_parts[1] if len(host_db_parts) > 1 else ''
            else:
                # 假设是简单格式或已经解析好的配置
                parts = self.db_url.split(':')
                username = parts[0] if len(parts) > 0 else ''
                password = parts[1] if len(parts) > 1 else ''
                host = parts[2] if len(parts) > 2 else 'localhost'
                port = int(parts[3]) if len(parts) > 3 else 3306
                database = parts[4] if len(parts) > 4 else ''

            # 建立单一连接而非使用连接池
            conn = pymysql.connect(
                host=host,
                user=username,
                password=password,
                database=database,
                port=port,
                connect_timeout=10,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            logger.info("数据库连接成功")
            yield conn
        except Exception as e:
            logger.error(f"数据库连接错误: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
                logger.info("数据库连接已正确关闭")

    def _safe_remove_dir(self, path: Path):
        """
        安全地删除目录
        
        Args:
            path (Path): 要删除的目录路径
        """
        try:
            if path.exists():
                shutil.rmtree(path)
                logger.info(f"Removed directory: {path}")
        except Exception as e:
            logger.error(f"Error removing directory {path}: {str(e)}")

    def _get_or_create_index(self):
        """获取或创建文档索引，改为直接创建新索引"""
        try:
            # 如果目录为空，创建测试文档
            if not any(self.docs_dir.iterdir()):
                test_doc = self.docs_dir / "test.txt"
                test_doc.write_text("This is a test document for indexing purposes.")
                logger.info(f"Created test document: {test_doc}")

            # 加载文档
            reader = SimpleDirectoryReader(
                input_dir=str(self.docs_dir),
                recursive=True,
                exclude_hidden=True,
                filename_as_id=True
            )
            
            documents = reader.load_data()
            
            if not documents:
                logger.warning("No documents loaded")
                return None

            # 创建节点解析器和解析文档
            parser = SimpleNodeParser.from_defaults(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            logger.info(f"Parsing documents with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
            nodes = parser.get_nodes_from_documents(documents)
            logger.info(f"Created {len(nodes)} nodes from {len(documents)} documents")
            
            # 创建全新的存储上下文，不从硬盘加载
            from llama_index.core.storage import StorageContext
            storage_context = StorageContext.from_defaults()

            # 创建索引
            logger.info("Creating new vector store index")
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )

            # 不尝试持久化存储，避免文件访问问题
            logger.info("Index created in memory")
            return index

        except Exception as e:
            logger.error(f"Error in index creation: {str(e)}", exc_info=True)
            return None

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        base=2
    )
    def get_database_tool(self) -> Optional[MySQLSearchTool]:
        """获取数据库搜索工具，增强了表选择和错误处理"""
        try:
            # 测试数据库连接
            with self._db_connection() as conn:
                logger.info("Database connection successful")
                
                # 验证数据库表是否存在
                cursor = conn.cursor()
                
                # 查询所有表名
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = DATABASE()
                """)
                tables = [row['table_name'] for row in cursor.fetchall()]
                logger.info(f"找到数据库表: {tables}")
                
                if not tables:
                    logger.warning("数据库中没有找到任何表，创建示例表")
                    
                    # 尝试创建一个简单的示例表
                    try:
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS test_users (
                                id INT AUTO_INCREMENT PRIMARY KEY,
                                name VARCHAR(100),
                                email VARCHAR(100),
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """)
                        conn.commit()
                        logger.info("创建了示例表 test_users")
                        tables = ['test_users']
                    except Exception as e:
                        logger.error(f"创建示例表失败: {str(e)}")
                        return None
            
            # 获取默认表
            default_table = tables[0] if tables else None
            
            if not default_table:
                logger.error("无法确定默认表")
                return None
                
            logger.info(f"使用表 '{default_table}' 作为默认搜索表")

            return MySQLSearchTool(
                db_uri=self.db_url,
                table_name=default_table,
                verbose=True,
                config=dict(
                    embedder=dict(
                        provider="openai",
                        config=dict(
                            model="text-embedding-ada-002",
                        ),
                    ),
                ),
                chunk_size=5,
                similarity_threshold=0.7,
                max_results=10,
                connection_timeout=10,
                query_timeout=30,
                retry_strategy={
                    "max_attempts": 3,
                    "backoff_factor": 2
                }
            )

        except Exception as e:
            logger.error(f"Failed to create database search tool: {str(e)}")
            return None

    def get_web_tool(self) -> Optional[SerperDevTool]:
        """
        获取网络搜索工具
        
        Returns:
            Optional[SerperDevTool]: 网络搜索工具实例
        """
        try:
            return SerperDevTool(
                api_key=os.getenv("SERPER_API_KEY"),
                retry_on_failure=True,
                max_retries=3
            )
        except Exception as e:
            logger.error(f"Failed to create web search tool: {str(e)}")
            return None

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        base=2
    )
    def get_document_tool(self) -> Optional[LlamaIndexTool]:
        """
        获取文档搜索工具
        
        Returns:
            Optional[LlamaIndexTool]: 文档搜索工具实例
        """
        try:
            index = self._get_or_create_index()
            if not index:
                logger.warning("No index created, document search tool unavailable")
                return None

            # 创建查询引擎
            query_engine = index.as_query_engine(
                similarity_top_k=3,
                node_postprocessors=[],
                response_mode="tree_summarize",
                streaming=True,
                verbose=True
            )

            # 创建工具
            tool = LlamaIndexTool.from_query_engine(
                query_engine,
                name="Document Search",
                description="Search through documents for relevant information",
                metadata={
                    "source": "local_documents",
                    "type": "text_search",
                    "capabilities": [
                        "document_search",
                        "content_retrieval",
                        "semantic_search"
                    ]
                }
            )
            
            # 添加错误处理装饰器
            @tool.wrap_tool
            def formatted_tool(*args, **kwargs):
                try:
                    formatted_kwargs = {
                        k: str(v) if v is not None else "" 
                        for k, v in kwargs.items()
                    }
                    return tool.run(**formatted_kwargs)
                except Exception as e:
                    logger.error(f"Error in document tool: {str(e)}")
                    return f"Error searching documents: {str(e)}"
                    
            return formatted_tool
                
        except Exception as e:
            logger.error(f"Failed to create document search tool: {str(e)}")
            return None
            
    def cleanup(self):
        """清理资源"""
        try:
            # 清理索引目录
            self._safe_remove_dir(self.index_dir)
            
            # 清理嵌入模型资源
            if hasattr(self, 'embed_model') and hasattr(self.embed_model, 'cleanup'):
                try:
                    self.embed_model.cleanup()
                    logger.info("嵌入模型资源已清理")
                except Exception as e:
                    logger.error(f"清理嵌入模型时出错: {str(e)}")
            
            logger.info("清理操作完成")
        except Exception as e:
            logger.error(f"清理过程中出错: {str(e)}")

