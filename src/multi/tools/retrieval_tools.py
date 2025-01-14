from typing import Optional, Dict, Any
from pathlib import Path
import os
import json
import logging
import time
import shutil
import backoff
import psycopg
from contextlib import contextmanager
from tenacity import retry, wait_exponential, stop_after_attempt

from crewai_tools import PGSearchTool, SerperDevTool, LlamaIndexTool
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
        
        # 验证和初始化OpenAI API密钥
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # 初始化 embeddings 模型
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
        
    def _initialize_storage(self) -> StorageContext:
        """
        初始化必要的存储目录和文件
        
        Returns:
            StorageContext: LlamaIndex存储上下文
        """
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
        
        for filename in required_files:
            file_path = self.index_dir / filename
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump({}, f)
        
        # 创建存储上下文
        storage_context = StorageContext.from_defaults(
            persist_dir=str(self.index_dir)
        )
        storage_context.persist()
        
        return storage_context
    
    @contextmanager
    def _db_connection(self):
        """数据库连接上下文管理器"""
        conn = None
        try:
            conn = psycopg.connect(
                self.db_url,
                connect_timeout=10
            )
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
                logger.info("Database connection closed properly")

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
        """
        获取或创建文档索引
        
        Returns:
            VectorStoreIndex: 文档索引对象
        """
        try:
            # 如果目录为空，创建测试文档
            if not any(self.docs_dir.iterdir()):
                test_doc = self.docs_dir / "test.txt"
                test_doc.write_text("This is a test document.")
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
            parser = SimpleNodeParser.from_defaults()
            nodes = parser.get_nodes_from_documents(documents)
            
            # 创建或加载存储上下文
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.index_dir)
            )

            # 创建索引
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )

            # 持久化存储
            index.storage_context.persist(persist_dir=str(self.index_dir))
            logger.info(f"Index saved to {self.index_dir}")

            return index

        except Exception as e:
            logger.error(f"Error in index creation/loading: {str(e)}", exc_info=True)
            return None

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        base=2
    )
    def get_database_tool(self) -> Optional[PGSearchTool]:
        """
        获取数据库搜索工具
        
        Returns:
            Optional[PGSearchTool]: 数据库搜索工具实例
        """
        try:
            # 测试数据库连接
            with self._db_connection() as conn:
                logger.info("Database connection successful")
                
                # 验证表是否存在
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'visits'
                    );
                """)
                table_exists = cursor.fetchone()[0]
                if not table_exists:
                    raise ValueError("Required table 'visits' does not exist")

            return PGSearchTool(
                db_uri=self.db_url,
                table_name='visits',
                embed_model=self.embed_model,
                verbose=True,
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
        """
        清理资源
        """
        try:
            # 清理索引目录
            self._safe_remove_dir(self.index_dir)
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")