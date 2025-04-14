from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
import os
import logging
from typing import Optional, List
from multi.tools.retrieval_tools import RetrievalTools
from multi.tools.document_retrieval_tool import DocumentRetrievalTool
from crewai.tools import BaseTool
from typing import Type, Dict, Any
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置环境变量
if not os.getenv('MYSQL_DATABASE_URL'):
    os.environ['MYSQL_DATABASE_URL'] = 'mysql://agent:123@localhost:3306/multi'

@CrewBase
class RetrievalCrew:
    """数据检索小组，负责从不同来源检索和处理数据"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, dataset_path=None):
        """初始化检索小组"""
        # 先调用父类初始化方法
        super().__init__()
        
        # 存储Spider数据集路径
        self.dataset_path = dataset_path
        logger.info(f'RetrievalCrew初始化，数据集路径: {self.dataset_path}')
        
        # 默认设置空工具
        self.doc_tool = None
        self.db_tool = None
        self.web_tool = None  # 禁用此工具以避免消耗Serper tokens
        self.tools = None
        
        try:
            # 验证环境变量
            self._validate_environment()
            
            # 初始化工具集
            logger.info("正在初始化检索工具...")
            self.tools = self._initialize_tools()
            
            if self.tools:
                # 只初始化文档工具和数据库工具，不初始化web工具
                try:
                    # 不初始化web_tool以避免使用Serper tokens
                    # self.web_tool = self.tools.get_web_tool()
                    self.web_tool = self._create_fallback_web_tool()  # 使用后备工具代替
                    logger.info("使用后备网络检索工具以避免消耗Serper tokens")
                    
                    # 文档工具初始化
                    if self.dataset_path:
                        # 如果提供了数据集路径，使用Spider数据集
                        self.doc_tool = DocumentRetrievalTool(
                            document_path=self.dataset_path
                        )
                        logger.info(f'使用Spider数据集初始化文档检索工具: {self.dataset_path}')
                    else:
                        # 否则使用默认文档工具
                        self.doc_tool = self.tools.get_document_tool()
                    
                    # 数据库工具初始化
                    self.db_tool = self.tools.get_database_tool()
                    
                except Exception as e:
                    logger.error(f"工具初始化失败: {str(e)}")
            else:
                logger.warning("检索工具集初始化失败")
        except Exception as e:
            logger.error(f"检索工具初始化发生错误: {str(e)}")

    def _create_fallback_db_tool(self):
        """创建一个后备数据库工具，在主要工具失败时使用"""
        # 创建输入模型
        class FallbackDBInput(BaseModel):
            query: str = Field(..., description="The search query")
        
        # 定义后备工具
        class FallbackDBTool(BaseTool):
            name: str = "Fallback Database Search"
            description: str = "A fallback tool for database searches when the primary tool is unavailable"
            args_schema: Type[BaseModel] = FallbackDBInput
            
            def _run(self, query: str) -> str:
                return f"Database tool is currently unavailable. Your query was: {query}"
        
        return FallbackDBTool()

    def _create_fallback_web_tool(self):
        """创建一个功能性的后备网络工具，可以返回Spider相关信息而不是使用Serper"""
        
        from crewai.tools import BaseTool
        from typing import Type
        from pydantic import BaseModel, Field
        import json
        import os
        # 创建输入模型
        class SpiderWebInput(BaseModel):
            query: str = Field(..., description="The search query")
        
        # 定义包含Spider信息的后备工具
        class SpiderInfoTool(BaseTool):
            name: str = "Spider Database Info Tool"
            description: str = "Provides information about database schemas and SQL queries from the Spider dataset instead of searching the web"
            args_schema: Type[BaseModel] = SpiderWebInput
            dataset_path: str = self.dataset_path if hasattr(self, 'dataset_path') else None
            
            def _run(self, query: str) -> str:
                """从Spider数据集返回相关信息，而不是搜索网络"""
                if not self.dataset_path or not os.path.exists(self.dataset_path):
                    return f"[后备工具] 无法提供Spider信息因为数据集路径不可用。您的查询是: {query}"
                
                try:
                    # 尝试从数据集加载一些相关信息
                    response_parts = []
                    
                    # 检查tables.json
                    tables_path = os.path.join(self.dataset_path, 'tables.json')
                    if os.path.exists(tables_path):
                        with open(tables_path, 'r') as f:
                            tables_data = json.load(f)
                        response_parts.append(f"Spider数据集包含{len(tables_data)}个数据库模式。")
                        
                        # 找到与查询相关的数据库（简单匹配）
                        relevant_dbs = []
                        for db_schema in tables_data:
                            if any(keyword in db_schema.get('db_id', '').lower() for keyword in query.lower().split()):
                                relevant_dbs.append(db_schema.get('db_id', ''))
                        
                        if relevant_dbs:
                            response_parts.append(f"与您的查询相关的数据库可能包括: {', '.join(relevant_dbs)}")
                    
                    # 检查dev.json中的示例
                    dev_path = os.path.join(self.dataset_path, 'dev.json')
                    if os.path.exists(dev_path):
                        with open(dev_path, 'r') as f:
                            dev_data = json.load(f)
                        response_parts.append(f"开发集包含{len(dev_data)}个查询示例。")
                        
                        # 找到与查询相关的示例（简单匹配）
                        query_keywords = query.lower().split()
                        for i, item in enumerate(dev_data[:5]):  # 只检查前5个以保持响应简短
                            example_query = item.get('question', '').lower()
                            if any(keyword in example_query for keyword in query_keywords):
                                response_parts.append(f"相关示例 #{i+1}:")
                                response_parts.append(f"问题: {item.get('question', '')}")
                                response_parts.append(f"SQL: {item.get('query', '')}")
                                response_parts.append(f"数据库: {item.get('db_id', '')}")
                                break
                    
                    if not response_parts:
                        return f"[后备工具] 无法在Spider数据集中找到与'{query}'相关的信息。"
                    
                    return "\n".join(response_parts)
                except Exception as e:
                    return f"[后备工具] 处理Spider数据集时出错: {str(e)}"
        
        return SpiderInfoTool()

    def _validate_environment(self) -> None:
        """验证必要的环境变量，包括默认值处理"""
        required_vars = {
            'OPENAI_API_KEY': None,  # 必需，无默认值
            'MYSQL_DATABASE_URL': 'mysql://agent:123@localhost:3306/multi',  # 带默认值
            'SERPER_API_KEY': None,  # 必需，无默认值
        }
        
        missing = []
        for var, default in required_vars.items():
            value = os.getenv(var)
            if not value:
                if default:
                    os.environ[var] = default
                    logger.warning(f"Using default value for {var}: {default}")
                else:
                    missing.append(var)
        
        if missing:
            message = f"Missing required environment variables: {missing}"
            logger.error(message)
            raise ValueError(message)

    def _initialize_tools(self) -> Optional[RetrievalTools]:
        """初始化工具集"""
        try:
            return RetrievalTools(
                db_url=os.getenv('MYSQL_DATABASE_URL'),
                docs_dir=os.getenv('DOCS_DIR', './documents'),
                index_dir=os.getenv('INDEX_PERSIST_DIR', './index_storage'),
                embedding_model="text-embedding-3-small",  # 更新为新版嵌入模型
                chunk_size=512,  # 调整块大小
                chunk_overlap=50
            )
        except Exception as e:
            logger.error(f"Failed to initialize tools: {str(e)}")
            return None

    @agent
    def database_agent(self) -> Agent:
        """数据库检索代理"""
        tools = []
        if self.db_tool:
            tools.append(self.db_tool)
            
        return Agent(
            config=self.agents_config["database_agent"],
            tools=tools,
            max_iter=1,
            max_rpm=10,
            max_execution_time=300,
            verbose=True,
            allow_delegation=False
        )

    @agent
    def web_agent(self) -> Agent:
        """网络检索代理"""
        tools = []
        if self.web_tool:
            tools.append(self.web_tool)
            
        return Agent(
            config=self.agents_config["web_agent"],
            tools=tools,
            max_iter=1,
            max_rpm=10, 
            verbose=True,
            allow_delegation=False
        )

    @agent
    def document_agent(self) -> Agent:
        """文档检索代理"""
        tools = []
        if self.doc_tool:
            tools.append(self.doc_tool)
            
        return Agent(
            config=self.agents_config["document_agent"],
            tools=tools,
            max_iter=1,
            max_rpm=10,
            verbose=True,
            allow_delegation=False
        )

    @task
    def search_database(self) -> Task:
        """数据库搜索任务"""
        return Task(
            description="""
            根据提供的查询和计划搜索数据库信息。
            Query: {query}
            Plan: {plan}
            """,
            expected_output="检索到的数据库信息",
            agent=self.database_agent(),
            context=[],  # 可以添加依赖的上下文
            output_file="outputs/database_search_result.md"
        )

    @task
    def search_web(self) -> Task:
        """网络搜索任务"""
        return Task(
            description="""
            根据提供的查询和计划在网络上搜索相关信息。
            Query: {query}
            Plan: {plan}
            """,
            expected_output="相关的网络搜索结果",
            agent=self.web_agent(),
            context=[],
            output_file="outputs/web_search_result.md"
        )

    @task
    def search_documents(self) -> Task:
        """文档搜索任务"""
        return Task(
            description="""
            在文档中搜索相关信息。
            Query: {query}
            Plan: {plan}
            """,
            expected_output="从文档中提取的相关信息",
            agent=self.document_agent(),
            context=[],
            output_file="outputs/document_search_result.md"
        )

    def create_manager_agent(self) -> Agent:
        """创建管理者代理"""
        return Agent(
            config=self.agents_config["manager_agent"],
            max_iter=3,  # 管理者可以有更多迭代次数
            verbose=True
        )

    @crew
    def crew(self) -> Crew:
        """创建并配置检索小组"""
        try:
            return Crew(
                agents=self.agents,
                tasks=self.tasks,
                process=Process.sequential,  # 使用顺序处理
                manager_agent=self.create_manager_agent(),
                verbose=True,
                max_rpm=30,  # 整个小组的最大请求频率
                full_output=True  # 启用完整输出捕获
            )
        except Exception as e:
            logger.error(f"Failed to create crew: {str(e)}")
            raise

    def cleanup(self):
        """清理资源"""
        if self.tools:
            self.tools.cleanup()

    def __del__(self):
        """析构函数，确保资源被清理"""
        self.cleanup()


