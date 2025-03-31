from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
import os
import logging
from typing import Optional, List
from multi.tools.retrieval_tools import RetrievalTools
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

    def __init__(self):
        """初始化检索小组"""
        # 先调用父类初始化方法
        super().__init__()
        
        # 默认设置空工具
        self.doc_tool = None
        self.db_tool = None
        self.web_tool = None
        self.tools = None
        
        try:
            # 验证环境变量
            self._validate_environment()
            
            # 初始化工具集
            logger.info("正在初始化检索工具...")
            self.tools = self._initialize_tools()
            
            if self.tools:
                # 分别获取每个工具，不抛出错误
                try:
                    self.web_tool = self.tools.get_web_tool()
                    logger.info("网络检索工具初始化成功" if self.web_tool else "网络检索工具获取失败")
                except Exception as e:
                    logger.error(f"网络工具初始化失败: {str(e)}")
                    self.web_tool = None
                    
                try:
                    self.doc_tool = self.tools.get_document_tool()
                    logger.info("文档检索工具初始化成功" if self.doc_tool else "文档检索工具获取失败")
                except Exception as e:
                    logger.error(f"文档工具初始化失败: {str(e)}")
                    self.doc_tool = None
                    
                try:
                    self.db_tool = self.tools.get_database_tool()
                    logger.info("数据库检索工具初始化成功" if self.db_tool else "数据库检索工具获取失败")
                except Exception as e:
                    logger.error(f"数据库工具初始化失败: {str(e)}")
                    self.db_tool = None
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
        """创建一个后备网络搜索工具，在主要工具失败时使用"""
        from crewai.tools import BaseTool
        from typing import Type, Dict, Any
        from pydantic import BaseModel, Field
        
        # 创建输入模型
        class FallbackWebInput(BaseModel):
            query: str = Field(..., description="The search query")
        
        # 定义后备工具
        class FallbackWebTool(BaseTool):
            name: str = "Fallback Web Search"
            description: str = "A fallback tool for web searches when the primary tool is unavailable"
            args_schema: Type[BaseModel] = FallbackWebInput
            
            def _run(self, query: str) -> str:
                return f"Web search tool is currently unavailable. Your query was: {query}"
        
        return FallbackWebTool()

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
                index_dir=os.getenv('INDEX_PERSIST_DIR', './index_storage')
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

