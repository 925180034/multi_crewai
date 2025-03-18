from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
import os
import logging
from typing import Optional, List
from multi.tools.retrieval_tools import RetrievalTools

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
        # 验证环境变量
        self._validate_environment()
        
        # 初始化工具集
        self.tools = self._initialize_tools()
        
        # 获取具体工具实例
        self.doc_tool = self.tools.get_document_tool() if self.tools else None
        self.db_tool = self.tools.get_database_tool() if self.tools else None
        self.web_tool = self.tools.get_web_tool() if self.tools else None

    def _validate_environment(self) -> None:
        """验证必要的环境变量"""
        required_vars = [
            'OPENAI_API_KEY',
            'MYSQL_DATABASE_URL',
            'SERPER_API_KEY'
        ]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

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
        return Agent(
            config=self.agents_config["database_agent"],
            tools=[self.db_tool] if self.db_tool else [],
            max_iter=1,  # 限制最大迭代次数
            max_rpm=10,  # 限制API请求频率
            max_execution_time=300,  # 限制执行时间为5分钟
            verbose=True,
            allow_delegation=False  # 禁止委派
        )

    @agent
    def web_agent(self) -> Agent:
        """网络检索代理"""
        return Agent(
            config=self.agents_config["web_agent"],
            tools=[self.web_tool] if self.web_tool else [],
            max_iter=1,  # 限制最大迭代次数
            max_rpm=10, 
            verbose=True,
            allow_delegation=False
        )

    @agent
    def document_agent(self) -> Agent:
        """文档检索代理"""
        return Agent(
            config=self.agents_config["document_agent"],
            tools=[self.doc_tool] if self.doc_tool else [],
            max_iter=1,  # 限制最大迭代次数
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

# from crewai import Agent, Crew, Task, Process
# from crewai.project import CrewBase, agent, crew, task
# import os
# import logging
# from typing import Optional, List
# from multi.tools.retrieval_tools import RetrievalTools

# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# @CrewBase
# class RetrievalCrew:
#     """数据检索小组，负责从不同来源检索和处理数据"""
    
#     agents_config = "config/agents.yaml"
#     tasks_config = "config/tasks.yaml"

#     def __init__(self):
#         """初始化检索小组"""
#         # 验证环境变量
#         self._validate_environment()
        
#         # 初始化工具集
#         self.tools = self._initialize_tools()
        
#         # 获取具体工具实例
#         self.doc_tool = self.tools.get_document_tool() if self.tools else None
#         self.db_tool = self.tools.get_database_tool() if self.tools else None
#         self.web_tool = self.tools.get_web_tool() if self.tools else None

#     def _validate_environment(self) -> None:
#         """验证必要的环境变量"""
#         required_vars = [
#             'OPENAI_API_KEY',
#             'PG_DATABASE_URL',
#             'SERPER_API_KEY'
#         ]
#         missing = [var for var in required_vars if not os.getenv(var)]
#         if missing:
#             raise ValueError(f"Missing required environment variables: {missing}")

#     def _initialize_tools(self) -> Optional[RetrievalTools]:
#         """初始化工具集"""
#         try:
#             return RetrievalTools(
#                 db_url=os.getenv('PG_DATABASE_URL'),
#                 docs_dir=os.getenv('DOCS_DIR', './documents'),
#                 index_dir=os.getenv('INDEX_PERSIST_DIR', './index_storage')
#             )
#         except Exception as e:
#             logger.error(f"Failed to initialize tools: {str(e)}")
#             return None

#     @agent
#     def database_agent(self) -> Agent:
#         """数据库检索代理"""
#         return Agent(
#             config=self.agents_config["database_agent"],
#             tools=[self.db_tool] if self.db_tool else [],
#             max_iter=1,  # 限制最大迭代次数
#             max_rpm=10,  # 限制API请求频率
#             max_execution_time=300,  # 限制执行时间为5分钟
#             verbose=True,
#             allow_delegation=False  # 禁止委派
#         )

#     @agent
#     def web_agent(self) -> Agent:
#         """网络检索代理"""
#         return Agent(
#             config=self.agents_config["web_agent"],
#             tools=[self.web_tool] if self.web_tool else [],
#             max_iter=1,  # 限制最大迭代次数
#             max_rpm=10, 
#             verbose=True,
#             allow_delegation=False
#         )

#     @agent
#     def document_agent(self) -> Agent:
#         """文档检索代理"""
#         return Agent(
#             config=self.agents_config["document_agent"],
#             tools=[self.doc_tool] if self.doc_tool else [],
#             max_iter=1,  # 限制最大迭代次数
#             max_rpm=10,
#             verbose=True,
#             allow_delegation=False
#         )

#     @task
#     def search_database(self) -> Task:
#         """数据库搜索任务"""
#         return Task(
#             description="""
#             根据提供的查询和计划搜索数据库信息。
#             Query: {query}
#             Plan: {plan}
#             """,
#             expected_output="检索到的数据库信息",
#             agent=self.database_agent(),
#             context=[],  # 可以添加依赖的上下文
#             output_file="outputs/database_search_result.md"
#         )

#     @task
#     def search_web(self) -> Task:
#         """网络搜索任务"""
#         return Task(
#             description="""
#             根据提供的查询和计划在网络上搜索相关信息。
#             Query: {query}
#             Plan: {plan}
#             """,
#             expected_output="相关的网络搜索结果",
#             agent=self.web_agent(),
#             context=[],
#             output_file="outputs/web_search_result.md"
#         )

#     @task
#     def search_documents(self) -> Task:
#         """文档搜索任务"""
#         return Task(
#             description="""
#             在文档中搜索相关信息。
#             Query: {query}
#             Plan: {plan}
#             """,
#             expected_output="从文档中提取的相关信息",
#             agent=self.document_agent(),
#             context=[],
#             output_file="outputs/document_search_result.md"
#         )

#     def create_manager_agent(self) -> Agent:
#         """创建管理者代理"""
#         return Agent(
#             config=self.agents_config["manager_agent"],
#             max_iter=3,  # 管理者可以有更多迭代次数
#             verbose=True
#         )

#     @crew
#     def crew(self) -> Crew:
#         """创建并配置检索小组"""
#         try:
#             return Crew(
#                 agents=self.agents,
#                 tasks=self.tasks,
#                 process=Process.sequential,  # 使用顺序处理
#                 manager_agent=self.create_manager_agent(),
#                 verbose=True,
#                 max_rpm=30,  # 整个小组的最大请求频率
#                 full_output=True  # 启用完整输出捕获
#             )
#         except Exception as e:
#             logger.error(f"Failed to create crew: {str(e)}")
#             raise

#     def cleanup(self):
#         """清理资源"""
#         if self.tools:
#             self.tools.cleanup()

#     def __del__(self):
#         """析构函数，确保资源被清理"""
#         self.cleanup()