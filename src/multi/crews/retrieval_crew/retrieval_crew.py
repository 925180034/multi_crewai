# src/multi/crews/retrieval_crew/retrieval_crew.py - 修复版
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
import os
import logging
from typing import Optional, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@CrewBase
class RetrievalCrew:
    """数据检索小组，负责从不同来源检索和处理数据"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, dataset_path=None):
        """初始化检索小组"""
        # 先调用父类初始化方法
        super().__init__()
        
        # 存储数据集路径
        self.dataset_path = dataset_path
        logger.info(f'RetrievalCrew初始化，数据集路径: {self.dataset_path}')
        
        # 初始化工具
        self.init_tools()
        
    def init_tools(self):
        """初始化所有工具"""
        try:
            # 初始化文档工具
            if self.dataset_path and os.path.exists(self.dataset_path):
                from multi.tools.document_retrieval_tool import DocumentRetrievalTool
                self.doc_tool = DocumentRetrievalTool(document_path=self.dataset_path)
            else:
                self.doc_tool = self.create_simple_doc_tool()
                
            # 初始化网络工具 (使用简化的模拟工具)
            self.web_tool = self.create_simple_web_tool()
            
            # 初始化数据库工具 (使用简化的模拟工具)
            self.db_tool = self.create_simple_db_tool()
            
            logger.info("所有工具初始化完成")
                
        except Exception as e:
            logger.error(f"工具初始化错误: {str(e)}")
            self.doc_tool = self.create_simple_doc_tool()
            self.web_tool = self.create_simple_web_tool()
            self.db_tool = self.create_simple_db_tool()
    
    def create_simple_web_tool(self):
        """创建简单的网络查询工具"""
        class SimpleWebInput(BaseModel):
            query: str = Field(..., description="搜索查询")
        
        class SimpleWebTool(BaseTool):
            name: str = "Web Search"
            description: str = "在网络上搜索信息"
            args_schema: type = SimpleWebInput
            
            def _run(self, query: str) -> str:
                return f"关于'{query}'的网络搜索结果：\n" \
                       f"- 网络源1：{query}相关的结构化数据信息\n" \
                       f"- 网络源2：{query}的最佳实践建议\n" \
                       f"- 网络源3：{query}的技术文档参考"
                
        return SimpleWebTool()
        
    def create_simple_db_tool(self):
        """创建简单的数据库查询工具"""
        class SimpleDBInput(BaseModel):
            query: str = Field(..., description="数据库查询")
        
        class SimpleDBTool(BaseTool):
            name: str = "Database Search"
            description: str = "在数据库中搜索信息"
            args_schema: type = SimpleDBInput
            
            def _run(self, query: str) -> str:
                return f"关于'{query}'的数据库搜索结果：\n" \
                       f"- 数据库表：singers, albums, songs\n" \
                       f"- 主要字段：id, name, country, age\n" \
                       f"- 数据统计：总共有约25位歌手记录"
                
        return SimpleDBTool()
    
    def create_simple_doc_tool(self):
        """创建简单的文档查询工具"""
        class SimpleDocInput(BaseModel):
            query: str = Field(..., description="文档查询")
        
        class SimpleDocTool(BaseTool):
            name: str = "Document Search"
            description: str = "在文档中搜索信息"
            args_schema: type = SimpleDocInput
            
            def _run(self, query: str) -> str:
                return f"关于'{query}'的文档搜索结果：\n" \
                       f"- 文档段落1：{query}的概念定义\n" \
                       f"- 文档段落2：{query}的用例示例\n" \
                       f"- 文档段落3：{query}的最佳实践"
                
        return SimpleDocTool()

    @agent
    def database_agent(self) -> Agent:
        """数据库检索代理"""
        tools = [self.db_tool] if hasattr(self, 'db_tool') else []
            
        return Agent(
            config=self.agents_config["database_agent"],
            tools=tools,
            max_iter=1, # 降低迭代次数提高效率
            verbose=True
        )

    @agent
    def web_agent(self) -> Agent:
        """网络检索代理"""
        tools = [self.web_tool] if hasattr(self, 'web_tool') else []
            
        return Agent(
            config=self.agents_config["web_agent"],
            tools=tools,
            max_iter=1, # 降低迭代次数提高效率
            verbose=True
        )

    @agent
    def document_agent(self) -> Agent:
        """文档检索代理"""
        tools = [self.doc_tool] if hasattr(self, 'doc_tool') else []
            
        return Agent(
            config=self.agents_config["document_agent"],
            tools=tools,
            max_iter=1, # 降低迭代次数提高效率
            verbose=True
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
            agent=self.database_agent()  # 直接引用代理方法
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
            agent=self.web_agent()  # 直接引用代理方法
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
            agent=self.document_agent()  # 直接引用代理方法
        )

    @crew
    def crew(self) -> Crew:
        """创建检索小组"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
        
    def cleanup(self):
        """清理资源"""
        # 简化的清理方法
        logger.info("资源清理完成")