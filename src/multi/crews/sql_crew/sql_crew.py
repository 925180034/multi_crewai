# src/multi/crews/sql_crew/sql_crew.py
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
import logging


logger = logging.getLogger(__name__)

@CrewBase
class SQLCrew:
    """SQL生成小组，将自然语言查询转换为SQL语句"""
    
    # 配置文件路径
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    # 初始化日志记录器
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    def __init__(self, spider_dataset_path=None):
        """使用Spider数据集路径初始化SQL小组"""
        super().__init__()
        self.spider_dataset_path = spider_dataset_path
        logger.info(f'SQLCrew初始化，Spider数据集路径: {self.spider_dataset_path}')
        
        # 如果提供了Spider数据集路径，加载数据库模式
        self.db_schemas = []
        if self.spider_dataset_path:
            self.load_spider_schemas()
    
    def load_spider_schemas(self):
        """加载Spider数据库模式"""
        try:
            import json
            import os
            
            # 从Spider数据集加载数据库模式
            schema_path = os.path.join(self.spider_dataset_path, 'tables.json')
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    self.db_schemas = json.load(f)
                logger.info(f'已加载 {len(self.db_schemas)} 个数据库模式来自Spider数据集')
            else:
                logger.warning(f'Spider数据库模式文件未在 {schema_path} 找到')
                self.db_schemas = []
        except Exception as e:
            logger.error(f'加载Spider数据库模式时出错: {str(e)}')
            self.db_schemas = []

    @agent
    def sql_generator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["sql_generator_agent"],
        )

    @task
    def generate_sql(self) -> Task:
        return Task(
            config=self.tasks_config["generate_sql"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the SQL Generation Crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )