# src/multi/crews/sql_crew/sql_crew.py - 优化版
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
import logging
import os
import json

logger = logging.getLogger(__name__)

@CrewBase
class SQLCrew:
    """SQL生成小组，将自然语言查询转换为SQL语句"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, spider_dataset_path=None):
        """使用Spider数据集路径初始化SQL小组"""
        super().__init__()
        self.spider_dataset_path = spider_dataset_path
        self.db_schemas = []
        
        # 如果提供了Spider数据集路径，加载数据库模式
        if self.spider_dataset_path and os.path.exists(self.spider_dataset_path):
            self.load_spider_schemas()
    
    def load_spider_schemas(self):
        """加载Spider数据库模式"""
        try:
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
            max_iter=2,  # 降低迭代次数提高效率
            verbose=True
        )

    @task
    def generate_sql(self) -> Task:
        return Task(
            config=self.tasks_config["generate_sql"]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )