# src/multi/crews/matcher_crew/matcher_crew.py - 优化版
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task
import logging

logger = logging.getLogger(__name__)

@CrewBase
class MatcherCrew:
    """Schema匹配小组，识别数据结构关系"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def schema_matcher(self) -> Agent:
        return Agent(
            config=self.agents_config["schema_matcher"],
            max_iter=2,  # 降低迭代次数提高效率
            verbose=True
        )

    @task
    def match_schemas(self) -> Task:
        return Task(
            config=self.tasks_config["match_schemas"]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )