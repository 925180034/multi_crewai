# src/multi/crews/planner_crew/planner_crew.py - 优化版
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
import logging

logger = logging.getLogger(__name__)

class PlanOutput(BaseModel):
    timestamp: str
    query: str
    plan_details: str
    execution_steps: List[str]

@CrewBase
class PlannerCrew:
    """规划小组，分析查询并创建执行计划"""

    @agent
    def planner_agent(self) -> Agent:
        return Agent(
            role="Query Planner",
            goal="创建简洁有效的查询执行计划",
            backstory="你是一个专业的数据查询规划专家，能够分解复杂查询为可执行步骤",
            verbose=True,
            max_iter=2  # 限制迭代次数提高效率
        )

    @task
    def planning_task(self) -> Task:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return Task(
            description="分析用户查询并创建执行计划。查询：{query}",
            expected_output="详细的查询执行计划，包含必要的数据源和步骤",
            agent=self.planner_agent(),
            output_json=PlanOutput  # 使用结构化输出
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.planner_agent()],
            tasks=[self.planning_task()],
            process=Process.sequential,
            verbose=True
        )