from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task

class PlanOutput(BaseModel):
    timestamp: str
    query: str
    plan_details: str
    execution_steps: List[str]

@CrewBase
class PlannerCrew:
    """Planning crew for query analysis"""

    @agent
    def planner_agent(self) -> Agent:
        return Agent(
            role="Query Planner",
            goal="Analyze user queries and create execution plans",
            backstory="Expert at understanding user requirements and planning query execution",
            verbose=True
        )

    @task
    def planning_task(self) -> Task:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return Task(
            description="Analyze the user query and create a plan",
            expected_output="Query execution plan with required steps",
            agent=self.planner_agent(),
            output_json=PlanOutput,  # Use Pydantic model for structured output
            output_file=f"outputs/planner_task_{timestamp}.md"
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.planner_agent()],
            tasks=[self.planning_task()],
            process=Process.sequential,
            verbose=True,
            full_output=True  # Enable full output capturing
        )