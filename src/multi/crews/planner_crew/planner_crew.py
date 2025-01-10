# src/query_system/crews/planner_crew/planner_crew.py
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task

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
        return Task(
            description="Analyze the user query and create a plan",
            expected_output="Query execution plan with required steps",
            agent=self.planner_agent()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.planner_agent()],
            tasks=[self.planning_task()],
            process=Process.sequential,
            verbose=True
        )