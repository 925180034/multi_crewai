# src/multi/crews/retrieval_crew/retrieval_crew.py
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class RetrievalCrew:
    """Data retrieval crew for accessing and processing data sources"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def retrieval_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["retrieval_agent"],
        )

    @task
    def retrieve_data(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_data"],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )