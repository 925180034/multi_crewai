# src/multi/crews/matcher_crew/matcher_crew.py
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class MatcherCrew:
    """Schema matching crew for identifying data structure relationships"""
    
    # Paths to config files
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def schema_matcher_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["schema_matcher_agent"],
        )

    @task
    def match_schemas(self) -> Task:
        return Task(
            config=self.tasks_config["match_schemas"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Schema Matching Crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,    # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )