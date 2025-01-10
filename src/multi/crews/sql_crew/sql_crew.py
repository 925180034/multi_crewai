# src/multi/crews/sql_crew/sql_crew.py
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class SQLCrew:
    """SQL generation crew for converting queries into SQL statements"""
    
    # Paths to config files
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

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