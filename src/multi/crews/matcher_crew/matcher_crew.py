from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class MatcherCrew:
    """Schema matching crew for identifying data structure relationships"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def schema_matcher(self) -> Agent:  # 方法名要和配置中的agent名称一致
        return Agent(
            config=self.agents_config["schema_matcher"]
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