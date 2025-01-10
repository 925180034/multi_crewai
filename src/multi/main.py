#!/usr/bin/env python
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start

# Fix the imports to use relative paths
from .crews.planner_crew.planner_crew import PlannerCrew
from .crews.retrieval_crew.retrieval_crew import RetrievalCrew
from .crews.matcher_crew.matcher_crew import MatcherCrew
from .crews.sql_crew.sql_crew import SQLCrew

class QueryState(BaseModel):
    query: str = ""
    plan: str = ""
    retrieved_data: str = ""
    schema_matches: str = ""
    sql_query: str = ""

class QueryFlow(Flow[QueryState]):
    @start()
    def process_query(self):
        print("Starting query processing")
        # Here we would get the query from somewhere
        self.state.query = "Sample query"

    @listen(process_query)
    def create_plan(self):
        print("Creating execution plan")
        result = (
            PlannerCrew()
            .crew()
            .kickoff(inputs={"query": self.state.query})
        )
        print("Plan created:", result.raw)
        self.state.plan = result.raw

    @listen(create_plan)
    def retrieve_data(self):
        print("Retrieving relevant data")
        result = (
            RetrievalCrew()
            .crew()
            .kickoff(inputs={"query": self.state.query, "plan": self.state.plan})
        )
        print("Data retrieved:", result.raw)
        self.state.retrieved_data = result.raw

    @listen(retrieve_data)
    def match_schemas(self):
        print("Matching schemas")
        result = (
            MatcherCrew()
            .crew()
            .kickoff(inputs={
                "query": self.state.query,
                "plan": self.state.plan,
                "retrieved_data": self.state.retrieved_data
            })
        )
        print("Schema matches found:", result.raw)
        self.state.schema_matches = result.raw

    @listen(match_schemas)
    def generate_sql(self):
        print("Generating SQL query")
        result = (
            SQLCrew()
            .crew()
            .kickoff(inputs={
                "query": self.state.query,
                "schema_matches": self.state.schema_matches,
                "retrieved_data": self.state.retrieved_data
            })
        )
        print("SQL query generated:", result.raw)
        self.state.sql_query = result.raw

def kickoff():
    query_flow = QueryFlow()
    query_flow.kickoff()

def plot():
    query_flow = QueryFlow()
    query_flow.plot()

if __name__ == "__main__":
    kickoff()