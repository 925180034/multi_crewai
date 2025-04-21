#!/usr/bin/env python
# Improved src/multi/main.py - Enhanced for Spider dataset evaluation
import json
import os
import time
import re
import logging
import traceback
import sqlite3
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple

from crewai.flow.flow import Flow, listen, start

from multi.crews.planner_crew.planner_crew import PlannerCrew
from multi.crews.retrieval_crew.retrieval_crew import RetrievalCrew
from multi.crews.matcher_crew.matcher_crew import MatcherCrew  
from multi.crews.sql_crew.sql_crew import SQLCrew

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryState(BaseModel):
    """Enhanced query state model"""
    query: str = ""
    plan: str = ""
    db_data: str = ""
    web_data: str = ""
    doc_data: str = ""
    schema_matches: str = ""
    sql_query: str = ""
    error: Optional[str] = None
    status: str = "initialized"
    execution_start: Optional[datetime] = None
    execution_end: Optional[datetime] = None
    step_metrics: Dict[str, Any] = Field(default_factory=dict)
    # Spider-specific fields
    db_schema: Optional[Dict[str, Any]] = None
    db_id: str = ""
    gold_sql: str = ""  # Reference SQL from Spider
    execution_results: Optional[Dict[str, Any]] = None

class QueryFlow(Flow[QueryState]):
    """Enhanced query processing flow"""
    
    def __init__(self, dataset_path=None, dataset_type="default"):
        """Initialize query flow with better cleanup"""
        super().__init__()
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Store dataset info
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        
        # Initialize crews with proper tracking for cleanup
        self.planner_crew = None
        self.retrieval_crew = None
        self.matcher_crew = None
        self.sql_crew = None
        
        # Immediately initialize all crews
        self._initialize_crews()
        
        # Add timestamp for tracking execution time
        self.state.execution_start = datetime.now()

    def _initialize_crews(self):
        """Initialize all crews with better error handling"""
        try:
            self.planner_crew = PlannerCrew()
            logger.info("Planner crew initialized")
            
            self.retrieval_crew = RetrievalCrew(dataset_path=self.dataset_path)
            logger.info("Retrieval crew initialized")
            
            self.matcher_crew = MatcherCrew()
            logger.info("Matcher crew initialized")
            
            self.sql_crew = SQLCrew(spider_dataset_path=self.dataset_path)
            logger.info("SQL crew initialized")
            
            logger.info("All crews initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize crews: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def cleanup_crews(self):
        """Enhanced cleanup of all crews' resources"""
        try:
            # Add a specific cleanup order
            if hasattr(self.retrieval_crew, 'cleanup') and self.retrieval_crew is not None:
                try:
                    self.retrieval_crew.cleanup()
                    logger.info("Retrieval crew cleaned up")
                except Exception as e:
                    logger.warning(f"Error cleaning up retrieval crew: {str(e)}")
            
            # Add explicit cleanup for other crews if they implement cleanup methods
            for crew_name, crew in [
                ("planner_crew", self.planner_crew),
                ("matcher_crew", self.matcher_crew),
                ("sql_crew", self.sql_crew)
            ]:
                if crew is not None and hasattr(crew, 'cleanup'):
                    try:
                        crew.cleanup()
                        logger.info(f"{crew_name} cleaned up")
                    except Exception as e:
                        logger.warning(f"Error cleaning up {crew_name}: {str(e)}")
                        
            # Set all crews to None to help garbage collection
            self.planner_crew = None
            self.retrieval_crew = None
            self.matcher_crew = None
            self.sql_crew = None
            
            logger.info("All crews cleaned up")
        except Exception as e:
            logger.error(f"Error in cleanup_crews: {str(e)}")

    @start()
    def process_query(self):
        """Process initial query with better error handling"""
        try:
            # Record start time
            self.state.execution_start = datetime.now()
            self.state.status = "processing"
            
            # Load data
            self._load_query_data()
            logger.info(f"Processing query: {self.state.query}")
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.state.error = error_msg
            self.state.status = "error"
            raise

    def _load_query_data(self):
        """Load query data with improved dataset handling"""
        # From Spider dataset
        if self.dataset_path and self.dataset_type == "spider":
            self._load_from_spider()
        # From custom dataset
        elif self.dataset_path and self.dataset_type == "custom":
            self._load_from_custom()
        # Use default query
        else:
            self.state.query = "Find all customers' names and their total spending, ordered by amount in descending order."

    def _load_from_spider(self):
        """Load from Spider dataset with better error handling"""
        try:
            # Check if dataset path exists
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Spider dataset path not found: {self.dataset_path}")
            
            # Load dev.json
            dev_file = os.path.join(self.dataset_path, 'dev.json')
            if not os.path.exists(dev_file):
                raise FileNotFoundError(f"Spider dev file not found: {dev_file}")
                
            with open(dev_file, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            
            if not queries or len(queries) < 1:
                raise ValueError("No queries found in Spider dev file")
                
            # Use first query
            self.state.query = queries[0]['question']
            self.state.db_id = queries[0]['db_id']
            self.state.gold_sql = queries[0].get('query', '')
            
            # Load database schema
            tables_file = os.path.join(self.dataset_path, 'tables.json')
            if not os.path.exists(tables_file):
                raise FileNotFoundError(f"Spider tables file not found: {tables_file}")
                
            with open(tables_file, 'r', encoding='utf-8') as f:
                schemas = json.load(f)
            
            # Find matching schema
            found_schema = False
            for schema in schemas:
                if schema['db_id'] == self.state.db_id:
                    self.state.db_schema = schema
                    found_schema = True
                    break
            
            if not found_schema:
                logger.warning(f"Schema not found for db_id: {self.state.db_id}")
                
            logger.info(f"Successfully loaded query from Spider dataset, db_id: {self.state.db_id}")
        except Exception as e:
            error_msg = f"Failed to load from Spider: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.state.error = error_msg
            raise

    def _load_from_custom(self):
        """Load from custom dataset with better error handling"""
        try:
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Custom dataset path not found: {self.dataset_path}")
                
            data_file = os.path.join(self.dataset_path, 'data.json')
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Custom data file not found: {data_file}")
                
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'query' not in data:
                raise ValueError("No query found in custom dataset")
                
            self.state.query = data['query']
            self.state.db_schema = data.get('schema', {})
            logger.info(f"Successfully loaded query from custom dataset")
        except Exception as e:
            error_msg = f"Failed to load from custom dataset: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.state.error = error_msg
            raise

    @listen(process_query)
    def create_plan(self):
        """Create execution plan with better error handling"""
        if self.state.error:
            logger.warning(f"Skipping create_plan due to error: {self.state.error}")
            return
            
        try:
            self.state.step_metrics['create_plan_start'] = datetime.now()
            result = self.planner_crew.crew().kickoff(
                inputs={"query": self.state.query}
            )
            self.state.plan = result.raw if hasattr(result, 'raw') else str(result)
            self.state.step_metrics['create_plan_end'] = datetime.now()
            logger.info("Plan created successfully")
        except Exception as e:
            error_msg = f"Failed to create plan: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.state.error = error_msg

    @listen(create_plan)
    def retrieve_data(self):
        """Retrieve data with better error handling"""
        if self.state.error:
            logger.warning(f"Skipping retrieve_data due to error: {self.state.error}")
            return
            
        try:
            self.state.step_metrics['retrieve_data_start'] = datetime.now()
            result = self.retrieval_crew.crew().kickoff(
                inputs={
                    "query": self.state.query,
                    "plan": self.state.plan
                }
            )
            
            # Process results
            if hasattr(result, 'tasks_output'):
                self.state.db_data = result.tasks_output[0].raw if len(result.tasks_output) > 0 else ""
                self.state.web_data = result.tasks_output[1].raw if len(result.tasks_output) > 1 else ""
                self.state.doc_data = result.tasks_output[2].raw if len(result.tasks_output) > 2 else ""
            
            self.state.step_metrics['retrieve_data_end'] = datetime.now()
            logger.info("Data retrieved successfully")
        except Exception as e:
            error_msg = f"Failed to retrieve data: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.state.error = error_msg

    @listen(retrieve_data)
    def match_schemas(self):
        """Match schemas with better error handling"""
        if self.state.error:
            logger.warning(f"Skipping match_schemas due to error: {self.state.error}")
            return
            
        try:
            self.state.step_metrics['match_schemas_start'] = datetime.now()
            result = self.matcher_crew.crew().kickoff(
                inputs={
                    "query": self.state.query,
                    "db_data": self.state.db_data,
                    "web_data": self.state.web_data,
                    "doc_data": self.state.doc_data
                }
            )
            self.state.schema_matches = result.raw if hasattr(result, 'raw') else str(result)
            self.state.step_metrics['match_schemas_end'] = datetime.now()
            logger.info("Schemas matched successfully")
        except Exception as e:
            error_msg = f"Failed to match schemas: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.state.error = error_msg

    @listen(match_schemas)
    def generate_sql(self):
        """Generate SQL with better error handling and database schema integration"""
        if self.state.error:
            logger.warning(f"Skipping SQL generation due to error: {self.state.error}")
            return
            
        try:
            self.state.step_metrics['generate_sql_start'] = datetime.now()
            
            # Prepare database schema information
            db_schema_info = ""
            db_id = ""
            
            # Use db_schema or load from SQL crew
            if hasattr(self.state, 'db_schema') and self.state.db_schema:
                db_schema_info = json.dumps(self.state.db_schema, ensure_ascii=False, indent=2)
                if hasattr(self.state, 'db_id'):
                    db_id = self.state.db_id
            elif self.sql_crew and hasattr(self.sql_crew, 'db_schemas') and self.sql_crew.db_schemas:
                # Find the correct schema for this db_id if available
                if hasattr(self.state, 'db_id') and self.state.db_id:
                    for schema in self.sql_crew.db_schemas:
                        if schema.get('db_id') == self.state.db_id:
                            db_schema_info = json.dumps(schema, ensure_ascii=False, indent=2)
                            db_id = self.state.db_id
                            break
                
                # Fallback to first schema if no matching schema found
                if not db_schema_info and self.sql_crew.db_schemas:
                    db_schema_info = json.dumps(self.sql_crew.db_schemas[0], ensure_ascii=False, indent=2)
                    db_id = self.sql_crew.db_schemas[0].get('db_id', '')
            
            result = self.sql_crew.crew().kickoff(
                inputs={
                    "query": self.state.query,
                    "db_schema": db_schema_info,
                    "db_id": db_id,
                    "schema_matches": self.state.schema_matches
                }
            )
            
            self.state.sql_query = result.raw if hasattr(result, 'raw') else str(result)
            
            # Clean up the SQL (remove explanations or headers)
            self.state.sql_query = self._extract_sql_query(self.state.sql_query)
            
            self.state.step_metrics['generate_sql_end'] = datetime.now()
            self.state.execution_end = datetime.now()
            self.state.status = "completed"
            
            logger.info("SQL generated successfully")
        except Exception as e:
            error_msg = f"Failed to generate SQL: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.state.error = error_msg
            self.state.status = "error"
            self.state.execution_end = datetime.now()

    def _extract_sql_query(self, raw_output):
        """Extract clean SQL query from raw output text"""
        # Search for SQL enclosed in triple backticks
        sql_pattern = r"```sql\s*(.*?)\s*```"
        sql_match = re.search(sql_pattern, raw_output, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Search for any code blocks
        code_pattern = r"```(.*?)```"
        code_match = re.search(code_pattern, raw_output, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Try to find SQL keywords and extract the SQL statement
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
        lines = raw_output.split('\n')
        for i, line in enumerate(lines):
            for keyword in sql_keywords:
                if line.strip().upper().startswith(keyword):
                    # Extract from this line until a blank line or end
                    sql_lines = []
                    j = i
                    while j < len(lines) and lines[j].strip():
                        sql_lines.append(lines[j])
                        j += 1
                    return ' '.join([l.strip() for l in sql_lines])
        
        # If no clear SQL is found, return the raw output
        return raw_output.strip()

def execute_sql(sql_query, db_path):
    """Execute SQL query and get results"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description] if cursor.description else []
        conn.close()
        
        return {
            "success": True,
            "results": results,
            "columns": column_names
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def evaluate_sql_execution(generated_sql, gold_sql, db_path):
    """Evaluate SQL by executing both queries and comparing results"""
    if not os.path.exists(db_path):
        return (0.0, f"Database file not found: {db_path}")
    
    try:
        # Execute generated SQL
        generated_results = execute_sql(generated_sql, db_path)
        
        # Execute gold SQL
        gold_results = execute_sql(gold_sql, db_path)
        
        # Check for execution errors
        if not generated_results.get("success", False):
            return (0.0, f"Generated SQL execution error: {generated_results.get('error', 'Unknown error')}")
            
        if not gold_results.get("success", False):
            return (0.0, f"Gold SQL execution error: {gold_results.get('error', 'Unknown error')}")
        
        # Compare results
        gen_results = generated_results.get("results", [])
        gold_results = gold_results.get("results", [])
        
        # Sort results for consistent comparison
        gen_results = sorted([tuple(row) for row in gen_results])
        gold_results = sorted([tuple(row) for row in gold_results])
        
        # Check if results match
        if gen_results == gold_results:
            return (1.0, "Execution results match")
        
        # Partial match - check if returned columns match
        gen_cols = set(generated_results.get("columns", []))
        gold_cols = set(gold_results.get("columns", []))
        
        if gen_cols == gold_cols:
            return (0.5, "Column names match but data differs")
            
        # Calculate overlap of results
        if len(gen_results) > 0 and len(gold_results) > 0:
            common_rows = set(gen_results).intersection(set(gold_results))
            if common_rows:
                overlap = len(common_rows) / max(len(gen_results), len(gold_results))
                return (overlap, f"Partial match: {len(common_rows)} rows match out of {max(len(gen_results), len(gold_results))}")
        
        return (0.0, "Results do not match")
    except Exception as e:
        return (0.0, f"Evaluation error: {str(e)}")

def evaluate_sql_structure(generated_sql, gold_sql):
    """Improved evaluation of SQL structure with normalization and component matching"""
    if not generated_sql or not gold_sql:
        return (0.0, "Empty SQL query")
    
    def normalize_sql(sql):
        """Normalize SQL for better comparison"""
        # Convert to lowercase
        sql = sql.lower()
        
        # Remove extra whitespace
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # Remove quotes around identifiers
        sql = re.sub(r'["`\']([^"`\']+)["`\']', r'\1', sql)
        
        # Normalize JOIN syntax
        sql = re.sub(r'join\s+', ' join ', sql)
        
        return sql
    
    def extract_components(sql):
        """Extract key components from SQL query"""
        # Basic extraction of SQL components
        components = {
            "select": [],
            "from": [],
            "where": [],
            "group_by": [],
            "order_by": [],
            "limit": None
        }
        
        # Extract SELECT clause
        select_match = re.search(r'select\s+(.*?)\s+from', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_items = select_match.group(1).split(',')
            components["select"] = [item.strip() for item in select_items]
        
        # Extract FROM clause
        from_match = re.search(r'from\s+(.*?)(?:\s+where|\s+group\s+by|\s+order\s+by|\s+limit|$)', sql, re.IGNORECASE | re.DOTALL)
        if from_match:
            from_items = from_match.group(1).split(',')
            components["from"] = [item.strip() for item in from_items]
        
        # Extract WHERE clause
        where_match = re.search(r'where\s+(.*?)(?:\s+group\s+by|\s+order\s+by|\s+limit|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_conditions = where_match.group(1).split('and')
            components["where"] = [cond.strip() for cond in where_conditions]
        
        # Extract GROUP BY clause
        group_by_match = re.search(r'group\s+by\s+(.*?)(?:\s+having|\s+order\s+by|\s+limit|$)', sql, re.IGNORECASE | re.DOTALL)
        if group_by_match:
            group_by_items = group_by_match.group(1).split(',')
            components["group_by"] = [item.strip() for item in group_by_items]
        
        # Extract ORDER BY clause
        order_by_match = re.search(r'order\s+by\s+(.*?)(?:\s+limit|$)', sql, re.IGNORECASE | re.DOTALL)
        if order_by_match:
            order_by_items = order_by_match.group(1).split(',')
            components["order_by"] = [item.strip() for item in order_by_items]
        
        # Extract LIMIT clause
        limit_match = re.search(r'limit\s+(\d+)', sql, re.IGNORECASE)
        if limit_match:
            components["limit"] = limit_match.group(1)
        
        return components
    
    # Normalize SQL queries
    norm_generated = normalize_sql(generated_sql)
    norm_gold = normalize_sql(gold_sql)
    
    # Exact match after normalization
    if norm_generated == norm_gold:
        return (1.0, "Exact match after normalization")
    
    # Extract and compare components
    gen_components = extract_components(norm_generated)
    gold_components = extract_components(norm_gold)
    
    # Calculate component matches
    component_scores = []
    
    for component, gold_items in gold_components.items():
        gen_items = gen_components[component]
        
        # Skip empty components
        if not gold_items:
            continue
            
        if component == "limit":
            # Direct comparison for limit
            component_scores.append(1.0 if gen_items == gold_items else 0.0)
        else:
            # Set comparison for lists, accounting for ordering differences
            if not gen_items:
                component_scores.append(0.0)
            else:
                gold_set = set(str(item) for item in gold_items)
                gen_set = set(str(item) for item in gen_items)
                
                if gold_set and gen_set:
                    # Calculate Jaccard similarity
                    overlap = len(gold_set.intersection(gen_set))
                    union = len(gold_set.union(gen_set))
                    score = overlap / union if union > 0 else 0.0
                    component_scores.append(score)
                else:
                    component_scores.append(0.0)
    
    # Calculate overall score
    if component_scores:
        overall_score = sum(component_scores) / len(component_scores)
        return (overall_score, f"Component similarity: {overall_score:.2f}")
    else:
        return (0.0, "No matching components found")

def evaluate_sql(generated_sql, gold_sql, db_path=None):
    """
    Comprehensive SQL evaluation combining structure and execution methods
    
    Args:
        generated_sql: Generated SQL query
        gold_sql: Reference SQL query
        db_path: Path to the database file (optional)
        
    Returns:
        dict: Evaluation results
    """
    results = {
        "structure_score": 0.0,
        "execution_score": 0.0,
        "combined_score": 0.0,
        "structure_details": "",
        "execution_details": "",
        "status": "success"
    }
    
    try:
        # Evaluate SQL structure
        structure_score, structure_details = evaluate_sql_structure(generated_sql, gold_sql)
        results["structure_score"] = structure_score
        results["structure_details"] = structure_details
        
        # Evaluate SQL execution if database is available
        if db_path and os.path.exists(db_path):
            execution_score, execution_details = evaluate_sql_execution(generated_sql, gold_sql, db_path)
            results["execution_score"] = execution_score
            results["execution_details"] = execution_details
            
            # Combined score (weighted more towards execution)
            results["combined_score"] = (structure_score * 0.4) + (execution_score * 0.6)
        else:
            # If no database, use structure score only
            results["execution_details"] = "No database available for execution testing"
            results["combined_score"] = structure_score
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
    
    return results

def batch_test_spider(spider_dataset_path, limit=10, start_index=0):
    """
    Enhanced batch testing on Spider dataset with better progress reporting and resource management
    
    Args:
        spider_dataset_path: Path to the Spider dataset
        limit: Maximum number of queries to test
        start_index: Starting index for testing
        
    Returns:
        dict: Test results summary
    """
    # Create output directories
    results_dir = Path("outputs/spider_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Store execution start time
    batch_start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load Spider dataset
    dev_file = os.path.join(spider_dataset_path, 'dev.json')
    if not os.path.exists(dev_file):
        raise FileNotFoundError(f'Spider dev file not found: {dev_file}')
    
    with open(dev_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    # Load tables.json for schema info
    tables_file = os.path.join(spider_dataset_path, 'tables.json')
    if not os.path.exists(tables_file):
        raise FileNotFoundError(f'Spider tables file not found: {tables_file}')
        
    with open(tables_file, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    
    # Organize schemas by db_id for quick lookup
    schemas_by_id = {}
    for schema in tables_data:
        schemas_by_id[schema['db_id']] = schema
    
    # Limit test data
    if limit <= 0 or limit > len(dev_data):
        limit = len(dev_data)
    
    test_data = dev_data[start_index:start_index + limit]
    logger.info(f"Running tests for {len(test_data)} queries from index {start_index} to {start_index + len(test_data) - 1}")
    
    # Initialize results
    results = []
    total_structure_score = 0
    total_execution_score = 0
    total_combined_score = 0
    successful_tests = 0
    
    # Test each query
    for i, query_item in enumerate(test_data):
        query_start_time = time.time()
        nl_query = query_item['question']
        db_id = query_item['db_id']
        gold_sql = query_item['query']
        
        # Find database path
        db_path = os.path.join(spider_dataset_path, 'database', db_id, f'{db_id}.sqlite')
        
        # Progress report
        print(f"\n{'='*80}")
        print(f"Processing query {start_index + i + 1}/{start_index + len(test_data)} ({i + 1}/{len(test_data)})")
        print(f"Database: {db_id}")
        print(f"Query: {nl_query}")
        print(f"Gold SQL: {gold_sql}")
        print(f"{'-'*80}")
        
        # Store query results
        query_result = {
            "index": start_index + i,
            "db_id": db_id,
            "question": nl_query,
            "gold_sql": gold_sql,
            "generated_sql": "",
            "evaluation": {},
            "elapsed_time": 0,
            "error": None
        }
        
        try:
            # Initialize flow
            query_flow = QueryFlow(dataset_path=spider_dataset_path, dataset_type="spider")
            
            # Set query state
            query_flow.state.query = nl_query
            query_flow.state.db_id = db_id
            query_flow.state.gold_sql = gold_sql
            
            # Load schema
            if db_id in schemas_by_id:
                query_flow.state.db_schema = schemas_by_id[db_id]
            
            # Run flow
            logger.info(f"Starting flow for query {start_index + i + 1}")
            flow_result = query_flow.kickoff()
            
            # Get generated SQL
            generated_sql = query_flow.state.sql_query
            query_result["generated_sql"] = generated_sql
            
            # Evaluate SQL
            if os.path.exists(db_path):
                evaluation_result = evaluate_sql(generated_sql, gold_sql, db_path)
            else:
                logger.warning(f"Database file not found: {db_path}, skipping execution evaluation")
                evaluation_result = evaluate_sql(generated_sql, gold_sql)
            
            query_result["evaluation"] = evaluation_result
            
            # Update scores
            if evaluation_result["status"] == "success":
                total_structure_score += evaluation_result["structure_score"]
                total_execution_score += evaluation_result["execution_score"]
                total_combined_score += evaluation_result["combined_score"]
                successful_tests += 1
            
            # Print progress
            print(f"Generated SQL: {generated_sql}")
            print(f"Structure Score: {evaluation_result['structure_score']:.2f}")
            print(f"Execution Score: {evaluation_result['execution_score']:.2f}")
            print(f"Combined Score: {evaluation_result['combined_score']:.2f}")
            
        except Exception as e:
            error_msg = f"Error processing query {start_index + i + 1}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            query_result["error"] = error_msg
            print(f"ERROR: {error_msg}")
        finally:
            # Clean up resources
            if 'query_flow' in locals() and hasattr(query_flow, 'cleanup_crews'):
                query_flow.cleanup_crews()
            
            # Calculate elapsed time
            query_end_time = time.time()
            query_elapsed_time = query_end_time - query_start_time
            query_result["elapsed_time"] = query_elapsed_time
            
            print(f"Time: {query_elapsed_time:.2f} seconds")
            print(f"{'='*80}")
            
            # Save incremental results
            results.append(query_result)
            
            # Save after each query to avoid losing results
            incremental_summary = {
                "timestamp": timestamp,
                "total_queries": len(test_data),
                "processed_queries": i + 1,
                "successful_queries": successful_tests,
                "average_structure_score": total_structure_score / successful_tests if successful_tests > 0 else 0,
                "average_execution_score": total_execution_score / successful_tests if successful_tests > 0 else 0,
                "average_combined_score": total_combined_score / successful_tests if successful_tests > 0 else 0,
                "results": results
            }
            
            incremental_file = results_dir / f"spider_results_{timestamp}_incremental.json"
            with open(incremental_file, 'w', encoding='utf-8') as f:
                json.dump(incremental_summary, f, ensure_ascii=False, indent=2)
    
    # Calculate batch elapsed time
    batch_end_time = time.time()
    batch_elapsed_time = batch_end_time - batch_start_time
    
    # Create final summary
    if successful_tests > 0:
        average_structure_score = total_structure_score / successful_tests
        average_execution_score = total_execution_score / successful_tests
        average_combined_score = total_combined_score / successful_tests
    else:
        average_structure_score = 0
        average_execution_score = 0
        average_combined_score = 0
    
    summary = {
        "timestamp": timestamp,
        "total_queries": len(test_data),
        "successful_queries": successful_tests,
        "average_structure_score": average_structure_score,
        "average_execution_score": average_execution_score,
        "average_combined_score": average_combined_score,
        "total_elapsed_time": batch_elapsed_time,
        "results": results
    }
    
    # Save final results
    final_file = results_dir / f"spider_results_{timestamp}_final.json"
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\nBatch Test Summary:")
    print(f"Total Queries: {len(test_data)}")
    print(f"Successful Queries: {successful_tests}")
    print(f"Average Structure Score: {average_structure_score:.4f}")
    print(f"Average Execution Score: {average_execution_score:.4f}")
    print(f"Average Combined Score: {average_combined_score:.4f}")
    print(f"Total Time: {batch_elapsed_time:.2f} seconds")
    print(f"Results saved to: {final_file}")
    
    return summary