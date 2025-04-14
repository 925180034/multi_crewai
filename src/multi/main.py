#!/usr/bin/env python
from datetime import datetime
from pathlib import Path
import json
import logging
import os
import time
import traceback
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from crewai.flow.flow import Flow, listen, start

from multi.crews.planner_crew.planner_crew import PlannerCrew
from multi.crews.retrieval_crew.retrieval_crew import RetrievalCrew
from multi.crews.matcher_crew.matcher_crew import MatcherCrew  
from multi.crews.sql_crew.sql_crew import SQLCrew

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryState(BaseModel):
    """查询状态模型，增强版"""
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
    gold_sql: str = ""  # 参考SQL（从Spider获取）

class QueryFlow(Flow[QueryState]):
    """查询处理流程"""
    
    def __init__(self, spider_dataset_path=None):
        """初始化查询流程"""
        super().__init__()
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # 存储Spider数据集路径
        self.spider_dataset_path = spider_dataset_path
        logger.info(f'QueryFlow初始化，Spider数据集路径: {self.spider_dataset_path}')
        
        # 初始化crews
        self.planner_crew = None
        self.retrieval_crew = None
        self.matcher_crew = None
        self.sql_crew = None

    def _initialize_crews(self):
        """初始化所有crews"""
        try:
            if not self.planner_crew:
                self.planner_crew = PlannerCrew()
            
            if not self.retrieval_crew:
                # 检查路径是否可用
                if self.spider_dataset_path and os.path.exists(self.spider_dataset_path):
                    logger.info(f'正在使用Spider数据集初始化检索小组: {self.spider_dataset_path}')
                    self.retrieval_crew = RetrievalCrew(dataset_path=self.spider_dataset_path)
                else:
                    logger.info('初始化检索小组，无Spider数据集')
                    self.retrieval_crew = RetrievalCrew()
            
            if not self.matcher_crew:
                self.matcher_crew = MatcherCrew()
            
            if not self.sql_crew:
                # 检查路径是否可用
                if self.spider_dataset_path and os.path.exists(self.spider_dataset_path):
                    logger.info(f'正在使用Spider数据集初始化SQL小组: {self.spider_dataset_path}')
                    self.sql_crew = SQLCrew(spider_dataset_path=self.spider_dataset_path)
                else:
                    logger.info('初始化SQL小组，无Spider数据集')
                    self.sql_crew = SQLCrew()
        except Exception as e:
            logger.error(f"Error initializing crews: {str(e)}")
            raise

    def cleanup_crews(self):
        """清理所有crews的资源"""
        try:
            if self.retrieval_crew:
                self.retrieval_crew.cleanup()
            # 其他crew的清理逻辑...
        except Exception as e:
            logger.error(f"Error cleaning up crews: {str(e)}")

    def serialize_crew_output(self, crew_output) -> Dict[str, Any]:
        """序列化crew输出"""
        try:
            # 处理token使用情况
            token_usage = {}
            if hasattr(crew_output, 'token_usage') and crew_output.token_usage:
                token_usage = {
                    'total_tokens': getattr(crew_output.token_usage, 'total_tokens', 0),
                    'prompt_tokens': getattr(crew_output.token_usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(crew_output.token_usage, 'completion_tokens', 0)
                }
            
            # 处理任务输出
            tasks_output = []
            if hasattr(crew_output, 'tasks_output'):
                for task in crew_output.tasks_output:
                    tasks_output.append({
                        "description": task.description if hasattr(task, 'description') else None,
                        "output": task.raw if hasattr(task, 'raw') else str(task)
                    })
            
            return {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "raw_output": crew_output.raw if hasattr(crew_output, 'raw') else str(crew_output),
                "tasks_output": tasks_output,
                "token_usage": token_usage
            }
        except Exception as e:
            logger.error(f"Error serializing crew output: {str(e)}")
            return {"error": str(e)}

    def save_crew_output(self, name: str, crew_output):
        """保存crew输出到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{name}_{timestamp}.json"
            
            output_data = self.serialize_crew_output(crew_output)
            
            if hasattr(crew_output, 'model_dump'):
                output_data['state'] = crew_output.model_dump()
            else:
                output_data['state'] = vars(crew_output)
                
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved crew output to {output_file}")
        except Exception as e:
            logger.error(f"Error saving crew output: {str(e)}")

    # @start()
    # def process_query(self):
    #     """处理初始查询"""
    #     try:
    #         self._initialize_crews()
            
    #         self.state.query = """
    #         查找所有患者中最近一次就诊诊断为糖尿病的记录，包括：
    #         1. 患者的基本信息
    #         2. 就诊的具体诊断内容
    #         3. 相关的用药建议
    #         并结合治疗指南提供用药参考。
    #         """
            
    #         # 保存初始查询
    #         with open(self.output_dir / f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w", encoding="utf-8") as f:
    #             json.dump({"query": self.state.query}, f, ensure_ascii=False, indent=2)
                
    #         logger.info("Initial query processed")
    #     except Exception as e:
    #         logger.error(f"Error processing query: {str(e)}")
    #         raise

    @start()
    def process_query(self):
        """处理初始查询"""
        try:
            self._initialize_crews()
            
            # 如果已提供Spider数据集路径
            if self.spider_dataset_path:
                # 尝试使用第一个Spider查询作为示例
                try:
                    import json
                    import os
                    
                    # 从Spider dev集加载查询
                    dev_file = os.path.join(self.spider_dataset_path, 'dev.json')
                    if os.path.exists(dev_file):
                        with open(dev_file, 'r') as f:
                            dev_data = json.load(f)
                        
                        if dev_data and len(dev_data) > 0:
                            # 使用第一个查询作为示例
                            self.state.query = dev_data[0]['question']
                            self.state.db_id = dev_data[0]['db_id']
                            logger.info(f"使用Spider示例查询: {self.state.query}")
                            
                            # 加载对应的数据库模式
                            tables_file = os.path.join(self.spider_dataset_path, 'tables.json')
                            with open(tables_file, 'r') as f:
                                db_schemas = json.load(f)
                            
                            # 找到此数据库的模式
                            for schema in db_schemas:
                                if schema['db_id'] == self.state.db_id:
                                    self.state.db_schema = schema
                                    break
                except Exception as e:
                    logger.error(f"使用Spider示例查询时出错: {str(e)}")
                    # 回退到默认查询
                    self.state.query = """
                    查找所有客户的姓名和他们的总消费金额，按金额降序排列。
                    """
            else:
                # 使用默认查询
                self.state.query = """
                查找所有客户的姓名和他们的总消费金额，按金额降序排列。
                """
                
            # 保存初始查询
            with open(self.output_dir / f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w", encoding="utf-8") as f:
                query_data = {"query": self.state.query}
                if hasattr(self.state, 'db_schema'):
                    query_data["db_schema"] = self.state.db_schema
                if hasattr(self.state, 'db_id'):
                    query_data["db_id"] = self.state.db_id
                
                json.dump(query_data, f, ensure_ascii=False, indent=2)
                
            logger.info("Initial query processed")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    @listen(process_query)
    def create_plan(self):
        """创建执行计划"""
        try:
            result = self.planner_crew.crew().kickoff(
                inputs={"query": self.state.query}
            )
            self.state.plan = result.raw if hasattr(result, 'raw') else str(result)
            self.save_crew_output("plan", result)
            logger.info("Plan created successfully")
        except Exception as e:
            logger.error(f"Error in create_plan: {str(e)}")
            self.save_crew_output("plan_error", {"error": str(e)})
            raise

    @listen(create_plan)
    def retrieve_data(self):
        """检索数据，增强错误处理和状态验证"""
        try:
            # 验证必要的前置条件
            if not self.state.query or not self.state.plan:
                logger.error("Missing required state for data retrieval")
                self.state.error = "Missing query or plan for data retrieval"
                return
                
            logger.info(f"Starting data retrieval with query length: {len(self.state.query)}")
            
            # 尝试初始化检索小组
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    result = self.retrieval_crew.crew().kickoff(
                        inputs={
                            "query": self.state.query,
                            "plan": self.state.plan
                        }
                    )
                    
                    # 处理结果并验证数据完整性
                    if hasattr(result, 'tasks_output'):
                        self.state.db_data = result.tasks_output[0].raw if len(result.tasks_output) > 0 else ""
                        self.state.web_data = result.tasks_output[1].raw if len(result.tasks_output) > 1 else ""
                        self.state.doc_data = result.tasks_output[2].raw if len(result.tasks_output) > 2 else ""
                        
                        # 验证数据完整性
                        if not self.state.db_data and not self.state.web_data and not self.state.doc_data:
                            logger.warning("No data retrieved from any source")
                            # 继续执行，但记录警告
                    
                    self.save_crew_output("retrieve_data", result)
                    logger.info("Data retrieved successfully")
                    break
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Retrieval attempt {retry_count} failed: {str(e)}")
                    if retry_count >= max_retries:
                        raise
                    time.sleep(2 ** retry_count)  # 指数退避
            
        except Exception as e:
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
            logger.error(f"Error in retrieve_data: {str(e)}", exc_info=True)
            self.save_crew_output("retrieve_data_error", error_details)
            self.state.error = f"Data retrieval failed: {str(e)}"

    @listen(retrieve_data)
    def match_schemas(self):
        """匹配数据模式"""
        try:
            result = self.matcher_crew.crew().kickoff(
                inputs={
                    "query": self.state.query,
                    "db_data": self.state.db_data,
                    "web_data": self.state.web_data,
                    "doc_data": self.state.doc_data
                }
            )
            self.state.schema_matches = result.raw if hasattr(result, 'raw') else str(result)
            self.save_crew_output("schema_matches", result)
            logger.info("Schema matching completed")
        except Exception as e:
            logger.error(f"Error in match_schemas: {str(e)}")
            self.save_crew_output("schema_matches_error", {"error": str(e)})
            raise

    @listen(match_schemas)
    def generate_sql(self):
        """生成SQL查询"""
        try:
            # 准备数据库模式信息
            db_schema_info = ""
            db_id = ""
            
            # 如果状态中有db_schema（从Spider加载的），使用它
            if hasattr(self.state, 'db_schema') and self.state.db_schema:
                db_schema_info = json.dumps(self.state.db_schema, ensure_ascii=False, indent=2)
                if hasattr(self.state, 'db_id'):
                    db_id = self.state.db_id
            
            # 如果SQL crew加载了Spider模式但状态中没有特定模式
            elif self.sql_crew and hasattr(self.sql_crew, 'db_schemas') and self.sql_crew.db_schemas:
                # 使用第一个模式作为示例
                db_schema_info = json.dumps(self.sql_crew.db_schemas[0], ensure_ascii=False, indent=2)
                db_id = self.sql_crew.db_schemas[0].get('db_id', '')
            
            result = self.sql_crew.crew().kickoff(
                inputs={
                    "query": self.state.query,
                    "db_data": self.state.db_data,
                    "web_data": self.state.web_data,
                    "doc_data": self.state.doc_data,
                    "schema_matches": self.state.schema_matches,
                    "db_schema": db_schema_info,
                    "db_id": db_id
                }
            )
            self.state.sql_query = result.raw if hasattr(result, 'raw') else str(result)
            self.save_crew_output("sql_query", result)
            logger.info("SQL query generated")
        except Exception as e:
            logger.error(f"Error in generate_sql: {str(e)}")
            self.save_crew_output("sql_query_error", {"error": str(e)})
            raise

def kickoff(spider_dataset_path=None):
    """启动查询流程"""

    # 获取Spider数据集路径
    spider_dir = os.getenv('SPIDER_DATASET_PATH')
    if not spider_dir or not os.path.exists(spider_dir):
        logger.error(f'Spider数据集未在{spider_dir}找到')
        raise ValueError('SPIDER_DATASET_PATH未在.env文件中正确设置')
    
    query_flow = QueryFlow(spider_dataset_path=spider_dataset_path)
    try:
        result = query_flow.kickoff()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_data = {
            "timestamp": timestamp,
            "state": query_flow.state.model_dump(),
        }
        
        if result is not None:
            output_data.update(query_flow.serialize_crew_output(result))
        
        with open(f"outputs/final_result_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        logger.info("Query flow completed successfully")
            
    except Exception as e:
        logger.error(f"Error in query flow: {str(e)}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"outputs/error_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "error": str(e),
                "state": query_flow.state.model_dump() if hasattr(query_flow, 'state') else {}
            }, f, ensure_ascii=False, indent=2)
        raise
    finally:
        query_flow.cleanup_crews()

def plot():
    """绘制流程图"""
    query_flow = QueryFlow()
    try:
        query_flow.plot()
        logger.info("Flow plot generated successfully")
    except Exception as e:
        logger.error(f"Error generating flow plot: {str(e)}")
        raise

def test_with_spider(spider_dataset_path, query_index=0):
    """使用Spider数据集中的查询测试框架"""
    import json
    import os
    
    # 从Spider开发集加载查询
    dev_file = os.path.join(spider_dataset_path, 'dev.json')
    if not os.path.exists(dev_file):
        raise FileNotFoundError(f'Spider dev文件未在 {dev_file} 找到')
    
    with open(dev_file, 'r') as f:
        dev_data = json.load(f)
    
    if not dev_data or query_index >= len(dev_data):
        raise ValueError(f'无效的查询索引 {query_index}。数据集有 {len(dev_data)} 个查询')
    
    # 获取选定的查询
    query_item = dev_data[query_index]
    nl_query = query_item['question']
    db_id = query_item['db_id']
    gold_sql = query_item['query']
    
    # 加载相应的数据库模式
    tables_file = os.path.join(spider_dataset_path, 'tables.json')
    with open(tables_file, 'r') as f:
        db_schemas = json.load(f)
    
    # 找到此数据库的模式
    db_schema = None
    for schema in db_schemas:
        if schema['db_id'] == db_id:
            db_schema = schema
            break
    
    if not db_schema:
        raise ValueError(f'未找到数据库 {db_id} 的模式')
    
    # 使用Spider数据集初始化查询流程
    query_flow = QueryFlow(spider_dataset_path=spider_dataset_path)
    
    # 覆盖查询
    query_flow.state.query = nl_query
    query_flow.state.db_schema = db_schema
    
    # 设置数据库ID到状态以便SQLCrew可以访问
    query_flow.state.db_id = db_id
    
    # 运行流程
    result = query_flow.kickoff()
    
    # 比较结果
    print(f'自然语言查询: {nl_query}')
    print(f'标准SQL: {gold_sql}')
    print(f'生成的SQL: {query_flow.state.sql_query}')
    
    return result

if __name__ == "__main__":
    kickoff()