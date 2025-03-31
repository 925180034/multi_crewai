#!/usr/bin/env python
from datetime import datetime
from pathlib import Path
import json
import logging
import time
import traceback
from pydantic import BaseModel
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
    """查询状态模型"""
    query: str = ""
    plan: str = ""
    db_data: str = ""
    web_data: str = ""
    doc_data: str = ""
    schema_matches: str = ""
    sql_query: str = ""

class QueryFlow(Flow[QueryState]):
    """查询处理流程"""
    
    def __init__(self):
        """初始化查询流程"""
        super().__init__()
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
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
                self.retrieval_crew = RetrievalCrew()
            if not self.matcher_crew:
                self.matcher_crew = MatcherCrew()
            if not self.sql_crew:
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
            
            self.state.query = """
            1.查询购买金额最高的前5名客户的姓名、联系方式和总消费金额
            2.显示每种产品的平均售价和销售总量
            3.找出评分最高和最低的商品的详细信息，包括名称、类型和价格
            4.生成一份报告, 列出消费金额超过2000元的高价值客户及其购买的所有商品
            5.分析哪些商品被金卡会员购买次数最多，并按受欢迎程度排序
            """
            
            # 保存初始查询
            with open(self.output_dir / f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w", encoding="utf-8") as f:
                json.dump({"query": self.state.query}, f, ensure_ascii=False, indent=2)
                
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
            result = self.sql_crew.crew().kickoff(
                inputs={
                    "query": self.state.query,
                    "db_data": self.state.db_data,
                    "web_data": self.state.web_data,
                    "doc_data": self.state.doc_data,
                    "schema_matches": self.state.schema_matches
                }
            )
            self.state.sql_query = result.raw if hasattr(result, 'raw') else str(result)
            self.save_crew_output("sql_query", result)
            logger.info("SQL query generated")
        except Exception as e:
            logger.error(f"Error in generate_sql: {str(e)}")
            self.save_crew_output("sql_query_error", {"error": str(e)})
            raise

def kickoff():
    """启动查询流程"""
    query_flow = QueryFlow()
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

if __name__ == "__main__":
    kickoff()