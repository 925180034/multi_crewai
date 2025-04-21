#!/usr/bin/env python
# 文件：src/multi/main.py 的优化版本
import json
import os
import time
import re
import logging
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


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
# litellm.set_verbose = True
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
    """查询处理流程，简化版"""
    
    def __init__(self, dataset_path=None, dataset_type="default"):
        """初始化查询流程"""
        super().__init__()
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # 存储数据集信息
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        
        # 初始化crews
        self.planner_crew = None
        self.retrieval_crew = None
        self.matcher_crew = None
        self.sql_crew = None
        
        # 立即初始化所有crews
        self._initialize_crews()

    def _initialize_crews(self):
        """初始化所有crews，简化逻辑"""
        try:
            self.planner_crew = PlannerCrew()
            self.retrieval_crew = RetrievalCrew(dataset_path=self.dataset_path)
            self.matcher_crew = MatcherCrew()
            self.sql_crew = SQLCrew(spider_dataset_path=self.dataset_path)
            logger.info("所有crews初始化完成")
        except Exception as e:
            logger.error(f"初始化crews失败: {str(e)}")
            raise

    def cleanup_crews(self):
        """清理所有crews的资源"""
        try:
            if hasattr(self.retrieval_crew, 'cleanup'):
                self.retrieval_crew.cleanup()
            # 其他清理逻辑...
        except Exception as e:
            logger.error(f"清理crews资源失败: {str(e)}")

    @start()
    def process_query(self):
        """处理初始查询，简化逻辑"""
        try:
            # 加载数据
            self._load_query_data()
            logger.info(f"使用查询: {self.state.query}")
        except Exception as e:
            logger.error(f"处理查询失败: {str(e)}")
            raise

    def _load_query_data(self):
        """加载查询数据，封装逻辑"""
        # 从Spider数据集加载
        if self.dataset_path and self.dataset_type == "spider":
            self._load_from_spider()
        # 从自定义数据集加载
        elif self.dataset_path and self.dataset_type == "custom":
            self._load_from_custom()
        # 使用默认查询
        else:
            self.state.query = "查找所有客户的姓名和他们的总消费金额，按金额降序排列。"

    def _load_from_spider(self):
        """从Spider数据集加载查询"""
        try:
            import json
            
            # 加载查询
            dev_file = os.path.join(self.dataset_path, 'dev.json')
            with open(dev_file, 'r') as f:
                queries = json.load(f)
            
            if queries and len(queries) > 0:
                # 使用第一个查询
                self.state.query = queries[0]['question']
                self.state.db_id = queries[0]['db_id']
                
                # 加载数据库模式
                tables_file = os.path.join(self.dataset_path, 'tables.json')
                with open(tables_file, 'r') as f:
                    schemas = json.load(f)
                
                # 查找对应的模式
                for schema in schemas:
                    if schema['db_id'] == self.state.db_id:
                        self.state.db_schema = schema
                        break
        except Exception as e:
            logger.error(f"从Spider加载数据失败: {str(e)}")
            raise

    def _load_from_custom(self):
        """从自定义数据集加载查询"""
        try:
            import json
            
            data_file = os.path.join(self.dataset_path, 'data.json')
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            if 'query' in data:
                self.state.query = data['query']
                self.state.db_schema = data.get('schema', {})
        except Exception as e:
            logger.error(f"从自定义数据集加载失败: {str(e)}")
            raise

    @listen(process_query)
    def create_plan(self):
        """创建执行计划，简化错误处理"""
        try:
            result = self.planner_crew.crew().kickoff(
                inputs={"query": self.state.query}
            )
            self.state.plan = result.raw if hasattr(result, 'raw') else str(result)
            logger.info("计划创建成功")
        except Exception as e:
            logger.error(f"创建计划失败: {str(e)}")
            self.state.error = f"创建计划失败: {str(e)}"

    @listen(create_plan)
    def retrieve_data(self):
        """检索数据，简化错误处理"""
        if self.state.error:
            logger.warning(f"跳过数据检索，因为存在错误: {self.state.error}")
            return
            
        try:
            result = self.retrieval_crew.crew().kickoff(
                inputs={
                    "query": self.state.query,
                    "plan": self.state.plan
                }
            )
            
            # 处理结果
            if hasattr(result, 'tasks_output'):
                self.state.db_data = result.tasks_output[0].raw if len(result.tasks_output) > 0 else ""
                self.state.web_data = result.tasks_output[1].raw if len(result.tasks_output) > 1 else ""
                self.state.doc_data = result.tasks_output[2].raw if len(result.tasks_output) > 2 else ""
            
            logger.info("数据检索成功")
        except Exception as e:
            logger.error(f"数据检索失败: {str(e)}")
            self.state.error = f"数据检索失败: {str(e)}"

    @listen(retrieve_data)
    def match_schemas(self):
        """匹配数据模式，简化错误处理"""
        if self.state.error:
            logger.warning(f"跳过模式匹配，因为存在错误: {self.state.error}")
            return
            
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
            logger.info("模式匹配成功")
        except Exception as e:
            logger.error(f"模式匹配失败: {str(e)}")
            self.state.error = f"模式匹配失败: {str(e)}"

    @listen(match_schemas)
    def generate_sql(self):
        """生成SQL查询，简化错误处理"""
        if self.state.error:
            logger.warning(f"跳过SQL生成，因为存在错误: {self.state.error}")
            return
            
        try:
            # 准备数据库模式信息
            db_schema_info = ""
            db_id = ""
            
            # 使用db_schema或从SQL crew加载
            if hasattr(self.state, 'db_schema') and self.state.db_schema:
                db_schema_info = json.dumps(self.state.db_schema, ensure_ascii=False, indent=2)
                if hasattr(self.state, 'db_id'):
                    db_id = self.state.db_id
            elif self.sql_crew and hasattr(self.sql_crew, 'db_schemas') and self.sql_crew.db_schemas:
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
            logger.info("SQL生成成功")
        except Exception as e:
            logger.error(f"SQL生成失败: {str(e)}")
            self.state.error = f"SQL生成失败: {str(e)}"

def evaluate_sql(generated_sql, gold_sql, db_path=None):
    """
    评估生成的SQL与标准SQL的相似度
    
    Args:
        generated_sql: 生成的SQL
        gold_sql: 标准SQL
        db_path: 数据库路径(可选)
        
    Returns:
        float: 相似度得分(0-1)
    """
    # 简单的字符串比较
    if generated_sql.strip().lower() == gold_sql.strip().lower():
        return 1.0
    
    # 基本的结构化比较 (移除空格、大小写等差异)
    def normalize_sql(sql):
        sql = sql.lower().strip()
        sql = re.sub(r'\s+', ' ', sql)
        sql = re.sub(r'`', '', sql)
        sql = re.sub(r'"', '', sql)
        sql = re.sub(r'\'', '', sql)
        return sql
    
    norm_generated = normalize_sql(generated_sql)
    norm_gold = normalize_sql(gold_sql)
    
    if norm_generated == norm_gold:
        return 0.9
    
    # 检查关键词和表名是否匹配
    gold_tokens = set(re.findall(r'\b\w+\b', norm_gold))
    generated_tokens = set(re.findall(r'\b\w+\b', norm_generated))
    
    # 计算tokens的重叠度
    common_tokens = gold_tokens.intersection(generated_tokens)
    if len(gold_tokens) > 0:
        token_overlap = len(common_tokens) / len(gold_tokens)
    else:
        token_overlap = 0
        
    # 如果数据库路径存在，可以尝试执行SQL验证结果是否相同
    # 注意：这是更复杂的评估，可能需要专门的工具
    
    return token_overlap * 0.8  # 降低权重，因为这只是基于token的简单对比

def batch_test_spider(spider_dataset_path, limit=10, start_index=0):
    """
    批量测试Spider数据集查询
    
    Args:
        spider_dataset_path: Spider数据集路径
        limit: 测试查询数量
        start_index: 起始索引
        
    Returns:
        dict: 测试结果摘要
    """
    # 加载Spider数据集
    dev_file = os.path.join(spider_dataset_path, 'dev.json')
    if not os.path.exists(dev_file):
        raise FileNotFoundError(f'Spider dev文件未在 {dev_file} 找到')
    
    with open(dev_file, 'r') as f:
        dev_data = json.load(f)
    
    # 限制测试数量
    if limit <= 0 or limit > len(dev_data):
        limit = len(dev_data)
    
    test_data = dev_data[start_index:start_index + limit]
    
    # 创建结果目录
    results_dir = Path("outputs/spider_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试单个查询
    def test_single_query(query_item, index):
        try:
            start_time = time.time()
            
            nl_query = query_item['question']
            db_id = query_item['db_id']
            gold_sql = query_item['query']
            
            # 初始化查询流程
            query_flow = QueryFlow(dataset_path=spider_dataset_path, dataset_type="spider")
            
            # 覆盖查询
            query_flow.state.query = nl_query
            query_flow.state.db_id = db_id
            
            # 运行流程
            result = query_flow.kickoff()
            
            # 获取生成的SQL
            generated_sql = query_flow.state.sql_query
            
            # 评估结果
            accuracy = evaluate_sql(generated_sql, gold_sql)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            return {
                "index": index,
                "db_id": db_id,
                "question": nl_query,
                "gold_sql": gold_sql,
                "generated_sql": generated_sql,
                "accuracy": accuracy,
                "elapsed_time": elapsed_time
            }
            
        except Exception as e:
            import traceback
            return {
                "index": index,
                "db_id": db_id if 'db_id' in locals() else "unknown",
                "question": nl_query if 'nl_query' in locals() else "unknown",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "accuracy": 0,
                "elapsed_time": -1
            }
    
    # 串行执行所有测试
    results = []
    total_accuracy = 0
    successful_tests = 0
    
    for i, query_item in enumerate(test_data):
        print(f"测试查询 {start_index + i + 1}/{start_index + limit} ({i + 1}/{len(test_data)})")
        result = test_single_query(query_item, start_index + i)
        results.append(result)
        
        if "error" not in result:
            total_accuracy += result["accuracy"]
            successful_tests += 1
            print(f"精确度: {result['accuracy']:.2f}")
        else:
            print(f"错误: {result['error']}")
        
        print("-" * 50)
    
    # 计算总体准确率
    if successful_tests > 0:
        average_accuracy = total_accuracy / successful_tests
    else:
        average_accuracy = 0
    
    # 保存详细结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"spider_results_{timestamp}.json"
    
    summary = {
        "total_queries": len(test_data),
        "successful_queries": successful_tests,
        "average_accuracy": average_accuracy,
        "results": results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n测试完成!")
    print(f"总查询数: {len(test_data)}")
    print(f"成功查询数: {successful_tests}")
    print(f"平均准确率: {average_accuracy:.4f}")
    print(f"详细结果已保存至: {results_file}")
    
    return summary

def kickoff(dataset_path=None, dataset_type="default"):
    """
    启动查询流程
    
    Args:
        dataset_path (str, optional): 数据集路径. Defaults to None.
        dataset_type (str, optional): 数据集类型 ("spider", "custom", "default"). Defaults to "default".
    """
    try:
        # 检查数据集路径
        if dataset_path and os.path.exists(dataset_path):
            logger.info(f'使用{dataset_type}数据集路径: {dataset_path}')
        else:
            logger.warning(f'{dataset_type}数据集路径无效或未提供，将使用默认查询')
            dataset_path = None
        
        query_flow = QueryFlow(dataset_path=dataset_path, dataset_type=dataset_type)
        result = query_flow.kickoff()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_data = {
            "timestamp": timestamp,
            "state": query_flow.state.model_dump(),
        }
        
        # 保存结果
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / f"final_result_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        logger.info("Query flow completed successfully")
        
        if query_flow.state.sql_query:
            print("\n生成的SQL查询:")
            print("-" * 50)
            print(query_flow.state.sql_query)
            print("-" * 50)
            
        return result
            
    except Exception as e:
        logger.error(f"Error in query flow: {str(e)}")
        raise
    finally:
        if 'query_flow' in locals() and hasattr(query_flow, 'cleanup_crews'):
            query_flow.cleanup_crews()