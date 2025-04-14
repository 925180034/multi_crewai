# src/multi/tools/document_retrieval_tool.py - 优化版
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import logging
import os
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentRetrievalInput(BaseModel):
    """文档检索工具的输入模式"""
    query: str = Field(..., description="用于在文档中搜索的查询")
    document_path: str = Field(None, description="文档或目录路径，如果不提供则使用默认路径")

class DocumentRetrievalTool(BaseTool):
    """文档检索工具，用于在文档中搜索相关信息"""
    
    name: str = "文档信息检索"
    description: str = """
    此工具根据给定的查询在文档中搜索相关信息。
    它可以处理多种文档格式并提供上下文相关的响应。
    """
    args_schema: Type[BaseModel] = DocumentRetrievalInput
    max_iter: int = 1  # 默认最大迭代次数
    verbose: bool = True  # 启用详细日志

    def _run(self, query: str, document_path: str = None) -> str:
        """
        执行文档检索
        
        Args:
            query: 搜索查询
            document_path: 文档路径(可选)
            
        Returns:
            str: 检索结果
        """
        try:
            # 使用提供的文档路径或设置默认路径
            doc_path = document_path or getattr(self, 'document_path', None)
            
            if not doc_path:
                return f"错误: 未提供文档路径"
                
            # 验证路径
            doc_path = Path(doc_path)
            if not doc_path.exists():
                return f"错误: 文档路径不存在 - {doc_path}"
                
            # 根据查询模拟文档检索结果
            # 这里简化了实际检索逻辑，实际应用中应使用更复杂的检索方法
            return self._simulate_search(query, doc_path)
                
        except Exception as e:
            logger.error(f"文档检索过程中出错: {str(e)}")
            return f"文档检索失败: {str(e)}"
            
    def _simulate_search(self, query: str, doc_path: Path) -> str:
        """
        模拟文档搜索结果
        
        Args:
            query: 搜索查询
            doc_path: 文档路径
            
        Returns:
            str: 模拟的检索结果
        """
        # 检查是目录还是文件
        if doc_path.is_dir():
            # 列出目录中的文件
            files = list(doc_path.glob('*.txt')) + list(doc_path.glob('*.pdf')) + list(doc_path.glob('*.json'))
            files_info = '\n'.join([f"- {f.name}" for f in files[:5]])
            
            return f"关于'{query}'的文档检索结果:\n\n" \
                   f"在目录 {doc_path} 中找到的相关文件:\n{files_info}\n\n" \
                   f"相关内容摘要:\n" \
                   f"1. {query}的基本概念和定义\n" \
                   f"2. {query}的应用示例和用例\n" \
                   f"3. {query}相关的技术细节和最佳实践"
        else:
            # 获取文件扩展名
            ext = doc_path.suffix.lower()
            
            if ext in ['.pdf', '.txt', '.md', '.json']:
                return f"从文件 {doc_path.name} 中提取的关于'{query}'的信息:\n\n" \
                       f"1. 文档包含与'{query}'相关的以下信息:\n" \
                       f"   - 定义: {query}是指...\n" \
                       f"   - 使用方法: 在以下场景中使用{query}...\n" \
                       f"   - 示例: {query}的应用示例包括...\n\n" \
                       f"2. 相关技术细节:\n" \
                       f"   - 实现方式\n" \
                       f"   - 限制和注意事项\n" \
                       f"   - 优化建议"
            else:
                return f"不支持的文件类型 {ext}"