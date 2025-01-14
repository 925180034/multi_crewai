# src/multi/tools/document_retrieval_tool.py
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from pathlib import Path
import logging
import os

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DocumentRetrievalInput(BaseModel):
    """Input schema for DocumentRetrievalTool."""
    query: str = Field(..., description="The query to search for in the documents.")
    document_path: str = Field(..., description="Path to the document or directory to search in.")


class DocumentRetrievalTool(BaseTool):
    name: str = "Document Information Retrieval"
    description: str = """
    This tool searches through documents (PDFs, TXTs, etc.) to find relevant information 
    based on a given query. It can handle multiple document formats and provides 
    context-aware responses.
    """
    args_schema: Type[BaseModel] = DocumentRetrievalInput
    max_iter: int = 2  # Default max iterations
    iter_count: int = 0  # Track iterations

    def _run(self, query: str, document_path: str) -> str:
        """
        Main function to execute the document retrieval tool.
        """
        try:
            self.iter_count += 1
            if self.iter_count > self.max_iter:
                logger.warning(f"Maximum iterations ({self.max_iter}) reached. Aborting.")
                return f"Maximum iterations ({self.max_iter}) reached. Aborting."

            # 验证目录路径
            document_dir = Path(document_path)
            if not document_dir.exists() or not document_dir.is_dir():
                logger.error(f"Invalid document directory: {document_path}")
                return f"Error: Invalid document directory: {document_path}"

            # 加载文档
            documents = []
            try:
                reader = SimpleDirectoryReader(input_dir=document_path)
                documents = reader.load_data()
                logger.info(f"Loaded {len(documents)} documents from {document_path}")
            except Exception as e:
                logger.error(f"Error loading documents: {str(e)}")
                return f"Error loading documents: {str(e)}"

            if not documents:
                logger.warning(f"No documents found in the specified path: {document_path}")
                return f"No documents found in the specified path: {document_path}"

            # 创建索引
            try:
                index = VectorStoreIndex.from_documents(documents)
                query_engine = index.as_query_engine()
            except Exception as e:
                logger.error(f"Error creating VectorStoreIndex: {str(e)}")
                return f"Error creating VectorStoreIndex: {str(e)}"

            # 设置元数据并创建查询工具
            metadata = ToolMetadata(
                name="Document Search",
                description=f"Search through documents (Attempt {self.iter_count}/{self.max_iter})"
            )
            query_tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=metadata
            )

            # 执行查询
            try:
                response = query_tool.query(query)
                logger.info(f"Query successful: {response}")
                return f"(Iteration {self.iter_count}/{self.max_iter}): {str(response)}"
            except Exception as e:
                logger.error(f"Error during query execution: {str(e)}")
                return f"Error during query execution: {str(e)}"

        except Exception as e:
            logger.critical(f"Unexpected error during document retrieval: {str(e)}")
            return f"Critical error during document retrieval: {str(e)}"