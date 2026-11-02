"""负责将目录中的文件统一加载为LangChain Document。"""

import os
import sys
from datetime import datetime
from pathlib import Path

# 允许直接运行本文件；作为包导入时，项目根目录通常已经在搜索路径中。
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system_learning_version.edu_document_loaders import DocumentLoaderManager
from logger import get_logger


logger = get_logger(__name__)


class DocumentLoaderProcessor:
    """完成“遍历目录 → 选择Loader → 生成Document列表”的加载流程。"""

    def __init__(self, loader_manager=None):
        self.loader_manager = loader_manager or DocumentLoaderManager()

    def load_documents(self, directory_path):
        """加载目录中的全部受支持文件，返回统一的Document列表。"""
        documents = []

        for file_path in self._get_file_paths(directory_path):
            try:
                loaded_docs = self._load_single_document(file_path)
                documents.extend(loaded_docs)
                if loaded_docs:
                    logger.info(f"成功加载文件: {file_path}")
            except Exception:
                logger.exception(f"加载文件失败: {file_path}")

        logger.info(f"加载的Document数量: {len(documents)}")
        return documents

    @staticmethod
    def _get_file_paths(directory_path):
        """递归遍历目录，返回所有文件的绝对路径。"""
        absolute_directory = os.path.abspath(directory_path)
        if not os.path.isdir(absolute_directory):
            raise NotADirectoryError(f"文档目录不存在: {absolute_directory}")

        file_paths = []
        for root, _, files in os.walk(absolute_directory):
            for file_name in files:
                file_paths.append(os.path.join(root, file_name))
        return file_paths

    def _load_single_document(self, file_path):
        """加载单个文件，并为其产生的Document补充统一元数据。"""
        file_extension = os.path.splitext(file_path)[1].lower()
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        loader = self.loader_manager.get_loader(file_path, file_extension)

        if loader is None:
            logger.warning(f"不支持的文件类型: {file_path}")
            return []

        loaded_docs = loader.load()
        loaded_at = datetime.now().isoformat()

        for doc in loaded_docs:
            doc.metadata.pop("source", None)
            doc.metadata["file_path"] = file_path
            doc.metadata["file_name"] = file_name
            doc.metadata["file_extension"] = file_extension
            doc.metadata["timestamp"] = loaded_at

        return loaded_docs


if __name__ == "__main__":
    # 使用轻量的Markdown文件测试，避免运行本文件时触发PDF/Word OCR。
    test_file_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "ai_data"
        / "人工智能就业课课程大纲.md"
    )
    if not test_file_path.is_file():
        raise FileNotFoundError(f"测试文件不存在: {test_file_path}")

    document_loader = DocumentLoaderProcessor()
    test_documents = document_loader._load_single_document(str(test_file_path))

    print(f"加载得到的Document数量: {len(test_documents)}")
    if test_documents:
        first_document = test_documents[0]
        print(f"第一条Document元数据: {first_document.metadata}")
        print(f"第一条Document正文预览: {first_document.page_content[:100]}")
