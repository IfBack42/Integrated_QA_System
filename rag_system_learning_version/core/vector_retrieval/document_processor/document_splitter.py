"""负责将LangChain Document依次切分为父块和子块。"""

import sys
from pathlib import Path

# 允许直接运行本文件；作为包导入时，项目根目录通常已经在搜索路径中。
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from config import config
from logger import get_logger
from rag_system_learning_version.edu_text_spliter import TextSplitterManager


logger = get_logger(__name__)


class DocumentSplitterProcessor:
    """完成“原始Document → 父块 → 子块”的分层切分流程。"""

    def __init__(
        self,
        parent_chunk_size=config.PARENT_CHUNK_SIZE,
        child_chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        splitter_manager=None,
    ):
        self.splitter_manager = splitter_manager or TextSplitterManager(
            parent_chunk_size=parent_chunk_size,
            child_chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents):
        """先生成全部父块，再生成全部子块，返回二者组成的元组。"""
        parent_chunks = self._process_parent_chunks(documents)
        child_chunks = self._process_child_chunks(parent_chunks)

        logger.info(
            f"切分完成: 父块 {len(parent_chunks)} 个, 子块 {len(child_chunks)} 个"
        )
        return parent_chunks, child_chunks

    @staticmethod
    def _validate_document(doc, required_metadata, stage):
        """检查当前切分阶段接收的Document是否符合数据约定。"""
        if not isinstance(doc, Document):
            raise TypeError(
                f"{stage}只接受LangChain Document，当前类型: {type(doc).__name__}"
            )

        if not isinstance(doc.page_content, str) or not doc.page_content.strip():
            raise ValueError(f"{stage}收到了空白或非字符串page_content")

        if not isinstance(doc.metadata, dict):
            raise TypeError(f"{stage}要求metadata为字典")

        missing_fields = [
            field_name
            for field_name in required_metadata
            if not doc.metadata.get(field_name)
        ]
        if missing_fields:
            raise ValueError(f"{stage}缺少必需元数据: {missing_fields}")

    def _process_parent_chunks(self, documents):
        """逐个切分原始Document，使用文件名为每个父块生成parent_id。"""
        parent_chunks = []

        for document_index, doc in enumerate(documents):
            file_path = getattr(doc, "metadata", {}).get("file_path", "")
            try:
                self._validate_document(
                    doc,
                    required_metadata=("file_path", "file_name", "file_extension"),
                    stage="父块切分",
                )
                file_name = doc.metadata["file_name"]
                file_extension = doc.metadata["file_extension"]
                parent_splitter, splitter_name = (
                    self.splitter_manager.get_parent_splitter(file_extension)
                )
                logger.info(
                    f"处理文档: {file_path}, 使用父块切分器: {splitter_name}"
                )

                current_parent_chunks = parent_splitter.split_documents([doc])
                if not current_parent_chunks:
                    raise ValueError("非空Document未生成任何父块")

                for parent_index, parent_doc in enumerate(current_parent_chunks):
                    parent_doc.metadata["parent_id"] = (
                        f"{file_name}_parent_{parent_index}"
                    )

                parent_chunks.extend(current_parent_chunks)
            except Exception:
                logger.exception(
                    "父块切分失败: "
                    f"document_index={document_index}, "
                    f"file_path={file_path or '<missing>'}"
                )
                raise

        return parent_chunks

    def _process_child_chunks(self, parent_chunks):
        """逐个切分父块，子块继承parent_id并获得自己的child_id。"""
        child_chunks = []

        for parent_index, parent_doc in enumerate(parent_chunks):
            metadata = getattr(parent_doc, "metadata", {})
            file_path = metadata.get("file_path", "")
            parent_id = metadata.get("parent_id", "")

            try:
                self._validate_document(
                    parent_doc,
                    required_metadata=(
                        "file_path",
                        "file_name",
                        "file_extension",
                        "parent_id",
                    ),
                    stage="子块切分",
                )
                file_extension = parent_doc.metadata["file_extension"]
                child_splitter, splitter_name = (
                    self.splitter_manager.get_child_splitter(file_extension)
                )
                logger.info(
                    f"处理父块: {parent_id}, 使用子块切分器: {splitter_name}"
                )

                current_child_chunks = child_splitter.split_documents([parent_doc])
                if not current_child_chunks:
                    raise ValueError("非空父块未生成任何子块")

                for child_index, child_doc in enumerate(current_child_chunks):
                    child_doc.metadata["child_id"] = (
                        f"{parent_id}_child_{child_index}"
                    )

                child_chunks.extend(current_child_chunks)
            except Exception:
                logger.exception(
                    "子块切分失败: "
                    f"parent_index={parent_index}, "
                    f"parent_id={parent_id or '<missing>'}, "
                    f"file_path={file_path or '<missing>'}"
                )
                raise

        return child_chunks


if __name__ == "__main__":
    # 先加载一份轻量Markdown文档，再单独验证父子切分与ID传递。
    from rag_system_learning_version.core.vector_retrieval.document_processor.document_loader import (
        DocumentLoaderProcessor,
    )

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

    document_splitter = DocumentSplitterProcessor()
    test_parent_chunks, test_child_chunks = document_splitter.split_documents(
        test_documents
    )

    print(f"父块数量: {len(test_parent_chunks)}")
    print(f"子块数量: {len(test_child_chunks)}")
    if test_parent_chunks:
        print(f"第一个父块ID: {test_parent_chunks[0].metadata['parent_id']}")
    if test_child_chunks:
        print(f"第一个子块ID: {test_child_chunks[0].metadata['child_id']}")
