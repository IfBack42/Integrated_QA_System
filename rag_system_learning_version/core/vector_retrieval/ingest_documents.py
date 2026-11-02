"""独立执行完整的文档入库流程。

数据流：文档目录 → Document加载 → 父子块切分 → BGE-M3编码
→ 组装Milvus数据 → 写入当前配置的Collection。
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAG_SYSTEM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENT_DIRECTORY = PROJECT_ROOT / "sample_data"

from logger import get_logger
from rag_system_learning_version.core.vector_retrieval.document_processor import (
    DocumentLoaderProcessor,
    DocumentSplitterProcessor,
)
from rag_system_learning_version.core.vector_retrieval.vector_service import (
    create_vector_service,
)


logger = get_logger(__name__)


def ingest_documents(document_directory=DEFAULT_DOCUMENT_DIRECTORY):
    """处理指定目录中的文档，并将生成的子块写入Milvus。"""
    document_directory = Path(document_directory).resolve()

    # 1. 将目录中的文件加载为统一的Document对象。
    loader = DocumentLoaderProcessor()
    documents = loader.load_documents(str(document_directory))
    if not documents:
        raise ValueError(f"目录中没有成功加载的文档: {document_directory}")

    # 2. 把原始Document依次切成父块和子块。
    splitter = DocumentSplitterProcessor()
    parent_chunks, child_chunks = splitter.split_documents(documents)

    # 3. 入库只需要Embedding和Milvus，不加载查询阶段的Reranker。
    vector_service = create_vector_service(load_reranker=False)
    try:
        result = vector_service.add_documents(parent_chunks, child_chunks)
    finally:
        vector_service.milvus_storage.close()

    logger.info(
        "完整入库流程结束: 原始Document=%d, 父块=%d, 子块=%d",
        len(documents),
        len(parent_chunks),
        len(child_chunks),
    )
    return result


def main():
    """单文件运行入口；需要换目录时修改上方默认路径即可。"""
    print(f"本次入库目录: {DEFAULT_DOCUMENT_DIRECTORY}")
    result = ingest_documents()
    print(f"Milvus写入结果: {result}")


if __name__ == "__main__":
    main()
