"""独立执行一次Milvus混合检索的示例脚本。

当前查询流程：查询编码 → Milvus稠密/稀疏混合检索
→ 父块去重 → Reranker重排序。各组件隐藏自己的第三方API格式，
VectorService只负责串联业务步骤。
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system_learning_version.core.vector_retrieval.vector_service import (
    create_vector_service,
)


def query_documents(query, file_path_filter=None):
    """执行一次查询并返回重排序后的父Document列表。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("查询内容不能为空")

    vector_service = create_vector_service(load_reranker=True)
    try:
        return vector_service.hybrid_search_with_rerank(
            query=query.strip(),
            file_path_filter=file_path_filter,
        )
    finally:
        vector_service.milvus_storage.close()


def print_results(results):
    """用便于学习和核对的格式打印查询结果。"""
    if not results:
        print("没有检索到相关父块。")
        return

    for rank, document in enumerate(results, start=1):
        print(f"\n===== 结果 {rank} =====")
        print(f"parent_id: {document.metadata.get('parent_id')}")
        print(f"file_path: {document.metadata.get('file_path')}")
        print(document.page_content)


def main():
    """单文件交互入口。"""
    query = input("请输入查询: ").strip()
    results = query_documents(query)
    print_results(results)


if __name__ == "__main__":
    main()
