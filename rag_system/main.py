"""RAG交互式查询入口。

本文件只负责输入输出、创建RAGService和关闭资源。
文档入库请单独运行 ``core/vector_retrieval/ingest_documents.py``。
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from rag_system.core.rag_service import create_rag_service

logger = get_logger(__name__)


def main(file_path_filter=None):
    """启动交互式RAG查询；可选限制为某个完整文件路径。"""
    rag_service = None
    history = []
    try:
        rag_service = create_rag_service()
        time.sleep(0.25)
        print("\n欢迎使用RAG问答系统")
        print("输入问题进行查询，输入 q 退出")

        while True:
            query = input("\n请输入问题: ").strip()
            if query.lower() == "q":
                print("再见")
                break
            if not query:
                print("问题不能为空，请重新输入")
                continue

            try:
                answer = rag_service.answer(
                    query=query,
                    history=history,
                    file_path_filter=file_path_filter,
                )
                print(f"答案: {answer}")
                history.append({"question": query, "answer": answer})
                history = history[-5:]
            except Exception:
                logger.exception("RAG查询失败")
                print("系统暂时无法完成查询，请查看日志")
    except Exception:
        logger.exception("RAG系统启动失败")
        print("RAG系统启动失败，请查看日志")
    finally:
        if rag_service is not None:
            rag_service.close()


if __name__ == "__main__":
    main()
