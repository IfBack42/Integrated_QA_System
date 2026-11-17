"""回溯问题策略：同时检索原问题与其依赖的背景问题。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from rag_system.core.llm import LLMCallError, PromptLoader

logger = get_logger(__name__)


class StepBackStrategy:
    def __init__(
        self,
        vector_service,
        llm_client,
        prompt_loader=None,
        prompt_path=None,
        retrieval_k=None,
        final_limit=None,
    ):
        self.vector_service = vector_service
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader or PromptLoader()
        self.prompt_path = prompt_path or (
            Path(__file__).resolve().parents[1] / "prompts" / "step_back.md"
        )
        self.retrieval_k = retrieval_k
        self.final_limit = final_limit

    def retrieve(self, query, file_path_filter=None):
        prompt = self.prompt_loader.render(self.prompt_path, query=query)
        try:
            background_query = self.llm_client.chat(prompt, temperature=0.1)
        except LLMCallError as exc:
            logger.warning(
                "回溯问题生成失败，退回原始query直接检索: %s",
                exc,
            )
            return self.vector_service.hybrid_search_with_rerank(
                query=query,
                k=self.retrieval_k,
                file_path_filter=file_path_filter,
            )

        documents = []
        for search_query in (query, background_query):
            documents.extend(
                self.vector_service.hybrid_search_with_rerank(
                    query=search_query,
                    k=self.retrieval_k,
                    file_path_filter=file_path_filter,
                    final_limit=self.retrieval_k,
                )
            )

        unique_documents = self._deduplicate(documents)
        if not unique_documents:
            return []
        return self.vector_service.rerank_documents(
            query=query,
            documents=unique_documents,
            limit=self.final_limit,
        )

    @staticmethod
    def _deduplicate(documents):
        unique = []
        seen = set()
        for document in documents:
            identity = document.metadata.get("parent_id") or document.page_content
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(document)
        return unique


if __name__ == "__main__":
    from langchain_core.documents import Document

    class FakeLLMClient:
        @staticmethod
        def chat(prompt, temperature=0.1):
            return "这个问题依赖的基础概念是什么？"

    class FakeVectorService:
        @staticmethod
        def hybrid_search_with_rerank(query, **kwargs):
            return [Document(page_content=query, metadata={"parent_id": query})]

        @staticmethod
        def rerank_documents(query, documents, limit=None):
            return documents[:limit]

    strategy = StepBackStrategy(
        FakeVectorService(), FakeLLMClient(), retrieval_k=3, final_limit=2
    )
    print(strategy.retrieve("测试问题"))
