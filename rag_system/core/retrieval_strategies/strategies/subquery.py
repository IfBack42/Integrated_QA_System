"""子查询策略：拆分复杂问题，多轮检索后按原问题全局重排。"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from rag_system.core.llm import LLMCallError, PromptLoader

logger = get_logger(__name__)


class SubqueryStrategy:
    def __init__(
        self,
        vector_service,
        llm_client,
        prompt_loader=None,
        prompt_path=None,
        retrieval_k=None,
        final_limit=None,
        max_subqueries=3,
    ):
        self.vector_service = vector_service
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader or PromptLoader()
        self.prompt_path = prompt_path or (
            Path(__file__).resolve().parents[1] / "prompts" / "subquery.md"
        )
        self.retrieval_k = retrieval_k
        self.final_limit = final_limit
        self.max_subqueries = max_subqueries

    def retrieve(self, query, file_path_filter=None):
        prompt = self.prompt_loader.render(self.prompt_path, query=query)
        try:
            raw_subqueries = self.llm_client.chat(prompt, temperature=0.1)
            subqueries = self._parse_subqueries(raw_subqueries)
        except LLMCallError as exc:
            logger.warning(
                "子查询生成失败，退回原始query直接检索: %s",
                exc,
            )
            subqueries = [query]

        if not subqueries:
            logger.warning("LLM未生成有效子查询，退回原始query直接检索")
            subqueries = [query]

        all_documents = []
        for subquery in subqueries:
            documents = self.vector_service.hybrid_search_with_rerank(
                query=subquery,
                k=self.retrieval_k,
                file_path_filter=file_path_filter,
                final_limit=self.retrieval_k,
            )
            all_documents.extend(documents)

        unique_documents = self._deduplicate(all_documents)
        if not unique_documents:
            return []

        # 各子查询的局部得分不能直接互相比较，最后必须按用户原问题全局重排。
        return self.vector_service.rerank_documents(
            query=query,
            documents=unique_documents,
            limit=self.final_limit,
        )

    def _parse_subqueries(self, raw_text):
        """清理模型偶尔输出的编号和项目符号，并限制子查询数量。"""
        subqueries = []
        for line in raw_text.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.、)])\s*", "", line).strip()
            if cleaned and cleaned not in subqueries:
                subqueries.append(cleaned)
            if len(subqueries) >= self.max_subqueries:
                break
        return subqueries

    @staticmethod
    def _deduplicate(documents):
        """优先使用parent_id去重，缺少ID时退回正文去重。"""
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
            return "1. 第一部分是什么\n2. 第二部分是什么"

    class FakeVectorService:
        @staticmethod
        def hybrid_search_with_rerank(query, **kwargs):
            return [Document(page_content=query, metadata={"parent_id": query})]

        @staticmethod
        def rerank_documents(query, documents, limit=None):
            return documents[:limit]

    strategy = SubqueryStrategy(
        FakeVectorService(), FakeLLMClient(), retrieval_k=3, final_limit=2
    )
    print(strategy.retrieve("比较两个部分"))
