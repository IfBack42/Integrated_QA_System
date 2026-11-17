"""HyDE策略：先生成假设答案，再用假设答案执行向量检索。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from rag_system.core.llm import LLMCallError, PromptLoader

logger = get_logger(__name__)


class HyDEStrategy:
    def __init__(
        self,
        vector_service,
        llm_client,
        prompt_loader=None,
        prompt_path=None,
        retrieval_k=None,
    ):
        self.vector_service = vector_service
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader or PromptLoader()
        self.prompt_path = prompt_path or (
            Path(__file__).resolve().parents[1] / "prompts" / "hyde.md"
        )
        self.retrieval_k = retrieval_k

    def retrieve(self, query, file_path_filter=None):
        prompt = self.prompt_loader.render(self.prompt_path, query=query)
        try:
            hypothetical_answer = self.llm_client.chat(prompt, temperature=0.2)
        except LLMCallError as exc:
            # HyDE只是检索增强步骤。生成失败时不能伪装成“知识库未命中”，
            # 记录故障后使用原始query退回直接检索；Milvus故障仍会继续抛出。
            logger.warning(
                "HyDE假设答案生成失败，退回原始query直接检索: %s",
                exc,
            )
            return self.vector_service.hybrid_search_with_rerank(
                query=query,
                k=self.retrieval_k,
                file_path_filter=file_path_filter,
            )

        logger.info("HyDE生成的假设答案: %s", hypothetical_answer)
        return self.vector_service.hybrid_search_with_rerank(
            query=hypothetical_answer,
            k=self.retrieval_k,
            file_path_filter=file_path_filter,
        )


if __name__ == "__main__":
    class FakeLLMClient:
        @staticmethod
        def chat(prompt, temperature=0.2):
            return "一段适合检索的假设答案"

    class FakeVectorService:
        @staticmethod
        def hybrid_search_with_rerank(**kwargs):
            return [kwargs["query"]]

    print(HyDEStrategy(FakeVectorService(), FakeLLMClient()).retrieve("测试问题"))
