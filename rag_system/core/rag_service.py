"""RAG系统总业务入口。

数据流：query → 意图识别 → 通用回答或专业检索 → LLM生成最终答案。
本层负责串联业务，不实现向量查询、策略细节或底层API调用。
"""

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from logger import get_logger
from rag_system.core.llm import PromptLoader

logger = get_logger(__name__)


@dataclass(frozen=True)
class RAGTraceResult:
    """RAGService为评估模块提供的可观测结果。"""

    answer: str
    intent_label: str
    intent_confidence: float
    retrieved_contexts: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()


class RAGService:
    """向项目上层提供 ``answer()`` 一个完整RAG入口。"""

    GENERAL_LABEL = "通用知识"
    PROFESSIONAL_LABEL = "专业咨询"
    NO_CONTEXT_ANSWER = "当前知识库资料不足以回答该问题。"

    def __init__(
        self,
        query_classifier,
        retrieval_service,
        llm_client,
        prompt_loader=None,
        prompt_dir=None,
    ):
        self.query_classifier = query_classifier
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader or PromptLoader()
        self.prompt_dir = prompt_dir or (Path(__file__).resolve().parent / "llm" / "prompts")

    def answer(self, query, history=None, file_path_filter=None):
        """正常业务入口，只向上层返回最终答案字符串。"""
        return self.answer_with_trace(
            query=query,
            history=history,
            file_path_filter=file_path_filter,
        ).answer

    def answer_with_trace(self, query, history=None, file_path_filter=None):
        """执行完整RAG流程，并为离线评估额外返回意图和检索上下文。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query必须是非空字符串")
        clean_query = query.strip()
        history_text = self._format_history(history)

        label, confidence = self.query_classifier.classify(clean_query)
        logger.info(
            "RAG意图识别完成: label=%s, confidence=%.4f",
            label,
            confidence,
        )

        if label == self.GENERAL_LABEL:
            prompt = self.prompt_loader.render(
                self.prompt_dir / "general_answer.md",
                history=history_text,
                question=clean_query,
            )
            answer = self.llm_client.chat(prompt, temperature=0.2)
            return RAGTraceResult(
                answer=answer,
                intent_label=label,
                intent_confidence=float(confidence),
            )

        if label != self.PROFESSIONAL_LABEL:
            raise RuntimeError(f"意图分类器返回了未知类别: {label}")

        documents = self.retrieval_service.retrieve(
            query=clean_query,
            file_path_filter=file_path_filter,
        )
        if not documents:
            logger.info("专业问题未检索到可用文档")
            return RAGTraceResult(
                answer=self.NO_CONTEXT_ANSWER,
                intent_label=label,
                intent_confidence=float(confidence),
            )

        context = self._format_context(documents)
        prompt = self.prompt_loader.render(
            self.prompt_dir / "rag_answer.md",
            history=history_text,
            context=context,
            question=clean_query,
        )
        answer = self.llm_client.chat(prompt, temperature=0.1)
        return RAGTraceResult(
            answer=answer,
            intent_label=label,
            intent_confidence=float(confidence),
            retrieved_contexts=tuple(
                document.page_content for document in documents
            ),
            source_files=tuple(
                document.metadata.get("file_path") or "未知来源"
                for document in documents
            ),
        )

    @staticmethod
    def _format_history(history):
        """把最近5轮合法历史整理为提示词文本。"""
        if history is None:
            return "无"
        if not isinstance(history, list):
            raise TypeError("history必须是列表或None")

        lines = []
        for item in history[-5:]:
            if not isinstance(item, dict):
                raise ValueError("history中的每一项必须是字典")
            question = item.get("question")
            answer = item.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError("history每一项必须包含字符串question和answer")
            lines.append(f"用户：{question}\n助手：{answer}")
        return "\n\n".join(lines) if lines else "无"

    @staticmethod
    def _format_context(documents):
        """把Document列表整理成带编号和来源的最终上下文。"""
        sections = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("file_path") or "未知来源"
            sections.append(
                f"[资料{index} | 来源：{source}]\n{document.page_content}"
            )
        return "\n\n".join(sections)

    def close(self):
        """统一释放RAG检索链路持有的资源。"""
        try:
            self.retrieval_service.close()
        finally:
            self.llm_client.close()


def create_rag_service(settings=None, device=None):
    """装配RAG依赖；离线评估可通过device='cpu'避免占用在线服务显存。"""
    # 真实模型和数据库依赖只在正式装配时导入，单元测试导入RAGService时
    # 不会无意义地加载BERT、Embedding和Reranker相关模块。
    from rag_system.core.intent_classifier import QueryClassifier
    from rag_system.core.llm import LLMClient
    from rag_system.core.retrieval_service import RetrievalService
    from rag_system.core.vector_retrieval.vector_service import create_vector_service

    settings = settings or config
    vector_service = None
    llm_client = None
    try:
        llm_client = LLMClient(settings=settings)
        query_classifier = QueryClassifier(device=device)
        vector_service = create_vector_service(
            load_reranker=True,
            device=device,
        )
        retrieval_service = RetrievalService(
            vector_service=vector_service,
            llm_client=llm_client,
            settings=settings,
        )
        return RAGService(
            query_classifier=query_classifier,
            retrieval_service=retrieval_service,
            llm_client=llm_client,
        )
    except Exception:
        if vector_service is not None:
            vector_service.close()
        if llm_client is not None:
            llm_client.close()
        raise


if __name__ == "__main__":
    from langchain_core.documents import Document

    class FakeClassifier:
        @staticmethod
        def classify(query):
            return "专业咨询", 0.99

    class FakeRetrievalService:
        @staticmethod
        def retrieve(query, file_path_filter=None):
            return [
                Document(
                    page_content="测试知识库内容",
                    metadata={"file_path": "测试.txt"},
                )
            ]

        @staticmethod
        def close():
            pass

    class FakeLLMClient:
        @staticmethod
        def chat(prompt, temperature=0.1):
            return "根据测试资料生成的回答"

        @staticmethod
        def close():
            pass

    service = RAGService(
        query_classifier=FakeClassifier(),
        retrieval_service=FakeRetrievalService(),
        llm_client=FakeLLMClient(),
    )
    print(service.answer("测试专业问题"))
