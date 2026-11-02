"""编排一次正式FAQ查询，不负责数据库底层操作或RAG决策。"""

from logger import get_logger
from faq_system.domain import FAQResult, FAQStatus
from faq_system.retrieval import RedisSearchRetriever


logger = get_logger(__name__)


class FAQService:
    """调用Retriever完成FAQ检索，并统一包装业务返回结果。"""

    def __init__(
        self,
        redis_client,
        threshold,
        limit=3,
        retriever=None,
    ):
        self.retriever = retriever or RedisSearchRetriever(
            redis_client=redis_client,
            threshold=threshold,
            limit=limit,
        )

    def query(self, query, threshold=None):
        """返回FAQ命中、未命中或无效输入；基础设施异常继续向上抛出。"""
        if not isinstance(query, str) or not query.strip():
            logger.debug("FAQ查询输入无效")
            return FAQResult(
                status=FAQStatus.INVALID,
                message="查询内容必须是非空字符串",
            )

        match = self.retriever.retrieve(
            query.strip(),
            threshold=threshold,
        )
        if match is None:
            logger.debug("FAQService查询结果: MISS")
            return FAQResult(status=FAQStatus.MISS)

        logger.debug(
            "FAQService查询结果: FAQ_HIT，qa_id=%s",
            match.qa_id,
        )
        return FAQResult(
            status=FAQStatus.FAQ_HIT,
            answer=match.answer,
            matched_question=match.question,
            score=match.score,
        )


def main():
    """使用假的Retriever独立验证FAQService的流程编排。"""
    from faq_system.domain import RetrievalMatch

    class FakeRetriever:
        def retrieve(self, query, threshold=None):
            if query == "命中问题":
                return RetrievalMatch(
                    qa_id="1",
                    question="标准问题",
                    answer="标准答案",
                    score=3.0,
                )
            return None

    service = FAQService(
        redis_client=None,
        threshold=0.85,
        retriever=FakeRetriever(),
    )
    print(service.query("命中问题"))
    print(service.query("未命中问题"))
    print(service.query(""))


if __name__ == "__main__":
    main()
