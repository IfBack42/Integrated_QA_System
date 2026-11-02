"""Redis Search FAQ检索器：编排搜索并判断候选是否可靠命中。"""

from logger import get_logger
from faq_system.domain import FAQDocument, RetrievalMatch
from faq_system.utils.preprocess import preprocess_text


logger = get_logger(__name__)


class RedisSearchRetriever:
    """调用Redis Search检索正式FAQ，并对最佳候选执行阈值判断。"""

    def __init__(self, redis_client, threshold, limit=3):
        if threshold < 0:
            raise ValueError("FAQ BM25匹配阈值不能小于0")
        if limit <= 0:
            raise ValueError("FAQ候选数量limit必须大于0")

        self.redis_client = redis_client
        self.threshold = float(threshold)
        self.limit = limit

    def retrieve(self, query, threshold=None):
        """检索最佳FAQ；最高分未达到阈值时返回None。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("FAQ检索文本必须是非空字符串")

        effective_threshold = (
            self.threshold if threshold is None else float(threshold)
        )
        if effective_threshold < 0:
            raise ValueError("FAQ BM25匹配阈值不能小于0")

        # Python只对当前query分词，用于构造“任意词命中”的OR查询。
        # FAQ语料分词、倒排索引和BM25评分仍由Redis Search维护和执行。
        query_tokens = [
            token.strip()
            for token in preprocess_text(query)
            if token.strip()
        ]
        if not query_tokens:
            return None

        candidates = self.redis_client.search_faq(
            query_tokens,
            limit=self.limit,
        )
        if not candidates:
            logger.debug("Redis Search未返回FAQ候选")
            return None

        best_candidate = candidates[0]
        best_score = best_candidate.score
        logger.debug(
            "Redis Search最佳候选：qa_id=%s，question=%r，"
            "BM25得分=%.3f，最低命中分=%.3f",
            best_candidate.qa_id,
            best_candidate.question,
            best_score,
            effective_threshold,
        )
        if best_score < effective_threshold:
            logger.debug(
                "FAQ未可靠命中，最高BM25得分: %.3f，阈值: %.3f",
                best_score,
                effective_threshold,
            )
            return None

        logger.debug("FAQ可靠命中，qa_id=%s", best_candidate.qa_id)
        return RetrievalMatch(
            qa_id=best_candidate.qa_id,
            question=best_candidate.question,
            answer=best_candidate.answer,
            score=best_score,
        )




def main():
    """使用独立测试索引验证Retriever命中与未命中流程。"""
    from faq_system.redis_client.redis_client import RedisClient

    test_client = None
    try:
        test_client = RedisClient(
            faq_index_name="idx:test:faq:retriever",
            faq_document_key_prefix="test:faq:retriever:",
            faq_version_key="test:faq:meta:retriever:version",
        )
        test_client._drop_faq_index(delete_documents=True)

        documents = [
            FAQDocument(
                qa_id="1",
                subject_name="运维",
                question="安装VMware Tools时显示灰色如何解决",
                answer="关闭虚拟机并调整虚拟设备设置。",
            ),
            FAQDocument(
                qa_id="2",
                subject_name="Python",
                question="Python列表和元组有什么区别",
                answer="列表可以修改，元组通常不可修改。",
            ),
        ]
        test_client.replace_official_faqs(
            documents,
            version="test-retriever-version-1",
        )

        retriever = RedisSearchRetriever(
            redis_client=test_client,
            threshold=0.85,
            limit=3,
        )
        match = retriever.retrieve("安装 VMware Tools")
        miss = retriever.retrieve("量子宇宙飞船天气")

        assert match is not None
        assert match.qa_id == "1"
        assert match.answer == "关闭虚拟机并调整虚拟设备设置。"
        assert miss is None

        print("Retriever命中结果:")
        print(match)
        print(f"无关问题检索结果: {miss}")
    finally:
        if test_client is not None:
            test_client._drop_faq_index(delete_documents=True)
            test_client.close()


if __name__ == "__main__":
    main()
