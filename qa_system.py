"""项目级问答编排入口：先查询正式FAQ，未命中时进入RAG。"""

from config import config
from logger import get_logger
from faq_system.redis_client import RedisClient
from faq_system.mysql_client import MySQLClient
from faq_system.domain import FAQStatus
from faq_system.faq_service import FAQService
from domain import AnswerSource, QAResult, QAStatus
from rag_system.core.rag_service import RAGService, create_rag_service


logger = get_logger(__name__)


class QASystem:
    """创建项目依赖，并串联FAQ与RAG两条问答链路。"""

    # 构造函数依赖注入：测试时可以传入假的数据库和RAG服务。
    def __init__(
        self,
        settings=None,
        mysql_client=None,
        redis_client=None,
        rag_service=None,
    ):
        self.settings = settings or config
        self.mysql_client = None
        self.redis_client = None
        self.faq_service = None
        self.rag_service = None

        try:
            self.mysql_client = mysql_client or MySQLClient(self.settings)
            self.redis_client = redis_client or RedisClient(self.settings)

            # 启动时从MySQL生成权威FAQ快照。Redis版本相同则跳过；
            # 版本不同才全量刷新正式FAQ Hash文档及其搜索索引。
            snapshot = self.mysql_client.fetch_faq_snapshot()
            is_refreshed = self.redis_client.replace_official_faqs(
                documents=snapshot.documents,
                version=snapshot.version,
            )
            logger.debug(
                "正式FAQ启动同步完成，是否刷新Redis=%s",
                is_refreshed,
            )

            self.faq_service = FAQService(
                redis_client=self.redis_client,
                threshold=self.settings.FAQ_BM25_THRESHOLD,
                limit=self.settings.RETRIEVAL_K,
            )
            # 用户接受服务启动时一次性加载RAG模型，因此这里采用显式初始化；
            # 测试时可通过rag_service参数注入轻量假对象。
            self.rag_service = rag_service or create_rag_service(self.settings)
            logger.info("问答系统初始化完成")
        except Exception:
            self.close()
            raise

    def query(
        self,
        query,
        threshold=None,
        history=None,
        file_path_filter=None,
    ):
        """返回整套系统统一的QAResult；系统异常继续向调用方抛出。"""
        faq_result = self.faq_service.query(query, threshold=threshold)

        if faq_result.status == FAQStatus.INVALID:
            return QAResult(
                status=QAStatus.INVALID,
                message=faq_result.message,
            )

        if faq_result.is_hit:
            return QAResult(
                status=QAStatus.ANSWERED,
                answer=faq_result.answer,
                source=AnswerSource.FAQ,
                matched_question=faq_result.matched_question,
                score=faq_result.score,
            )

        clean_query = query.strip()
        # TODO 带历史或文件过滤条件时，相同query可能对应不同答案。
        # 在上下文指纹功能完成前，这类请求暂时绕过精确临时缓存。
        can_use_temporary_cache = not history and not file_path_filter
        if can_use_temporary_cache:
            cached_answer = self.redis_client.get_temporary_answer(clean_query)
            if cached_answer is not None:
                logger.info("FAQ未命中，命中Redis临时RAG答案")
                return QAResult(
                    status=QAStatus.ANSWERED,
                    answer=cached_answer,
                    source=AnswerSource.RAG_CACHE,
                )
        else:
            logger.debug("当前请求包含history或file_path_filter，跳过临时缓存")

        logger.info("FAQ未命中，开始进入RAG系统")
        answer = self.rag_service.answer(
            query=clean_query,
            history=history,
            file_path_filter=file_path_filter,
        )
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("RAGService返回了空答案")
        clean_answer = answer.strip()

        # RAGService抛出异常时，程序不会执行到这里；空答案也已在上面拒绝。
        # “知识库资料不足”属于正常拒答，不应作为可复用答案写入临时缓存。
        if (
            can_use_temporary_cache
            and clean_answer != RAGService.NO_CONTEXT_ANSWER
        ):
            self.redis_client.set_temporary_answer(
                query=clean_query,
                answer=clean_answer,
            )
        elif clean_answer == RAGService.NO_CONTEXT_ANSWER:
            logger.debug("RAG未获得可靠上下文，不写入Redis临时缓存")

        return QAResult(
            status=QAStatus.ANSWERED,
            answer=clean_answer,
            source=AnswerSource.RAG,
        )

    def close(self):
        """尽量释放全部资源；某一个关闭失败不阻断其他清理操作。"""
        resources = (
            ("RAG服务", self.rag_service),
            ("Redis连接", self.redis_client),
            ("MySQL连接", self.mysql_client),
        )
        for resource_name, resource in resources:
            if resource is None:
                continue
            try:
                resource.close()
            except Exception:
                logger.exception("%s关闭失败", resource_name)
