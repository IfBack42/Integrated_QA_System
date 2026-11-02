"""项目级问答编排入口。后续在这里接入RAGService。"""

from config import config
from logger import get_logger
from faq_system.redis_client import RedisClient
from faq_system.mysql_client import MySQLClient
from faq_system.domain import FAQStatus
from faq_system.faq_service import FAQService


logger = get_logger(__name__)


class QASystem:
    """创建基础设施并决定FAQ未命中后的下一步。"""
                    # 构造函数依赖注入
    def __init__(self, settings=None, mysql_client=None, redis_client=None):
        self.settings = settings or config
        self.mysql_client = None
        self.redis_client = None

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
            logger.info("FAQ系统初始化完成")
        except Exception:
            self.close()
            raise
    # TODO 此处整合faq和rag
    def query(self, query, threshold=None):
        result = self.faq_service.query(query, threshold=threshold)
        if result.status == FAQStatus.MISS:
            logger.info("FAQ未命中，后续应进入RAG系统")
        return result

    def close(self):
        """统一释放项目持有的数据库连接。"""
        if self.redis_client is not None:
            self.redis_client.close()
        if self.mysql_client is not None:
            self.mysql_client.close()
