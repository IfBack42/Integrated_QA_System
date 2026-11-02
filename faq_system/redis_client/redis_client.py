# 该脚本用于: Redis缓存技术.

# 导包
import sys
from pathlib import Path

import redis
import re

from redis.commands.search.field import TagField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config
from faq_system.domain import FAQDocument, RetrievalCandidate
from logger import get_logger


logger = get_logger(__name__)


# todo 1.定义Redis客户端类 -> 封装Redis连接, 数据存储, 数据获取, 答案查询等功能.
class RedisClient:
    # todo 1.1 初始化方法, 获取Redis连接, 初始化日志, 处理连接异常.
    def __init__(
        self,
        settings=None,
        temporary_answer_key_prefix=None,
        faq_index_name=None,
        faq_document_key_prefix=None,
        faq_version_key=None,
    ):
        self.settings = settings or config
        self.temporary_answer_key_prefix = (
            temporary_answer_key_prefix
            or self.settings.REDIS_TEMPORARY_ANSWER_KEY_PREFIX
        )
        self.faq_index_name = (
            faq_index_name or self.settings.REDIS_FAQ_INDEX_NAME
        )
        self.faq_document_key_prefix = (
            faq_document_key_prefix
            or self.settings.REDIS_FAQ_DOCUMENT_KEY_PREFIX
        )
        self.faq_version_key = (
            faq_version_key or self.settings.REDIS_FAQ_VERSION_KEY
        )
        # 建立Redis连接.
        try:
            self.client = redis.StrictRedis(
                host=self.settings.REDIS_HOST,           # Redis数据库主机地址
                port=self.settings.REDIS_PORT,           # Redis数据库端口号
                password=self.settings.REDIS_PASSWORD,   # Redis数据库密码
                db=self.settings.REDIS_DB,               # Redis数据库编号
                decode_responses=True               # 自动将返回的字节数据转换为字符串(避免手动解码)
            )
            # 创建客户端并不会立刻访问Redis，ping()才会真正检测连接是否可用.
            self.client.ping()
            logger.info('Redis连接成功')
        except redis.RedisError as e:
            logger.error(f'Redis连接异常: {e}')
            raise   # 声明抛出异常.


    def _temporary_answer_key(self, query):
        """拼接RAG生成但尚未审核的临时答案Key。"""
        return f"{self.temporary_answer_key_prefix}{query}"


    def _faq_document_key(self, qa_id):
        """拼接生成权威数据索引键"""
        return f"{self.faq_document_key_prefix}{qa_id}"


    def get_redis_info(self, pattern="*"):
        """统一返回符合pattern的普通Redis Key和全部Search索引名称。"""
        return {
            "keys": sorted(self.client.scan_iter(match=pattern)),
            "search_indexes": list(
                self.client.execute_command("FT._LIST")
            ),
        }


    def create_faq_index(self):
        """确保正式FAQ搜索索引存在；新建时返回True，已存在返回False。"""
        index_names = list(self.client.execute_command("FT._LIST"))
        if self.faq_index_name in index_names:
            return False

        schema = (
            TagField("qa_id"),
            TagField("subject_name"),
            TextField("question", weight=5.0),
        )
        definition = IndexDefinition(
            prefix=[self.faq_document_key_prefix],
            index_type=IndexType.HASH,
            language="chinese",
        )
        self.client.ft(self.faq_index_name).create_index(
            schema,
            definition=definition,
        )
        logger.info("Redis Search正式FAQ索引创建成功: %s", self.faq_index_name)
        return True


    def replace_official_faqs(self, documents, version):
        """版本不同时替换Redis正式FAQ副本；版本相同时直接跳过。"""
        documents = list(documents)
        if not documents:
            raise ValueError("拒绝使用空FAQ列表覆盖Redis正式副本")
        if not all(isinstance(item, FAQDocument) for item in documents):
            raise TypeError("documents中的每一项都必须是FAQDocument")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("权威FAQ版本必须是非空字符串")

        self.create_faq_index()
        # MySQL快照版本与Redis保存的版本一致，说明正式副本无需刷新。
        if self.client.get(self.faq_version_key) == version:
            logger.debug("Redis正式FAQ已是最新版本: %s", version)
            return False

        old_keys = list(
            self.client.scan_iter(
                match=f"{self.faq_document_key_prefix}*"
            )
        )

        pipeline = self.client.pipeline(transaction=True)
        if old_keys:
            pipeline.delete(*old_keys)
        for document in documents:
            pipeline.hset(
                self._faq_document_key(document.qa_id),
                mapping=document.to_mapping(),
            )
        pipeline.execute()
        # 文档批量写入没有抛出异常后，再把Redis标记为这个内容版本。
        # 如果进程在这里之前中断，旧版本会促使下次启动重新同步。
        self.client.set(self.faq_version_key, version)

        logger.info(
            "Redis正式FAQ副本刷新成功，版本=%s，共%d条",
            version,
            len(documents),
        )
        return True


    @staticmethod
    def _escape_search_text(text):
        """转义可能被Redis Search当成查询语法的标点。"""
        special_chars = re.compile(
            r"([\\,\.<>\{\}\[\]\"'`:;!@#$%^&*()\-+=~|])"
        )
        return special_chars.sub(r"\\\1", text)


    def search_faq(self, query_terms, limit=3):
        """执行正式FAQ全文检索，返回按BM25得分排序的候选文档。"""
        if not isinstance(query_terms, (list, tuple)):
            raise TypeError("FAQ查询词必须使用列表或元组传入")
        if limit <= 0:
            raise ValueError("FAQ检索数量limit必须大于0")

        escaped_terms = [
            self._escape_search_text(str(term).strip())
            for term in query_terms
            if str(term).strip()
        ]
        if not escaped_terms:
            raise ValueError("FAQ查询词不能为空")

        # “|”是Redis Search查询语法中的OR：任意词命中即可进入候选集，
        # 再由BM25根据各个词的贡献计算总分并完成排序。
        query_expression = "|".join(escaped_terms)
        logger.debug(
            "开始执行Redis Search FAQ查询，terms=%r，limit=%d",
            escaped_terms,
            limit,
        )
        query = (
            # 只在Schema定义的question全文字段中搜索用户文本。
            Query(f"@question:({query_expression})")
            # 使用中文查询分析规则处理本次输入。
            .language("chinese")
            # 使用BM25STD计算候选FAQ的相关性得分并排序。
            .scorer("BM25STD")
            # 要求Redis Search把每篇候选文档的得分一并返回。
            .with_scores()
            # question参与检索；answer虽然不参与索引，但命中后可以返回。
            .return_fields("qa_id", "subject_name", "question", "answer")
            # 从第0条开始，最多返回limit条候选，也就是Top-K。
            .paging(0, limit)
            # 使用Redis Search第2版查询语法。
            .dialect(2)
        )
        result = self.client.ft(self.faq_index_name).search(query)
        candidates = [
            RetrievalCandidate(
                qa_id=getattr(document, "qa_id", None),
                question=getattr(document, "question", None),
                answer=getattr(document, "answer", None),
                score=float(document.score),
            )
            for document in result.docs
        ]
        logger.debug(
            "Redis Search FAQ查询完成，返回候选数=%d",
            len(candidates),
        )
        return candidates


    def _drop_faq_index(self, delete_documents=False):
        """删除当前客户端指向的FAQ索引；主要用于独立测试清理。"""
        is_dropped = False
        try:
            self.client.ft(self.faq_index_name).dropindex(
                delete_documents=delete_documents
            )
            is_dropped = True
        except ResponseError as exc:
            message = str(exc).lower()
            index_not_found = (
                "unknown index name" in message
                or "no such index" in message
            )
            if not index_not_found:
                raise

        if delete_documents:
            remaining_keys = list(
                self.client.scan_iter(
                    match=f"{self.faq_document_key_prefix}*"
                )
            )
            if remaining_keys:
                self.client.delete(*remaining_keys)
            self.client.delete(self.faq_version_key)
        return is_dropped


    def close(self):
        try:
            self.client.close()
            logger.info("Redis连接已关闭")
        except redis.RedisError as e:
            logger.error(f"Redis连接关闭失败: {e}")


def main():
    """使用独立测试索引验证创建、写入和检索，结束后自动清理。"""
    test_client = None
    try:
        test_client = RedisClient(
            faq_index_name="idx:test:faq:redis-client",
            faq_document_key_prefix="test:faq:redis-client:",
            faq_version_key="test:faq:meta:redis-client:version",
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

        is_new_created = test_client.create_faq_index()
        is_synchronized = test_client.replace_official_faqs(
            documents,
            version="test-content-version-1",
        )
        is_second_sync_skipped = not test_client.replace_official_faqs(
            documents,
            version="test-content-version-1",
        )
        matches = test_client.search_faq(["安装", "VMware", "Tools"])
        redis_info = test_client.get_redis_info("test:faq:*")

        print(f"是否新建测试索引: {is_new_created}")
        print(f"首次是否执行同步: {is_synchronized}")
        print(f"相同版本再次同步是否跳过: {is_second_sync_skipped}")

        print(f"测试Redis Key: {redis_info['keys']}")
        print(f"Search索引: {redis_info['search_indexes']}")

        print("检索结果:")
        for match in matches:
            print(match)
    finally:
        if test_client is not None:
            test_client._drop_faq_index(delete_documents=True)
            test_client.close()


if __name__ == '__main__':
    # main()

    client = RedisClient(
            faq_index_name="idx:faq:official",
            faq_document_key_prefix="faq:official:",
            faq_version_key="faq:meta:official:version",
        )
    redis_info = client.get_redis_info("faq:official:*")
    print(f"测试Redis Key: {redis_info['keys']}",end="\n"*5)
    print(f"Search索引: {redis_info['search_indexes']}")

