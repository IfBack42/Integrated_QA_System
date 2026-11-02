"""独立验证Redis Search是否能够索引并检索中文FAQ。

直接运行本文件即可。它不会读取MySQL，也不会修改正式FAQ数据，只会操作
``verify:faq:*`` 前缀下的测试数据和 ``idx:verify:faq`` 测试索引。
"""

import sys
from pathlib import Path

import redis
from redis.commands.search.field import TagField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError


# 允许直接运行当前文件时导入项目根目录下的config.py。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config


TEST_INDEX_NAME = "idx:verify:faq"
TEST_KEY_PREFIX = "verify:faq:doc:"

# True：验证结束后删除测试索引和测试文档；False：保留数据，方便在Redis中查看。
CLEANUP_AFTER_TEST = True


TEST_DOCUMENTS = {
    f"{TEST_KEY_PREFIX}1": {
        "qa_id": "1",
        "subject_name": "运维",
        "question": "安装VMware Tools时菜单显示灰色如何解决",
        "answer": "关闭虚拟机，将CD/DVD和软盘改为自动检测，再重新启动虚拟机。",
    },
    f"{TEST_KEY_PREFIX}2": {
        "qa_id": "2",
        "subject_name": "Python",
        "question": "Python列表和元组有什么区别",
        "answer": "列表可以修改，元组创建后通常不可修改。",
    },
    f"{TEST_KEY_PREFIX}3": {
        "qa_id": "3",
        "subject_name": "Redis",
        "question": "Redis缓存有什么作用",
        "answer": "缓存常用数据可以减少后端数据库访问并缩短响应时间。",
    },
}


def create_client() -> redis.Redis:
    """根据项目配置创建Redis连接，并真正检查服务是否可用。"""
    client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD,
        db=config.REDIS_DB,
        decode_responses=True,
    )
    client.ping()
    return client


def remove_test_data(client: redis.Redis) -> None:
    """只删除本文件创建的测试索引和测试文档。"""
    try:
        client.ft(TEST_INDEX_NAME).dropindex(delete_documents=True)
    except ResponseError as exc:
        # 索引第一次运行时本来就不存在，这种情况不算失败。
        error_message = str(exc).lower()
        index_not_found = (
            "unknown index name" in error_message
            or "no such index" in error_message
        )
        if not index_not_found:
            raise

    test_keys = list(client.scan_iter(match=f"{TEST_KEY_PREFIX}*"))
    if test_keys:
        client.delete(*test_keys)


def create_test_index(client: redis.Redis) -> None:
    """创建只索引指定测试前缀的中文Hash全文索引。
        index = schema + definition
    """
    schema = ( # schema:指定字段建立倒排索引
        TagField("qa_id"),
        TagField("subject_name"),
        TextField("question", weight=5.0),
    )
    definition = IndexDefinition( # definition:规定索引应用范围
        prefix=[TEST_KEY_PREFIX],
        index_type=IndexType.HASH,
        language="chinese",
    )
    client.ft(TEST_INDEX_NAME).create_index(schema, definition=definition)


def write_test_documents(client: redis.Redis) -> None:
    """把三条FAQ作为Redis Hash写入；符合前缀的文档会被自动索引。"""
    pipeline = client.pipeline(transaction=True) # transaction：使用事务方式执行这一批命令
    for key, document in TEST_DOCUMENTS.items():
        pipeline.hset(key, mapping=document)
    pipeline.execute()


def search_test_question(client: redis.Redis) -> None:
    """执行一次中文BM25检索并打印排名结果。"""
    user_query = "安装 VMware Tools"
    query = (
        Query(f"@question:({user_query})")
        .language("chinese")
        .scorer("BM25STD")
        .with_scores()
        .return_fields("qa_id", "subject_name", "question", "answer")
        .paging(0, 3)
        .dialect(2)
    )

    result = client.ft(TEST_INDEX_NAME).search(query)
    print(f"查询文本: {user_query}")
    print(f"命中文档数量: {result.total}")

    for position, document in enumerate(result.docs, start=1):
        print(f"\n第{position}名")
        print(f"Redis键: {document.id}")
        print(f"BM25分数: {getattr(document, 'score', '未返回')}")
        print(f"标准问题: {document.question}")
        print(f"答案: {document.answer}")

    if result.total == 0:
        raise RuntimeError("Redis Search没有检索到预期的中文FAQ")

    first_document = result.docs[0]
    if first_document.id != f"{TEST_KEY_PREFIX}1":
        raise RuntimeError("检索结果存在，但第一名不是预期的VMware FAQ")


def main() -> None:
    client = None
    try:
        print("1. 连接Redis并检查Search能力")
        client = create_client()
        result = client.execute_command("FT._LIST") #检查：Redis Search 是否可用；当前有哪些搜索索引。
        print(f"现有索引：{result}")

        print("2. 清理可能残留的同名测试数据")
        remove_test_data(client)

        print("3. 创建中文FAQ测试索引")
        create_test_index(client)
        result = client.execute_command("FT._LIST") #检查：Redis Search 是否可用；当前有哪些搜索索引。
        print(f"现有索引：{result}")

        print("4. 写入三条测试FAQ文档")
        write_test_documents(client)

        print("5. 使用Redis Search执行BM25检索")
        search_test_question(client)

        print("\nRedis Search中文FAQ验证成功")
    except redis.RedisError as exc:
        print(f"\nRedis或Redis Search验证失败: {exc}")
        raise
    finally:
        if client is not None:
            if CLEANUP_AFTER_TEST:
                remove_test_data(client)
                print("测试索引和测试文档已清理")
            client.close()


if __name__ == "__main__":
    main()
