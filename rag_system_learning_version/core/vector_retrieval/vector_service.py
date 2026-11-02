"""Milvus向量业务服务。

职责：组合EmbeddingModel、RerankerModel和MilvusStorage，向上层提供
文档录入与混合检索两个业务入口。
不负责：加载/切分原始文件、管理Milvus连接细节或生成最终LLM答案。

录入数据流：父块与子块 → 编码 → 组装Milvus data → 写入Collection。
查询数据流：query编码 → 稠密/稀疏混合检索 → 父块去重 → 重排序。

说明：查询算法暂时沿用教学版本，本轮只适配新的模型对象、客户端对象和Schema字段。
"""

import sys
from pathlib import Path

# 允许本文件直接运行时找到项目级config和logger包。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from logger import get_logger
from rag_system_learning_version.core.vector_retrieval.db.milvus_client import MilvusStorage
from rag_system_learning_version.core.vector_retrieval.models.embedding_model import (
    EmbeddingModel,
)
from rag_system_learning_version.core.vector_retrieval.models.reranker_model import (
    RerankerModel,
)

# 初始化日志实例.
logger = get_logger(__name__)

from langchain_core.documents import Document # 用于创建文档对象, 统一文档数据格式(含内容 + 元数据)


class VectorService:
    """整合向量编码、Milvus读写和候选重排序。"""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        milvus_storage: MilvusStorage,
        reranker_model: RerankerModel | None = None,
    ):
        self.embedding_model = embedding_model
        self.milvus_storage = milvus_storage
        self.reranker_model = reranker_model

    # 将父子块编码并组装成data，再交给MilvusStorage持久化.
    def add_documents(self, parent_chunks, child_chunks):
        """将父子Document加工为Collection所需数据并写入Milvus。"""
        if not child_chunks:
            logger.info("没有待入库的子块，跳过Milvus写入")
            return None

        # 1. 父块只保存一份在当前映射中；子块通过parent_id领取对应正文。
        parent_content_by_id = {
            parent_doc.metadata["parent_id"]: parent_doc.page_content
            for parent_doc in parent_chunks
        }

        # 2. 一次批量编码全部子块，得到稠密向量和稀疏向量。
        child_texts = [child_doc.page_content for child_doc in child_chunks]
        embeddings = self.embedding_model.encode(child_texts)
        if (
            len(embeddings["dense"]) != len(child_chunks)
            or len(embeddings["sparse"]) != len(child_chunks)
        ):
            raise RuntimeError("编码向量数量与待入库子块数量不一致")

        # 3. 将Document和模型输出整理成与Milvus Schema一致的data。
        data = []
        for index, child_doc in enumerate(child_chunks):
            metadata = child_doc.metadata
            parent_id = metadata["parent_id"]
            if parent_id not in parent_content_by_id:
                raise ValueError(
                    f"子块 {metadata.get('child_id', '<missing>')} "
                    f"找不到对应父块: {parent_id}"
                )

            # EmbeddingModel已经统一整理好Python list与dict，不泄露NumPy/CSR细节。
            dense_vector = embeddings["dense"][index]
            sparse_vector = embeddings["sparse"][index]

            data.append(
                {
                    "child_id": metadata["child_id"],
                    "child_text": child_doc.page_content,
                    "dense_vector": dense_vector,
                    "sparse_vector": sparse_vector,
                    "parent_id": parent_id,
                    "parent_content": parent_content_by_id[parent_id],
                    "file_path": metadata["file_path"],
                    "timestamp": metadata["timestamp"],
                }
            )

        # 4. 业务层只把整理好的data交给底层客户端。
        return self.milvus_storage.add_documents(data)



    # todo 3.4 定义函数 -> 混合检索(稠密 + 稀疏)  + 重排序, 返回精准父文档.
    def hybrid_search_with_rerank(
        self,
        query,
        k=None,
        file_path_filter=None,
    ):
        """
        该函数用于执行: 混合检索(稠密 + 稀疏向量) + 结果重排序, 返回精准父文档
        :param query: 用户查询文本, 例如: 'AI学科的课程内容是什么'
        :param k: 混合检索返回的TopK子块数量
        :param file_path_filter: 文件路径过滤条件, 默认None不过滤.
        :return: 重排序后的Top-M个父文档列表(M是从conf中获取的)
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query必须是非空字符串")
        effective_k = config.RETRIEVAL_K if k is None else k
        if (
            not isinstance(effective_k, int)
            or isinstance(effective_k, bool)
            or effective_k <= 0
        ):
            raise ValueError("k必须是正整数")

        # 1. EmbeddingModel负责隐藏BGE-M3、NumPy和CSR格式细节。
        query_vectors = self.embedding_model.encode_query(query.strip())

        # 2. 上层只表达过滤条件；MilvusStorage负责转换表达式和构造PyMilvus请求。
        filters = {"file_path": file_path_filter} if file_path_filter else None

        # 3. MilvusStorage构造并执行稠密/稀疏混合检索，返回统一命中记录。
        hits = self.milvus_storage.hybrid_search(
            dense_vector=query_vectors["dense"],
            sparse_vector=query_vectors["sparse"],
            top_k=effective_k,
            filters=filters,
        )
        if not hits:
            logger.debug("Milvus混合检索未返回候选子块")
            return []

        # 4. 将统一命中记录转换成子Document，供父子块业务继续处理。
        sub_chunks = [self._to_child_document(hit) for hit in hits]

        # 5. 从子块中提取去重的父文档 -> 避免同一父块的多个子块重复返回.
        parent_docs = self._restore_unique_parent_documents(sub_chunks)

        # 6. 父文档少于2个时没有重排价值，直接返回。
        if len(parent_docs) < 2:
            return parent_docs[:config.CANDIDATE_M]

        # 7. RerankerModel负责构造CrossEncoder文本对并返回对应分数。
        if self.reranker_model is None:
            raise RuntimeError(
                "当前VectorService未加载RerankerModel，不能执行候选重排序"
            )
        scores = self.reranker_model.score(
            query=query.strip(),
            documents=[doc.page_content for doc in parent_docs],
        )
        if len(scores) != len(parent_docs):
            raise RuntimeError("Reranker分数数量与父文档数量不一致")

        # 8. 业务层负责按相关性分数排序，并返回最终Top-M父文档。
        ranked_parent_docs = [
            doc
            for _, doc in sorted(
                zip(scores, parent_docs),
                key=lambda score_and_doc: score_and_doc[0],
                reverse=True,
            )
        ]
        return ranked_parent_docs[:config.CANDIDATE_M]



    # todo 3.5 定义函数 -> 从子块列表中提取去重的父文档.
    def _restore_unique_parent_documents(self, sub_chunks):
        # 1. 保存已经处理过的父块ID，用稳定ID去重而不是比较整段正文。
        parent_ids = set()
        # 2. 保存最终恢复出的唯一父Document。
        unique_docs = []

        # 3. 遍历Milvus命中的所有子Document。
        for chunk in sub_chunks:
            # 3.1 从子块元数据中取回父块ID和父块正文。
            parent_id = chunk.metadata.get("parent_id")
            parent_content = chunk.metadata.get("parent_content")

            # 3.2 缺少父子关联信息说明入库数据或查询结果不完整，不能静默继续。
            if not parent_id or not parent_content:
                raise ValueError("检索结果缺少parent_id或parent_content")

            # 3.3 同一父块可能命中多个子块；已经恢复过时直接跳过。
            if parent_id in parent_ids:
                continue

            # 3.4 父块正文放入page_content；metadata只保留溯源字段，
            # 不再重复保存一份parent_content。
            metadata = {
                "parent_id": parent_id,
                "file_path": chunk.metadata.get("file_path"),
                "timestamp": chunk.metadata.get("timestamp"),
            }
            unique_docs.append(
                Document(
                    page_content=parent_content,
                    metadata=metadata,
                )
            )
            parent_ids.add(parent_id)

        # 4. 返回按parent_id去重后的父Document列表。
        return unique_docs



    # todo 3.6 定义函数 -> 将Milvus结果(hit) 转换为 LangChain Document对象.
    def _to_child_document(self, hit):
        # MilvusStorage已经统一了Hit结构；这里转换为父子块业务使用的Document。
        child_text = hit.get("child_text")
        if not child_text:
            raise ValueError("Milvus命中记录缺少child_text")
        return Document(
            page_content=child_text,
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "file_path": hit.get("file_path"),
                "timestamp": hit.get("timestamp"),
                "retrieval_score": hit.get("score"),
            }
        )


def create_vector_service(load_reranker=True):
    """创建向量业务服务；纯入库场景可跳过Reranker模型。"""

    milvus_storage = None
    try:
        embedding_model = EmbeddingModel()

        milvus_storage = MilvusStorage(
            dense_dim=embedding_model.dense_dim,
        )
        reranker_model = RerankerModel() if load_reranker else None
    except Exception:
        # 防止Milvus已经连接、Reranker初始化却失败时遗留连接。
        if milvus_storage is not None:
            milvus_storage.close()
        raise

    return VectorService(
        embedding_model=embedding_model,
        milvus_storage=milvus_storage,
        reranker_model=reranker_model,
    )



# 单文件测试：使用假依赖验证VectorService的数据加工职责，不连接真实Milvus。
if __name__ == "__main__":
    class FakeEmbeddingModel:
        dense_dim = 2

        @staticmethod
        def encode(texts):
            return {
                "dense": [[0.1, 0.2] for _ in texts],
                "sparse": [{1: 0.7, 8: 0.3} for _ in texts],
            }

        @staticmethod
        def encode_query(query):
            return {
                "dense": [0.1, 0.2],
                "sparse": {1: 0.7, 8: 0.3},
            }

    class FakeRerankerModel:
        @staticmethod
        def score(query, documents):
            return [0.2, 0.9][:len(documents)]

    class FakeMilvusStorage:
        collection_name = "test_collection"

        @staticmethod
        def add_documents(data):
            print(f"底层客户端收到 {len(data)} 条整理后的data")
            print(data[0])
            return {"upsert_count": len(data)}

        @staticmethod
        def hybrid_search(
            dense_vector,
            sparse_vector,
            top_k,
            filters=None,
        ):
            return [
                {
                    "child_text": "第一条测试命中的子块",
                    "parent_id": "示例_parent_0",
                    "parent_content": "这是第一条用于补充上下文的父块正文。",
                    "file_path": "D:/data/示例.txt",
                    "timestamp": "2026-10-25T20:00:00",
                    "score": 0.8,
                },
                {
                    "child_text": "第二条测试命中的子块",
                    "parent_id": "示例_parent_1",
                    "parent_content": "这是第二条用于补充上下文的父块正文。",
                    "file_path": "D:/data/示例.txt",
                    "timestamp": "2026-10-25T20:00:00",
                    "score": 0.7,
                },
            ][:top_k]

    parent = Document(
        page_content="这是第一条用于补充上下文的父块正文。",
        metadata={"parent_id": "示例_parent_0"},
    )
    child = Document(
        page_content="这是用于向量检索的子块。",
        metadata={
            "child_id": "示例_parent_0_child_0",
            "parent_id": "示例_parent_0",
            "file_path": "D:/data/示例.txt",
            "timestamp": "2026-10-25T20:00:00",
        },
    )

    vector_service = VectorService(
        embedding_model=FakeEmbeddingModel(),
        milvus_storage=FakeMilvusStorage(),
        reranker_model=FakeRerankerModel(),
    )
    print(vector_service.add_documents([parent], [child]))
    print(vector_service.hybrid_search_with_rerank("测试查询"))
