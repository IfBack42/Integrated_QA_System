
"""Milvus底层客户端。

职责：从项目config读取Milvus及Collection配置；实例化时连接Milvus、检查或创建Collection，
并读写上层已经整理好的数据。
不负责：文档切分、ID生成、Embedding编码、向量转换和业务流程判断。

数据流：模型维度 + config中的连接/Collection配置 → MilvusStorage
→ 自动连接并准备Collection
→ 上层业务提交data或检索请求 → 返回数据库结果。
"""

import sys
import re
from pathlib import Path

# 允许直接运行本文件；作为包导入时，项目根目录通常已经在搜索路径中。
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from logger import get_logger

# 初始化日志实例.
logger = get_logger(__name__)
# Client只依赖Milvus连接、Schema和查询所需的PyMilvus类型。
from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker

# 定义MilvusStorage类: 封装Milvus底层操作 -> 连接、集合管理和数据读写.
class MilvusStorage:
    # Collection字段名称：属于Milvus Schema数据合同，不作为普通运行参数修改。
    CHILD_ID_FIELD = "child_id"
    CHILD_TEXT_FIELD = "child_text"
    DENSE_VECTOR_FIELD = "dense_vector"
    SPARSE_VECTOR_FIELD = "sparse_vector"
    PARENT_ID_FIELD = "parent_id"
    PARENT_CONTENT_FIELD = "parent_content"
    FILE_PATH_FIELD = "file_path"
    TIMESTAMP_FIELD = "timestamp"

    # 索引名称和度量方式：创建索引与查询时必须保持一致。
    DENSE_INDEX_NAME = "child_dense_index"
    SPARSE_INDEX_NAME = "child_sparse_index"
    DENSE_METRIC_TYPE = "IP"
    SPARSE_METRIC_TYPE = "IP"

    # 当前混合检索参数：集中放在Milvus业务类顶部，便于学习和调整。
    DENSE_NPROBE = 10
    DENSE_WEIGHT = 1.0
    SPARSE_WEIGHT = 0.7

    SEARCH_OUTPUT_FIELDS = (
        CHILD_TEXT_FIELD,
        PARENT_ID_FIELD,
        PARENT_CONTENT_FIELD,
        FILE_PATH_FIELD,
        TIMESTAMP_FIELD,
    )

    def __init__(self, dense_dim, settings=None):
        """接收模型维度、读取数据库配置并完成Milvus初始化。"""
        if not isinstance(dense_dim, int) or dense_dim <= 0:
            raise ValueError("dense_dim必须是正整数")

        self.settings = settings or config
        self.collection_name = self.settings.MILVUS_COLLECTION_NAME
        self.dense_dim = dense_dim
        self.host = self.settings.MILVUS_HOST
        self.port = self.settings.MILVUS_PORT
        self.database = self.settings.MILVUS_DATABASE_NAME
        self.client = None
        self._initialize()

    def _initialize(self):
        """连接Milvus，并创建或加载当前对象绑定的Collection。"""
        if self.client is not None:
            return

        client = MilvusClient(
            uri=f"http://{self.host}:{self.port}",
            db_name=self.database,
        )
        try:
            self.client = client
            self._load_collection()
        except Exception:
            self.client = None
            client.close()
            raise

        logger.info(
            f"MilvusStorage初始化完成，当前集合: {self.collection_name}"
        )

    def _require_initialized(self):
        """阻止未初始化的客户端直接执行数据库操作。"""
        if self.client is None:
            raise RuntimeError(
                "MilvusStorage尚未成功初始化"
            )


    # todo 3.2 私有方法: 创建子块集合(若不存在) 或校验并加载已有集合.
    def _load_collection(self):
        self._require_initialized()
        expected_fields = {
            self.CHILD_ID_FIELD,
            self.CHILD_TEXT_FIELD,
            self.DENSE_VECTOR_FIELD,
            self.SPARSE_VECTOR_FIELD,
            self.PARENT_ID_FIELD,
            self.PARENT_CONTENT_FIELD,
            self.FILE_PATH_FIELD,
            self.TIMESTAMP_FIELD,
        }

        # 已存在的集合必须先核对Schema。Milvus 2.5不能直接修改既有字段，
        # 因此不自动删除旧集合，避免误删数据。
        if self.client.has_collection(self.collection_name):
            collection_info = self.client.describe_collection(self.collection_name)
            actual_fields = {
                field["name"] for field in collection_info.get("fields", [])
            }
            if actual_fields != expected_fields:
                raise RuntimeError(
                    f"集合 {self.collection_name} 仍使用旧Schema。"
                    f"当前字段: {sorted(actual_fields)}；"
                    f"期望字段: {sorted(expected_fields)}。"
                    "请确认旧数据不再需要后，手动删除并重新创建集合。"
                )

            logger.info(f"{self.collection_name}集合已存在且Schema匹配，将加载集合")
            self.client.load_collection(self.collection_name)
            logger.info(f"已加载集合 {self.collection_name}")
            return

        logger.info(f"{self.collection_name}集合不存在，将创建子块集合")

        # 子块是检索单元；父块正文只在入库阶段根据parent_id补入。
        # 动态字段为后续小说的书籍、卷、章节等业务元数据预留扩展空间。
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(
            field_name=self.CHILD_ID_FIELD,
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=100,
        )
        schema.add_field(
            field_name=self.CHILD_TEXT_FIELD,
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name=self.DENSE_VECTOR_FIELD,
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dense_dim,
        )
        schema.add_field(
            field_name=self.SPARSE_VECTOR_FIELD,
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_field(
            field_name=self.PARENT_ID_FIELD,
            datatype=DataType.VARCHAR,
            max_length=100,
        )
        schema.add_field(
            field_name=self.PARENT_CONTENT_FIELD,
            datatype=DataType.VARCHAR,
            max_length=65535,
        )
        schema.add_field(
            field_name=self.FILE_PATH_FIELD,
            datatype=DataType.VARCHAR,
            max_length=1000,
        )
        schema.add_field(
            field_name=self.TIMESTAMP_FIELD,
            datatype=DataType.VARCHAR,
            max_length=50,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=self.DENSE_VECTOR_FIELD,
            index_name=self.DENSE_INDEX_NAME,
            index_type="IVF_FLAT",
            metric_type=self.DENSE_METRIC_TYPE,
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name=self.SPARSE_VECTOR_FIELD,
            index_name=self.SPARSE_INDEX_NAME,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type=self.SPARSE_METRIC_TYPE,
            params={"drop_ratio_build": 0.2},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"已创建子块集合 {self.collection_name}")
        self.client.load_collection(self.collection_name)
        logger.info(f"已加载集合 {self.collection_name}")


    # 将上层已经整理好的数据写入Milvus.
    def add_documents(self, data):
        """将符合Collection Schema的数据原样插入或更新到Milvus。"""
        if not data:
            logger.info("未收到待写入数据，跳过Milvus upsert")
            return None

        self._require_initialized()
        result = self.client.upsert(
            collection_name=self.collection_name,
            data=data,
        )
        logger.info(
            f"已插入/更新 {len(data)} 条数据到集合 {self.collection_name}"
        )
        return result

    def hybrid_search(
        self,
        dense_vector,
        sparse_vector,
        top_k,
        filters=None,
    ):
        """构造并执行Milvus混合检索，返回与PyMilvus格式解耦的命中记录。"""
        self._require_initialized()
        self._validate_search_inputs(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k,
        )
        filter_expr = self._build_filter_expr(filters)

        # 稠密向量使用IVF_FLAT索引，nprobe控制查询时探查的聚类数量。
        dense_request = AnnSearchRequest(
            data=[dense_vector],
            anns_field=self.DENSE_VECTOR_FIELD,
            param={
                "metric_type": self.DENSE_METRIC_TYPE,
                "params": {"nprobe": self.DENSE_NPROBE},
            },
            limit=top_k,
            expr=filter_expr,
        )

        # 稀疏向量使用倒排索引，不需要nprobe等IVF参数。
        sparse_request = AnnSearchRequest(
            data=[sparse_vector],
            anns_field=self.SPARSE_VECTOR_FIELD,
            param={"metric_type": self.SPARSE_METRIC_TYPE, "params": {}},
            limit=top_k,
            expr=filter_expr,
        )

        # 权重顺序必须和请求顺序保持一致：dense在前，sparse在后。
        ranker = WeightedRanker(self.DENSE_WEIGHT, self.SPARSE_WEIGHT)
        raw_results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=top_k,
            output_fields=list(self.SEARCH_OUTPUT_FIELDS),
        )

        # 一条query对应外层结果的第0组；正常零召回直接返回空列表。
        if not raw_results or not raw_results[0]:
            return []
        return self._normalize_search_hits(raw_results[0])

    def _validate_search_inputs(
        self,
        dense_vector,
        sparse_vector,
        top_k,
    ):
        """在调用Milvus前检查向量和Top-K。"""
        if not isinstance(dense_vector, (list, tuple)) or not dense_vector:
            raise ValueError("dense_vector必须是非空列表或元组")
        if len(dense_vector) != self.dense_dim:
            raise ValueError(
                f"dense_vector维度错误: 当前{len(dense_vector)}，"
                f"期望{self.dense_dim}"
            )
        if not isinstance(sparse_vector, dict) or not sparse_vector:
            raise ValueError("sparse_vector必须是非空字典")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k必须是正整数")

    @staticmethod
    def _build_filter_expr(filters):
        """将简单等值过滤字典转换成Milvus表达式；复杂小说范围后续扩展。"""
        if not filters:
            return ""
        if not isinstance(filters, dict):
            raise TypeError("filters必须是字典")

        clauses = []
        for field_name, value in filters.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field_name):
                raise ValueError(f"非法过滤字段名: {field_name}")

            if isinstance(value, str):
                escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
                clauses.append(f'{field_name} == "{escaped_value}"')
            elif isinstance(value, bool):
                clauses.append(f"{field_name} == {str(value).lower()}")
            elif isinstance(value, (int, float)):
                clauses.append(f"{field_name} == {value}")
            else:
                raise TypeError(
                    f"过滤字段{field_name}暂不支持{type(value).__name__}类型"
                )

        return " and ".join(clauses)

    def _normalize_search_hits(self, raw_hits):
        """把MilvusClient返回的字典式Hit转换成上层稳定使用的普通字典。"""
        hits = []
        required_fields = {
            self.CHILD_TEXT_FIELD,
            self.PARENT_ID_FIELD,
            self.PARENT_CONTENT_FIELD,
        }

        for raw_hit in raw_hits:
            entity = raw_hit.get("entity", {})
            hit = {
                field_name: entity.get(field_name)
                for field_name in self.SEARCH_OUTPUT_FIELDS
            }
            missing_fields = [
                field_name
                for field_name in required_fields
                if not hit.get(field_name)
            ]
            if missing_fields:
                raise ValueError(
                    f"Milvus命中结果缺少必需字段: {sorted(missing_fields)}"
                )

            score = raw_hit.get("distance")
            if score is None:
                score = raw_hit.get("score")
            hit["score"] = float(score) if score is not None else None
            hits.append(hit)

        return hits

    def close(self):
        """关闭当前Milvus连接；尚未初始化时无需处理。"""
        if self.client is None:
            return
        self.client.close()
        self.client = None
        logger.info("Milvus连接已关闭")


if __name__ == "__main__":
    class FakeMilvusClient:
        """验证写入和混合检索封装，不连接真实Milvus。"""

        @staticmethod
        def upsert(collection_name, data):
            print(f"测试集合: {collection_name}")
            print(f"收到数据: {data}")
            return {"upsert_count": len(data)}

        @staticmethod
        def hybrid_search(collection_name, reqs, ranker, limit, output_fields):
            print(f"混合检索集合: {collection_name}")
            print(f"检索请求数量: {len(reqs)}，最终Top-K: {limit}")
            print(f"返回字段: {output_fields}")
            return [[
                {
                    "distance": 0.88,
                    "entity": {
                        "child_text": "测试命中的子块",
                        "parent_id": "文件1_parent_0",
                        "parent_content": "测试父块正文",
                        "file_path": "D:/data/文件1.txt",
                        "timestamp": "2026-10-26T20:00:00",
                    },
                }
            ]]

    # 使用__new__绕过真实连接，只测试当前文件的写入职责。
    class FakeSettings:
        MILVUS_COLLECTION_NAME = "test_collection"
        MILVUS_HOST = "localhost"
        MILVUS_PORT = "19530"
        MILVUS_DATABASE_NAME = "test_database"

    test_store = MilvusStorage.__new__(MilvusStorage)
    test_store.settings = FakeSettings()
    test_store.collection_name = test_store.settings.MILVUS_COLLECTION_NAME
    test_store.dense_dim = 2
    test_store.host = test_store.settings.MILVUS_HOST
    test_store.port = test_store.settings.MILVUS_PORT
    test_store.database = test_store.settings.MILVUS_DATABASE_NAME
    test_store.client = None
    try:
        test_store.add_documents([{"child_id": "尚未初始化"}])
    except RuntimeError as error:
        print(f"未初始化写入保护: {error}")

    test_store.client = FakeMilvusClient()

    test_data = [
        {
            "child_id": "文件1_parent_0_child_0",
            "child_text": "测试子块",
        }
    ]
    print(f"写入结果: {test_store.add_documents(test_data)}")
    print(
        "检索结果:",
        test_store.hybrid_search(
            dense_vector=[0.1, 0.2],
            sparse_vector={1: 0.7, 8: 0.3},
            top_k=3,
            filters={"file_path": "D:/data/文件1.txt"},
        ),
    )
