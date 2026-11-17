"""BGE-M3编码模型封装。

职责：加载一次BGE-M3，并向业务层提供统一的文本编码接口和稠密向量维度。
不负责：文档切分、Milvus连接、Collection管理或入库数据组装。

数据流：文本列表 → BGE-M3原始输出 → Python稠密向量列表与稀疏向量字典列表。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAG_SYSTEM_ROOT = Path(__file__).resolve().parents[3]

import torch.cuda
from milvus_model.hybrid import BGEM3EmbeddingFunction

from logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """管理BGE-M3模型，并提供文档和查询共用的编码能力。"""

    def __init__(self, model_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path) if model_path else (
            RAG_SYSTEM_ROOT / "models" / "bge-m3"
        )

        logger.info(f"BGE-M3模型使用设备: {self.device}")
        self.model = BGEM3EmbeddingFunction(
            model_name_or_path=str(self.model_path),
            use_fp16=(self.device == "cuda"),
            device=self.device,
        )

    @property
    def dense_dim(self):
        """返回当前模型生成的稠密向量维度。"""
        return self.model.dim["dense"]

    def encode(self, texts):
        """批量编码文本，并把模型原始格式整理成业务可直接使用的向量。"""
        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("待编码文本不能为空")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("待编码文本必须全部为非空字符串")

        raw_embeddings = self.model(list(texts))
        return self._normalize_embeddings(raw_embeddings, len(texts))

    def encode_query(self, query):
        """编码单条查询，直接返回一条稠密向量和一条稀疏向量字典。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询文本必须是非空字符串")
        batch_embeddings = self.encode([query.strip()])
        return {
            "dense": batch_embeddings["dense"][0],
            "sparse": batch_embeddings["sparse"][0],
        }

    @staticmethod
    def _normalize_embeddings(raw_embeddings, expected_count):
        """隐藏NumPy和CSR细节，统一返回Milvus可直接接收的Python结构。"""
        if not isinstance(raw_embeddings, dict):
            raise TypeError("BGE-M3编码结果必须是字典")
        if "dense" not in raw_embeddings or "sparse" not in raw_embeddings:
            raise ValueError("BGE-M3编码结果缺少dense或sparse")

        raw_dense_vectors = raw_embeddings["dense"]
        raw_sparse_matrix = raw_embeddings["sparse"]
        if len(raw_dense_vectors) != expected_count:
            raise RuntimeError("稠密向量数量与输入文本数量不一致")
        if raw_sparse_matrix.shape[0] != expected_count:
            raise RuntimeError("稀疏向量数量与输入文本数量不一致")

        dense_vectors = []
        sparse_vectors = []
        for index in range(expected_count):
            # 稠密向量转换为普通list，避免向上层泄露NumPy数组。
            dense_vector = raw_dense_vectors[index]
            if hasattr(dense_vector, "tolist"):
                dense_vector = dense_vector.tolist()
            dense_vectors.append(list(dense_vector))

            # CSR矩阵取一行并保持二维结构，再提取非零维度及其权重。
            sparse_row = raw_sparse_matrix[[index]]
            sparse_vector = {
                int(term_index): float(weight)
                for term_index, weight in zip(
                    sparse_row.indices,
                    sparse_row.data,
                )
            }
            sparse_vectors.append(sparse_vector)

        return {
            "dense": dense_vectors,
            "sparse": sparse_vectors,
        }


if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    test_embeddings = embedding_model.encode(["这是一段测试文本。"])
    print(f"稠密向量维度: {embedding_model.dense_dim}")
    print(f"稠密向量数量: {len(test_embeddings['dense'])}")
    print(f"稀疏向量示例: {test_embeddings['sparse'][0]}")
