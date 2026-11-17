"""BGE-Reranker重排序模型封装。

职责：加载一次CrossEncoder，并为查询与候选文档对计算相关性分数。
不负责：Milvus检索、候选构造或最终业务流程判断。

数据流：query + 候选正文列表 → 查询-正文对 → CrossEncoder → 相关性分数列表。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAG_SYSTEM_ROOT = Path(__file__).resolve().parents[3]

import torch.cuda
from sentence_transformers import CrossEncoder

from logger import get_logger

logger = get_logger(__name__)


class RerankerModel:
    """管理CrossEncoder模型，并提供统一的候选评分接口。"""

    def __init__(self, model_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path) if model_path else (
            RAG_SYSTEM_ROOT / "models" / "bge-reranker-large"
        )

        logger.info(f"Reranker模型使用设备: {self.device}")
        self.model = CrossEncoder(
            str(self.model_path),
            device=self.device,
        )

    def score(self, query, documents):
        """为query与每篇候选正文计算相关性分数，并保持返回顺序一致。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("重排序query必须是非空字符串")
        if not isinstance(documents, (list, tuple)):
            raise TypeError("重排序documents必须是列表或元组")
        if not documents:
            return []
        if any(
            not isinstance(document, str) or not document.strip()
            for document in documents
        ):
            raise ValueError("重排序候选正文必须全部为非空字符串")

        # CrossEncoder要求每个输入元素都是[query, document]文本对。
        pairs = [[query, document] for document in documents]
        raw_scores = self.model.predict(pairs)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        if not isinstance(raw_scores, list):
            raw_scores = list(raw_scores)

        scores = [float(score) for score in raw_scores]
        if len(scores) != len(documents):
            raise RuntimeError("Reranker返回分数数量与候选文档数量不一致")
        return scores


if __name__ == "__main__":
    reranker_model = RerankerModel()
    test_scores = reranker_model.score(
        "什么是大语言模型？",
        ["大语言模型是一类基于海量文本训练的模型。"],
    )
    print(f"测试相关性分数: {test_scores}")
