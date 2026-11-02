"""在线查询分类服务。

职责：检查业务输入、调用BERT意图模型，并向上层返回稳定的分类接口。
模型不存在时直接报错，提示先运行训练入口；不会使用随机初始化模型假装预测。

数据流：用户查询 -> QueryClassifier -> BertIntentModel -> 类别/置信度。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from rag_system_learning_version.core.intent_classifier.bert_intent_model import (
    BertIntentModel,
)

logger = get_logger(__name__)


class QueryClassifier:
    """RAG业务使用的在线意图分类入口。"""

    def __init__(self, model_path=None, intent_model=None):
        self.intent_model = intent_model or BertIntentModel(model_path=model_path)

    def classify(self, query):
        """返回 ``(类别名称, 置信度)``。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("用户查询必须是非空字符串")

        label, confidence = self.intent_model.predict(query.strip())
        logger.debug(
            "查询意图分类完成: label=%s, confidence=%.4f", label, confidence
        )
        return label, confidence

    def predict_category(self, query):
        """兼容现有RAG代码：只返回类别名称。"""
        label, _ = self.classify(query)
        return label


def main():
    """单独验证在线分类服务。"""
    classifier = QueryClassifier()
    query = input("请输入待分类问题: ").strip()
    label, confidence = classifier.classify(query)
    print(f"分类结果: {label}，置信度: {confidence:.4f}")


if __name__ == "__main__":
    main()
