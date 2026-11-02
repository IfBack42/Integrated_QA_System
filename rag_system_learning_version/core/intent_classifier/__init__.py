"""查询意图分类模块的统一导出入口。"""

from .bert_intent_model import BertIntentModel
from .query_classifier import QueryClassifier

__all__ = ["BertIntentModel", "QueryClassifier"]
