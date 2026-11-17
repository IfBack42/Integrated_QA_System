"""查询意图分类模块的统一导出入口。"""

from .bert_intent_model import BertIntentModel

__all__ = ["BertIntentModel", "QueryClassifier"]
