"""BERT意图分类模型的独立训练入口。

修改下方训练集路径后，可直接运行本文件完成训练、评估和模型保存。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.core.intent_classifier.classifier_trainer import (
    BertIntentTrainer,
)

RAG_SYSTEM_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_FILE = (
    RAG_SYSTEM_ROOT
    / "core"
    / "intent_classifier"
    / "classify_data"
    / "model_enhanced.jsonl"
)
CANDIDATE_MODEL_PATH = (
    RAG_SYSTEM_ROOT / "models" / "bert_query_classifier_candidate_v2"
)


def main():
    """训练候选模型；验证通过前不覆盖当前正式模型。"""
    trainer = BertIntentTrainer(output_model_path=CANDIDATE_MODEL_PATH)
    trainer.train(TRAINING_DATA_FILE)


if __name__ == "__main__":
    main()
