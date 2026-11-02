"""BERT意图分类模型的独立训练入口。

修改下方训练集路径后，可直接运行本文件完成训练、评估和模型保存。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system_learning_version.core.intent_classifier.classifier_trainer import (
    BertIntentTrainer,
)

RAG_SYSTEM_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_FILE = PROJECT_ROOT / "sample_data" / "intent_training.example.jsonl"


def main():
    """装配训练器并执行一次完整训练。"""
    trainer = BertIntentTrainer()
    trainer.train(TRAINING_DATA_FILE)


if __name__ == "__main__":
    main()
