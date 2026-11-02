"""BERT意图分类模型适配器。

职责：加载已经训练好的分类模型，并把单条文本转换为“类别 + 置信度”。
不负责：读取训练集、划分数据、训练、评估或决定RAG业务流程。

数据流：查询文本 -> Tokenizer编码 -> BERT推理 -> 类别和置信度。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from logger import get_logger

logger = get_logger(__name__)

RAG_SYSTEM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = RAG_SYSTEM_ROOT / "models" / "bert_query_classifier"

LABEL_TO_ID = {"通用知识": 0, "专业咨询": 1}
ID_TO_LABEL = {label_id: label for label, label_id in LABEL_TO_ID.items()}


class BertIntentModel:
    """封装已经训练好的BERT意图分类模型。"""

    def __init__(self, model_path=None, device=None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"未找到已训练的意图分类模型: {self.model_path}。"
                "请先运行 train_classifier.py 完成训练。"
            )

        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "已加载BERT意图分类模型: %s，使用设备: %s",
            self.model_path,
            self.device,
        )

    def predict(self, query):
        """预测单条查询，返回 ``(类别名称, 置信度)``。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("待分类查询必须是非空字符串")

        encoding = self.tokenizer(
            query.strip(),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        encoding = {name: value.to(self.device) for name, value in encoding.items()}

        with torch.no_grad():
            logits = self.model(**encoding).logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_id = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(probabilities[0, predicted_id].item())

        try:
            label = ID_TO_LABEL[predicted_id]
        except KeyError as exc:
            raise ValueError(f"模型返回了未知类别编号: {predicted_id}") from exc

        return label, confidence


def main():
    """单独验证已训练模型能否完成一次预测。"""
    model = BertIntentModel()
    query = input("请输入待分类问题: ").strip()
    label, confidence = model.predict(query)
    print(f"分类结果: {label}，置信度: {confidence:.4f}")


if __name__ == "__main__":
    main()
