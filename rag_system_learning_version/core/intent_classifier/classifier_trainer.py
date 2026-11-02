"""BERT意图分类模型训练器。

职责：读取和预处理训练数据、构造Dataset、训练、评估并保存模型。
不负责：在线查询分类，也不决定查询进入FAQ、RAG或其他业务分支。

数据流：JSONL训练集 -> Tokenizer编码 -> Dataset -> Trainer训练/评估 -> 模型目录。
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from logger import get_logger
from rag_system_learning_version.core.intent_classifier.bert_intent_model import (
    DEFAULT_MODEL_PATH,
    ID_TO_LABEL,
    LABEL_TO_ID,
)

logger = get_logger(__name__)

RAG_SYSTEM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL_PATH = RAG_SYSTEM_ROOT / "models" / "bert-base-chinese"
DEFAULT_TRAINING_OUTPUT_DIR = RAG_SYSTEM_ROOT / "training_artifacts" / "bert_results"
DEFAULT_TRAINING_LOG_DIR = RAG_SYSTEM_ROOT / "training_artifacts" / "bert_logs"


class QueryDataset(torch.utils.data.Dataset):
    """把Tokenizer批量编码结果包装成Trainer可逐条读取的数据集。"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {name: values[index] for name, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        """
        item =
        {"input_ids": 第一条文本的编号数据,
        "token_type_ids": 第一条文本的辅助数据,
        "attention_mask": 第一条文本的有效位置数据,
        "labels": 0,}
        """
        return item

    def __len__(self):
        return len(self.labels)


class BertIntentTrainer:
    """整合BERT意图分类模型的离线训练和评估流程。"""

    def __init__(self, base_model_path=None, output_model_path=None):
        self.base_model_path = (
            Path(base_model_path) if base_model_path else DEFAULT_BASE_MODEL_PATH
        )
        self.output_model_path = (
            Path(output_model_path) if output_model_path else DEFAULT_MODEL_PATH
        )

        if not self.base_model_path.exists():
            raise FileNotFoundError(f"未找到BERT基础模型: {self.base_model_path}")

        self.tokenizer = BertTokenizer.from_pretrained(self.base_model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            self.base_model_path,
            num_labels=len(LABEL_TO_ID),
            label2id=LABEL_TO_ID,
            id2label=ID_TO_LABEL,
        )

    def _load_training_data(self, data_file):
        """读取JSONL训练集，返回文本列表和标签列表。"""
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"训练集文件不存在: {data_path}")

        records = []
        with data_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    query = record["query"]
                    label = record["label"]
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(
                        f"训练集第{line_number}行格式错误，需要query和label字段"
                    ) from exc

                if label not in LABEL_TO_ID:
                    raise ValueError(f"训练集第{line_number}行包含未知标签: {label}")
                records.append((query, label))

        if not records:
            raise ValueError("训练集为空")

        texts, labels = zip(*records)
        return list(texts), list(labels)

    def _preprocess_data(self, texts, labels):
        """把文本和文字标签转换为模型输入与数字标签。"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        label_ids = [LABEL_TO_ID[label] for label in labels]
        return encodings, label_ids

    @staticmethod
    def _compute_metrics(eval_prediction):
        """供Trainer回调，计算验证集准确率。"""
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": float((predictions == labels).mean())}

    def _save_model(self):
        """保存训练后的模型、标签映射和Tokenizer。"""
        self.output_model_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.output_model_path)
        self.tokenizer.save_pretrained(self.output_model_path)
        logger.info("模型已保存至: %s", self.output_model_path)

    def evaluate(self, texts, label_ids):
        """在指定数据上输出分类报告和混淆矩阵。"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        dataset = QueryDataset(encodings, label_ids)
        predictions = Trainer(model=self.model).predict(dataset)
        predicted_ids = np.argmax(predictions.predictions, axis=-1)
        label_names = [ID_TO_LABEL[index] for index in sorted(ID_TO_LABEL)]

        logger.info(
            "分类报告:\n%s",
            classification_report(label_ids, predicted_ids, target_names=label_names),
        )
        logger.info("混淆矩阵:\n%s", confusion_matrix(label_ids, predicted_ids))

    def train(self, data_file):
        """执行完整训练流程并保存最佳模型。"""
        texts, labels = self._load_training_data(data_file)
        # texts = ["Python是什么","课程如何报名","什么是机器学习",]
        # labels = ["通用知识","专业咨询","通用知识",]

        train_texts, validation_texts, train_labels, validation_labels = (
            train_test_split(texts, labels, test_size=0.2, random_state=42)
        )

        train_encodings, train_label_ids = self._preprocess_data(
            train_texts, train_labels
        )
        validation_encodings, validation_label_ids = self._preprocess_data(
            validation_texts, validation_labels
        )
        train_dataset = QueryDataset(train_encodings, train_label_ids)
        validation_dataset = QueryDataset(
            validation_encodings, validation_label_ids
        )

        training_args = TrainingArguments(
            # 保存训练检查点和训练状态的目录。
            output_dir=str(DEFAULT_TRAINING_OUTPUT_DIR),
            # 完整遍历训练集3轮。
            num_train_epochs=3,
            # Trainer内部创建训练DataLoader时，每个设备每批读取8条数据。
            per_device_train_batch_size=8,
            # Trainer内部创建验证DataLoader时，每个设备每批读取8条数据。
            per_device_eval_batch_size=8,
            # 训练开始后的前20步逐渐提高学习率，减少初期震荡。
            warmup_steps=20,
            # 对模型权重施加衰减，降低过拟合风险。
            weight_decay=0.01,
            # TensorBoard等训练日志的保存目录。
            logging_dir=str(DEFAULT_TRAINING_LOG_DIR),
            # 每训练10步记录一次日志。
            logging_steps=10,
            # 每轮训练结束后，使用验证集评估一次。
            evaluation_strategy="epoch",
            # 每轮训练结束后保存一次模型检查点。
            save_strategy="epoch",
            # 训练结束后，自动恢复评估指标最好的检查点。
            load_best_model_at_end=True,
            # 最多保留1个训练检查点，避免长期占用磁盘。
            save_total_limit=1,
            # 使用验证集损失判断哪个检查点最好。
            metric_for_best_model="eval_loss",
            # 当前使用FP32训练，不启用FP16混合精度。
            fp16=False,
            # 不向Weights & Biases等外部平台上传训练日志。
            report_to=[],
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=self._compute_metrics, # 这种“把函数传进去，等对方需要时调用”的函数叫回调函数
        )

        logger.info("开始训练BERT意图分类模型")
        trainer.train()
        self.model = trainer.model
        self._save_model()
        self.evaluate(validation_texts, validation_label_ids)


def main():
    """单独检查训练组件能否加载基础BERT模型。"""
    trainer = BertIntentTrainer()
    print(f"训练组件初始化成功，基础模型: {trainer.base_model_path}")


if __name__ == "__main__":
    main()
