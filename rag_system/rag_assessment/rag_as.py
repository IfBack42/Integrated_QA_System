"""使用RAGAS评估当前系统已经采集的真实问答与检索上下文。

数据流：ragas_runtime_data.json → HuggingFace Dataset
→ RAGAS四项指标 → retrained_ragas_evaluation_result.csv。
"""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from config import config
from logger import get_logger
from rag_system.core.vector_retrieval.models.embedding_model import EmbeddingModel


logger = get_logger(__name__)

ASSESSMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = ASSESSMENT_DIR / "ragas_runtime_data.json"
DEFAULT_RESULT_FILE = ASSESSMENT_DIR / "retrained_ragas_evaluation_result.csv"


class BGEEmbeddingAdapter(Embeddings):
    """把项目已有的EmbeddingModel适配为RAGAS认识的LangChain接口。"""

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or EmbeddingModel()

    def embed_documents(self, texts):
        """批量返回BGE-M3稠密向量；RAGAS不使用项目的稀疏向量。"""
        return self.embedding_model.encode(texts)["dense"]

    def embed_query(self, text):
        """返回单条文本的BGE-M3稠密向量。"""
        return self.embedding_model.encode_query(text)["dense"]


class RagasEvaluator:
    """加载静态评估数据，配置裁判模型，并执行RAGAS四项评分。"""

    REQUIRED_FIELDS = {"question", "answer", "context", "ground_truth"}

    def __init__(
        self,
        settings=None,
        data_file=None,
        judge_llm=None,
        embeddings=None,
    ):
        self.settings = settings or config
        self.data_file = Path(data_file) if data_file else DEFAULT_DATA_FILE
        # RAGAS需要LangChain模型接口，不能直接使用项目的LLMClient。
        self.judge_llm = judge_llm or ChatOpenAI(
            model=self.settings.LLM_MODEL,
            api_key=self.settings.LLM_API_KEY,
            base_url=self.settings.BASE_URL,
            temperature=0,
            max_retries=2,
        )
        self.embeddings = embeddings or BGEEmbeddingAdapter()

    def load_dataset(self, sample_limit=None):
        """读取并校验JSON，转换为RAGAS 0.2.6要求的Dataset字段。"""
        if not self.data_file.exists():
            raise FileNotFoundError(f"未找到RAGAS评估数据: {self.data_file}")
        if sample_limit is not None and sample_limit <= 0:
            raise ValueError("sample_limit必须是大于0的整数或None")

        with self.data_file.open("r", encoding="utf-8") as file:
            samples = json.load(file)
        if not isinstance(samples, list) or not samples:
            raise ValueError("RAGAS评估数据必须是非空列表")

        samples = samples[:sample_limit] if sample_limit else samples
        for index, sample in enumerate(samples, start=1):
            if not isinstance(sample, dict):
                raise TypeError(f"第{index}条评估样本必须是字典")
            missing_fields = self.REQUIRED_FIELDS - sample.keys()
            if missing_fields:
                raise ValueError(
                    f"第{index}条评估样本缺少字段: {sorted(missing_fields)}"
                )
            if not isinstance(sample["context"], list) or not sample["context"]:
                raise ValueError(f"第{index}条样本的context必须是非空列表")

        logger.info("RAGAS评估数据加载成功，共%d条", len(samples))
        return Dataset.from_dict(
            {
                "question": [item["question"] for item in samples],
                "answer": [item["answer"] for item in samples],
                "contexts": [item["context"] for item in samples],
                "ground_truth": [item["ground_truth"] for item in samples],
            }
        )

    def evaluate(self, sample_limit=None):
        """执行忠实度、答案相关性、上下文精确率和上下文召回率评估。"""
        dataset = self.load_dataset(sample_limit=sample_limit)
        logger.info("开始执行RAGAS评估")
        return evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self.judge_llm,
            embeddings=self.embeddings,
            # 快速验证阶段串行执行，避免一次并发大量LLM请求。
            run_config=RunConfig(
                timeout=180,
                max_retries=2,
                max_wait=20,
                max_workers=1,
            ),
            raise_exceptions=True,
            show_progress=True,
        )

    @staticmethod
    def save_result(result, result_file=None):
        """把逐条得分和整体指标保存为CSV，作为后续对比基线。"""
        output_path = Path(result_file) if result_file else DEFAULT_RESULT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_pandas().to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("RAGAS评估结果已保存: %s", output_path)
        return output_path


def main():
    """默认用1条真实样本快速验证环境；全量评估时改为None。"""
    evaluator = RagasEvaluator()
    result = evaluator.evaluate(sample_limit=1)
    result_file = evaluator.save_result(result)
    print(f"RAGAS快速验证结果: {result}")
    print(f"逐条结果文件: {result_file}")


if __name__ == "__main__":
    main()
