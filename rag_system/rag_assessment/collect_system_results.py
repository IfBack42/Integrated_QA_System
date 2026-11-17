"""运行黄金测试集，采集当前RAG系统的真实答案、上下文和耗时。

输入：system_test_cases.json中的用户问题和人工参考答案。
输出：
1. collected_system_results.json：完整业务记录，便于检查路由和异常。
2. ragas_runtime_data.json：只保留RAGAS需要的真实答案与检索上下文。
"""

import json
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from rag_system.core.rag_service import RAGService, create_rag_service


logger = get_logger(__name__)

ASSESSMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_FILE = ASSESSMENT_DIR / "system_test_cases.json"
DEFAULT_RESULTS_FILE = ASSESSMENT_DIR / "collected_system_results.json"
DEFAULT_RAGAS_FILE = ASSESSMENT_DIR / "ragas_runtime_data.json"


class SystemResultCollector:
    """逐条调用真实RAGService，并把运行轨迹整理成评估数据。"""

    REFUSAL_MARKER = "当前知识库资料不足以回答"

    def __init__(self, rag_service=None, cases_file=None, device="cpu"):
        # 离线评估默认使用CPU，避免与正在运行的GPU问答服务争抢显存。
        self.rag_service = rag_service or create_rag_service(device=device)
        self.cases_file = Path(cases_file) if cases_file else DEFAULT_CASES_FILE

    def load_cases(self, sample_limit=None):
        """读取黄金测试集；sample_limit用于先小批量验证。"""
        if not self.cases_file.exists():
            raise FileNotFoundError(f"未找到黄金测试集: {self.cases_file}")
        if sample_limit is not None and sample_limit <= 0:
            raise ValueError("sample_limit必须是大于0的整数或None")

        with self.cases_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        cases = payload.get("cases") if isinstance(payload, dict) else None
        if not isinstance(cases, list) or not cases:
            raise ValueError("黄金测试集必须包含非空cases列表")
        return cases[:sample_limit] if sample_limit else cases

    def collect(self, sample_limit=5):
        """运行测试问题；单条失败会记录error，不中断其他样本。"""
        cases = self.load_cases(sample_limit=sample_limit)
        records = []

        for index, case in enumerate(cases, start=1):
            query = case.get("query")
            logger.info(
                "开始采集测试样本 %d/%d: %s",
                index,
                len(cases),
                case.get("case_id"),
            )
            started_at = perf_counter()
            try:
                trace = self.rag_service.answer_with_trace(query=query)
                elapsed_seconds = round(perf_counter() - started_at, 4)
                actual_should_answer = not self.is_refusal_answer(trace.answer)
                record = {
                    **case,
                    "actual_intent": trace.intent_label,
                    "intent_confidence": trace.intent_confidence,
                    "response": trace.answer,
                    "retrieved_contexts": list(trace.retrieved_contexts),
                    "source_files": list(trace.source_files),
                    "elapsed_seconds": elapsed_seconds,
                    "behavior_matches": (
                        bool(case.get("should_answer"))
                        == actual_should_answer
                    ),
                    "error": None,
                }
            except Exception as exc:
                elapsed_seconds = round(perf_counter() - started_at, 4)
                logger.exception("测试样本执行失败: %s", case.get("case_id"))
                record = {
                    **case,
                    "actual_intent": None,
                    "intent_confidence": None,
                    "response": None,
                    "retrieved_contexts": [],
                    "source_files": [],
                    "elapsed_seconds": elapsed_seconds,
                    "behavior_matches": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            records.append(record)

        return records

    @classmethod
    def is_refusal_answer(cls, answer):
        """判断系统是否按提示词要求明确拒绝回答资料外问题。"""
        clean_answer = (answer or "").strip()
        return (
            clean_answer == RAGService.NO_CONTEXT_ANSWER
            or cls.REFUSAL_MARKER in clean_answer
        )

    @staticmethod
    def save_records(
        records,
        results_file=None,
        ragas_file=None,
    ):
        """保存完整记录，并生成仅包含成功专业检索结果的RAGAS输入。"""
        results_path = Path(results_file) if results_file else DEFAULT_RESULTS_FILE
        ragas_path = Path(ragas_file) if ragas_file else DEFAULT_RAGAS_FILE

        with results_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)

        ragas_samples = [
            {
                "question": record["query"],
                "answer": record["response"],
                "context": record["retrieved_contexts"],
                "ground_truth": record["reference_answer"],
            }
            for record in records
            if record["error"] is None
            and bool(record["should_answer"])
            and record["actual_intent"] == RAGService.PROFESSIONAL_LABEL
            and record["response"]
            and record["retrieved_contexts"]
        ]
        with ragas_path.open("w", encoding="utf-8") as file:
            json.dump(ragas_samples, file, ensure_ascii=False, indent=2)

        logger.info("完整评估记录已保存: %s", results_path)
        logger.info("RAGAS运行数据已保存: %s，共%d条", ragas_path, len(ragas_samples))
        return results_path, ragas_path

    def close(self):
        self.rag_service.close()


def main():
    """默认先采集5条，确认数据流正确后再把sample_limit改为None。"""
    collector = None
    try:
        collector = SystemResultCollector()
        records = collector.collect(sample_limit=5)
        results_path, ragas_path = collector.save_records(records)
        success_count = sum(record["error"] is None for record in records)
        print(f"采集完成: 成功{success_count}/{len(records)}条")
        print(f"完整记录: {results_path}")
        print(f"RAGAS数据: {ragas_path}")
    finally:
        if collector is not None:
            collector.close()


if __name__ == "__main__":
    main()
