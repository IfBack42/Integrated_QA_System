"""根据专业问题选择检索策略。

数据流：query → 外部提示词 → LLM → 校验后的策略名称。
调用失败或输出不合规时退回直接检索，但不会吞掉其他业务异常。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from rag_system.core.llm import LLMCallError, PromptLoader

logger = get_logger(__name__)


class StrategySelector:
    """只负责把query转换为受约束的策略名称。"""

    DIRECT = "直接检索"
    HYDE = "假设答案检索"
    SUBQUERY = "子查询检索"
    STEP_BACK = "回溯问题检索"

    SUPPORTED_STRATEGIES = {DIRECT, HYDE, SUBQUERY, STEP_BACK}
    ALIASES = {
        "假设问题检索": HYDE,
        "HyDE检索": HYDE,
        "回溯检索": STEP_BACK,
    }

    def __init__(self, llm_client, prompt_loader=None, prompt_path=None):
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader or PromptLoader()
        self.prompt_path = prompt_path or (
            Path(__file__).resolve().parent / "prompts" / "select_strategy.md"
        )

    def select(self, query):
        """返回四种受支持策略之一，无法可靠选择时退回直接检索。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query必须是非空字符串")

        prompt = self.prompt_loader.render(self.prompt_path, query=query.strip())
        try:
            raw_strategy = self.llm_client.chat(prompt, temperature=0.0)
        except LLMCallError:
            logger.exception("检索策略选择失败，退回直接检索")
            return self.DIRECT

        strategy = self._normalize(raw_strategy)
        if strategy is None:
            logger.warning(
                "大模型返回了不支持的检索策略 %r，退回直接检索",
                raw_strategy,
            )
            return self.DIRECT

        logger.debug("检索策略选择完成: %s", strategy)
        return strategy

    def _normalize(self, raw_strategy):
        """去掉常见包装文本，并把旧名称映射为统一名称。"""
        if not isinstance(raw_strategy, str):
            return None

        candidate = raw_strategy.strip().splitlines()[0].strip()
        candidate = candidate.strip("`*#：:。！!？?\"' ")
        candidate = self.ALIASES.get(candidate, candidate)
        return candidate if candidate in self.SUPPORTED_STRATEGIES else None


if __name__ == "__main__":
    class FakeLLMClient:
        @staticmethod
        def chat(prompt, temperature=0.0):
            return "子查询检索"

    selector = StrategySelector(FakeLLMClient())
    print(selector.select("比较第一册和第二册的主要冲突"))
