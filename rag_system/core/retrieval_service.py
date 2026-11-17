"""专业问题的检索业务入口。

数据流：专业query → 选择策略 → 执行具体策略 → 返回父Document列表。
上层只调用本服务，不需要知道HyDE、子查询或Milvus的具体实现。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from logger import get_logger
from rag_system.core.retrieval_strategies.strategy_selector import StrategySelector
from rag_system.core.retrieval_strategies.strategies import (
    DirectStrategy,
    HyDEStrategy,
    StepBackStrategy,
    SubqueryStrategy,
)

logger = get_logger(__name__)


class RetrievalService:
    """封装策略选择和策略执行，对外只返回检索文档。"""

    def __init__(
        self,
        vector_service,
        llm_client,
        settings=None,
        selector=None,
        strategies=None,
    ):
        self.vector_service = vector_service
        self.llm_client = llm_client
        self.settings = settings or config
        self.selector = selector or StrategySelector(llm_client)
        self.strategies = strategies or self._build_strategies()

        missing = StrategySelector.SUPPORTED_STRATEGIES - self.strategies.keys()
        if missing:
            raise ValueError(f"缺少检索策略实现: {sorted(missing)}")

    def _build_strategies(self):
        """装配四种策略；具体执行逻辑仍留在各策略模块。"""
        common = {
            "vector_service": self.vector_service,
            "retrieval_k": self.settings.RETRIEVAL_K,
        }
        return {
            StrategySelector.DIRECT: DirectStrategy(**common),
            StrategySelector.HYDE: HyDEStrategy(
                llm_client=self.llm_client,
                **common,
            ),
            StrategySelector.SUBQUERY: SubqueryStrategy(
                llm_client=self.llm_client,
                final_limit=self.settings.CANDIDATE_M,
                **common,
            ),
            StrategySelector.STEP_BACK: StepBackStrategy(
                llm_client=self.llm_client,
                final_limit=self.settings.CANDIDATE_M,
                **common,
            ),
        }

    # TODO file_path_filter过滤器未使用
    def retrieve(self, query, file_path_filter=None, strategy_name=None):
        """选择并执行检索策略，返回 ``list[Document]``。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query必须是非空字符串")

        effective_strategy = strategy_name or self.selector.select(query.strip())
        if effective_strategy not in self.strategies:
            raise ValueError(f"不支持的检索策略: {effective_strategy}")

        logger.info("开始执行检索策略: %s", effective_strategy)
        documents = self.strategies[effective_strategy].retrieve(
            query=query.strip(),
            file_path_filter=file_path_filter,
        )
        logger.info(
            "检索策略执行完成: strategy=%s, documents=%s",
            effective_strategy,
            len(documents),
        )
        return documents

    def close(self):
        """释放检索业务持有的向量数据库连接。"""
        self.vector_service.close()


if __name__ == "__main__":
    from langchain_core.documents import Document

    class FakeSelector:
        @staticmethod
        def select(query):
            return StrategySelector.DIRECT

    class FakeStrategy:
        @staticmethod
        def retrieve(query, file_path_filter=None):
            return [Document(page_content="测试文档", metadata={})]

    class FakeVectorService:
        @staticmethod
        def close():
            pass

    fake_strategies = {
        name: FakeStrategy() for name in StrategySelector.SUPPORTED_STRATEGIES
    }
    service = RetrievalService(
        vector_service=FakeVectorService(),
        llm_client=object(),
        selector=FakeSelector(),
        strategies=fake_strategies,
    )
    print(service.retrieve("测试问题"))
