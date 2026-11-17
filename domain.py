"""项目级问答结果对象。

FAQResult只描述FAQ子系统内部结果；QAResult描述QASystem最终交给main的结果。
"""

from dataclasses import dataclass
from enum import Enum


class QAStatus(str, Enum):
    """整套问答系统对外暴露的正常业务状态。"""

    ANSWERED = "answered"
    INVALID = "invalid"


class AnswerSource(str, Enum):
    """标记最终答案来自哪一层，方便日志、测试和后续效果评估。"""

    FAQ = "faq"
    RAG_CACHE = "rag_cache"
    RAG = "rag"


@dataclass(frozen=True)
class QAResult:
    """QASystem返回给main的统一结果，不暴露FAQ或RAG内部对象。"""

    status: QAStatus
    answer: str | None = None
    source: AnswerSource | None = None
    message: str | None = None
    matched_question: str | None = None
    score: float | None = None

    @property
    def is_answered(self) -> bool:
        return self.status == QAStatus.ANSWERED and bool(self.answer)


def main():
    """直接运行本文件时，查看项目级返回对象的最小形状。"""
    result = QAResult(
        status=QAStatus.ANSWERED,
        answer="示例答案",
        source=AnswerSource.RAG,
    )
    print(result)
    print(f"是否已经回答: {result.is_answered}")


if __name__ == "__main__":
    main()
