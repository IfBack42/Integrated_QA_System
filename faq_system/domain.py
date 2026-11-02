"""FAQ领域对象：用有名字的状态替代含义不清的布尔值。"""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class FAQDocument:
    """MySQLClient根据查询行创建，临时存在Python内存。

    它作为一条权威FAQ在MySQLClient、QASystem和RedisClient之间传递；
    RedisClient调用to_mapping()后，才会把它写成Redis Hash文档。
    """

    qa_id: str
    subject_name: str
    question: str
    answer: str

    def __post_init__(self):
        if not self.qa_id.strip():
            raise ValueError("FAQ文档的qa_id不能为空")
        if not self.question.strip():
            raise ValueError("FAQ文档的question不能为空")
        if not self.answer.strip():
            raise ValueError("FAQ文档的answer不能为空")

    def to_mapping(self):
        """转换成可直接交给Redis HSET mapping参数的字典。"""
        return {
            "qa_id": self.qa_id,
            "subject_name": self.subject_name,
            "question": self.question,
            "answer": self.answer,
        }


@dataclass(frozen=True)
class FAQSnapshot:
    """MySQLClient创建、QASystem接收的启动同步对象。

    它临时存在Python内存，集中携带全部FAQDocument和内容版本；
    QASystem把它交给RedisClient完成版本判断与正式副本同步。
    """

    version: str
    documents: tuple[FAQDocument, ...]

    @property
    def document_count(self) -> int:
        return len(self.documents)


class FAQStatus(str, Enum):
    """FAQService与QASystem、main之间传递的业务状态枚举。

    枚举定义加载在Python内存中；它描述正常业务结果，不表示系统异常。
    """

    CACHE_HIT = "cache_hit"
    FAQ_HIT = "faq_hit"
    MISS = "miss"
    INVALID = "invalid"


@dataclass(frozen=True)  # 自动生成初始化等方法；frozen=True禁止实例字段被重新赋值。
class RetrievalCandidate:
    """RedisClient根据Redis Search结果创建的未判定候选。

    它临时存在Python内存，只在RedisClient到Retriever之间传递；
    此时已经有BM25得分，但尚未通过可靠命中规则。
    """

    qa_id: str
    question: str
    answer: str
    score: float

    def __post_init__(self):
        if not isinstance(self.qa_id, str) or not self.qa_id.strip():
            raise ValueError("检索候选的qa_id不能为空")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("检索候选的question不能为空")
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("检索候选的answer不能为空")


@dataclass(frozen=True)
class RetrievalMatch:
    """Retriever通过阈值判断后创建的可靠FAQ命中。

    它临时存在Python内存，从Retriever传给FAQService；
    与RetrievalCandidate相比，它已经获得“可靠命中”身份。
    """

    qa_id: str
    question: str
    answer: str
    score: float


@dataclass(frozen=True)
class FAQResult:
    """FAQService创建并返回给QASystem、main的最终FAQ业务结果。

    它临时存在Python内存，统一携带状态、答案、匹配问题和得分；
    QASystem据此决定是否进入RAG，main据此决定向用户输出什么。
    """

    status: FAQStatus
    answer: str | None = None
    matched_question: str | None = None
    score: float | None = None
    message: str | None = None

    @property # 将函数包装为属性，调用时不需要带括号
    def is_hit(self) -> bool:
        return self.status in {FAQStatus.CACHE_HIT, FAQStatus.FAQ_HIT}


def main():
    """直接运行本文件时，验证FAQ文档结构与字段映射。"""
    document = FAQDocument(
        qa_id="1",
        subject_name="运维",
        question="安装VMware Tools时显示灰色如何解决",
        answer="关闭虚拟机并调整虚拟设备设置。",
    )
    print("FAQDocument创建成功:")
    print(document)
    print("转换后的Hash字段:")
    print(document.to_mapping())


if __name__ == "__main__":
    main()
