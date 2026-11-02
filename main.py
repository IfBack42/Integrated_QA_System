"""命令行入口：只负责输入输出、启动和资源释放。"""

import sys
import time
from pathlib import Path


project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from logger import get_logger
from faq_system.domain import FAQStatus
from qa_system import QASystem


logger = get_logger(__name__)


def _print_result(result):
    if result.is_hit:
        print(f"答案: {result.answer}")
        return

    if result.status == FAQStatus.INVALID:
        print(f"输入无效: {result.message}")
        return

    print("FAQ中未找到可靠答案，后续需要进入RAG系统")


def main():
    qa_system = None
    try:
        qa_system = QASystem()
        time.sleep(0.1)
        print("\n欢迎使用FAQ问答系统:")
        print("输入查询进行回答，输入 q 退出")

        while True:
            query = input("\n请输入查询: ").strip()
            if query.lower() == "q":
                logger.info("退出FAQ系统")
                print("再见")
                break

            try:
                result = qa_system.query(query)
                # Logger默认写入stderr，print写入stdout；短暂等待可以减少
                # 两个输出流在开发控制台中显示顺序交错。
                time.sleep(0.1)
                _print_result(result)
            except Exception:
                logger.exception("FAQ查询发生系统异常")
                print("系统暂时无法完成查询，请查看日志")
    except Exception:
        logger.exception("FAQ系统启动失败")
        print("FAQ系统启动失败，请查看日志")
    finally:
        if qa_system is not None:
            qa_system.close()


if __name__ == "__main__":
    main()
