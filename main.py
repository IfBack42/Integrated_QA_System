"""命令行入口：只负责输入输出、启动和资源释放。"""

import sys
import time
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from logger import get_logger
from domain import QAStatus
from qa_system import QASystem


logger = get_logger(__name__)


def _print_result(result):
    if result.is_answered:
        print(f"答案:\n {result.answer}")
        return

    if result.status == QAStatus.INVALID:
        print(f"输入无效: {result.message}")
        return

    print(result.message or "系统未能生成答案")


def main():
    qa_system = None
    try:
        qa_system = QASystem()
        time.sleep(0.1)
        print("\n欢迎使用智能问答系统:")
        print("输入查询进行回答，输入 q 退出")

        while True:
            query = input("\n请输入查询: ").strip()
            if query.lower() == "q":
                logger.info("退出问答系统")
                print("再见")
                break

            try:
                result = qa_system.query(query)
                # Logger默认写入stderr，print写入stdout；短暂等待可以减少
                # 两个输出流在开发控制台中显示顺序交错。
                time.sleep(0.1)
                _print_result(result)
            except Exception:
                logger.exception("问答查询发生系统异常")
                print("系统暂时无法完成查询，请查看日志")
    except Exception:
        logger.exception("问答系统启动失败")
        print("问答系统启动失败，请查看日志")
    finally:
        if qa_system is not None:
            qa_system.close()


if __name__ == "__main__":
    main()
