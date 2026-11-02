"""项目日志配置。

项目只在这里配置控制台和文件处理器。业务模块通过 ``get_logger(__name__)``
取得带模块名称的 logger，日志会向上传递到 ``IntegratedQA``，避免重复添加处理器。
"""

import logging
from datetime import date
from pathlib import Path

from config import config


APP_LOGGER_NAME = "IntegratedQA"


def setup_logging() -> None:
    """初始化项目级日志处理器，同一进程只执行一次。"""
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    # 项目总Logger必须先允许DEBUG通过，再由不同Handler分别筛选。
    app_logger.setLevel(logging.DEBUG)
    app_logger.propagate = False

    if app_logger.handlers:
        return

    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{date.today().isoformat()}.log"

    console_handler = logging.StreamHandler()
    # 当前处于开发阶段，控制台显示完整调试信息。
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    # 日志文件暂时只保存INFO及以上信息，避免记录过多逐请求细节。
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    app_logger.addHandler(console_handler)
    app_logger.addHandler(file_handler)


def get_logger(module_name: str | None = None) -> logging.Logger:
    """返回模块专属 logger，并复用项目级处理器。"""
    if not module_name:
        return logging.getLogger(APP_LOGGER_NAME)

    module_logger = logging.getLogger(f"{APP_LOGGER_NAME}.{module_name}")
    module_logger.setLevel(logging.NOTSET)
    module_logger.propagate = True
    return module_logger


# 导入日志包时自动完成项目级配置，不自动领取任何模块 logger。
setup_logging()

if __name__ == '__main__':
    logger = get_logger(__name__)
    logger.info("创建logger成功")
