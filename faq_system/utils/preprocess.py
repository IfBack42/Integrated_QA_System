# 该脚本用于: 文本预处理.

# 导入分词库
import logging
import sys
import unicodedata
from pathlib import Path

import jieba

# 屏蔽Jieba加载词典和缓存时输出的内部日志；项目自己的日志不受影响。
jieba.setLogLevel(logging.ERROR)

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入日志
from logger import get_logger

logger = get_logger(__name__)

STOPWORDS_FILE = Path(__file__).with_name("stopwords.txt")


def _load_stopwords():
    """启动时读取一次停用词表，后续查询直接复用内存集合。"""
    with STOPWORDS_FILE.open("r", encoding="utf-8-sig") as file:
        return frozenset(
            line.strip().lower()
            for line in file
            if line.strip()
        )


STOPWORDS = _load_stopwords()


def _is_pure_punctuation(token):
    """判断一个分词结果是否全部由标点或符号组成。"""
    return all(
        unicodedata.category(character)[0] in {"P", "S"}
        for character in token
    )


# todo 1.定义文本预处理函数.
def preprocess_text(text):
    """分词、小写化，并移除空白、停用词和纯标点。"""
    logger.debug("开始预处理文本")
    try:
        tokens = jieba.lcut(text.lower())
        return [
            token.strip()
            for token in tokens
            if token.strip()
            and token.strip() not in STOPWORDS
            and not _is_pure_punctuation(token.strip())
        ]
    except AttributeError as e:
        # 记录预处理失败
        logger.error(f"文本预处理失败: {e}")
        # 返回空列表
        return []


# todo 2.程序的主入口.
if __name__ == '__main__':
    # 1. 调用preprocess_text()函数, 测试.
    print(preprocess_text("黑马程序员"))
