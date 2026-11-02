"""统一加载项目配置。

普通参数从 config.ini 读取，密码和 API Key 从环境变量读取。
本地开发时，项目根目录下的 .env 会被加载到当前 Python 进程。
"""

import configparser
import json
import os
from pathlib import Path

from dotenv import load_dotenv # 加载.env文件

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_ROOT / "config.ini"
ENV_FILE = PROJECT_ROOT / ".env"

# 系统环境变量优先；.env 只补充当前进程中不存在的变量。
load_dotenv(ENV_FILE)


def _require_env(name: str) -> str:
    """读取必需的敏感配置，缺失时给出明确提示。"""
    value = os.getenv(name)
    if not value or not value.strip():
        raise ValueError(f"缺少环境变量 {name}，请在项目根目录的 .env 中配置")
    return value


class Config:
    """提供统一的项目配置访问接口。"""

    def __init__(self, config_file: str | Path = CONFIG_FILE):
        self.config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        loaded_files = self.config.read(config_file, encoding="utf-8")
        if not loaded_files:
            raise FileNotFoundError(f"未找到配置文件: {config_file}")

        # 项目固定目录：由代码根据项目根目录计算，不需要写入 INI。
        self.PROJECT_ROOT = str(PROJECT_ROOT)
        self.LOG_DIR = str(PROJECT_ROOT / "logger" / "logs")
        self.DATA_DIR = str(PROJECT_ROOT / "sample_data")
        self.MODELS_DIR = str(PROJECT_ROOT / "rag_system_learning_version" / "models")
        self.EDU_DOCUMENT_LOADERS_DIR = str(
            PROJECT_ROOT / "rag_system_learning_version" / "edu_document_loaders"
        )

        # MySQL：连接参数来自 INI，密码只来自环境变量。
        self.MYSQL_HOST = self.config.get("mysql", "host", fallback="localhost")
        self.MYSQL_USER = self.config.get("mysql", "user", fallback="root")
        self.MYSQL_PASSWORD = _require_env("MYSQL_PASSWORD")
        self.MYSQL_DATABASE = self.config.get(
            "mysql", "database", fallback="subjects_kg"
        )
        # 权威FAQ数据表名保持稳定，版本信息不写进表名。
        self.MYSQL_FAQ_TABLE_NAME = self.config.get(
            "mysql", "faq_table_name", fallback="jpkb"
        )

        # Redis：连接参数来自 INI，密码只来自环境变量。
        self.REDIS_HOST = self.config.get("redis", "host", fallback="localhost")
        self.REDIS_PORT = self.config.getint("redis", "port", fallback=6379)
        self.REDIS_PASSWORD = _require_env("REDIS_PASSWORD")
        self.REDIS_DB = self.config.getint("redis", "db", fallback=0)
        # RAG生成但尚未审核的临时答案Key前缀，不属于正式FAQ搜索索引。
        self.REDIS_TEMPORARY_ANSWER_KEY_PREFIX = self.config.get(
            "redis",
            "temporary_answer_key_prefix",
            fallback="faq:interim:",
        )
        # Redis Search正式FAQ搜索索引名称。
        self.REDIS_FAQ_INDEX_NAME = self.config.get(
            "redis",
            "faq_index_name",
            fallback="idx:faq:official",
        )
        # 正式FAQ Hash文档Key前缀，也是搜索索引管理的数据范围。
        self.REDIS_FAQ_DOCUMENT_KEY_PREFIX = self.config.get(
            "redis",
            "faq_document_key_prefix",
            fallback="faq:official:",
        )
        # Redis中保存的权威FAQ内容版本，用于判断是否需要重新同步。
        self.REDIS_FAQ_VERSION_KEY = self.config.get(
            "redis",
            "faq_version_key",
            fallback="faq:meta:official:version",
        )

        # Milvus 普通连接参数。
        self.MILVUS_HOST = self.config.get("milvus", "host", fallback="localhost")
        self.MILVUS_PORT = self.config.get("milvus", "port", fallback="19530")
        self.MILVUS_DATABASE_NAME = self.config.get(
            "milvus", "database_name", fallback="rag_system"
        )
        self.MILVUS_COLLECTION_NAME = self.config.get(
            "milvus", "collection_name", fallback="edurag_final"
        )

        # LLM：模型参数来自 INI，API Key 只来自环境变量。
        self.LLM_MODEL = self.config.get("llm", "model", fallback="qwen-plus")
        self.DASHSCOPE_API_KEY = _require_env("DASHSCOPE_API_KEY")
        self.DASHSCOPE_BASE_URL = self.config.get(
            "llm",
            "dashscope_base_url",
            fallback="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 检索参数。
        # 父块最大长度（当前文本切分器按字符数计算）。
        # 父块较大，用来保留一段相对完整的上下文；检索到子块后，可把对应父块交给大模型。
        self.PARENT_CHUNK_SIZE = self.config.getint(
            "retrieval", "parent_chunk_size", fallback=1200
        )
        # 子块最大长度（按字符数计算）。
        # 子块较小，用来建立索引和执行检索，使系统能更准确地定位相关内容。
        self.CHILD_CHUNK_SIZE = self.config.getint(
            "retrieval", "child_chunk_size", fallback=300
        )
        # 相邻文本块之间重复保留的字符数。
        # 用来减少一句话刚好被切断时造成的语义丢失；当前代码会同时用于父块和子块。
        self.CHUNK_OVERLAP = self.config.getint(
            "retrieval", "chunk_overlap", fallback=50
        )
        # 每次检索先召回的候选文档数量（Top-K）。
        # 数量越大，漏掉相关内容的概率越低，但后续排序和处理的开销也越大。
        self.RETRIEVAL_K = self.config.getint(
            "retrieval", "retrieval_k", fallback=5
        )
        # 对候选文档重排后，最终保留的上下文数量（Top-M）。
        # 这些内容会继续交给大模型；通常应小于或等于 RETRIEVAL_K，以减少噪声和Token消耗。
        self.CANDIDATE_M = self.config.getint(
            "retrieval", "candidate_m", fallback=2
        )
        # Redis Search FAQ检索的原始BM25得分阈值（不是0到1的概率）。
        # 最高分达到该值时视为FAQ命中，否则继续进入RAG；后续应通过测试集调整。
        self.FAQ_BM25_THRESHOLD = self.config.getfloat(
            "retrieval", "faq_bm25_threshold", fallback=0.85
        )

        # 应用参数。JSON 解析比 eval 更安全。
        valid_sources = self.config.get(
            "app",
            "valid_sources",
            fallback='["ai", "java", "test", "ops", "bigdata"]',
        )
        self.VALID_SOURCES = json.loads(valid_sources)
        self.CUSTOMER_SERVICE_PHONE = self.config.get(
            "app", "customer_service_phone", fallback=""
        )


config = Config()


if __name__ == "__main__":
    print(CONFIG_FILE)
