"""RAG核心业务模块。

主要调用链：RAGService → RetrievalService → Strategy → VectorService。
本包入口保持轻量，不在导包时初始化模型或连接外部服务。
"""
