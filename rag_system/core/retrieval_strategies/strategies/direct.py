"""直接检索策略：不改写query，直接执行混合向量检索。"""


class DirectStrategy:
    def __init__(self, vector_service, retrieval_k=None):
        self.vector_service = vector_service
        self.retrieval_k = retrieval_k

    def retrieve(self, query, file_path_filter=None):
        """输入原始query，返回按相关性排序的父Document列表。"""
        return self.vector_service.hybrid_search_with_rerank(
            query=query,
            k=self.retrieval_k,
            file_path_filter=file_path_filter,
        )


if __name__ == "__main__":
    class FakeVectorService:
        @staticmethod
        def hybrid_search_with_rerank(**kwargs):
            return [f"直接检索收到: {kwargs['query']}"]

    print(DirectStrategy(FakeVectorService()).retrieve("测试问题"))
