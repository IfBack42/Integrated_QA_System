"""文本切分器管理器。

在一个处理流程中创建好父块/子块切分器，业务层只需根据文档元数据中的文件后缀领取切分器。
"""

from langchain_text_splitters import MarkdownTextSplitter

from .edu_chinese_recursive_text_splitter import ChineseRecursiveTextSplitter


class TextSplitterManager:
    """统一创建和选择项目中的文本切分器。"""

    def __init__(self, parent_chunk_size, child_chunk_size, chunk_overlap):
        # 记录当前主流程使用的轻量切分器类。
        self.splitter_classes = {
            "chinese": ChineseRecursiveTextSplitter,
            "markdown": MarkdownTextSplitter,
        }

        # 切分器只依赖切块参数，因此在当前处理流程中只初始化一次。
        self.chinese_parent = ChineseRecursiveTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.chinese_child = ChineseRecursiveTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.markdown_parent = MarkdownTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.markdown_child = MarkdownTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def get_parent_splitter(self, file_extension):
        """根据文件类型返回：(父块切分器, 切分器名称)。"""
        if file_extension == ".md":
            return self.markdown_parent, "Markdown"

        return self.chinese_parent, "ChineseRecursive"

    def get_child_splitter(self, file_extension):
        """根据文件类型返回：(子块切分器, 切分器名称)。"""
        if file_extension == ".md":
            return self.markdown_child, "Markdown"

        return self.chinese_child, "ChineseRecursive"

    @staticmethod
    def create_ali_splitter(**kwargs):
        """需要语义切分时再加载ModelScope相关依赖，避免拖慢普通启动。"""
        from .edu_model_text_spliter import AliTextSplitter

        return AliTextSplitter(**kwargs)
