"""文档加载器管理器。

这里只负责两件事：
1. 统一管理“扩展名 -> Loader类”的对应关系。
2. 根据单个文件路径创建对应的Loader实例。
"""

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

from .edu_docloader import OCRDOCLoader
from .edu_imgloader import OCRIMGLoader
from .edu_pdfloader import OCRPDFLoader
from .edu_pptloader import OCRPPTLoader


class DocumentLoaderManager:
    """统一管理项目中的LangChain Loader和自定义OCR Loader。"""

    def __init__(self):
        # 这里保存的是Loader类，不是Loader对象。
        # 因为每个Loader对象都需要绑定一个具体文件路径。
        self.loader_classes = {
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".pdf": OCRPDFLoader,
            ".docx": OCRDOCLoader,
            ".ppt": OCRPPTLoader,
            ".pptx": OCRPPTLoader,
            ".jpg": OCRIMGLoader,
            ".png": OCRIMGLoader,
        }

    def get_loader(self, file_path, file_extension):
        """使用已解析的文件后缀创建Loader实例；不支持时返回None。"""
        loader_class = self.loader_classes.get(file_extension)

        if loader_class is None:
            return None

        if file_extension == ".txt":
            return loader_class(file_path, encoding="utf-8")
        return loader_class(file_path)


