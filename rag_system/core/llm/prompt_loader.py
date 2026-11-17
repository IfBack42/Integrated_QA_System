"""Markdown提示词读取器。

职责：读取、缓存并填充外部提示词文件；不判断提示词属于哪项业务。
"""

from pathlib import Path


class PromptLoader:
    """读取UTF-8提示词并使用命名参数填充占位符。"""

    def __init__(self):
        self._cache = {}

    def _load(self, prompt_path):
        """读取提示词原文；同一路径在当前进程中只读磁盘一次。"""
        path = Path(prompt_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"未找到提示词文件: {path}")

        if path not in self._cache:
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                raise ValueError(f"提示词文件不能为空: {path}")
            self._cache[path] = content
        return self._cache[path]

    def render(self, prompt_path, **values):
        """读取模板并填充 ``{name}`` 形式的占位符。"""
        template = self._load(prompt_path)
        try:
            return template.format(**values)
        except KeyError as exc:
            raise ValueError(
                f"提示词缺少占位参数 {exc.args[0]}: {Path(prompt_path)}"
            ) from exc


if __name__ == "__main__":
    import tempfile

    prompt_file = Path(tempfile.gettempdir()) / "prompt_loader_test.md"
    prompt_file.write_text("问题：{query}", encoding="utf-8")
    try:
        print(PromptLoader().render(prompt_file, query="测试问题"))
    finally:
        prompt_file.unlink(missing_ok=True)
