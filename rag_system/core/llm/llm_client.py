"""通用大模型调用客户端。

职责：把已经构造好的消息发送给兼容OpenAI协议的大模型，并返回文本。
不负责：选择检索策略、读取业务提示词、处理Document或生成RAG上下文。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from config import config
from logger import get_logger

logger = get_logger(__name__)


class LLMCallError(RuntimeError):
    """大模型API调用或响应格式异常。"""


class LLMClient:
    """向上层提供稳定的文本生成接口。"""

    DEFAULT_SYSTEM_MESSAGE = "你是一个可靠的中文助手，请严格遵循用户给出的任务要求。"

    def __init__(self, settings=None, client=None):
        self.settings = settings or config
        self.client = client or OpenAI(
            api_key=self.settings.LLM_API_KEY,
            base_url=self.settings.BASE_URL,
        )

    def chat(
        self,
        user_message,
        system_message=None,
        temperature=0.1,
    ):
        """发送一次对话请求并返回模型生成的非空文本。"""
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message必须是非空字符串")
        if system_message is not None and not isinstance(system_message, str):
            raise TypeError("system_message必须是字符串或None")

        try:
            completion = self.client.chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_message or self.DEFAULT_SYSTEM_MESSAGE,
                    },
                    {"role": "user", "content": user_message.strip()},
                ],
                temperature=temperature,
            )
        except Exception as exc:
            logger.error("LLM API调用失败: %s", exc)
            raise LLMCallError("大模型服务调用失败") from exc

        if not completion.choices or not completion.choices[0].message:
            raise LLMCallError("大模型返回了空响应")

        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise LLMCallError("大模型返回内容为空")

        logger.debug("LLM调用完成，返回字符数=%s", len(content))
        return content.strip()

    def close(self):
        """关闭底层OpenAI客户端持有的HTTP连接池。"""
        close_method = getattr(self.client, "close", None)
        if callable(close_method):
            close_method()


if __name__ == "__main__":
    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            message = type("Message", (), {"content": "测试回答"})()
            choice = type("Choice", (), {"message": message})()
            return type("Completion", (), {"choices": [choice]})()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    client = LLMClient(client=FakeClient())
    print(client.chat("请回答测试问题"))
    client.close()
