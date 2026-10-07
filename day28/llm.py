"""Two chat functions for one RAG pipeline: Ollama locally, DeepSeek in the cloud."""

from __future__ import annotations

from day22.agent import ChatFn
from day22.llm import chat as cloud_chat
from day26.client import LocalLLMClient


def local_chat(client: LocalLLMClient | None = None) -> ChatFn:
    model = client or LocalLLMClient()

    def ask(messages: list[dict[str, str]]) -> str:
        prompt = "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)
        return model.ask(prompt, think=False)

    return ask


def cloud_chat_fn() -> ChatFn:
    return cloud_chat
