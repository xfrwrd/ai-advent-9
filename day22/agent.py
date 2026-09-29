"""Question with or without retrieved chunks, then one LLM call."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from day21.config import INDEX_DIR
from day21.embeddings import Embedder, LocalHashEmbedder
from day21.index_store import LoadedIndex, load_index
from day22.llm import chat
from day22.retrieve import Hit, search

ChatFn = Callable[[list[dict[str, str]]], str]


@dataclass
class Answer:
    question: str
    use_rag: bool
    text: str
    hits: list[Hit] = field(default_factory=list)

    @property
    def sources(self) -> list[str]:
        return [hit.chunk.source for hit in self.hits]


class RagAgent:
    def __init__(
        self,
        index: LoadedIndex,
        chat_fn: ChatFn = chat,
        embedder: Embedder | None = None,
        top_k: int = 5,
    ) -> None:
        self.index = index
        self.chat_fn = chat_fn
        self.embedder = embedder or LocalHashEmbedder()
        self.top_k = top_k

    def ask(self, question: str, use_rag: bool) -> Answer:
        hits = search(self.index, self.embedder, question, self.top_k) if use_rag else []
        text = self.chat_fn(_messages(question, hits if use_rag else None))
        return Answer(question=question, use_rag=use_rag, text=text, hits=hits if use_rag else [])


def load_agent(index_path: Path | None = None, chat_fn: ChatFn = chat) -> RagAgent:
    path = index_path or (INDEX_DIR / "structure.sqlite")
    return RagAgent(load_index(path), chat_fn=chat_fn)


def _messages(question: str, hits: list[Hit] | None) -> list[dict[str, str]]:
    if not hits:
        return [
            {
                "role": "system",
                "content": (
                    "Ответь кратко на вопрос о проекте ai-advent-9. "
                    "Документов проекта у тебя нет. Если не знаешь точный факт из кода, так и скажи."
                ),
            },
            {"role": "user", "content": question},
        ]
    blocks = []
    for index, hit in enumerate(hits, start=1):
        snippet = hit.chunk.text.strip()
        if len(snippet) > 2000:
            snippet = snippet[:2000] + "\n..."
        blocks.append(
            f"[{index}] {hit.chunk.source} / {hit.chunk.section}\n{snippet}"
        )
    context = "\n\n".join(blocks)
    return [
        {
            "role": "system",
            "content": (
                "Отвечай только по фрагментам ниже. Назови файл-источник. "
                "Если ответа во фрагментах нет, так и скажи."
            ),
        },
        {
            "role": "user",
            "content": f"Фрагменты:\n{context}\n\nВопрос: {question}",
        },
    ]
