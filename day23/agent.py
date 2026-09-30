"""Day 22 retrieval plus an optional rewrite and a separate relevance filter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from day21.config import INDEX_DIR
from day21.embeddings import Embedder, LocalHashEmbedder
from day21.index_store import LoadedIndex, load_index
from day22.agent import ChatFn, _messages
from day22.llm import chat
from day22.retrieve import Hit, search
from day23.config import (
    BASELINE_TOP_K,
    FILTERED_TOP_K,
    HIGHER_IS_BETTER,
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD,
)
from day23.filter import FILTERED_OUT, KEPT, FilterResult, RankedChunk, apply_filter
from day23.rewrite import rewrite_query

WITHOUT_RAG = "without_rag"
BASELINE = "baseline"
FILTERED = "filtered"
REWRITE_FILTERED = "rewrite_filtered"
MODES = (WITHOUT_RAG, BASELINE, FILTERED, REWRITE_FILTERED)

SearchFn = Callable[[LoadedIndex, Embedder, str, int], list[Hit]]

NO_CONTEXT_SYSTEM = (
    "Релевантный контекст не найден: поиск вернул фрагменты, "
    "но все они отсечены порогом релевантности. "
    "Не отвечай так, будто факты взяты из базы проекта. "
    "Прямо скажи, что релевантных фрагментов нет."
)


@dataclass
class ModeResult:
    mode: str
    original_query: str
    rewritten_query: str | None
    retrieved_before_filter: list[RankedChunk] = field(default_factory=list)
    filtered_out: list[RankedChunk] = field(default_factory=list)
    retrieved_after_filter: list[RankedChunk] = field(default_factory=list)
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    no_relevant_context: bool = False
    threshold: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "retrieved_before_filter": [chunk.as_dict() for chunk in self.retrieved_before_filter],
            "filtered_out": [chunk.as_dict() for chunk in self.filtered_out],
            "retrieved_after_filter": [chunk.as_dict() for chunk in self.retrieved_after_filter],
            "answer": self.answer,
            "sources": self.sources,
            "no_relevant_context": self.no_relevant_context,
            "threshold": self.threshold,
        }


class Day23Agent:
    def __init__(
        self,
        index: LoadedIndex,
        chat_fn: ChatFn = chat,
        embedder: Embedder | None = None,
        search_fn: SearchFn = search,
        retrieval_top_k: int = RETRIEVAL_TOP_K,
        filtered_top_k: int = FILTERED_TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        higher_is_better: bool = HIGHER_IS_BETTER,
        baseline_top_k: int = BASELINE_TOP_K,
    ) -> None:
        self.index = index
        self.chat_fn = chat_fn
        self.embedder = embedder or LocalHashEmbedder()
        self.search_fn = search_fn
        self.retrieval_top_k = retrieval_top_k
        self.filtered_top_k = filtered_top_k
        self.similarity_threshold = similarity_threshold
        self.higher_is_better = higher_is_better
        self.baseline_top_k = baseline_top_k

    def run(self, question: str, mode: str) -> ModeResult:
        if mode == WITHOUT_RAG:
            text = self.chat_fn(_messages(question, None))
            return ModeResult(mode=mode, original_query=question, rewritten_query=None, answer=text)
        if mode == BASELINE:
            hits = self.search_fn(self.index, self.embedder, question, self.baseline_top_k)
            chunks = _mark_kept(hits)
            text = self.chat_fn(_messages(question, hits))
            return ModeResult(
                mode=mode,
                original_query=question,
                rewritten_query=None,
                retrieved_before_filter=chunks,
                filtered_out=[],
                retrieved_after_filter=chunks,
                answer=text,
                sources=_sources(chunks),
            )
        if mode not in (FILTERED, REWRITE_FILTERED):
            raise ValueError(f"unknown mode: {mode}")
        rewritten = None
        search_query = question
        if mode == REWRITE_FILTERED:
            rewritten = rewrite_query(question, self.chat_fn)
            search_query = rewritten
        hits = self.search_fn(self.index, self.embedder, search_query, self.retrieval_top_k)
        filtered = apply_filter(
            hits,
            self.similarity_threshold,
            self.filtered_top_k,
            self.higher_is_better,
        )
        no_context = not filtered.kept_hits
        if no_context:
            text = self.chat_fn(
                [
                    {"role": "system", "content": NO_CONTEXT_SYSTEM},
                    {"role": "user", "content": question},
                ]
            )
        else:
            text = self.chat_fn(_messages(question, list(filtered.kept_hits)))
        return _from_filter(mode, question, rewritten, filtered, text, no_context, self.similarity_threshold)


def load_agent(index_path: Path | None = None, chat_fn: ChatFn = chat) -> Day23Agent:
    path = index_path or (INDEX_DIR / "structure.sqlite")
    return Day23Agent(load_index(path), chat_fn=chat_fn)


def _mark_kept(hits: list[Hit]) -> list[RankedChunk]:
    return [
        RankedChunk(
            rank=rank,
            chunk_id=hit.chunk.chunk_id,
            source=hit.chunk.source,
            section=hit.chunk.section,
            score=hit.score,
            status=KEPT,
            reason="baseline",
        )
        for rank, hit in enumerate(hits, start=1)
    ]


def _sources(chunks: list[RankedChunk] | tuple[RankedChunk, ...]) -> list[str]:
    sources: list[str] = []
    for chunk in chunks:
        if chunk.status == KEPT and chunk.source not in sources:
            sources.append(chunk.source)
    return sources


def _from_filter(
    mode: str,
    question: str,
    rewritten: str | None,
    filtered: FilterResult,
    answer: str,
    no_context: bool,
    threshold: float,
) -> ModeResult:
    kept = list(filtered.kept)
    dropped = list(filtered.dropped)
    return ModeResult(
        mode=mode,
        original_query=question,
        rewritten_query=rewritten,
        retrieved_before_filter=list(filtered.chunks),
        filtered_out=dropped,
        retrieved_after_filter=kept,
        answer=answer,
        sources=_sources(kept),
        no_relevant_context=no_context,
        threshold=threshold,
    )
