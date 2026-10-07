"""Day 24 retrieval and grounding, timed, with the answer model swapped."""

from __future__ import annotations

import time
from dataclasses import dataclass

from day21.config import INDEX_DIR
from day21.embeddings import Embedder, LocalHashEmbedder
from day21.index_store import LoadedIndex, load_index
from day22.agent import ChatFn
from day22.retrieve import search
from day23.agent import Day23Agent, SearchFn
from day23.config import FILTERED_TOP_K, HIGHER_IS_BETTER, RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD
from day23.filter import apply_filter
from day23.rewrite import rewrite_query
from day24.ground import (
    UNKNOWN_ANSWER,
    fallback_quotes,
    grounded_messages,
    parse_grounded,
    retrieved_from_hits,
    sources_from_hits,
    validate_quotes,
)
from day24.response import RAGResponse
from day28.llm import cloud_chat_fn, local_chat

LOCAL = "local"
CLOUD = "cloud"
PROVIDERS = (LOCAL, CLOUD)


@dataclass
class TimedRAG:
    provider: str
    response: RAGResponse
    retrieval_s: float
    generation_s: float
    total_s: float


class Day28RAG:
    def __init__(
        self,
        index: LoadedIndex,
        local_fn: ChatFn,
        cloud_fn: ChatFn,
        embedder: Embedder | None = None,
        search_fn: SearchFn = search,
    ) -> None:
        shared = embedder or LocalHashEmbedder()
        self.pipelines = {
            LOCAL: Day23Agent(
                index,
                chat_fn=local_fn,
                embedder=shared,
                search_fn=search_fn,
                retrieval_top_k=RETRIEVAL_TOP_K,
                filtered_top_k=FILTERED_TOP_K,
                similarity_threshold=SIMILARITY_THRESHOLD,
                higher_is_better=HIGHER_IS_BETTER,
            ),
            CLOUD: Day23Agent(
                index,
                chat_fn=cloud_fn,
                embedder=shared,
                search_fn=search_fn,
                retrieval_top_k=RETRIEVAL_TOP_K,
                filtered_top_k=FILTERED_TOP_K,
                similarity_threshold=SIMILARITY_THRESHOLD,
                higher_is_better=HIGHER_IS_BETTER,
            ),
        }

    def ask(self, question: str, provider: str) -> TimedRAG:
        if provider not in self.pipelines:
            raise ValueError(f"unknown provider: {provider}")
        pipeline = self.pipelines[provider]
        total_started = time.perf_counter()
        generation_s = 0.0
        gen_started = time.perf_counter()
        rewritten = rewrite_query(question, pipeline.chat_fn)
        generation_s += time.perf_counter() - gen_started
        retrieval_started = time.perf_counter()
        hits = pipeline.search_fn(
            pipeline.index,
            pipeline.embedder,
            rewritten,
            pipeline.retrieval_top_k,
        )
        filtered = apply_filter(
            hits,
            pipeline.similarity_threshold,
            pipeline.filtered_top_k,
            pipeline.higher_is_better,
        )
        retrieval_s = time.perf_counter() - retrieval_started
        top_score = max((hit.score for hit in hits), default=None)
        if not filtered.kept_hits:
            response = RAGResponse(
                answer=UNKNOWN_ANSWER,
                sources=[],
                quotes=[],
                retrieved_chunks=[],
                is_grounded=False,
                insufficient_context=True,
                original_query=question,
                rewritten_query=rewritten,
                threshold=pipeline.similarity_threshold,
                top_score=top_score,
            )
        else:
            kept = list(filtered.kept_hits)
            gen_started = time.perf_counter()
            raw = pipeline.chat_fn(grounded_messages(question, kept))
            generation_s += time.perf_counter() - gen_started
            answer, proposed = parse_grounded(raw)
            quotes = validate_quotes(proposed, kept) or fallback_quotes(answer, kept)
            response = RAGResponse(
                answer=answer,
                sources=sources_from_hits(kept),
                quotes=quotes,
                retrieved_chunks=retrieved_from_hits(kept),
                is_grounded=True,
                insufficient_context=False,
                original_query=question,
                rewritten_query=rewritten,
                threshold=pipeline.similarity_threshold,
                top_score=top_score,
            )
        return TimedRAG(
            provider=provider,
            response=response,
            retrieval_s=retrieval_s,
            generation_s=generation_s,
            total_s=time.perf_counter() - total_started,
        )


def load_day28() -> Day28RAG:
    index = load_index(INDEX_DIR / "structure.sqlite")
    return Day28RAG(index, local_chat(), cloud_chat_fn())
