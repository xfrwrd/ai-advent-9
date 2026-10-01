"""Day 23 retrieval and filter, then a grounded answer with checked quotes."""

from __future__ import annotations

from day22.llm import chat
from day23.agent import Day23Agent, load_agent
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


class Day24Agent:
    def __init__(self, pipeline: Day23Agent) -> None:
        self.pipeline = pipeline

    def ask(self, question: str) -> RAGResponse:
        rewritten = rewrite_query(question, self.pipeline.chat_fn)
        hits = self.pipeline.search_fn(
            self.pipeline.index,
            self.pipeline.embedder,
            rewritten,
            self.pipeline.retrieval_top_k,
        )
        filtered = apply_filter(
            hits,
            self.pipeline.similarity_threshold,
            self.pipeline.filtered_top_k,
            self.pipeline.higher_is_better,
        )
        top_score = max((hit.score for hit in hits), default=None)
        if not filtered.kept_hits:
            return RAGResponse(
                answer=UNKNOWN_ANSWER,
                sources=[],
                quotes=[],
                retrieved_chunks=[],
                is_grounded=False,
                insufficient_context=True,
                original_query=question,
                rewritten_query=rewritten,
                threshold=self.pipeline.similarity_threshold,
                top_score=top_score,
            )
        kept = list(filtered.kept_hits)
        raw = self.pipeline.chat_fn(grounded_messages(question, kept))
        answer, proposed = parse_grounded(raw)
        quotes = validate_quotes(proposed, kept) or fallback_quotes(answer, kept)
        return RAGResponse(
            answer=answer,
            sources=sources_from_hits(kept),
            quotes=quotes,
            retrieved_chunks=retrieved_from_hits(kept),
            is_grounded=True,
            insufficient_context=False,
            original_query=question,
            rewritten_query=rewritten,
            threshold=self.pipeline.similarity_threshold,
            top_score=top_score,
        )


def load_day24(chat_fn=chat) -> Day24Agent:
    return Day24Agent(load_agent(chat_fn=chat_fn))
