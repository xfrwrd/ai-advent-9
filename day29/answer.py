"""Generate one answer from chunks that were retrieved earlier."""

from __future__ import annotations

import time
from dataclasses import dataclass

from day24.ground import (
    UNKNOWN_ANSWER,
    fallback_quotes,
    parse_grounded,
    retrieved_from_hits,
    sources_from_hits,
    validate_quotes,
)
from day24.response import RAGResponse
from day29.config import CONFIGS, GenerateConfig
from day29.ollama import Generation, generate
from day29.prompts import context_budget, render_prompt
from day29.retrieve import FrozenContext


@dataclass
class ModeAnswer:
    config: GenerateConfig
    response: RAGResponse
    retrieval_s: float
    generation_s: float
    total_s: float
    called_model: bool
    context_budget: dict[str, object]
    generation: Generation | None

    def as_dict(self) -> dict[str, object]:
        return {
            "config": self.config.as_dict(),
            "response": self.response.as_dict(),
            "retrieval_s": self.retrieval_s,
            "generation_s": self.generation_s,
            "total_s": self.total_s,
            "called_model": self.called_model,
            "context_budget": self.context_budget,
            "generation": None if self.generation is None else self.generation.as_dict(),
        }


def answer_with(context: FrozenContext, config: GenerateConfig) -> ModeAnswer:
    started = time.perf_counter()
    prompt = render_prompt(context.question, context.hits, config)
    budget = context_budget(prompt, config.num_ctx)
    if not context.hits:
        response = RAGResponse(
            answer=UNKNOWN_ANSWER,
            sources=[],
            quotes=[],
            retrieved_chunks=[],
            is_grounded=False,
            insufficient_context=True,
            original_query=context.question,
            rewritten_query=context.search_query,
            threshold=context.threshold,
            top_score=context.top_score,
        )
        total_s = time.perf_counter() - started
        return ModeAnswer(config, response, context.retrieval_s, 0.0, total_s, False, budget, None)
    gen_started = time.perf_counter()
    generated = generate(prompt, config)
    generation_s = time.perf_counter() - gen_started
    text, proposed = parse_grounded(generated.text)
    quotes = validate_quotes(proposed, context.hits) or fallback_quotes(text, context.hits)
    response = RAGResponse(
        answer=text,
        sources=sources_from_hits(context.hits),
        quotes=quotes,
        retrieved_chunks=retrieved_from_hits(context.hits),
        is_grounded=True,
        insufficient_context=False,
        original_query=context.question,
        rewritten_query=context.search_query,
        threshold=context.threshold,
        top_score=context.top_score,
    )
    return ModeAnswer(
        config,
        response,
        context.retrieval_s,
        generation_s,
        time.perf_counter() - started,
        True,
        budget,
        generated,
    )


def config_for(mode: str) -> GenerateConfig:
    if mode not in CONFIGS:
        raise ValueError(f"unknown mode: {mode}")
    return CONFIGS[mode]
