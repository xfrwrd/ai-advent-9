"""One local retrieval. Both generation modes reuse the kept chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass

from day21.config import INDEX_DIR
from day21.embeddings import LocalHashEmbedder
from day21.index_store import LoadedIndex, load_index
from day22.retrieve import Hit, search
from day23.config import FILTERED_TOP_K, HIGHER_IS_BETTER, RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD
from day23.filter import apply_filter


@dataclass
class FrozenContext:
    question: str
    search_query: str
    hits: list[Hit]
    retrieval_s: float
    top_score: float | None
    threshold: float


def load_frozen_index() -> LoadedIndex:
    return load_index(INDEX_DIR / "structure.sqlite")


def retrieve(question: str, index: LoadedIndex | None = None) -> FrozenContext:
    loaded = index or load_frozen_index()
    embedder = LocalHashEmbedder()
    started = time.perf_counter()
    hits = search(loaded, embedder, question, RETRIEVAL_TOP_K)
    filtered = apply_filter(hits, SIMILARITY_THRESHOLD, FILTERED_TOP_K, HIGHER_IS_BETTER)
    return FrozenContext(
        question=question,
        search_query=question,
        hits=list(filtered.kept_hits),
        retrieval_s=time.perf_counter() - started,
        top_score=max((hit.score for hit in hits), default=None),
        threshold=SIMILARITY_THRESHOLD,
    )
