"""Separate relevance stage after retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from day22.retrieve import Hit

KEPT = "KEPT"
FILTERED_OUT = "FILTERED OUT"


@dataclass(frozen=True)
class RankedChunk:
    rank: int
    chunk_id: str
    source: str
    section: str
    score: float
    status: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "section": self.section,
            "score": self.score,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FilterResult:
    chunks: tuple[RankedChunk, ...]
    kept: tuple[RankedChunk, ...]
    kept_hits: tuple[Hit, ...]

    @property
    def dropped(self) -> tuple[RankedChunk, ...]:
        return tuple(chunk for chunk in self.chunks if chunk.status == FILTERED_OUT)


def apply_filter(
    hits: list[Hit],
    threshold: float,
    filtered_top_k: int,
    higher_is_better: bool = True,
) -> FilterResult:
    """Drop hits outside the threshold, then keep at most filtered_top_k.

    Input order is the retrieval order. Passing hits are sorted so the better
    score comes first. A larger score is better only when higher_is_better is set.
    """
    passing: list[tuple[int, Hit]] = []
    failing: list[tuple[int, Hit]] = []
    for rank, hit in enumerate(hits, start=1):
        if _passes(hit.score, threshold, higher_is_better):
            passing.append((rank, hit))
        else:
            failing.append((rank, hit))
    passing.sort(key=lambda item: _sort_key(item[1].score, item[0], higher_is_better))
    limit = max(filtered_top_k, 0)
    kept_pairs = passing[:limit]
    overflow = passing[limit:]
    by_rank: dict[int, RankedChunk] = {}
    for rank, hit in kept_pairs:
        by_rank[rank] = _chunk(rank, hit, KEPT, "kept")
    for rank, hit in overflow:
        by_rank[rank] = _chunk(rank, hit, FILTERED_OUT, "beyond filtered_top_k")
    for rank, hit in failing:
        reason = "below threshold" if higher_is_better else "above threshold"
        by_rank[rank] = _chunk(rank, hit, FILTERED_OUT, reason)
    ordered = tuple(by_rank[rank] for rank in sorted(by_rank))
    kept = tuple(by_rank[rank] for rank, _ in kept_pairs)
    kept_hits = tuple(hit for _, hit in kept_pairs)
    return FilterResult(ordered, kept, kept_hits)


def _passes(score: float, threshold: float, higher_is_better: bool) -> bool:
    if higher_is_better:
        return score >= threshold
    return score <= threshold


def _sort_key(score: float, rank: int, higher_is_better: bool) -> tuple[float, int]:
    return (-score if higher_is_better else score, rank)


def _chunk(rank: int, hit: Hit, status: str, reason: str) -> RankedChunk:
    return RankedChunk(
        rank=rank,
        chunk_id=hit.chunk.chunk_id,
        source=hit.chunk.source,
        section=hit.chunk.section,
        score=hit.score,
        status=status,
        reason=reason,
    )
