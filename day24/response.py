"""Structured RAG answer. Sources and quotes are data, not a markdown blob."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    source: str
    section: str
    chunk_id: str

    def as_dict(self) -> dict[str, str]:
        return {"source": self.source, "section": self.section, "chunk_id": self.chunk_id}


@dataclass(frozen=True)
class Quote:
    text: str
    source: str
    section: str
    chunk_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "text": self.text,
            "source": self.source,
            "section": self.section,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    chunk_id: str
    source: str
    section: str
    score: float
    text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "section": self.section,
            "score": self.score,
            "text": self.text,
        }


@dataclass
class RAGResponse:
    answer: str
    sources: list[Source] = field(default_factory=list)
    quotes: list[Quote] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    is_grounded: bool = False
    insufficient_context: bool = False
    original_query: str = ""
    rewritten_query: str | None = None
    threshold: float | None = None
    top_score: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "sources": [item.as_dict() for item in self.sources],
            "quotes": [item.as_dict() for item in self.quotes],
            "retrieved_chunks": [item.as_dict() for item in self.retrieved_chunks],
            "is_grounded": self.is_grounded,
            "insufficient_context": self.insufficient_context,
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "threshold": self.threshold,
            "top_score": self.top_score,
        }
