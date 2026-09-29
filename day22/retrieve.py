"""Rank indexed chunks for a question.

Hash embeddings from Day 21 collide, so the rank is BM25 over the chunk text
and file path, with the embedding cosine used as the tie-break.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from day21.chunking import Chunk
from day21.embeddings import Embedder
from day21.index_store import LoadedIndex

_TOKEN = re.compile(r"\w+")
_STOP = {
    "какой", "какая", "какие", "что", "где", "как", "чем", "чему",
    "в", "из", "за", "и", "или", "у", "на", "по", "для", "это",
    "а", "не", "ли", "да", "нет", "о", "об", "от", "до", "при",
    "со", "же", "бы", "то",
}


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def search(index: LoadedIndex, embedder: Embedder, question: str, k: int = 4) -> list[Hit]:
    if not index.chunks or k <= 0:
        return []
    documents = [_tokens(f"{chunk.source} {chunk.section} {chunk.text}") for chunk in index.chunks]
    df: Counter[str] = Counter()
    for tokens in documents:
        df.update(set(tokens))
    count = len(documents)
    average = sum(len(tokens) for tokens in documents) / count
    query = [token for token in _tokens(question) if token not in _STOP]
    query_vector = np.asarray(embedder.embed_batch([question])[0], dtype=np.float32)
    matrix = np.asarray(index.vectors, dtype=np.float32)
    cosine = matrix @ query_vector
    lexical = np.array([_bm25(query, tokens, df, count, average) for tokens in documents])
    order = np.lexsort((-cosine, -lexical))[:k]
    return [
        Hit(index.chunks[int(i)], float(lexical[int(i)]) + float(cosine[int(i)]))
        for i in order
    ]


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _bm25(query: list[str], tokens: list[str], df: Counter[str], count: int, average: float) -> float:
    if not query or not tokens:
        return 0.0
    frequency = Counter(tokens)
    length = len(tokens)
    score = 0.0
    k1 = 1.2
    b = 0.75
    for token in set(query):
        seen = frequency[token]
        if not seen:
            continue
        idf = math.log((count - df[token] + 0.5) / (df[token] + 0.5) + 1)
        score += idf * (seen * (k1 + 1)) / (seen + k1 * (1 - b + b * length / average))
    return score
