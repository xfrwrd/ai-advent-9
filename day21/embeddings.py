"""Local embeddings. No paid API.

local-token-hash-v1 hashes each word into a fixed-size vector and L2-normalizes it.
The same text always produces the same vector.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from day21.config import EMBEDDING_DIM, EMBEDDING_MODEL


class Embedder(Protocol):
    name: str
    dimension: int

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class LocalHashEmbedder:
    name = EMBEDDING_MODEL

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self.dimension = dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[index] += sign
            norm = float(np.linalg.norm(vec))
            if norm:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors
