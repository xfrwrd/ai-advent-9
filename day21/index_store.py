"""SQLite index: one row per chunk and one vector blob with the same chunk_id."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from day21.chunking import Chunk


@dataclass
class LoadedIndex:
    model: str
    dimension: int
    strategy: str
    chunks: list[Chunk]
    vectors: list[list[float]]


def save_index(
    path: Path,
    chunks: list[Chunk],
    vectors: list[list[float]],
    model: str,
    dimension: int,
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("vectors and chunks must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            """
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                section TEXT NOT NULL,
                chunking_strategy TEXT NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                file_type TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE embeddings (
                chunk_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL
            )
            """
        )
        strategy = chunks[0].chunking_strategy if chunks else ""
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("model", model),
                ("dimension", str(dimension)),
                ("strategy", strategy),
                ("count", str(len(chunks))),
            ],
        )
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != dimension:
                raise ValueError(f"{chunk.chunk_id} has dimension {len(vector)}, expected {dimension}")
            conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, text, source, title, section, chunking_strategy,
                    start_offset, end_offset, file_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.text,
                    chunk.source,
                    chunk.title,
                    chunk.section,
                    chunk.chunking_strategy,
                    chunk.start,
                    chunk.end,
                    chunk.file_type,
                ),
            )
            blob = sqlite3.Binary(_pack(vector))
            conn.execute(
                "INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)",
                (chunk.chunk_id, blob),
            )
        conn.commit()
    finally:
        conn.close()


def load_index(path: Path) -> LoadedIndex:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}
        dimension = int(meta["dimension"])
        rows = conn.execute(
            """
            SELECT c.*, e.vector
            FROM chunks c
            JOIN embeddings e ON e.chunk_id = c.chunk_id
            ORDER BY c.source, c.start_offset
            """
        ).fetchall()
    finally:
        conn.close()
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for row in rows:
        vector = _unpack(row["vector"], dimension)
        chunks.append(
            Chunk(
                chunk_id=row["chunk_id"],
                text=row["text"],
                source=row["source"],
                title=row["title"],
                section=row["section"],
                chunking_strategy=row["chunking_strategy"],
                start=row["start_offset"],
                end=row["end_offset"],
                file_type=row["file_type"],
            )
        )
        vectors.append(vector)
    return LoadedIndex(
        model=meta["model"],
        dimension=dimension,
        strategy=meta.get("strategy", ""),
        chunks=chunks,
        vectors=vectors,
    )


def _pack(vector: list[float]) -> bytes:
    import array

    return array.array("f", vector).tobytes()


def _unpack(blob: bytes, dimension: int) -> list[float]:
    import array

    values = array.array("f")
    values.frombytes(blob)
    if len(values) != dimension:
        raise ValueError(f"stored vector length {len(values)} != {dimension}")
    return list(values)
