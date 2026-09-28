"""Index the project with both chunking strategies.

Run from the repository root:

    python -m day21.index_documents
"""

from __future__ import annotations

from day21.chunking import Chunk, FixedSizeChunker, StructureChunker
from day21.config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    EMBEDDING_MODEL,
    INDEX_DIR,
    PROJECT_ROOT,
)
from day21.documents import dataset_stats, load_documents
from day21.embeddings import LocalHashEmbedder
from day21.index_store import load_index, save_index


def _lengths(chunks: list[Chunk]) -> tuple[int, int, int]:
    sizes = [len(chunk.text) for chunk in chunks] or [0]
    return min(sizes), max(sizes), round(sum(sizes) / len(sizes))


def _examples(chunks: list[Chunk], limit: int = 3) -> list[Chunk]:
    picked: list[Chunk] = []
    for chunk in chunks:
        if chunk.text.strip():
            picked.append(chunk)
        if len(picked) == limit:
            break
    return picked


def _line(text: str, last: bool = False) -> str:
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    if not rows:
        return ""
    return rows[-1] if last else rows[0]


def _boundary_example(chunks_fixed: list[Chunk], chunks_structure: list[Chunk]) -> str:
    source = "day20/agent.py"
    fixed = [chunk for chunk in chunks_fixed if chunk.source == source]
    whole = next(
        (chunk for chunk in chunks_structure if chunk.source == source and chunk.section == "Agent.run"),
        None,
    )
    if whole is None or not fixed:
        return "No shared example document was found."
    split = next(
        (
            chunk
            for chunk in fixed
            if chunk.start < whole.end and chunk.end > whole.start and chunk.text != whole.text
        ),
        fixed[0],
    )
    return "\n".join(
        [
            f"Document {source}",
            "",
            f"fixed       {len(split.text)} chars, offset {split.start}",
            f"  starts    {_line(split.text)}",
            f"  ends      {_line(split.text, last=True)}",
            "",
            f"structure   {whole.section}, {len(whole.text)} chars",
            f"  starts    {_line(whole.text)}",
            f"  ends      {_line(whole.text, last=True)}",
        ]
    )


def build(root=PROJECT_ROOT) -> None:
    documents = load_documents(root)
    stats = dataset_stats(documents)
    print("Day 21 index")
    print(f"  files        {stats['files']}")
    print(f"  extensions   {stats['extensions']}")
    print(f"  characters   {stats['characters']}")
    print(f"  words        {stats['words']}")
    print(f"  pages        {stats['pages']}  (~3000 characters)")
    print(f"  embeddings   {EMBEDDING_MODEL}")

    embedder = LocalHashEmbedder()
    strategies = [
        FixedSizeChunker(CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS),
        StructureChunker(),
    ]
    stored: dict[str, list[Chunk]] = {}
    for chunker in strategies:
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(chunker.chunk(doc))
        vectors = embedder.embed_batch([chunk.text for chunk in chunks])
        path = INDEX_DIR / f"{chunker.name}.sqlite"
        save_index(path, chunks, vectors, embedder.name, embedder.dimension)
        loaded = load_index(path)
        minimum, maximum, average = _lengths(chunks)
        print()
        print(chunker.name)
        print(f"  chunks       {len(chunks)}")
        print(f"  embeddings   {len(vectors)}")
        print(f"  dimension    {embedder.dimension}")
        print(f"  length       min {minimum}  max {maximum}  avg {average} chars")
        print(f"  saved        {path}")
        print(f"  reloaded     {len(loaded.chunks)} chunks, {len(loaded.vectors)} vectors")
        for sample in _examples(chunks):
            print(f"  sample       {sample.source}  [{sample.section}]")
            print(f"               {_line(sample.text)}")
        stored[chunker.name] = chunks

    print()
    print(_boundary_example(stored["fixed"], stored["structure"]))


if __name__ == "__main__":
    build()
