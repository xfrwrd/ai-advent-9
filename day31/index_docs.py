"""Build the documentation index. Running it again replaces the file.

    python -m day31.index_docs
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from day21.chunking import Chunk, StructureChunker
from day21.embeddings import LocalHashEmbedder
from day21.index_store import save_index
from day31.corpus import load_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = Path(__file__).resolve().parent / "index" / "docs.sqlite"


def build_index(root: Path | None = None, path: Path | None = None) -> Path:
    root = root or PROJECT_ROOT
    path = path or INDEX_PATH
    documents = load_documents(root)
    chunker = StructureChunker()
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for document in documents:
        for chunk in chunker.chunk(document):
            start_line = document.content.count("\n", 0, chunk.start) + 1
            end_line = document.content.count("\n", 0, max(chunk.end - 1, 0)) + 1
            labelled = replace(chunk, section=f"{chunk.section} · lines {start_line}-{end_line}")
            if labelled.chunk_id in seen:
                continue
            seen.add(labelled.chunk_id)
            chunks.append(labelled)
    embedder = LocalHashEmbedder()
    vectors = embedder.embed_batch([chunk.text for chunk in chunks]) if chunks else []
    save_index(path, chunks, vectors, embedder.name, embedder.dimension)
    return path


def main() -> None:
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT
    path = build_index(root)
    from day21.index_store import load_index

    loaded = load_index(path)
    sources = sorted({chunk.source for chunk in loaded.chunks})
    print(f"index {path}")
    print(f"files {len(sources)}")
    print(f"chunks {len(loaded.chunks)}")
    for source in sources:
        print(f"  {source}")


if __name__ == "__main__":
    main()
