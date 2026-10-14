"""Build the support knowledge index. Running it again replaces the file.

    python -m day33.index_support
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from day21.chunking import Chunk, StructureChunker
from day21.documents import Document
from day21.embeddings import LocalHashEmbedder
from day21.index_store import save_index

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
INDEX_PATH = Path(__file__).resolve().parent / "index" / "support.sqlite"


def load_knowledge(root: Path | None = None) -> list[Document]:
    folder = root or KNOWLEDGE_DIR
    documents: list[Document] = []
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        documents.append(
            Document(
                source=_source(path),
                title=path.stem,
                content=text,
                metadata={"file_type": "md"},
            )
        )
    return documents


def build_index(root: Path | None = None, path: Path | None = None) -> Path:
    folder = root or KNOWLEDGE_DIR
    path = path or INDEX_PATH
    documents = load_knowledge(folder)
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


def _source(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def main() -> None:
    from day21.index_store import load_index

    path = build_index()
    loaded = load_index(path)
    sources = sorted({chunk.source for chunk in loaded.chunks})
    print(f"index {path}")
    print(f"files {len(sources)}")
    print(f"chunks {len(loaded.chunks)}")
    for source in sources:
        print(f"  {source}")


if __name__ == "__main__":
    main()
