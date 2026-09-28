"""Read project files into a filesystem-independent Document."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from day21.config import SKIP_DIRS, TEXT_EXTENSIONS


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    content: str
    metadata: dict = field(default_factory=dict)


def load_documents(root: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS or part.startswith(".") for part in path.relative_to(root).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        relative = path.relative_to(root).as_posix()
        documents.append(
            Document(
                source=relative,
                title=path.name,
                content=text,
                metadata={"file_type": path.suffix.lower().lstrip(".")},
            )
        )
    return documents


def dataset_stats(documents: list[Document]) -> dict:
    extensions: dict[str, int] = {}
    characters = 0
    words = 0
    for doc in documents:
        kind = "." + str(doc.metadata.get("file_type", ""))
        extensions[kind] = extensions.get(kind, 0) + 1
        characters += len(doc.content)
        words += len(doc.content.split())
    return {
        "files": len(documents),
        "extensions": extensions,
        "characters": characters,
        "words": words,
        "pages": round(characters / 3000, 1),
    }
