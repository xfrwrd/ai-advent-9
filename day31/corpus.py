"""Documentation files for the Day 31 index.

The Day 21 loader walks the whole tree, including Python. This one does not.
"""

from __future__ import annotations

from pathlib import Path

from day21.documents import Document

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    ".index",
    "index",
}
SECRET_NAMES = {".env", "credentials.json", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
MAX_BYTES = 400_000


def collect(root: Path) -> list[Path]:
    """README, docs, day READMEs, and API descriptions. Secrets stay out."""
    found: list[Path] = []
    readme = root / "README.md"
    if _allowed(readme, root):
        found.append(readme)
    docs = root / "docs"
    if docs.is_dir():
        for path in sorted(docs.rglob("*")):
            if not _allowed(path, root):
                continue
            if path.suffix.lower() not in {".md", ".txt", ".yaml", ".yml", ".json"}:
                continue
            found.append(path)
    for path in sorted(root.glob("day*/README.md")):
        # The assistant's own page lists the demo questions. Indexing it
        # would make every question look "found" in this file.
        if path.parent.name == "day31":
            continue
        if _allowed(path, root):
            found.append(path)
    for name in ("openapi.yaml", "openapi.yml", "openapi.json"):
        path = root / name
        if _allowed(path, root):
            found.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def load_documents(root: Path) -> list[Document]:
    documents: list[Document] = []
    for path in collect(root):
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


def _allowed(path: Path, root: Path) -> bool:
    if not path.is_file():
        return False
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    if any(part in SKIP_DIRS or part.startswith(".") for part in relative.parts[:-1]):
        return False
    name = path.name
    if name.startswith(".") or name in SECRET_NAMES:
        return False
    if path.suffix.lower() in SECRET_SUFFIXES:
        return False
    try:
        if path.stat().st_size > MAX_BYTES:
            return False
    except OSError:
        return False
    return True
