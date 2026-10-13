"""Build docs and code indexes from one Git commit.

The blobs come from `git show`, not from the worktree, so an uncommitted edit
does not mix with the commit being reviewed.

    python -m day32.index_code
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from day21.chunking import Chunk, StructureChunker
from day21.documents import Document
from day21.embeddings import LocalHashEmbedder
from day21.index_store import save_index
from day31.corpus import MAX_BYTES, SECRET_NAMES, SECRET_SUFFIXES, SKIP_DIRS
from day32.gitdiff import run_git, run_git_bytes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOC_SUFFIXES = {".md", ".txt", ".yaml", ".yml", ".json"}


def index_dir(repo: Path) -> Path:
    return repo / "day32" / "index"


def manifest_path(repo: Path) -> Path:
    return index_dir(repo) / "manifest.json"


def build_snapshot(repo: Path, commit: str | None = None) -> dict:
    repo = repo.resolve()
    if commit is None:
        code, out, err = run_git(repo, ["rev-parse", "--verify", "--end-of-options", "HEAD^{commit}"])
        if code != 0 or not out.strip():
            raise RuntimeError(err.strip() or "HEAD не найден.")
        commit = out.strip()
    tree_code, tree, tree_err = run_git(repo, ["ls-tree", "-r", "--name-only", "--end-of-options", commit])
    if tree_code != 0:
        raise RuntimeError(tree_err.strip() or "git ls-tree не выполнился.")
    docs: list[Document] = []
    sources: list[Document] = []
    for path in tree.splitlines():
        if not path or path.startswith("-") or ".." in Path(path).parts:
            continue
        role = classify(path)
        if role == "skip":
            continue
        blob = _blob(repo, commit, path)
        if blob is None or not blob.strip():
            continue
        document = Document(
            source=path,
            title=Path(path).name,
            content=blob,
            metadata={"file_type": Path(path).suffix.lower().lstrip(".")},
        )
        if role == "doc":
            docs.append(document)
        else:
            sources.append(document)
    destination = index_dir(repo)
    docs_path = destination / "docs.sqlite"
    code_path = destination / "code.sqlite"
    docs_chunks = _store(docs_path, docs)
    code_chunks = _store(code_path, sources)
    manifest = {
        "commit": commit,
        "docs_index": docs_path.relative_to(repo).as_posix(),
        "code_index": code_path.relative_to(repo).as_posix(),
        "docs": len(docs),
        "code_files": len(sources),
        "chunks_docs": docs_chunks,
        "chunks_code": code_chunks,
    }
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path(repo).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def classify(path: str) -> str:
    """Return doc, code, or skip. Secrets and vendor trees are skip."""
    parts = Path(path).parts
    if any(part in SKIP_DIRS or part.startswith(".") for part in parts[:-1]):
        return "skip"
    name = parts[-1] if parts else path
    if name.startswith(".") or name in SECRET_NAMES or name.startswith(".env"):
        return "skip"
    suffix = Path(name).suffix.lower()
    if suffix in SECRET_SUFFIXES:
        return "skip"
    if path in {"README.md", "openapi.yaml", "openapi.yml", "openapi.json"}:
        return "doc"
    if path.startswith("docs/") and suffix in _DOC_SUFFIXES:
        return "doc"
    if re.fullmatch(r"day[^/]+/README\.md", path) and path not in {"day31/README.md", "day32/README.md"}:
        return "doc"
    if suffix == ".py":
        return "code"
    return "skip"


def _blob(repo: Path, commit: str, path: str) -> str | None:
    code, raw, _err = run_git_bytes(repo, ["show", "--end-of-options", f"{commit}:{path}"])
    if code != 0 or b"\0" in raw or len(raw) > MAX_BYTES:
        return None
    return raw.decode("utf-8", errors="ignore")


def _store(path: Path, documents: list[Document]) -> int:
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
    return len(chunks)


def main() -> None:
    manifest = build_snapshot(PROJECT_ROOT)
    print(f"commit {manifest['commit']}")
    print(f"docs {manifest['docs']} files, {manifest['chunks_docs']} chunks")
    print(f"code {manifest['code_files']} files, {manifest['chunks_code']} chunks")
    print(f"manifest {manifest_path(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
