"""Pick documentation and source chunks for a diff. Retrieval is Day 31's."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from day21.chunking import Chunk
from day21.index_store import LoadedIndex, load_index
from day31.retrieve import relevant
from day32.gitdiff import DiffBundle

EXCERPT_CHARS = 1_500


@dataclass
class ContextPack:
    docs: list[Chunk] = field(default_factory=list)
    code: list[Chunk] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    indexed_commit: str | None = None


def gather(repo: Path, diff: DiffBundle, *, k: int = 4) -> ContextPack:
    manifest_file = repo / "day32" / "index" / "manifest.json"
    if not manifest_file.is_file():
        return ContextPack(notes=["Индекс Day 32 не собран. Выполните python -m day32.index_code"])
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    indexed = str(manifest.get("commit") or "")
    pack = ContextPack(indexed_commit=indexed or None)
    if not indexed or indexed != diff.head_commit:
        short_index = indexed[:12] if indexed else "не записан"
        pack.notes.append(
            f"Индекс собран на {short_index}, ревью для {diff.head_commit[:12]}. "
            "RAG не используется, чтобы не смешать версии кода."
        )
        return pack
    docs_index = _load(repo / str(manifest.get("docs_index") or "day32/index/docs.sqlite"))
    code_index = _load(repo / str(manifest.get("code_index") or "day32/index/code.sqlite"))
    query = " ".join(change.path for change in diff.files)
    if docs_index is None or not docs_index.chunks:
        pack.notes.append("Релевантная документация не найдена.")
    else:
        pack.docs = [hit.chunk for hit in relevant(docs_index, query, k=k)]
        if not pack.docs:
            pack.notes.append("Релевантная документация не найдена.")
    paths = {change.path for change in diff.files}
    if change_old := {change.old_path for change in diff.files if change.old_path}:
        paths.update(change_old)
    if code_index is None or not code_index.chunks:
        pack.notes.append("Релевантный исходный код в индексе не найден.")
    else:
        pack.code = _code_chunks(code_index, paths, query, k)
        if not pack.code:
            pack.notes.append("Релевантный исходный код в индексе не найден.")
    return pack


def render_excerpt(chunk: Chunk) -> str:
    text = chunk.text.strip()
    note = ""
    if len(text) > EXCERPT_CHARS:
        text = text[:EXCERPT_CHARS].rstrip()
        note = "\n[фрагмент сокращён, показаны первые 1500 символов]"
    return f"Источник: {chunk.source}\nРаздел: {chunk.section}\nchunk_id: {chunk.chunk_id}\n{text}{note}"


def _load(path: Path) -> LoadedIndex | None:
    if not path.is_file():
        return None
    return load_index(path)


def _code_chunks(index: LoadedIndex, paths: set[str], query: str, k: int) -> list[Chunk]:
    chosen: list[Chunk] = []
    seen: set[str] = set()
    for chunk in index.chunks:
        if chunk.source in paths and chunk.chunk_id not in seen:
            chosen.append(chunk)
            seen.add(chunk.chunk_id)
        if len(chosen) == k:
            return chosen
    for hit in relevant(index, query, k=k):
        if hit.chunk.chunk_id in seen:
            continue
        chosen.append(hit.chunk)
        seen.add(hit.chunk.chunk_id)
        if len(chosen) == k:
            break
    return chosen
