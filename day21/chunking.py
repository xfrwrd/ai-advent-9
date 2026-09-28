"""Fixed-size and structure-aware chunking behind one interface."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from day21.config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS
from day21.documents import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    chunking_strategy: str
    start: int
    end: int
    file_type: str


def _chunk_id(strategy: str, source: str, section: str, start: int, text: str) -> str:
    digest = hashlib.sha1(f"{strategy}|{source}|{section}|{start}|{text}".encode()).hexdigest()[:12]
    return f"{strategy}:{source}:{start}:{digest}"


def _make(
    doc: Document,
    strategy: str,
    section: str,
    start: int,
    end: int,
    text: str,
) -> Chunk:
    cleaned = text.strip()
    return Chunk(
        chunk_id=_chunk_id(strategy, doc.source, section, start, cleaned),
        text=cleaned,
        source=doc.source,
        title=doc.title,
        section=section,
        chunking_strategy=strategy,
        start=start,
        end=end,
        file_type=str(doc.metadata.get("file_type", "")),
    )


class FixedSizeChunker:
    name = "fixed"

    def __init__(self, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> None:
        if size <= 0:
            raise ValueError("chunk size must be positive")
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must be smaller than chunk size")
        self.size = size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.content
        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.size, len(text))
            if end < len(text):
                cut = text.rfind(" ", start, end)
                if cut > start + self.size // 2:
                    end = cut
            piece = text[start:end]
            if piece.strip():
                chunks.append(_make(doc, self.name, "fixed-window", start, end, piece))
            if end >= len(text):
                break
            next_start = max(end - self.overlap, start + 1)
            if next_start < len(text) and text[next_start - 1].isalnum() and text[next_start].isalnum():
                space = text.find(" ", next_start)
                next_start = space + 1 if space != -1 else end
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PY_CLASS = re.compile(r"^class\s+(\w+)")
_PY_DEF = re.compile(r"^(?:async\s+)?def\s+(\w+)")
_PY_METHOD = re.compile(r"^([ \t]+)(?:async\s+)?def\s+(\w+)")


class StructureChunker:
    name = "structure"

    def chunk(self, doc: Document) -> list[Chunk]:
        kind = str(doc.metadata.get("file_type", ""))
        if kind == "md":
            return self._markdown(doc)
        if kind == "py":
            return self._python(doc)
        return self._paragraphs(doc)

    def _markdown(self, doc: Document) -> list[Chunk]:
        lines = doc.content.splitlines(keepends=True)
        blocks: list[tuple[str, int, list[str]]] = []
        section = "preamble"
        start = 0
        cursor = 0
        buf: list[str] = []
        for line in lines:
            match = _MD_HEADING.match(line)
            if match and buf:
                blocks.append((section, start, buf))
                buf = []
                start = cursor
            if match:
                section = match.group(2).strip()
            buf.append(line)
            cursor += len(line)
        if buf:
            blocks.append((section, start, buf))
        return self._blocks(doc, blocks)

    def _python(self, doc: Document) -> list[Chunk]:
        lines = doc.content.splitlines(keepends=True)
        blocks: list[tuple[str, int, list[str]]] = []
        section = "module"
        current_class = ""
        start = 0
        cursor = 0
        buf: list[str] = []

        def flush(at: int) -> None:
            nonlocal buf, start
            if any(line.strip() for line in buf):
                blocks.append((section, start, buf))
            buf = []
            start = at

        for line in lines:
            class_match = _PY_CLASS.match(line)
            def_match = _PY_DEF.match(line)
            method_match = _PY_METHOD.match(line)
            if class_match or def_match or method_match:
                flush(cursor)
                if class_match:
                    current_class = class_match.group(1)
                    section = current_class
                elif def_match:
                    current_class = ""
                    section = def_match.group(1)
                else:
                    section = f"{current_class}.{method_match.group(2)}" if current_class else method_match.group(2)
            buf.append(line)
            cursor += len(line)
        flush(cursor)
        return self._blocks(doc, blocks)

    def _paragraphs(self, doc: Document) -> list[Chunk]:
        parts = re.split(r"\n\s*\n", doc.content)
        chunks: list[Chunk] = []
        cursor = 0
        for index, part in enumerate(parts, start=1):
            start = doc.content.find(part, cursor)
            if start < 0:
                start = cursor
            end = start + len(part)
            cursor = end
            if part.strip():
                chunks.append(_make(doc, self.name, f"paragraph-{index}", start, end, part))
        return chunks

    def _blocks(self, doc: Document, blocks: list[tuple[str, int, list[str]]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section, start, lines in blocks:
            text = "".join(lines)
            if text.strip():
                chunks.append(_make(doc, self.name, section, start, start + len(text), text))
        return chunks
