"""Read-only file tools. Every path goes through policy.resolve first."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from pathlib import Path

from day34.policy import MAX_FILE_BYTES, SKIP_DIRS, PathDenied, is_secret, redact, repo_root, resolve

WALK_SKIP = SKIP_DIRS | {"artifacts", "reports", "index"}
MAX_LIST = 300
MAX_SCAN_FILES = 3000
MAX_RESULTS = 200
MAX_READ_LINES = 400
LINE_PREVIEW = 220


def list_files(path: str = ".", pattern: str = "", max_entries: int = 200, root: Path | None = None) -> dict:
    try:
        base = root or repo_root()
        folder, relative = resolve(path, base)
    except PathDenied as denied:
        return denied.payload()
    if not folder.is_dir():
        return {"error": "not_a_directory", "message": f"Это не каталог: {relative}"}
    cap = max(1, min(int(max_entries), MAX_LIST))
    files: list[str] = []
    hidden = 0
    truncated = False
    for current, rel in _walk(folder, base):
        if is_secret(current.name):
            hidden += 1
            continue
        if pattern and not (fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(current.name, pattern)):
            continue
        if len(files) >= cap:
            truncated = True
            break
        files.append(rel)
    return {
        "path": relative,
        "pattern": pattern,
        "files": files,
        "count": len(files),
        "truncated": truncated,
        "hidden_secrets": hidden,
    }


def search_in_files(
    query: str,
    glob: str = "",
    path: str = ".",
    max_results: int = 60,
    case_sensitive: bool = False,
    root: Path | None = None,
) -> dict:
    if not isinstance(query, str) or not query.strip():
        return {"error": "invalid_query", "message": "Пустой поисковый запрос."}
    try:
        base = root or repo_root()
        folder, relative = resolve(path, base)
    except PathDenied as denied:
        return denied.payload()
    if not folder.is_dir():
        return {"error": "not_a_directory", "message": f"Это не каталог: {relative}"}
    cap = max(1, min(int(max_results), MAX_RESULTS))
    needle = query if case_sensitive else query.lower()
    matches: list[dict] = []
    files_with_matches: list[str] = []
    skipped = {"secret": 0, "binary": 0, "large": 0}
    scanned = 0
    truncated_reason = ""
    for current, rel in _walk(folder, base):
        if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(current.name, glob)):
            continue
        if is_secret(current.name):
            skipped["secret"] += 1
            continue
        if scanned >= MAX_SCAN_FILES:
            truncated_reason = f"просмотрено не больше {MAX_SCAN_FILES} файлов"
            break
        text, problem = _text(current)
        if problem:
            skipped[problem] += 1
            continue
        scanned += 1
        found_here = False
        for number, line in enumerate(text.splitlines(), start=1):
            haystack = line if case_sensitive else line.lower()
            if needle not in haystack:
                continue
            if len(matches) >= cap:
                truncated_reason = f"показано не больше {cap} совпадений"
                break
            found_here = True
            matches.append({"path": rel, "line": number, "text": redact(line.strip())[:LINE_PREVIEW]})
        if found_here:
            files_with_matches.append(rel)
        if truncated_reason:
            break
    return {
        "query": query,
        "glob": glob,
        "path": relative,
        "matches": matches,
        "files_with_matches": files_with_matches,
        "files_scanned": scanned,
        "truncated": bool(truncated_reason),
        "truncated_reason": truncated_reason,
        "skipped": skipped,
    }


def read_file(path: str, start_line: int = 1, max_lines: int = 200, root: Path | None = None) -> dict:
    try:
        base = root or repo_root()
        target, relative = resolve(path, base)
    except PathDenied as denied:
        return denied.payload()
    if not target.exists():
        return {"error": "not_found", "message": f"Файл не найден: {relative}"}
    if not target.is_file():
        return {"error": "not_a_file", "message": f"Это не файл: {relative}"}
    text, problem = _text(target)
    if problem == "large":
        return {"error": "too_large", "message": f"Файл больше {MAX_FILE_BYTES} байт: {relative}"}
    if problem == "binary":
        return {"error": "binary_file", "message": f"Бинарный файл: {relative}"}
    lines = text.splitlines()
    start = max(1, int(start_line))
    count = max(1, min(int(max_lines), MAX_READ_LINES))
    chosen = lines[start - 1 : start - 1 + count]
    end = start + len(chosen) - 1
    numbered = "\n".join(f"{number:>5}| {redact(line)}" for number, line in enumerate(chosen, start=start))
    return {
        "path": relative,
        "start_line": start,
        "end_line": end,
        "total_lines": len(lines),
        "truncated": end < len(lines),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "content": numbered,
    }


def _walk(folder: Path, root: Path):
    root = root.resolve()
    for current, dirs, names in os.walk(folder, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in WALK_SKIP)
        for name in sorted(names):
            path = Path(current) / name
            try:
                real = path.resolve()
            except OSError:
                continue
            if not real.is_relative_to(root) or not real.is_file():
                continue
            yield path, path.relative_to(root).as_posix()


def _text(path: Path) -> tuple[str, str]:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return "", "large"
        raw = path.read_bytes()
    except OSError:
        return "", "binary"
    if b"\x00" in raw[:4096]:
        return "", "binary"
    return raw.decode("utf-8", errors="replace"), ""
