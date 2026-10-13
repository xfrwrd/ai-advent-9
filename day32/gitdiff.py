"""Read a diff between two Git refs. Commands are fixed. shell is never used."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from day31.corpus import SECRET_NAMES, SECRET_SUFFIXES, SKIP_DIRS

_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./~^\-]*$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_STATUS = {"A": "added", "M": "modified", "D": "deleted", "T": "modified"}

MAX_FILE_CHARS = 12_000
MAX_TOTAL_CHARS = 48_000
MAX_FILES = 40


@dataclass
class FileChange:
    status: str
    path: str
    old_path: str | None
    binary: bool
    diff: str
    omitted_reason: str | None
    new_spans: list[tuple[int, int]] = field(default_factory=list)
    old_spans: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class DiffBundle:
    base_ref: str
    head_ref: str
    base_commit: str
    head_commit: str
    files: list[FileChange]
    error: str | None = None


def load_diff(
    repo: Path,
    base: str,
    head: str,
    *,
    max_file_chars: int = MAX_FILE_CHARS,
    max_total_chars: int = MAX_TOTAL_CHARS,
    max_files: int = MAX_FILES,
) -> DiffBundle:
    repo = repo.resolve()
    if not _valid_ref(base) or not _valid_ref(head):
        return _failed(base, head, "Некорректный Git ref. Ожидается имя ветки, тег или commit.")
    code, out, _err = run_git(repo, ["rev-parse", "--is-inside-work-tree"])
    if code != 0 or out.strip() != "true":
        return _failed(base, head, "Директория не является Git-репозиторием.")
    base_commit, base_error = _resolve(repo, base)
    if base_error:
        return _failed(base, head, base_error)
    head_commit, head_error = _resolve(repo, head)
    if head_error:
        return _failed(base, head, head_error)
    assert base_commit is not None and head_commit is not None

    code, raw, err = run_git(
        repo,
        ["diff", "--name-status", "-z", "--find-renames", "--end-of-options", base_commit, head_commit],
    )
    if code != 0:
        return _failed(base, head, err.strip() or "git diff --name-status не выполнился.")
    listed = _parse_name_status(raw)
    binary = _binary_paths(repo, base_commit, head_commit)
    files: list[FileChange] = []
    used = 0
    for index, (status, path, old_path) in enumerate(listed):
        if _secret(path) or (old_path and _secret(old_path)):
            files.append(FileChange(status, path, old_path, False, "", "секрет, в модель не отправляется"))
            continue
        if index >= max_files:
            files.append(FileChange(status, path, old_path, False, "", f"слишком много файлов, лимит {max_files}"))
            continue
        if path in binary or (old_path and old_path in binary):
            files.append(FileChange(status, path, old_path, True, "", "бинарный файл"))
            continue
        diff_text = _file_diff(repo, base_commit, head_commit, path)
        if "Binary files" in diff_text or "GIT binary patch" in diff_text:
            files.append(FileChange(status, path, old_path, True, "", "бинарный файл"))
            continue
        if len(diff_text) > max_file_chars:
            files.append(
                FileChange(
                    status,
                    path,
                    old_path,
                    False,
                    "",
                    f"diff {len(diff_text)} символов, лимит {max_file_chars}",
                )
            )
            continue
        if used + len(diff_text) > max_total_chars:
            files.append(
                FileChange(
                    status,
                    path,
                    old_path,
                    False,
                    "",
                    f"превышен общий лимит анализа {max_total_chars} символов",
                )
            )
            continue
        old_spans, new_spans = _spans(diff_text)
        used += len(diff_text)
        files.append(FileChange(status, path, old_path, False, diff_text, None, new_spans, old_spans))
    return DiffBundle(base, head, base_commit, head_commit, files)


def run_git(repo: Path, args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return 127, "", "git не установлен"
    except subprocess.TimeoutExpired:
        return 124, "", "git не завершился вовремя"
    return completed.returncode, completed.stdout, completed.stderr


def run_git_bytes(repo: Path, args: list[str], timeout: int = 30) -> tuple[int, bytes, str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=False,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return 127, b"", "git не установлен"
    except subprocess.TimeoutExpired:
        return 124, b"", "git не завершился вовремя"
    return completed.returncode, completed.stdout, completed.stderr.decode("utf-8", errors="replace")


def _failed(base: str, head: str, message: str) -> DiffBundle:
    return DiffBundle(base, head, "", "", [], message)


def _valid_ref(ref: str) -> bool:
    if not ref or len(ref) > 200 or ref.startswith("-") or ".." in ref.split("/"):
        return False
    return _REF.fullmatch(ref) is not None


def _resolve(repo: Path, ref: str) -> tuple[str | None, str | None]:
    code, out, _err = run_git(repo, ["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"])
    if code != 0 or not out.strip():
        return None, f"Ref не найден: {ref}"
    return out.strip(), None


def _parse_name_status(raw: str) -> list[tuple[str, str, str | None]]:
    parts = raw.split("\0")
    if parts and parts[-1] == "":
        parts.pop()
    changes: list[tuple[str, str, str | None]] = []
    index = 0
    while index < len(parts):
        status = parts[index]
        index += 1
        if not status:
            continue
        if status[0] in {"R", "C"}:
            if index + 1 >= len(parts):
                break
            old, new = parts[index], parts[index + 1]
            index += 2
            kind = "renamed" if status[0] == "R" else "added"
            changes.append((kind, new, old if kind == "renamed" else None))
            continue
        if index >= len(parts):
            break
        path = parts[index]
        index += 1
        changes.append((_STATUS.get(status[0], "modified"), path, None))
    return changes


def _binary_paths(repo: Path, base: str, head: str) -> set[str]:
    code, out, _err = run_git(repo, ["diff", "--numstat", "-z", "--find-renames", "--end-of-options", base, head])
    if code != 0:
        return set()
    found: set[str] = set()
    for part in out.split("\0"):
        fields = part.split("\t")
        if len(fields) >= 3 and fields[0] == "-" and fields[1] == "-":
            found.add(fields[-1])
    return found


def _file_diff(repo: Path, base: str, head: str, path: str) -> str:
    _code, out, _err = run_git(
        repo,
        [
            "diff",
            "--unified=3",
            "--no-color",
            "--no-ext-diff",
            "--find-renames",
            "--end-of-options",
            base,
            head,
            "--",
            path,
        ],
    )
    return out


def _spans(diff_text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    old_spans: list[tuple[int, int]] = []
    new_spans: list[tuple[int, int]] = []
    for line in diff_text.splitlines():
        match = _HUNK.match(line)
        if not match:
            continue
        old_start = int(match.group(1))
        old_count = int(match.group(2) or "1")
        new_start = int(match.group(3))
        new_count = int(match.group(4) or "1")
        if old_count > 0:
            old_spans.append((old_start, old_start + old_count - 1))
        if new_count > 0:
            new_spans.append((new_start, new_start + new_count - 1))
    return old_spans, new_spans


def _secret(path: str) -> bool:
    name = Path(path).name
    if name.startswith(".env") or name in SECRET_NAMES:
        return True
    if Path(path).suffix.lower() in SECRET_SUFFIXES:
        return True
    return any(part in SKIP_DIRS for part in Path(path).parts)
