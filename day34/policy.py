"""Which paths the file tools may touch. Checked in Python, never by the model."""

from __future__ import annotations

import os
import re
from pathlib import Path

from day31.corpus import SECRET_NAMES as DAY31_SECRET_NAMES
from day31.corpus import SECRET_SUFFIXES as DAY31_SECRET_SUFFIXES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    ".idea",
    ".cursor",
}
NO_WRITE_DIRS = SKIP_DIRS | {"index", "reports", "artifacts", "data", ".github"}
SECRET_NAMES = DAY31_SECRET_NAMES | {".netrc", ".npmrc", ".pypirc", "secrets.json", "token.json"}
SECRET_SUFFIXES = DAY31_SECRET_SUFFIXES | {".token", ".secret", ".keystore", ".jks"}
WRITE_SUFFIXES = {".md", ".txt"}
MAX_FILE_BYTES = 300_000

_SECRET_VALUE = re.compile(
    r"(?i)((?:api[_-]?key|secret|token|password|passwd)\w*\s*[:=]\s*)([\"'])([^\"'\s]{8,})\2"
)
_KNOWN_TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b")


class PathDenied(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def payload(self) -> dict:
        return {"error": self.code, "message": self.message}


def repo_root() -> Path:
    raw = os.environ.get("DAY34_REPO", "").strip()
    root = Path(raw).resolve() if raw else PROJECT_ROOT
    if not (root / ".git").exists():
        raise PathDenied("not_a_git_repo", f"Каталог не является Git-репозиторием: {root}")
    return root


def artifacts_dir() -> Path:
    raw = os.environ.get("DAY34_ARTIFACTS", "").strip()
    return Path(raw).resolve() if raw else PROJECT_ROOT / "day34" / "artifacts"


def resolve(raw: object, root: Path, *, write: bool = False) -> tuple[Path, str]:
    """Absolute real path and repo-relative posix path, or PathDenied."""
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise PathDenied("invalid_path", "Путь должен быть непустой строкой.")
    text = raw.strip()
    given = Path(os.path.abspath(root))
    root = root.resolve()
    candidate = Path(text) if Path(text).is_absolute() else root / text
    lexical = Path(os.path.normpath(candidate))
    if not (lexical.is_relative_to(root) or lexical.is_relative_to(given)):
        raise PathDenied("outside_root", f"Путь вне репозитория: {text}")
    real = lexical.resolve()
    if not real.is_relative_to(root):
        raise PathDenied("outside_root", f"Путь ведёт за пределы репозитория через symlink: {text}")
    relative = real.relative_to(root).as_posix() or "."
    parts = real.relative_to(root).parts
    if any(part in SKIP_DIRS for part in parts):
        raise PathDenied("forbidden_dir", f"Каталог закрыт для инструментов: {relative}")
    if parts and is_secret(real.name):
        raise PathDenied("secret_file", f"Файл похож на секрет и не читается: {relative}")
    if write:
        if any(part in NO_WRITE_DIRS for part in parts[:-1]):
            raise PathDenied("write_forbidden", f"Запись в этот каталог запрещена: {relative}")
        if real.suffix.lower() not in WRITE_SUFFIXES:
            raise PathDenied("write_forbidden", "Записывать можно только .md и .txt.")
        if lexical.is_symlink():
            raise PathDenied("write_forbidden", f"Запись через symlink запрещена: {relative}")
    return real, relative


def is_secret(name: str) -> bool:
    lowered = name.lower()
    if lowered.startswith(".env") or lowered in SECRET_NAMES:
        return True
    if any(lowered.endswith(suffix) for suffix in SECRET_SUFFIXES):
        return True
    if lowered.startswith(("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519")):
        return True
    stem = lowered.split(".", 1)[0]
    return "secret" in stem or "credential" in stem


def redact(text: str) -> str:
    text = _SECRET_VALUE.sub(lambda match: f"{match.group(1)}{match.group(2)}***{match.group(2)}", text)
    return _KNOWN_TOKEN.sub("***", text)
