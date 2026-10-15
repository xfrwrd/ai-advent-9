"""Two-phase edits. propose stores a diff, apply writes it after confirmation.

The target file is written only in apply_file_change, and only if its content
still matches the hash taken when the diff was prepared.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

from day34.policy import PathDenied, artifacts_dir, repo_root, resolve

MAX_EDIT_CHARS = 8000
_PROPOSAL_ID = re.compile(r"^[0-9a-f]{16}$")
_NUMBERED = re.compile(r"^\s*\d+\| ?")


def propose_file_change(
    path: str,
    new_text: str,
    old_text: str = "",
    create: bool = False,
    reason: str = "",
    root: Path | None = None,
    artifacts: Path | None = None,
) -> dict:
    try:
        base = root or repo_root()
        target, relative = resolve(path, base, write=True)
    except PathDenied as denied:
        return denied.payload()
    if not isinstance(new_text, str) or not isinstance(old_text, str):
        return {"error": "invalid_args", "message": "old_text и new_text должны быть строками."}
    if len(new_text) > MAX_EDIT_CHARS or len(old_text) > MAX_EDIT_CHARS:
        return {"error": "too_large", "message": f"Правка длиннее {MAX_EDIT_CHARS} символов. Сделайте её меньше."}

    if create:
        if target.exists():
            return {"error": "already_exists", "message": f"Файл уже есть: {relative}. Используйте old_text и new_text."}
        if not new_text.strip():
            return {"error": "invalid_args", "message": "Для нового файла нужен new_text."}
        original = ""
        updated = new_text if new_text.endswith("\n") else new_text + "\n"
        base_sha = None
    else:
        if not target.is_file():
            return {"error": "not_found", "message": f"Файл не найден: {relative}. Для нового файла передайте create=true."}
        if not old_text:
            return {
                "error": "old_text_required",
                "message": "Укажите точный фрагмент old_text. Документ целиком не переписывается.",
            }
        raw = target.read_bytes()
        try:
            original = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {"error": "not_utf8", "message": f"Файл не в UTF-8: {relative}"}
        base_sha = hashlib.sha256(raw).hexdigest()
        old, new = old_text, new_text
        if original.count(old) == 0 and all(_NUMBERED.match(line) for line in old.splitlines() if line):
            old = _strip_numbers(old)
            new = _strip_numbers(new)
        found = original.count(old)
        if found == 0:
            if new and new in original:
                return {"status": "already_applied", "path": relative, "message": "Этот текст уже есть в файле. Правка не нужна."}
            return {
                "error": "old_text_not_found",
                "message": "old_text не найден. Скопируйте фрагмент из read_file без номеров строк.",
            }
        if found > 1:
            return {"error": "old_text_not_unique", "message": f"old_text встречается {found} раз. Возьмите фрагмент длиннее."}
        updated = original.replace(old, new, 1)
        if updated == original:
            return {"status": "no_change", "path": relative, "message": "Изменений нет."}

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile="/dev/null" if create else f"a/{relative}",
            tofile=f"b/{relative}",
        )
    )
    new_sha = hashlib.sha256(updated.encode("utf-8")).hexdigest()
    proposal_id = hashlib.sha256(f"{relative}|{base_sha}|{new_sha}".encode()).hexdigest()[:16]
    folder = (artifacts or artifacts_dir()) / "proposals"
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / f"{proposal_id}.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    if record.get("status") != "pending":
        record = {
            "proposal_id": proposal_id,
            "path": relative,
            "mode": "create" if create else "modify",
            "base_sha256": base_sha,
            "new_sha256": new_sha,
            "diff": diff,
            "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
            "reason": reason if isinstance(reason, str) else "",
            "new_content": updated,
            "status": "pending",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    return {
        "proposal_id": proposal_id,
        "path": relative,
        "mode": record["mode"],
        "status": record["status"],
        "diff": record["diff"],
        "diff_sha256": record["diff_sha256"],
        "reason": record["reason"],
        "lines_added": added,
        "lines_removed": removed,
        "written": False,
    }


def apply_file_change(
    proposal_id: str,
    confirm_token: str,
    root: Path | None = None,
    artifacts: Path | None = None,
) -> dict:
    record_path, record, problem = _load(proposal_id, artifacts)
    if problem:
        return problem
    if record["status"] == "applied":
        return {"status": "already_applied", "path": record["path"], "message": "Изменение уже применено."}
    if record["status"] == "cancelled":
        return {"error": "cancelled", "message": "Изменение отменено. Подготовьте его заново."}
    if not isinstance(confirm_token, str) or confirm_token != record["diff_sha256"]:
        return {"error": "confirmation_mismatch", "message": "Подтверждение не совпадает с подготовленным diff."}
    try:
        base = root or repo_root()
        target, relative = resolve(record["path"], base, write=True)
    except PathDenied as denied:
        return denied.payload()

    if record["mode"] == "create":
        if target.exists():
            return _stale(record_path, record, "Файл появился после подготовки diff.")
    else:
        if not target.is_file():
            return _stale(record_path, record, "Файл исчез после подготовки diff.")
        current = hashlib.sha256(target.read_bytes()).hexdigest()
        if current != record["base_sha256"]:
            return _stale(record_path, record, "Файл изменился после подготовки diff.")

    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".day34-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(record["new_content"])
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    written = target.read_bytes()
    verified = hashlib.sha256(written).hexdigest() == record["new_sha256"]
    record["status"] = "applied" if verified else "apply_failed"
    record["applied_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": record["status"],
        "path": relative,
        "verified": verified,
        "sha256": hashlib.sha256(written).hexdigest(),
        "lines": written.decode("utf-8").count("\n"),
    }


def cancel_file_change(proposal_id: str, artifacts: Path | None = None) -> dict:
    record_path, record, problem = _load(proposal_id, artifacts)
    if problem:
        return problem
    if record["status"] == "applied":
        return {"error": "already_applied", "message": "Изменение уже применено, отменять нечего."}
    record["status"] = "cancelled"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "cancelled", "path": record["path"]}


def _load(proposal_id: str, artifacts: Path | None) -> tuple[Path, dict, dict | None]:
    if not isinstance(proposal_id, str) or not _PROPOSAL_ID.fullmatch(proposal_id):
        return Path(), {}, {"error": "invalid_proposal", "message": "Некорректный proposal_id."}
    record_path = (artifacts or artifacts_dir()) / "proposals" / f"{proposal_id}.json"
    if not record_path.is_file():
        return record_path, {}, {"error": "not_found", "message": f"Изменение {proposal_id} не найдено."}
    return record_path, json.loads(record_path.read_text(encoding="utf-8")), None


def _stale(record_path: Path, record: dict, why: str) -> dict:
    record["status"] = "stale"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"error": "stale", "message": f"{why} Ничего не записано. Подготовьте diff заново."}


def _strip_numbers(text: str) -> str:
    return "\n".join(_NUMBERED.sub("", line, count=1) for line in text.split("\n"))
