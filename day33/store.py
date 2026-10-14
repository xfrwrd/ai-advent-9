"""Read-only DemoPortal records. Only the MCP server imports this module."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
USER_FIELDS = ("user_id", "display_name", "account_status", "contact_hint", "registered_on")
TICKET_FIELDS = (
    "ticket_id",
    "user_id",
    "status",
    "category",
    "error_code",
    "created_at",
    "summary",
    "messages",
)
BRIEF_FIELDS = (
    "ticket_id",
    "user_id",
    "status",
    "category",
    "error_code",
    "created_at",
    "summary",
)


def data_path() -> Path:
    raw = os.environ.get("DAY33_DATA", "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent / "data" / "support.json"


def invalid_id(value: str) -> bool:
    return not isinstance(value, str) or _ID.fullmatch(value) is None


def load_records(path: Path | None = None) -> tuple[dict | None, str]:
    file = path or data_path()
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"Файл данных не найден: {file}"
    except json.JSONDecodeError:
        return None, "Файл данных поддержки повреждён."
    except OSError as exc:
        return None, f"Файл данных не прочитан: {exc.__class__.__name__}"
    if not isinstance(payload, dict):
        return None, "Файл данных поддержки повреждён."
    return payload, ""


def get_user(user_id: str, path: Path | None = None) -> dict:
    if invalid_id(user_id):
        return {"error": "invalid_id", "message": "Некорректный user_id."}
    records, problem = load_records(path)
    if records is None:
        return {"error": "data_unavailable", "message": problem}
    for user in records.get("users") or []:
        if isinstance(user, dict) and user.get("user_id") == user_id:
            return {key: user.get(key) for key in USER_FIELDS}
    return {"error": "not_found", "message": f"Пользователь {user_id} не найден."}


def get_ticket(ticket_id: str, path: Path | None = None) -> dict:
    if invalid_id(ticket_id):
        return {"error": "invalid_id", "message": "Некорректный ticket_id."}
    ticket, problem = _ticket(ticket_id, path)
    if ticket is None:
        if problem:
            return {"error": "data_unavailable", "message": problem}
        return {"error": "not_found", "message": f"Тикет {ticket_id} не найден."}
    public = {key: ticket.get(key) for key in TICKET_FIELDS if key != "messages"}
    public["messages"] = _messages(ticket.get("messages"))
    return public


def list_user_tickets(user_id: str, path: Path | None = None) -> dict:
    if invalid_id(user_id):
        return {"error": "invalid_id", "message": "Некорректный user_id."}
    records, problem = load_records(path)
    if records is None:
        return {"error": "data_unavailable", "message": problem}
    known = False
    tickets = []
    for user in records.get("users") or []:
        if isinstance(user, dict) and user.get("user_id") == user_id:
            known = True
            break
    if not known:
        return {"error": "not_found", "message": f"Пользователь {user_id} не найден."}
    for ticket in records.get("tickets") or []:
        if isinstance(ticket, dict) and ticket.get("user_id") == user_id:
            tickets.append({key: ticket.get(key) for key in BRIEF_FIELDS})
    return {"user_id": user_id, "tickets": tickets}


def get_ticket_history(ticket_id: str, path: Path | None = None) -> dict:
    if invalid_id(ticket_id):
        return {"error": "invalid_id", "message": "Некорректный ticket_id."}
    ticket, problem = _ticket(ticket_id, path)
    if ticket is None:
        if problem:
            return {"error": "data_unavailable", "message": problem}
        return {"error": "not_found", "message": f"Тикет {ticket_id} не найден."}
    return {"ticket_id": ticket_id, "messages": _messages(ticket.get("messages"))}


def _ticket(ticket_id: str, path: Path | None) -> tuple[dict | None, str]:
    records, problem = load_records(path)
    if records is None:
        return None, problem
    for ticket in records.get("tickets") or []:
        if isinstance(ticket, dict) and ticket.get("ticket_id") == ticket_id:
            return ticket, ""
    return None, ""


def _messages(raw: object) -> list[dict]:
    if not isinstance(raw, list):
        return []
    messages = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        messages.append(
            {
                "at": item.get("at"),
                "author": item.get("author"),
                "text": item.get("text"),
            }
        )
    return messages
