"""Short dialog memory. A session is bound to one demo actor and one mode."""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_MESSAGES = 8
MAX_CHARS = 2000


@dataclass
class Session:
    session_id: str
    user_id: str
    mode: str
    messages: list[dict] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._items: dict[str, Session] = {}

    def open(self, session_id: str, user_id: str, mode: str) -> tuple[Session | None, str]:
        found = self._items.get(session_id)
        if found is not None and (found.user_id != user_id or found.mode != mode):
            return None, "Эта сессия принадлежит другому пользователю. История недоступна."
        if found is None:
            found = Session(session_id=session_id, user_id=user_id, mode=mode)
            self._items[session_id] = found
        return found, ""

    def add(self, session: Session, question: str, answer: str) -> None:
        session.messages.append({"role": "user", "content": question})
        session.messages.append({"role": "assistant", "content": answer})
        while len(session.messages) > MAX_MESSAGES:
            session.messages.pop(0)
        while session.messages and sum(len(item["content"]) for item in session.messages) > MAX_CHARS:
            session.messages.pop(0)

    def snapshot(self, session_id: str) -> list[dict]:
        found = self._items.get(session_id)
        if found is None:
            return []
        return [dict(item) for item in found.messages]


default_sessions = SessionStore()
