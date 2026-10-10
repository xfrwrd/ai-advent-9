"""In-memory chats. A known session_id is not enough to read one."""

from __future__ import annotations

import hmac
import secrets
import time
from dataclasses import dataclass, field

from day30.config import MAX_MESSAGES, MAX_SESSIONS, SESSION_TTL_S


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class Session:
    session_id: str
    token: str
    messages: list[Turn] = field(default_factory=list)
    updated_at: float = field(default_factory=time.monotonic)


class SessionStore:
    def __init__(self, max_sessions: int = MAX_SESSIONS, ttl_s: float = SESSION_TTL_S) -> None:
        self.max_sessions = max_sessions
        self.ttl_s = ttl_s
        self._sessions: dict[str, Session] = {}

    def open(self, session_id: str, token: str | None) -> tuple[Session, str | None]:
        """Return the session, or an error code: missing_token, bad_token."""
        self._purge()
        current = self._sessions.get(session_id)
        if current is None:
            if len(self._sessions) >= self.max_sessions:
                self._drop_oldest()
            created = Session(session_id=session_id, token=secrets.token_urlsafe(32))
            self._sessions[session_id] = created
            return created, None
        if not token or not hmac.compare_digest(token.encode(), current.token.encode()):
            return current, "bad_token"
        current.updated_at = time.monotonic()
        return current, None

    def remember(self, session: Session, user_text: str, answer: str) -> None:
        session.messages.append(Turn("user", user_text))
        session.messages.append(Turn("assistant", answer))
        overflow = len(session.messages) - MAX_MESSAGES
        if overflow > 0:
            del session.messages[:overflow]
        session.updated_at = time.monotonic()

    def _purge(self) -> None:
        cutoff = time.monotonic() - self.ttl_s
        stale = [key for key, item in self._sessions.items() if item.updated_at < cutoff]
        for key in stale:
            del self._sessions[key]

    def _drop_oldest(self) -> None:
        if not self._sessions:
            return
        oldest = min(self._sessions, key=lambda key: self._sessions[key].updated_at)
        del self._sessions[oldest]
