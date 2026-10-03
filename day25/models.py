"""Conversation history and the task memory that sits beside it."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class TaskState:
    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    clarified_facts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "goal": self.goal,
            "constraints": list(self.constraints),
            "terms": list(self.terms),
            "clarified_facts": list(self.clarified_facts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> TaskState:
        data = data or {}
        return cls(
            goal=str(data.get("goal") or ""),
            constraints=_strings(data.get("constraints")),
            terms=_strings(data.get("terms")),
            clarified_facts=_strings(data.get("clarified_facts")),
        )


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str
    sources: list[dict[str, str]] = field(default_factory=list)
    quotes: list[dict[str, str]] = field(default_factory=list)
    search_query: str = ""
    insufficient_context: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "sources": self.sources,
            "quotes": self.quotes,
            "search_query": self.search_query,
            "insufficient_context": self.insufficient_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ChatMessage:
        return cls(
            role=str(data["role"]),
            content=str(data["content"]),
            timestamp=str(data["timestamp"]),
            sources=list(data.get("sources") or []),
            quotes=list(data.get("quotes") or []),
            search_query=str(data.get("search_query") or ""),
            insufficient_context=bool(data.get("insufficient_context")),
        )


@dataclass
class Conversation:
    conversation_id: str
    messages: list[ChatMessage]
    task_state: TaskState
    created_at: str
    updated_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "conversation_id": self.conversation_id,
            "messages": [message.as_dict() for message in self.messages],
            "task_state": self.task_state.as_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Conversation:
        return cls(
            conversation_id=str(data["conversation_id"]),
            messages=[ChatMessage.from_dict(item) for item in data.get("messages") or []],
            task_state=TaskState.from_dict(data.get("task_state")),  # type: ignore[arg-type]
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
        )


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
