"""One Day 23 rewrite, with the task and the latest messages beside the question."""

from __future__ import annotations

import json

from day22.agent import ChatFn
from day23.rewrite import rewrite_query
from day25.models import ChatMessage, TaskState


def context_aware_query(
    message: str,
    state: TaskState,
    recent: list[ChatMessage],
    chat_fn: ChatFn,
) -> str:
    return rewrite_query(message, chat_fn, context=_context(state, recent))


def _context(state: TaskState, recent: list[ChatMessage]) -> str:
    lines = [f"TaskState: {json.dumps(state.as_dict(), ensure_ascii=False)}"]
    for item in recent:
        lines.append(f"{item.role}: {item.content}")
    return "\n".join(lines)
