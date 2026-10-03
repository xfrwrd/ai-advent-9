"""Update task memory from the latest user message without wiping earlier facts."""

from __future__ import annotations

import json

from day22.agent import ChatFn
from day25.models import ChatMessage, TaskState

STATE_SYSTEM = (
    "Обнови TaskState диалога по новому сообщению пользователя. "
    "Учитывай предыдущее состояние. "
    "Не очищай goal, constraints, terms и clarified_facts, если пользователь этого прямо не сделал. "
    "Меняй goal только если пользователь явно меняет цель. "
    "Не отвечай на вопрос. "
    'Верни один JSON: {"goal": "...", "constraints": ["..."], "terms": ["..."], "clarified_facts": ["..."]}'
)

GOAL_CHANGE_MARKERS = (
    "меняю цель",
    "меняем цель",
    "новая цель",
    "теперь хочу",
    "смени цель",
    "другая цель",
)


def update_task_state(
    state: TaskState,
    message: str,
    recent: list[ChatMessage],
    chat_fn: ChatFn,
) -> TaskState:
    proposed = _extract(state, message, recent, chat_fn)
    return merge_task_state(state, proposed, message)


def merge_task_state(old: TaskState, proposed: TaskState, message: str) -> TaskState:
    goal = old.goal
    suggested = proposed.goal.strip()
    if not old.goal:
        goal = suggested
    elif suggested and _changes_goal(message):
        goal = suggested
    return TaskState(
        goal=goal,
        constraints=_union(old.constraints, proposed.constraints),
        terms=_union(old.terms, proposed.terms),
        clarified_facts=_union(old.clarified_facts, proposed.clarified_facts),
    )


def working_messages(messages: list[ChatMessage], limit: int) -> list[ChatMessage]:
    if limit <= 0:
        return []
    return messages[-limit:]


def _extract(state: TaskState, message: str, recent: list[ChatMessage], chat_fn: ChatFn) -> TaskState:
    history = "\n".join(f"{item.role}: {item.content}" for item in recent)
    raw = chat_fn(
        [
            {"role": "system", "content": STATE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Предыдущее состояние:\n{json.dumps(state.as_dict(), ensure_ascii=False)}\n\n"
                    f"Последние сообщения:\n{history or '(пусто)'}\n\n"
                    f"Новое сообщение:\n{message}"
                ),
            },
        ]
    )
    return TaskState.from_dict(_parse_object(raw))


def _parse_object(raw: str) -> dict[str, object]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _changes_goal(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in GOAL_CHANGE_MARKERS)


def _union(old: list[str], new: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for item in [*old, *new]:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged
