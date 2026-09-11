"""Day 10 demo helpers — shared scenario for Sliding Window / Sticky Facts / Branching."""

from __future__ import annotations

import json
from pathlib import Path

from agent import estimate_messages_tokens

FINAL_QUESTION = (
    "Сформируй итоговое ТЗ и перечисли:\n"
    "- название проекта\n"
    "- backend\n"
    "- базу данных\n"
    "- срок MVP\n"
    "- ограничения\n"
    "- последнее принятое решение."
)

BRANCH_A_PROMPT = "Добавляем мобильное приложение."
BRANCH_B_PROMPT = "Мобильное приложение не делаем, оставляем только web."

CONTEXTFLOW_FACTS = {
    "project_name": "ContextFlow",
    "goal": "Сделать MVP за 2 недели",
    "constraints": "Нельзя использовать внешние SaaS-сервисы",
    "preferences": "Для первой версии оставляем только web-интерфейс",
    "decisions": "В первой версии только web-интерфейс",
}


def build_contextflow_scenario() -> list[dict]:
    """10–15 messages: key facts first, filler, then final decision."""
    messages: list[dict] = [
        {"role": "user", "content": "Проект называется ContextFlow."},
        {
            "role": "assistant",
            "content": "Запомнил: проект называется ContextFlow.",
        },
        {"role": "user", "content": "Backend написан на Go."},
        {"role": "assistant", "content": "Ок, backend — Go."},
        {"role": "user", "content": "База данных — Postgres."},
        {"role": "assistant", "content": "База данных: Postgres."},
        {"role": "user", "content": "MVP нужно сделать за 2 недели."},
        {"role": "assistant", "content": "Срок MVP — 2 недели."},
        {
            "role": "user",
            "content": "Нельзя использовать внешние SaaS-сервисы.",
        },
        {
            "role": "assistant",
            "content": "Ограничение: без внешних SaaS.",
        },
        {
            "role": "user",
            "content": "Цвета кнопок пока синие, логотип потом нарисуем.",
        },
        {
            "role": "assistant",
            "content": "Принял косметические детали — они не критичны для MVP.",
        },
        {
            "role": "user",
            "content": "Давай обсудим нейминг временных переменных в примерах README.",
        },
        {
            "role": "assistant",
            "content": "Можем вернуться к неймингу позже, на архитектуру не влияет.",
        },
        {
            "role": "user",
            "content": "Для первой версии оставляем только web-интерфейс.",
        },
        {
            "role": "assistant",
            "content": "Решение: в первой версии только web-интерфейс.",
        },
    ]
    return messages


def build_checkpoint_messages() -> list[dict]:
    """Base requirements only — used as branching checkpoint."""
    return [
        {"role": "user", "content": "Проект называется ContextFlow."},
        {"role": "assistant", "content": "Проект: ContextFlow."},
        {"role": "user", "content": "Backend написан на Go."},
        {"role": "assistant", "content": "Backend: Go."},
        {"role": "user", "content": "База данных — Postgres."},
        {"role": "assistant", "content": "БД: Postgres."},
        {"role": "user", "content": "MVP нужно сделать за 2 недели."},
        {"role": "assistant", "content": "Срок MVP: 2 недели."},
        {
            "role": "user",
            "content": "Нельзя использовать внешние SaaS-сервисы.",
        },
        {
            "role": "assistant",
            "content": "Ограничение: без внешних SaaS.",
        },
    ]


def load_shared_scenario(history_path: Path, facts_path: Path | None = None) -> int:
    messages = build_contextflow_scenario()
    history_path.write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if facts_path is not None:
        facts_path.write_text(
            json.dumps(CONTEXTFLOW_FACTS, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return estimate_messages_tokens(messages)
