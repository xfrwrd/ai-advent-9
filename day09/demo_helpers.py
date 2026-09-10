"""Day 09 demo helpers — long dialog with key facts for compression tests."""

from __future__ import annotations

import json
from pathlib import Path

from agent import estimate_messages_tokens


def build_contextflow_dialogue() -> list[dict]:
    """Long dialog: important facts at start/middle/end + filler in between."""
    messages: list[dict] = [
        {
            "role": "user",
            "content": "Проект называется ContextFlow.",
        },
        {
            "role": "assistant",
            "content": "Понял: проект — ContextFlow. Запомнил название.",
        },
        {
            "role": "user",
            "content": "Backend написан на Go.",
        },
        {
            "role": "assistant",
            "content": "Ок, backend ContextFlow — на Go.",
        },
    ]

    filler = (
        "Обсуждаем второстепенные детали: стиль логов, формат дат в UI, "
        "нейминг временных переменных и мелкие правки README. "
    ) * 8

    for index in range(8):
        messages.append(
            {
                "role": "user",
                "content": f"[filler-{index}] {filler}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"[filler-reply-{index}] Принял замечание #{index}. Продолжаем.",
            }
        )

    messages.extend(
        [
            {
                "role": "user",
                "content": "Для БД используется Postgres.",
            },
            {
                "role": "assistant",
                "content": "Записал: база данных — Postgres.",
            },
        ]
    )

    for index in range(4):
        messages.append(
            {
                "role": "user",
                "content": f"[more-filler-{index}] Ещё обсуждаем отступы в markdown и emoji в чате.",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"[more-filler-reply-{index}] Ок, учёл косметический пункт #{index}.",
            }
        )

    messages.extend(
        [
            {
                "role": "user",
                "content": "Теперь хотим добавить AI-планировщик.",
            },
            {
                "role": "assistant",
                "content": "Отлично: следующий шаг — добавить AI-планировщик в ContextFlow.",
            },
        ]
    )
    return messages


RECALL_QUESTION = (
    "Напомни название проекта, backend, базу данных и что мы хотим добавить."
)


def load_contextflow_demo(history_path: Path, summary_path: Path) -> int:
    """Write the ContextFlow dialogue to history.json and clear summary.json."""
    messages = build_contextflow_dialogue()
    history_path.write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps({"summary": ""}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return estimate_messages_tokens(messages)
