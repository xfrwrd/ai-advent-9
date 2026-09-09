"""Day 08 demo helpers — keep synthetic history out of Agent production logic."""

from __future__ import annotations

import json
from pathlib import Path

from agent import estimate_messages_tokens


def inflate_history_file(
    history_path: Path,
    pairs: int = 20,
    chars_per_message: int = 400,
) -> int:
    """Append synthetic user/assistant pairs to history.json without calling the API.

    Returns approximate estimated tokens of the resulting history.
    """
    if history_path.exists():
        try:
            messages = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(messages, list):
                messages = []
        except (OSError, json.JSONDecodeError):
            messages = []
    else:
        messages = []

    filler = ("Токен-тест. " * 40)[:chars_per_message]
    for index in range(pairs):
        messages.append(
            {
                "role": "user",
                "content": f"[demo-{index}] {filler}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"[demo-reply-{index}] {filler}",
            }
        )

    history_path.write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return estimate_messages_tokens(messages)
