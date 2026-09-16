"""Day 13: Explicit three-layer memory (Day 11/12 model + TASK_STATE_KEY)."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SHORT_TERM_PATH = BASE_DIR / "short_term.json"
WORKING_MEMORY_PATH = BASE_DIR / "working_memory.json"
LONG_TERM_MEMORY_PATH = BASE_DIR / "long_term_memory.json"

# Structured profile blob lives in Long-Term under this key.
PROFILE_KEY = "user_profile"
# Structured TaskState JSON lives in Working Memory under this key.
TASK_STATE_KEY = "task_state"


class MemoryLayer(str, Enum):
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"


LAYER_LABELS = {
    MemoryLayer.SHORT_TERM: "Short-Term Memory",
    MemoryLayer.WORKING: "Working Memory",
    MemoryLayer.LONG_TERM: "Long-Term Memory",
}


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(str(message.get("role", "")))
        total += estimate_tokens(str(message.get("content", "")))
        total += 4
    return total


def estimate_kv_tokens(data: dict[str, str]) -> int:
    if not data:
        return 0
    return estimate_tokens(json.dumps(data, ensure_ascii=False))


class MemoryStore:
    """Three independent memory layers with explicit write targeting."""

    def __init__(self, base_dir: Path | None = None) -> None:
        base = base_dir or BASE_DIR
        self.short_term_path = base / "short_term.json"
        self.working_path = base / "working_memory.json"
        self.long_term_path = base / "long_term_memory.json"

        self.short_term: list[dict] = self._load_list(self.short_term_path)
        self.working: dict[str, str] = self._load_dict(self.working_path)
        self.long_term: dict[str, str] = self._load_dict(self.long_term_path)

    # ----- persistence -----

    def _load_list(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _load_dict(self, path: Path) -> dict[str, str]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(key): "" if value is None else str(value)
            for key, value in data.items()
        }

    def _save_short_term(self) -> None:
        self.short_term_path.write_text(
            json.dumps(self.short_term, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_working(self) -> None:
        self.working_path.write_text(
            json.dumps(self.working, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save_long_term(self) -> None:
        self.long_term_path.write_text(
            json.dumps(self.long_term, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ----- short-term -----

    def append_message(self, role: str, content: str) -> None:
        self.short_term.append({"role": role, "content": content})
        self._save_short_term()

    def clear_short_term(self) -> None:
        self.short_term = []
        self._save_short_term()

    # ----- working -----

    def set_working_memory(self, key: str, value: str) -> None:
        key = key.strip()
        if not key:
            raise ValueError("Ключ Working Memory не может быть пустым.")
        self.working[key] = value
        self._save_working()

    def remove_working_memory(self, key: str) -> None:
        self.working.pop(key, None)
        self._save_working()

    def clear_working_memory(self) -> None:
        self.working = {}
        self._save_working()

    # ----- long-term -----

    def set_long_term_memory(self, key: str, value: str) -> None:
        key = key.strip()
        if not key:
            raise ValueError("Ключ Long-Term Memory не может быть пустым.")
        self.long_term[key] = value
        self._save_long_term()

    def remove_long_term_memory(self, key: str) -> None:
        self.long_term.pop(key, None)
        self._save_long_term()

    def clear_long_term_memory(self) -> None:
        self.long_term = {}
        self._save_long_term()

    def remember(self, layer: MemoryLayer, key: str, value: str) -> None:
        if layer == MemoryLayer.WORKING:
            self.set_working_memory(key, value)
        elif layer == MemoryLayer.LONG_TERM:
            self.set_long_term_memory(key, value)
        elif layer == MemoryLayer.SHORT_TERM:
            self.append_message("user", value)
        else:
            raise ValueError(f"Неизвестный слой памяти: {layer}")

    def clear_all(self) -> None:
        self.clear_short_term()
        self.clear_working_memory()
        self.clear_long_term_memory()

    # ----- prompt blocks -----

    def format_long_term_block(self, *, exclude_keys: set[str] | None = None) -> str:
        exclude = exclude_keys or set()
        items = {
            key: value
            for key, value in self.long_term.items()
            if key not in exclude
        }
        if not items:
            return "Long-term memory:\n(empty)"
        lines = ["Long-term memory:"]
        for key, value in items.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def format_working_block(self, *, exclude_keys: set[str] | None = None) -> str:
        exclude = exclude_keys or set()
        items = {
            key: value
            for key, value in self.working.items()
            if key not in exclude
        }
        if not items:
            return "Working memory:\n(empty)"
        lines = ["Working memory:"]
        for key, value in items.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def format_recent_dialogue_block(self) -> str:
        if not self.short_term:
            return "Recent dialogue:\n(empty)"
        lines = ["Recent dialogue:"]
        for message in self.short_term:
            role = message.get("role", "?")
            content = message.get("content", "")
            lines.append(f"- {role}: {content}")
        return "\n".join(lines)

    def build_system_prompt(
        self,
        *,
        exclude_lt_keys: set[str] | None = None,
        exclude_working_keys: set[str] | None = None,
    ) -> str:
        return "\n\n".join(
            [
                self.format_long_term_block(exclude_keys=exclude_lt_keys),
                self.format_working_block(exclude_keys=exclude_working_keys),
            ]
        )

    def layer_size_estimates(self) -> dict[str, int]:
        return {
            MemoryLayer.SHORT_TERM.value: estimate_messages_tokens(self.short_term),
            MemoryLayer.WORKING.value: estimate_kv_tokens(self.working),
            MemoryLayer.LONG_TERM.value: estimate_kv_tokens(self.long_term),
        }
