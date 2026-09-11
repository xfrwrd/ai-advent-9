"""Day 10: Context strategies — Sliding Window, Sticky Facts, Branching (no summary)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import requests

API_URL = "https://api.deepseek.com/chat/completions"
THINKING_DISABLED = {"type": "disabled"}

MAX_CONTEXT_TOKENS = 1_000_000
INPUT_PRICE_PER_MILLION = 0.22
OUTPUT_PRICE_PER_MILLION = 0.66

WINDOW_SIZE = 8

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BASE_DIR / "history.json"
FACTS_PATH = BASE_DIR / "facts.json"
BRANCHES_PATH = BASE_DIR / "branches.json"
STATS_PATH = BASE_DIR / "stats.json"

FACT_KEYS = ("project_name", "goal", "constraints", "preferences", "decisions")

EXTRACT_FACTS_SYSTEM = """Ты обновляешь key-value память фактов о проекте.
Верни ТОЛЬКО валидный JSON-объект с ключами:
project_name, goal, constraints, preferences, decisions.

Правила:
- сохраняй только важные факты, решения, ограничения, предпочтения;
- не выдумывай новых фактов;
- если факт неизвестен — оставь пустую строку "";
- объединяй с предыдущими фактами, не затирай старое без причины;
- не пиши ничего кроме JSON."""


class Strategy(str, Enum):
    SLIDING_WINDOW = "sliding_window"
    STICKY_FACTS = "sticky_facts"
    BRANCHING = "branching"


STRATEGY_LABELS = {
    Strategy.SLIDING_WINDOW: "Sliding Window",
    Strategy.STICKY_FACTS: "Sticky Facts / Key-Value Memory",
    Strategy.BRANCHING: "Branching",
}


@dataclass
class TokenStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    current_message_tokens: int = 0
    history_tokens: int = 0
    context_tokens_estimate: int = 0


@dataclass
class AgentResponse:
    content: str
    token_stats: TokenStats
    strategy: Strategy


@dataclass
class CumulativeStats:
    by_strategy: dict[str, dict] = field(default_factory=dict)

    def add(self, strategy: Strategy, total_tokens: int, cost: float) -> None:
        key = strategy.value
        bucket = self.by_strategy.setdefault(
            key, {"cumulative_tokens": 0, "cumulative_cost": 0.0}
        )
        bucket["cumulative_tokens"] += total_tokens
        bucket["cumulative_cost"] += cost


class AgentError(Exception):
    """Raised when the agent cannot complete a request."""


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


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * INPUT_PRICE_PER_MILLION
        + completion_tokens / 1_000_000 * OUTPUT_PRICE_PER_MILLION
    )


def empty_facts() -> dict[str, str]:
    return {key: "" for key in FACT_KEYS}


def default_branches() -> dict:
    return {
        "checkpoint": [],
        "current": "main",
        "branches": {
            "main": [],
        },
    }


class Agent:
    """Agent with pluggable context strategies (no summary compression)."""

    def __init__(
        self,
        api_key: str,
        strategy: Strategy = Strategy.SLIDING_WINDOW,
        model: str = "deepseek-v4-flash",
        window_size: int = WINDOW_SIZE,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.strategy = strategy
        self.window_size = window_size

        self.messages = self._load_json_list(HISTORY_PATH)
        self.facts = self._load_facts()
        self.branches_data = self._load_branches()
        self.cumulative = self._load_stats()

        if self.strategy == Strategy.BRANCHING:
            self.messages = list(self.get_current_branch_messages())

    # ----- persistence -----

    def _load_json_list(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _save_history(self) -> None:
        HISTORY_PATH.write_text(
            json.dumps(self.messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_facts(self) -> dict[str, str]:
        facts = empty_facts()
        if not FACTS_PATH.exists():
            return facts
        try:
            data = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return facts
        if not isinstance(data, dict):
            return facts
        for key in FACT_KEYS:
            value = data.get(key, "")
            facts[key] = str(value).strip() if value is not None else ""
        return facts

    def _save_facts(self) -> None:
        FACTS_PATH.write_text(
            json.dumps(self.facts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_branches(self) -> dict:
        if not BRANCHES_PATH.exists():
            return default_branches()
        try:
            data = json.loads(BRANCHES_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default_branches()
        if not isinstance(data, dict) or "branches" not in data:
            return default_branches()
        data.setdefault("checkpoint", [])
        data.setdefault("current", "main")
        data.setdefault("branches", {"main": []})
        return data

    def _save_branches(self) -> None:
        BRANCHES_PATH.write_text(
            json.dumps(self.branches_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_stats(self) -> CumulativeStats:
        if not STATS_PATH.exists():
            return CumulativeStats()
        try:
            data = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return CumulativeStats()
        return CumulativeStats(by_strategy=dict(data.get("by_strategy") or {}))

    def _save_stats(self) -> None:
        STATS_PATH.write_text(
            json.dumps({"by_strategy": self.cumulative.by_strategy}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_history(self) -> None:
        self.messages = []
        self._save_history()
        if self.strategy == Strategy.BRANCHING:
            current = self.get_current_branch()
            self.branches_data["branches"][current] = []
            self._save_branches()

    def clear_facts(self) -> None:
        self.facts = empty_facts()
        self._save_facts()

    def clear_all(self) -> None:
        self.messages = []
        self.facts = empty_facts()
        self.branches_data = default_branches()
        self.cumulative = CumulativeStats()
        self._save_history()
        self._save_facts()
        self._save_branches()
        self._save_stats()

    def replace_messages(self, messages: list[dict]) -> None:
        self.messages = list(messages)
        self._apply_window_if_needed()
        self._persist_messages()

    # ----- branching -----

    def get_current_branch(self) -> str:
        return str(self.branches_data.get("current") or "main")

    def list_branches(self) -> list[str]:
        return sorted(self.branches_data.get("branches", {}).keys())

    def get_current_branch_messages(self) -> list[dict]:
        name = self.get_current_branch()
        return list(self.branches_data["branches"].get(name, []))

    def create_checkpoint(self) -> None:
        self.branches_data["checkpoint"] = list(self.messages)
        self._sync_branch_from_messages()
        self._save_branches()

    def create_branch(self, name: str) -> None:
        name = name.strip()
        if not name:
            raise AgentError("Имя ветки не может быть пустым.")
        checkpoint = list(self.branches_data.get("checkpoint") or [])
        if not checkpoint and self.messages:
            checkpoint = list(self.messages)
            self.branches_data["checkpoint"] = checkpoint
        self.branches_data["branches"][name] = list(checkpoint)
        self._save_branches()

    def switch_branch(self, name: str) -> None:
        if name not in self.branches_data.get("branches", {}):
            raise AgentError(f"Ветка «{name}» не найдена.")
        self._sync_branch_from_messages()
        self.branches_data["current"] = name
        self.messages = list(self.branches_data["branches"][name])
        self._save_branches()
        self._save_history()

    def _sync_branch_from_messages(self) -> None:
        current = self.get_current_branch()
        self.branches_data.setdefault("branches", {})[current] = list(self.messages)

    # ----- core API -----

    def _post_chat(self, messages: list[dict]) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "thinking": THINKING_DISABLED,
        }
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=90,
            )
        except requests.Timeout as exc:
            raise AgentError("Таймаут при обращении к DeepSeek API.") from exc
        except requests.RequestException as exc:
            raise AgentError(
                f"Не удалось связаться с DeepSeek API: {exc.__class__.__name__}."
            ) from exc

        if not response.ok:
            raise AgentError(
                f"DeepSeek API вернул ошибку HTTP {response.status_code}."
            )
        try:
            return response.json()
        except ValueError as exc:
            raise AgentError("Не удалось разобрать ответ DeepSeek API.") from exc

    def _parse_answer(self, data: dict) -> tuple[str, TokenStats]:
        try:
            content = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            prompt_tokens = int(usage["prompt_tokens"])
            completion_tokens = int(usage["completion_tokens"])
            total_tokens = int(usage["total_tokens"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise AgentError("Не удалось разобрать ответ DeepSeek API.") from exc

        answer = content.strip()
        if not answer:
            raise AgentError("DeepSeek API вернул пустой ответ.")

        return answer, TokenStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=calculate_cost(prompt_tokens, completion_tokens),
        )

    def _apply_window_if_needed(self) -> None:
        if self.strategy in (Strategy.SLIDING_WINDOW, Strategy.STICKY_FACTS):
            if len(self.messages) > self.window_size:
                self.messages = self.messages[-self.window_size :]

    def _persist_messages(self) -> None:
        self._save_history()
        if self.strategy == Strategy.BRANCHING:
            self._sync_branch_from_messages()
            self._save_branches()

    def _format_facts_block(self) -> str:
        lines = ["Known facts:"]
        for key in FACT_KEYS:
            value = self.facts.get(key) or ""
            lines.append(f"- {key}: {value if value else '(empty)'}")
        return "\n".join(lines)

    def _update_facts_from_user(self, user_message: str) -> None:
        payload_messages = [
            {"role": "system", "content": EXTRACT_FACTS_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Текущие факты (JSON):\n"
                    f"{json.dumps(self.facts, ensure_ascii=False)}\n\n"
                    "Новое сообщение пользователя:\n"
                    f"{user_message}\n\n"
                    "Верни обновлённый JSON фактов."
                ),
            },
        ]
        data = self._post_chat(payload_messages)
        try:
            raw = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            total_tokens = int(usage.get("total_tokens", 0))
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise AgentError("Не удалось разобрать ответ extraction фактов.") from exc

        self.cumulative.add(
            Strategy.STICKY_FACTS,
            total_tokens,
            calculate_cost(prompt_tokens, completion_tokens),
        )
        self._save_stats()

        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AgentError("Модель вернула невалидный JSON фактов.") from exc

        if not isinstance(parsed, dict):
            raise AgentError("JSON фактов должен быть объектом.")

        updated = empty_facts()
        for key in FACT_KEYS:
            incoming = parsed.get(key, self.facts.get(key, ""))
            updated[key] = str(incoming).strip() if incoming is not None else ""
        self.facts = updated
        self._save_facts()

    def build_api_messages(self) -> list[dict]:
        if self.strategy == Strategy.STICKY_FACTS:
            return [
                {"role": "system", "content": self._format_facts_block()},
                *self.messages[-self.window_size :],
            ]
        if self.strategy == Strategy.SLIDING_WINDOW:
            return list(self.messages[-self.window_size :])
        return list(self.messages)

    def ask(self, user_message: str) -> AgentResponse:
        current_message_tokens = estimate_tokens(user_message)
        history_tokens = estimate_messages_tokens(self.messages)

        if self.strategy == Strategy.STICKY_FACTS:
            self._update_facts_from_user(user_message)

        self.messages.append({"role": "user", "content": user_message})
        self._apply_window_if_needed()

        api_messages = self.build_api_messages()
        context_estimate = estimate_messages_tokens(api_messages)

        if context_estimate > MAX_CONTEXT_TOKENS:
            self.messages.pop()
            self._persist_messages()
            raise AgentError(
                "Контекст превышает окно модели "
                f"(~{context_estimate} > {MAX_CONTEXT_TOKENS})."
            )

        try:
            data = self._post_chat(api_messages)
            answer, token_stats = self._parse_answer(data)
        except AgentError:
            self.messages.pop()
            self._persist_messages()
            raise

        self.messages.append({"role": "assistant", "content": answer})
        self._apply_window_if_needed()
        self._persist_messages()

        token_stats.current_message_tokens = current_message_tokens
        token_stats.history_tokens = history_tokens
        token_stats.context_tokens_estimate = context_estimate

        self.cumulative.add(
            self.strategy, token_stats.total_tokens, token_stats.estimated_cost
        )
        self._save_stats()

        return AgentResponse(
            content=answer,
            token_stats=token_stats,
            strategy=self.strategy,
        )
