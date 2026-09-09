"""Day 08: Agent with history persistence and token/cost statistics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

API_URL = "https://api.deepseek.com/chat/completions"
THINKING_DISABLED = {"type": "disabled"}

# Official DeepSeek pricing for deepseek-v4-flash (after 2026-08-16 tariff change).
# Day 8 uses a simplified estimate: OFF-PEAK + cache miss only.
# Does not account for peak hours or cache hits — label cost as estimated.
# Context length: 1M tokens.
MAX_CONTEXT_TOKENS = 1_000_000
INPUT_PRICE_PER_MILLION = 0.22
OUTPUT_PRICE_PER_MILLION = 0.66

HISTORY_PATH = Path(__file__).resolve().parent / "history.json"
STATS_PATH = Path(__file__).resolve().parent / "stats.json"


@dataclass
class TokenStats:
    current_message_tokens: int
    history_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float


@dataclass
class AgentResponse:
    content: str
    token_stats: TokenStats


@dataclass
class CumulativeStats:
    cumulative_tokens: int = 0
    cumulative_cost: float = 0.0


class AgentError(Exception):
    """Raised when the agent cannot get a valid answer from the API."""


def estimate_tokens(text: str) -> int:
    """Approximate token count when a DeepSeek-specific tokenizer is unavailable.

    Heuristic: ~4 characters per token. This is NOT an exact tokenizer result.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Approximate tokens for a list of chat messages (roles + content)."""
    total = 0
    for message in messages:
        total += estimate_tokens(str(message.get("role", "")))
        total += estimate_tokens(str(message.get("content", "")))
        total += 4  # rough per-message overhead
    return total


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    input_cost = prompt_tokens / 1_000_000 * INPUT_PRICE_PER_MILLION
    output_cost = completion_tokens / 1_000_000 * OUTPUT_PRICE_PER_MILLION
    return input_cost + output_cost


class Agent:
    """Agent with JSON history persistence and token/cost tracking."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.history_path = HISTORY_PATH
        self.stats_path = STATS_PATH
        self.messages = self._load_history()
        self.cumulative = self._load_stats()

    def _load_history(self) -> list[dict]:
        if not self.history_path.exists():
            return []

        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        if not isinstance(data, list):
            return []

        return data

    def _save_history(self) -> None:
        self.history_path.write_text(
            json.dumps(self.messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_stats(self) -> CumulativeStats:
        if not self.stats_path.exists():
            return CumulativeStats()

        try:
            data = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return CumulativeStats()

        return CumulativeStats(
            cumulative_tokens=int(data.get("cumulative_tokens", 0)),
            cumulative_cost=float(data.get("cumulative_cost", 0.0)),
        )

    def _save_stats(self) -> None:
        self.stats_path.write_text(
            json.dumps(asdict(self.cumulative), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_history(self) -> None:
        self.messages = []
        self.cumulative = CumulativeStats()
        self._save_history()
        self._save_stats()

    def ask(self, user_message: str) -> AgentResponse:
        history_before = list(self.messages)
        history_tokens = estimate_messages_tokens(history_before)
        current_message_tokens = estimate_tokens(user_message)

        self.messages.append({"role": "user", "content": user_message})
        estimated_prompt_tokens = estimate_messages_tokens(self.messages)

        if estimated_prompt_tokens > self.max_context_tokens:
            self.messages.pop()
            raise AgentError(
                "История диалога превышает контекстное окно модели "
                f"(~{estimated_prompt_tokens} estimated tokens > "
                f"{self.max_context_tokens} MAX_CONTEXT_TOKENS)."
            )

        payload = {
            "model": self.model,
            "messages": self.messages,
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
                timeout=60,
            )
        except requests.Timeout as exc:
            self.messages.pop()
            raise AgentError("Таймаут при обращении к DeepSeek API.") from exc
        except requests.RequestException as exc:
            self.messages.pop()
            raise AgentError(
                f"Не удалось связаться с DeepSeek API: {exc.__class__.__name__}."
            ) from exc

        if not response.ok:
            self.messages.pop()
            raise AgentError(
                f"DeepSeek API вернул ошибку HTTP {response.status_code}."
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            prompt_tokens = int(usage["prompt_tokens"])
            completion_tokens = int(usage["completion_tokens"])
            total_tokens = int(usage["total_tokens"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self.messages.pop()
            raise AgentError("Не удалось разобрать ответ DeepSeek API.") from exc

        answer = content.strip()
        if not answer:
            self.messages.pop()
            raise AgentError("DeepSeek API вернул пустой ответ.")

        self.messages.append({"role": "assistant", "content": answer})
        self._save_history()

        cost = calculate_cost(prompt_tokens, completion_tokens)
        self.cumulative.cumulative_tokens += total_tokens
        self.cumulative.cumulative_cost += cost
        self._save_stats()

        return AgentResponse(
            content=answer,
            token_stats=TokenStats(
                current_message_tokens=current_message_tokens,
                history_tokens=history_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost=cost,
            ),
        )
