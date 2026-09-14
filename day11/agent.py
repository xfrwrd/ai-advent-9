"""Day 11: Agent with explicit Short-Term / Working / Long-Term memory."""

from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from memory import (
    BASE_DIR,
    MemoryLayer,
    MemoryStore,
    estimate_messages_tokens,
    estimate_tokens,
)

API_URL = "https://api.deepseek.com/chat/completions"
THINKING_DISABLED = {"type": "disabled"}

MAX_CONTEXT_TOKENS = 1_000_000
INPUT_PRICE_PER_MILLION = 0.22
OUTPUT_PRICE_PER_MILLION = 0.66

STATS_PATH = BASE_DIR / "stats.json"

SYSTEM_INSTRUCTIONS = (
    "Ты помощник с явной трёхслойной памятью. "
    "Используй Long-term memory для устойчивых фактов о пользователе, "
    "Working memory для текущей задачи, "
    "Recent dialogue для темы текущего разговора. "
    "Не выдумывай факты, которых нет в этих блоках."
)

TEST_QUESTION = (
    "Что ты знаешь обо мне, о текущей задаче и о теме нашего разговора?"
)


@dataclass
class TokenStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    current_message_tokens: int = 0
    context_tokens_estimate: int = 0
    short_term_tokens: int = 0
    working_tokens: int = 0
    long_term_tokens: int = 0


@dataclass
class AgentResponse:
    content: str
    token_stats: TokenStats


@dataclass
class CumulativeStats:
    cumulative_tokens: int = 0
    cumulative_cost: float = 0.0

    def add(self, total_tokens: int, cost: float) -> None:
        self.cumulative_tokens += total_tokens
        self.cumulative_cost += cost


class AgentError(Exception):
    """Raised when the agent cannot complete a request."""


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * INPUT_PRICE_PER_MILLION
        + completion_tokens / 1_000_000 * OUTPUT_PRICE_PER_MILLION
    )


class Agent:
    """Chat agent that injects three memory layers as separate prompt blocks."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.memory = MemoryStore()
        self.cumulative = self._load_stats()

    # ----- stats persistence -----

    def _load_stats(self) -> CumulativeStats:
        if not STATS_PATH.exists():
            return CumulativeStats()
        try:
            data = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return CumulativeStats()
        return CumulativeStats(
            cumulative_tokens=int(data.get("cumulative_tokens", 0)),
            cumulative_cost=float(data.get("cumulative_cost", 0.0)),
        )

    def _save_stats(self) -> None:
        STATS_PATH.write_text(
            json.dumps(
                {
                    "cumulative_tokens": self.cumulative.cumulative_tokens,
                    "cumulative_cost": self.cumulative.cumulative_cost,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ----- memory API (delegates; explicit layer choice) -----

    def remember(self, layer: MemoryLayer, key: str, value: str) -> None:
        self.memory.remember(layer, key, value)

    def append_message(self, role: str, content: str) -> None:
        self.memory.append_message(role, content)

    def clear_short_term(self) -> None:
        self.memory.clear_short_term()

    def set_working_memory(self, key: str, value: str) -> None:
        self.memory.set_working_memory(key, value)

    def remove_working_memory(self, key: str) -> None:
        self.memory.remove_working_memory(key)

    def clear_working_memory(self) -> None:
        self.memory.clear_working_memory()

    def set_long_term_memory(self, key: str, value: str) -> None:
        self.memory.set_long_term_memory(key, value)

    def remove_long_term_memory(self, key: str) -> None:
        self.memory.remove_long_term_memory(key)

    def clear_long_term_memory(self) -> None:
        self.memory.clear_long_term_memory()

    def clear_all(self) -> None:
        self.memory.clear_all()
        self.cumulative = CumulativeStats()
        self._save_stats()

    # ----- prompt construction -----

    def build_api_messages(self, user_message: str | None = None) -> list[dict]:
        """
        System: Long-term + Working as separate text blocks.
        Then: Short-term dialogue as chat turns (+ optional pending user message).
        Layers are never merged into one JSON blob.
        """
        system_content = f"{SYSTEM_INSTRUCTIONS}\n\n{self.memory.build_system_prompt()}"
        messages: list[dict] = [{"role": "system", "content": system_content}]
        messages.extend(self.memory.short_term)
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
        return messages

    def preview_prompt(self) -> str:
        """Human-readable preview of all three layers as they enter the prompt."""
        dialogue_lines = []
        if not self.memory.short_term:
            dialogue_lines.append("(empty)")
        else:
            for message in self.memory.short_term:
                dialogue_lines.append(
                    f"- {message.get('role')}: {message.get('content')}"
                )
        return "\n".join(
            [
                "=== System ===",
                SYSTEM_INSTRUCTIONS,
                "",
                self.memory.format_long_term_block(),
                "",
                self.memory.format_working_block(),
                "",
                "=== Recent dialogue (short-term as chat turns) ===",
                *dialogue_lines,
            ]
        )

    def load_test_scenario(self) -> None:
        """Seed the three layers for the Day 11 verification scenario."""
        self.memory.clear_all()
        self.remember(MemoryLayer.LONG_TERM, "name", "Alex")
        self.remember(MemoryLayer.WORKING, "backend", "Go")
        self.remember(MemoryLayer.WORKING, "current_task", "API design")
        self.remember(MemoryLayer.SHORT_TERM, "", "Сейчас обсуждаем retry.")

    # ----- LLM -----

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

    def ask(self, user_message: str) -> AgentResponse:
        current_message_tokens = estimate_tokens(user_message)
        sizes = self.memory.layer_size_estimates()

        # Pending user turn is included in the API call, then stored in short-term.
        api_messages = self.build_api_messages(user_message=user_message)
        context_estimate = estimate_messages_tokens(api_messages)

        if context_estimate > MAX_CONTEXT_TOKENS:
            raise AgentError(
                "Контекст превышает окно модели "
                f"(~{context_estimate} > {MAX_CONTEXT_TOKENS})."
            )

        data = self._post_chat(api_messages)
        answer, token_stats = self._parse_answer(data)

        self.memory.append_message("user", user_message)
        self.memory.append_message("assistant", answer)

        token_stats.current_message_tokens = current_message_tokens
        token_stats.context_tokens_estimate = context_estimate
        token_stats.short_term_tokens = sizes[MemoryLayer.SHORT_TERM.value]
        token_stats.working_tokens = sizes[MemoryLayer.WORKING.value]
        token_stats.long_term_tokens = sizes[MemoryLayer.LONG_TERM.value]

        self.cumulative.add(token_stats.total_tokens, token_stats.estimated_cost)
        self._save_stats()

        return AgentResponse(content=answer, token_stats=token_stats)
