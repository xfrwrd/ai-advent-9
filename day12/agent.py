"""Day 12: Personalized agent — UserProfile on top of Day 11 memory layers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from memory import (
    BASE_DIR,
    PROFILE_KEY,
    MemoryLayer,
    MemoryStore,
    estimate_messages_tokens,
    estimate_tokens,
)
from profile import UserProfile

API_URL = "https://api.deepseek.com/chat/completions"
THINKING_DISABLED = {"type": "disabled"}

MAX_CONTEXT_TOKENS = 1_000_000
INPUT_PRICE_PER_MILLION = 0.22
OUTPUT_PRICE_PER_MILLION = 0.66

SYSTEM_INSTRUCTIONS = (
    "Ты персонализированный ассистент с явной трёхслойной памятью. "
    "USER PROFILE задаёт КАК отвечать этому пользователю — "
    "соблюдай style, format, expertise и constraints в каждом ответе. "
    "Long-term memory — устойчивые факты; Working memory — текущая задача; "
    "Recent dialogue — тема разговора. Не выдумывай факты вне этих блоков."
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
    """
    Memory layers (Day 11) + automatic UserProfile injection (Day 12).

    Profile is stored in Long-Term Memory under PROFILE_KEY and is injected
    into every LLM request without the user repeating preferences.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        memory: MemoryStore | None = None,
        base_dir: Path | None = None,
        llm_client: Callable[[list[dict]], dict] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.memory = memory or MemoryStore(base_dir=base_dir)
        self._llm_client = llm_client
        stats_dir = base_dir or BASE_DIR
        self.stats_path = stats_dir / "stats.json"
        self.cumulative = self._load_stats()

    # ----- stats -----

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

    # ----- profile (Long-Term) -----

    def get_profile(self) -> UserProfile:
        raw = self.memory.long_term.get(PROFILE_KEY)
        return UserProfile.from_json(raw)

    def save_profile(self, profile: UserProfile) -> None:
        self.memory.set_long_term_memory(PROFILE_KEY, profile.to_json())

    def update_profile(self, **kwargs: object) -> UserProfile:
        profile = self.get_profile()
        profile.update(**kwargs)
        self.save_profile(profile)
        return profile

    def clear_profile(self) -> None:
        self.memory.remove_long_term_memory(PROFILE_KEY)

    # ----- memory API -----

    def remember(self, layer: MemoryLayer, key: str, value: str) -> None:
        self.memory.remember(layer, key, value)

    def append_message(self, role: str, content: str) -> None:
        self.memory.append_message(role, content)

    def clear_short_term(self) -> None:
        self.memory.clear_short_term()

    def set_working_memory(self, key: str, value: str) -> None:
        self.memory.set_working_memory(key, value)

    def clear_working_memory(self) -> None:
        self.memory.clear_working_memory()

    def set_long_term_memory(self, key: str, value: str) -> None:
        self.memory.set_long_term_memory(key, value)

    def clear_long_term_memory(self) -> None:
        self.memory.clear_long_term_memory()

    def clear_all(self) -> None:
        self.memory.clear_all()
        self.cumulative = CumulativeStats()
        self._save_stats()

    # ----- prompt construction -----

    def build_system_content(self) -> str:
        profile = self.get_profile()
        parts = [
            SYSTEM_INSTRUCTIONS,
            "",
            profile.format_prompt_block(),
            "",
            self.memory.format_long_term_block(exclude_keys={PROFILE_KEY}),
            "",
            self.memory.format_working_block(),
        ]
        return "\n".join(parts)

    def build_api_messages(self, user_message: str | None = None) -> list[dict]:
        """
        System: instructions + USER PROFILE + LT (sans profile) + Working.
        Then: Short-term dialogue as chat turns (+ optional pending user message).
        Profile is always injected — user need not repeat preferences.
        """
        messages: list[dict] = [
            {"role": "system", "content": self.build_system_content()}
        ]
        messages.extend(self.memory.short_term)
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
        return messages

    def preview_prompt(self) -> str:
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
                self.build_system_content(),
                "",
                "=== Recent dialogue (short-term as chat turns) ===",
                *dialogue_lines,
            ]
        )

    # ----- LLM -----

    def _post_chat(self, messages: list[dict]) -> dict:
        if self._llm_client is not None:
            return self._llm_client(messages)

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
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            total_tokens = int(
                usage.get("total_tokens", prompt_tokens + completion_tokens)
            )
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
