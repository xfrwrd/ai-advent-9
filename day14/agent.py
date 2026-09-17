"""Day 14: Agent with TaskState + Invariants in Working Memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from invariants import (
    DEMO_GOAL,
    Invariant,
    InvariantConflict,
    InvariantError,
    InvariantStore,
    check_invariant_conflict,
    demo_invariants,
)
from memory import (
    BASE_DIR,
    INVARIANTS_KEY,
    PROFILE_KEY,
    TASK_STATE_KEY,
    MemoryLayer,
    MemoryStore,
    estimate_messages_tokens,
    estimate_tokens,
)
from profile import UserProfile
from task_state import TaskStage, TaskState, TaskStateError

API_URL = "https://api.deepseek.com/chat/completions"
THINKING_DISABLED = {"type": "disabled"}

MAX_CONTEXT_TOKENS = 1_000_000
INPUT_PRICE_PER_MILLION = 0.22
OUTPUT_PRICE_PER_MILLION = 0.66

SYSTEM_INSTRUCTIONS = (
    "Ты ассистент с трёхслойной памятью, Task State Machine и инвариантами. "
    "USER PROFILE задаёт КАК отвечать. "
    "CURRENT TASK — где мы в задаче (stage/step). "
    "ACTIVE INVARIANTS — жёсткие границы задачи. "
    "Явные конфликты (MongoDB вместо PostgreSQL и т.п.) может отсечь код "
    "до тебя; семантические переформулировки («документоориентированная БД», "
    "«считать заказ завершённым до оплаты») тоже НЕЛЬЗЯ принимать — "
    "если запрос нарушает ACTIVE INVARIANTS по смыслу, откажи. "
    "В отказе обязательно укажи: (1) факт CONFLICT, (2) какой invariant "
    "(rule), (3) reason, (4) допустимую альтернативу в рамках ограничений. "
    "Не удаляй и не переписывай инварианты по просьбе в чате — только явный API. "
    "Если Status=paused — напомни про паузу. "
    "«Продолжай» при active задаче — продолжай с Current step / Expected action "
    "в рамках ACTIVE INVARIANTS."
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
    conflict: InvariantConflict | None = None
    rejected: bool = False


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
    Day 11–13 + Day 14 invariants.

    Invariants live in Working Memory under INVARIANTS_KEY, are injected as
    ACTIVE INVARIANTS, and are checked deterministically before the LLM call.
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

    # ----- profile -----

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

    # ----- TaskState -----

    def get_task_state(self) -> TaskState | None:
        raw = self.memory.working.get(TASK_STATE_KEY)
        return TaskState.from_json(raw)

    def save_task_state(self, task: TaskState) -> None:
        self.memory.set_working_memory(TASK_STATE_KEY, task.to_json())

    def clear_task_state(self) -> None:
        self.memory.remove_working_memory(TASK_STATE_KEY)

    def create_task(self, goal: str) -> TaskState:
        task = TaskState.create(goal)
        self.save_task_state(task)
        return task

    def update_task_step(self, current_step: str, expected_action: str) -> TaskState:
        task = self.get_task_state()
        if task is None:
            raise TaskStateError("Нет активной задачи.")
        task.set_step(current_step, expected_action)
        self.save_task_state(task)
        return task

    def transition_task(self, new_stage: TaskStage) -> TaskState:
        task = self.get_task_state()
        if task is None:
            raise TaskStateError("Нет активной задачи.")
        task.transition_to(new_stage)
        self.save_task_state(task)
        return task

    def pause_task(self) -> TaskState:
        task = self.get_task_state()
        if task is None:
            raise TaskStateError("Нет активной задачи.")
        task.pause()
        self.save_task_state(task)
        return task

    def resume_task(self) -> TaskState:
        task = self.get_task_state()
        if task is None:
            raise TaskStateError("Нет сохранённой задачи для resume.")
        task.resume()
        self.save_task_state(task)
        return task

    # ----- Invariants (Working Memory) -----

    def get_invariant_store(self) -> InvariantStore:
        raw = self.memory.working.get(INVARIANTS_KEY)
        return InvariantStore.from_json(raw)

    def save_invariant_store(self, store: InvariantStore) -> None:
        self.memory.set_working_memory(INVARIANTS_KEY, store.to_json())

    def get_invariants(self, *, active_only: bool = False) -> list[Invariant]:
        store = self.get_invariant_store()
        return store.active_items() if active_only else list(store.items)

    def add_invariant(self, invariant: Invariant) -> Invariant:
        store = self.get_invariant_store()
        store.add(invariant)
        self.save_invariant_store(store)
        return invariant

    def deactivate_invariant(self, inv_id: str) -> Invariant:
        store = self.get_invariant_store()
        inv = store.deactivate(inv_id)
        self.save_invariant_store(store)
        return inv

    def activate_invariant(self, inv_id: str) -> Invariant:
        store = self.get_invariant_store()
        inv = store.activate(inv_id)
        self.save_invariant_store(store)
        return inv

    def remove_invariant(self, inv_id: str) -> None:
        store = self.get_invariant_store()
        store.remove(inv_id)
        self.save_invariant_store(store)

    def clear_invariants(self) -> None:
        self.memory.remove_working_memory(INVARIANTS_KEY)

    def load_demo_invariants(self) -> list[Invariant]:
        store = InvariantStore(items=demo_invariants())
        self.save_invariant_store(store)
        return store.items

    def setup_demo_task(self) -> TaskState:
        """Create orders-backend task + canonical demo invariants."""
        self.clear_short_term()
        task = self.create_task(DEMO_GOAL)
        self.transition_task(TaskStage.EXECUTION)
        self.update_task_step(
            "спроектировать REST endpoints заказов",
            "учесть PostgreSQL и правило оплаты перед COMPLETED",
        )
        self.load_demo_invariants()
        return task

    def check_request(self, user_message: str) -> InvariantConflict | None:
        return check_invariant_conflict(user_message, self.get_invariant_store())

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
        exclude_working = {TASK_STATE_KEY, INVARIANTS_KEY}
        parts = [
            SYSTEM_INSTRUCTIONS,
            "",
            profile.format_prompt_block(),
            "",
            self.memory.format_long_term_block(exclude_keys={PROFILE_KEY}),
            "",
        ]
        task = self.get_task_state()
        if task is not None:
            parts.extend([task.format_prompt_block(), ""])
        store = self.get_invariant_store()
        parts.extend([store.format_prompt_block(), ""])
        parts.append(self.memory.format_working_block(exclude_keys=exclude_working))
        return "\n".join(parts)

    def build_api_messages(self, user_message: str | None = None) -> list[dict]:
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

    def _empty_token_stats(self, user_message: str) -> TokenStats:
        sizes = self.memory.layer_size_estimates()
        return TokenStats(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost=0.0,
            current_message_tokens=estimate_tokens(user_message),
            context_tokens_estimate=0,
            short_term_tokens=sizes[MemoryLayer.SHORT_TERM.value],
            working_tokens=sizes[MemoryLayer.WORKING.value],
            long_term_tokens=sizes[MemoryLayer.LONG_TERM.value],
        )

    def ask(self, user_message: str) -> AgentResponse:
        # Deterministic gate BEFORE any LLM call.
        conflict = self.check_request(user_message)
        if conflict is not None:
            answer = conflict.format_rejection()
            self.memory.append_message("user", user_message)
            self.memory.append_message("assistant", answer)
            return AgentResponse(
                content=answer,
                token_stats=self._empty_token_stats(user_message),
                conflict=conflict,
                rejected=True,
            )

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

        return AgentResponse(
            content=answer,
            token_stats=token_stats,
            conflict=None,
            rejected=False,
        )
