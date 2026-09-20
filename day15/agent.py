"""Day 15: Agent — Day 14 + controlled transitions with guards."""

from __future__ import annotations

import json
import re
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
from task_state import (
    TaskStage,
    TaskState,
    TaskStateError,
    TransitionResult,
)

API_URL = "https://api.deepseek.com/chat/completions"
THINKING_DISABLED = {"type": "disabled"}

MAX_CONTEXT_TOKENS = 1_000_000
INPUT_PRICE_PER_MILLION = 0.22
OUTPUT_PRICE_PER_MILLION = 0.66

SYSTEM_INSTRUCTIONS = (
    "Ты ассистент с памятью, Task State Machine (rules+guards) и инвариантами. "
    "USER PROFILE — КАК отвечать. CURRENT TASK — где мы + TRANSITION GUARDS. "
    "ACTIVE INVARIANTS — что нельзя нарушать. "
    "Ты НЕ меняешь stage сам: переходы только через контролируемый API. "
    "Нельзя пропускать этапы lifecycle. "
    "«Пропустить утверждение» ≠ утверждение плана. "
    "Явное «План утверждаю» обрабатывает приложение, не ты."
)

# Explicit approval (may set plan_approved).
_APPROVE_PLAN_RE = re.compile(
    r"(план\s+утверждаю|утверждаю\s+план|approve\s+the\s+plan|approve\s+plan|"
    r"считай\s+план\s+утвержд[её]нным|план\s+утвержд[её]н)",
    re.IGNORECASE,
)
# Skip / bypass — must NOT approve.
_SKIP_APPROVAL_RE = re.compile(
    r"(пропусти\s+утверждение|пропусти\s+планирование|пропусти\s+план|"
    r"без\s+утверждения|skip\s+(the\s+)?plan|skip\s+approval)",
    re.IGNORECASE,
)
_MARK_EXEC_DONE_RE = re.compile(
    r"(execution\s+completed|реализация\s+завершена|execution\s+заверш|"
    r"отметь\s+execution\s+completed|работа\s+сделана)",
    re.IGNORECASE,
)
_MARK_VALIDATION_OK_RE = re.compile(
    r"(validation\s+passed|валидация\s+пройдена|проверка\s+успешн|"
    r"отметь\s+validation_passed)",
    re.IGNORECASE,
)
_START_EXEC_RE = re.compile(
    r"(начинай\s+реализац|начинай\s+писать\s+код|перейди\s+в\s+execution|"
    r"переходи\s+к\s+execution|start\s+execution|начинай\s+делать)",
    re.IGNORECASE,
)
_JUMP_DONE_RE = re.compile(
    r"(просто\s+поставь\s+done|сразу\s+(закрой|закрывай)\s+задач|"
    r"не\s+запускай\s+проверк|валидация\s+не\s+нужна|"
    r"сразу\s+done|поставь\s+done|mark\s+(as\s+)?done)",
    re.IGNORECASE,
)
_TO_VALIDATION_RE = re.compile(
    r"(перейди\s+в\s+validation|к\s+validation|start\s+validation)",
    re.IGNORECASE,
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
    transition: TransitionResult | None = None


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
    """Day 11–14 + Day 15 guarded transitions (single FSM)."""

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
        return UserProfile.from_json(self.memory.long_term.get(PROFILE_KEY))

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
        return TaskState.from_json(self.memory.working.get(TASK_STATE_KEY))

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

    def transition_task(self, new_stage: TaskStage) -> TransitionResult:
        """Only path to change stage — rules + guards in TaskState.transition_to."""
        task = self.get_task_state()
        if task is None:
            raise TaskStateError("Нет активной задачи.")
        result = task.transition_to(new_stage)
        if result.success:
            self.save_task_state(task)
        return result

    def approve_plan(self) -> TaskState:
        task = self.get_task_state()
        if task is None:
            raise TaskStateError("Нет активной задачи.")
        task.approve_plan()
        self.save_task_state(task)
        return task

    def mark_execution_completed(self) -> TaskState:
        task = self.get_task_state()
        if task is None:
            raise TaskStateError("Нет активной задачи.")
        task.mark_execution_completed()
        self.save_task_state(task)
        return task

    def mark_validation_passed(self, passed: bool = True) -> TaskState:
        task = self.get_task_state()
        if task is None:
            raise TaskStateError("Нет активной задачи.")
        task.mark_validation_passed(passed)
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

    # ----- Invariants -----

    def get_invariant_store(self) -> InvariantStore:
        return InvariantStore.from_json(self.memory.working.get(INVARIANTS_KEY))

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
        """Orders demo at EXECUTION with guards satisfied for that stage."""
        self.clear_short_term()
        task = self.create_task(DEMO_GOAL)
        self.approve_plan()
        result = self.transition_task(TaskStage.EXECUTION)
        if not result.success:
            raise TaskStateError(result.message)
        self.update_task_step(
            "спроектировать REST endpoints заказов",
            "учесть PostgreSQL и правило оплаты перед COMPLETED",
        )
        self.load_demo_invariants()
        task = self.get_task_state()
        assert task is not None
        return task

    def setup_lifecycle_demo(self) -> TaskState:
        """Fresh task in PLANNING with all guards false (Day 15 acceptance)."""
        self.clear_short_term()
        self.clear_invariants()
        return self.create_task("Контролируемый lifecycle REST API backend")

    def check_request(self, user_message: str) -> InvariantConflict | None:
        return check_invariant_conflict(user_message, self.get_invariant_store())

    # ----- memory -----

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

    # ----- lifecycle chat commands (code enforcement) -----

    def handle_lifecycle_command(self, user_message: str) -> AgentResponse | None:
        """
        Deterministic lifecycle intents. Returns AgentResponse if handled,
        else None (fall through to invariants/LLM).
        Never lets chat skip guards.
        """
        text = (user_message or "").strip()
        if not text:
            return None
        task = self.get_task_state()
        if task is None:
            return None

        # Skip planning / skip approval → attempt execution WITHOUT approving.
        if _SKIP_APPROVAL_RE.search(text):
            result = self.transition_task(TaskStage.EXECUTION)
            answer = (
                "Нельзя пропустить утверждение плана.\n\n" + result.message
                if not result.success
                else result.message
            )
            return self._local_reply(
                user_message,
                answer,
                rejected=not result.success,
                transition=result,
            )

        # Explicit approve (+ optional start execution).
        if _APPROVE_PLAN_RE.search(text) and not _SKIP_APPROVAL_RE.search(text):
            try:
                self.approve_plan()
            except TaskStateError as exc:
                return self._local_reply(user_message, str(exc), rejected=True)
            parts = ["План утверждён: plan_approved = true."]
            if _START_EXEC_RE.search(text) or "начинай" in text.lower():
                result = self.transition_task(TaskStage.EXECUTION)
                parts.append(result.message)
                return self._local_reply(
                    user_message,
                    "\n\n".join(parts),
                    rejected=not result.success,
                    transition=result,
                )
            parts.append("Можно выполнить переход planning → execution.")
            return self._local_reply(user_message, "\n".join(parts))

        if _MARK_EXEC_DONE_RE.search(text):
            try:
                self.mark_execution_completed()
            except TaskStateError as exc:
                return self._local_reply(user_message, str(exc), rejected=True)
            return self._local_reply(
                user_message,
                "execution_completed = true. Можно перейти execution → validation.",
            )

        if _MARK_VALIDATION_OK_RE.search(text):
            try:
                self.mark_validation_passed(True)
            except TaskStateError as exc:
                return self._local_reply(user_message, str(exc), rejected=True)
            return self._local_reply(
                user_message,
                "validation_passed = true. Можно перейти validation → done.",
            )

        if _JUMP_DONE_RE.search(text):
            result = self.transition_task(TaskStage.DONE)
            return self._local_reply(
                user_message,
                result.message,
                rejected=not result.success,
                transition=result,
            )

        if _TO_VALIDATION_RE.search(text):
            result = self.transition_task(TaskStage.VALIDATION)
            return self._local_reply(
                user_message,
                result.message,
                rejected=not result.success,
                transition=result,
            )

        if _START_EXEC_RE.search(text):
            # Only attempt the transition from planning; if already in execution,
            # let the message go to the LLM as normal implementation chat.
            if task.stage != TaskStage.PLANNING:
                return None
            result = self.transition_task(TaskStage.EXECUTION)
            return self._local_reply(
                user_message,
                result.message,
                rejected=not result.success,
                transition=result,
            )

        return None

    def _local_reply(
        self,
        user_message: str,
        answer: str,
        *,
        rejected: bool = False,
        transition: TransitionResult | None = None,
        conflict: InvariantConflict | None = None,
    ) -> AgentResponse:
        self.memory.append_message("user", user_message)
        self.memory.append_message("assistant", answer)
        return AgentResponse(
            content=answer,
            token_stats=self._empty_token_stats(user_message),
            conflict=conflict,
            rejected=rejected,
            transition=transition,
        )

    # ----- prompt -----

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
        # 1) Lifecycle commands (code) — before LLM, may reject transitions.
        handled = self.handle_lifecycle_command(user_message)
        if handled is not None:
            return handled

        # 2) Invariant gate (Day 14).
        conflict = self.check_request(user_message)
        if conflict is not None:
            return self._local_reply(
                user_message,
                conflict.format_rejection(),
                rejected=True,
                conflict=conflict,
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
