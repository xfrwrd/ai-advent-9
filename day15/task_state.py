"""Day 15: Task State Machine with transition rules + guards (extends Day 13)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class TaskStage(str, Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    DONE = "done"


class TaskStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"


# Transition RULES — is this edge in the graph at all?
ALLOWED_TRANSITIONS: dict[TaskStage, set[TaskStage]] = {
    TaskStage.PLANNING: {TaskStage.EXECUTION},
    TaskStage.EXECUTION: {TaskStage.VALIDATION},
    TaskStage.VALIDATION: {TaskStage.DONE, TaskStage.EXECUTION},
    TaskStage.DONE: set(),
}

STAGE_ORDER = (
    TaskStage.PLANNING,
    TaskStage.EXECUTION,
    TaskStage.VALIDATION,
    TaskStage.DONE,
)


class TaskStateError(Exception):
    """Invalid task state operation or transition."""


@dataclass
class TransitionRejection:
    """Structured explanation when a transition is refused."""

    current_stage: TaskStage
    requested_stage: TaskStage
    reason: str
    missing_condition: str
    expected_action: str

    def format_message(self) -> str:
        return "\n".join(
            [
                "TRANSITION REJECTED",
                "",
                f"Текущее состояние: {self.current_stage.value}",
                f"Запрошенное состояние: {self.requested_stage.value}",
                f"Причина отказа: {self.reason}",
                f"Условие, которое ещё не выполнено: {self.missing_condition}",
                f"Допустимое следующее действие: {self.expected_action}",
                "",
                "Stage не изменён.",
            ]
        )


@dataclass
class TransitionResult:
    success: bool
    task: "TaskState"
    rejection: TransitionRejection | None = None

    @property
    def message(self) -> str:
        if self.success:
            return (
                f"TRANSITION OK: {self.task.stage.value} "
                f"(status={self.task.status.value})"
            )
        assert self.rejection is not None
        return self.rejection.format_message()


@dataclass
class TaskState:
    """
    Day 13 FSM + Day 15 transition guards.

    Rules  = can this edge exist?
    Guards = are conditions true for this edge right now?
    Stage is only mutated via transition_to().
    """

    goal: str
    stage: TaskStage = TaskStage.PLANNING
    status: TaskStatus = TaskStatus.ACTIVE
    current_step: str = ""
    expected_action: str = ""
    # Transition guards (persisted with TaskState in Working Memory).
    plan_approved: bool = False
    execution_completed: bool = False
    validation_passed: bool = False

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "stage": self.stage.value,
            "status": self.status.value,
            "current_step": self.current_step,
            "expected_action": self.expected_action,
            "plan_approved": self.plan_approved,
            "execution_completed": self.execution_completed,
            "validation_passed": self.validation_passed,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def create(cls, goal: str) -> TaskState:
        goal = (goal or "").strip()
        if not goal:
            raise TaskStateError("Цель задачи не может быть пустой.")
        return cls(
            goal=goal,
            stage=TaskStage.PLANNING,
            status=TaskStatus.ACTIVE,
            current_step="уточнить требования и составить план",
            expected_action="утвердить план (plan_approved=true)",
            plan_approved=False,
            execution_completed=False,
            validation_passed=False,
        )

    @classmethod
    def from_dict(cls, data: dict | None) -> TaskState | None:
        if not data or not isinstance(data, dict):
            return None
        goal = str(data.get("goal") or "").strip()
        if not goal:
            return None
        try:
            stage = TaskStage(str(data.get("stage") or TaskStage.PLANNING.value))
            status = TaskStatus(str(data.get("status") or TaskStatus.ACTIVE.value))
        except ValueError as exc:
            raise TaskStateError(f"Некорректные поля TaskState: {exc}") from exc
        return cls(
            goal=goal,
            stage=stage,
            status=status,
            current_step=str(data.get("current_step") or ""),
            expected_action=str(data.get("expected_action") or ""),
            plan_approved=bool(data.get("plan_approved", False)),
            execution_completed=bool(data.get("execution_completed", False)),
            validation_passed=bool(data.get("validation_passed", False)),
        )

    @classmethod
    def from_json(cls, raw: str | None) -> TaskState | None:
        if not raw or not str(raw).strip():
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return cls.from_dict(data if isinstance(data, dict) else None)

    def set_step(self, current_step: str, expected_action: str) -> None:
        self.current_step = (current_step or "").strip()
        self.expected_action = (expected_action or "").strip()

    def can_transition_to(self, new_stage: TaskStage) -> bool:
        """Transition RULE only (graph edge), ignores guards."""
        return new_stage in ALLOWED_TRANSITIONS.get(self.stage, set())

    def _guard_for(self, new_stage: TaskStage) -> TransitionRejection | None:
        """Return rejection if a required guard fails; None if OK."""
        if self.stage == TaskStage.PLANNING and new_stage == TaskStage.EXECUTION:
            if not self.plan_approved:
                return TransitionRejection(
                    current_stage=self.stage,
                    requested_stage=new_stage,
                    reason="план ещё не утверждён",
                    missing_condition="plan_approved = true",
                    expected_action="утвердить текущий план",
                )
        if self.stage == TaskStage.EXECUTION and new_stage == TaskStage.VALIDATION:
            if not self.execution_completed:
                return TransitionRejection(
                    current_stage=self.stage,
                    requested_stage=new_stage,
                    reason="реализация ещё не завершена",
                    missing_condition="execution_completed = true",
                    expected_action="отметить execution как completed",
                )
        if self.stage == TaskStage.VALIDATION and new_stage == TaskStage.DONE:
            if not self.validation_passed:
                return TransitionRejection(
                    current_stage=self.stage,
                    requested_stage=new_stage,
                    reason="validation ещё не пройдена успешно",
                    missing_condition="validation_passed = true",
                    expected_action="отметить validation_passed = true",
                )
        # validation → execution (fix loop): no extra guard beyond rule + not paused
        return None

    def evaluate_transition(self, new_stage: TaskStage) -> TransitionResult:
        """Dry-run: rules + guards, does not mutate stage."""
        if self.status == TaskStatus.PAUSED:
            return TransitionResult(
                success=False,
                task=self,
                rejection=TransitionRejection(
                    current_stage=self.stage,
                    requested_stage=new_stage,
                    reason="задача на pause",
                    missing_condition="status = active (resume)",
                    expected_action="сначала Resume",
                ),
            )
        if self.stage == TaskStage.DONE:
            return TransitionResult(
                success=False,
                task=self,
                rejection=TransitionRejection(
                    current_stage=self.stage,
                    requested_stage=new_stage,
                    reason="задача уже в done",
                    missing_condition="(переходы из done запрещены)",
                    expected_action="создать новую задачу",
                ),
            )
        if not self.can_transition_to(new_stage):
            return TransitionResult(
                success=False,
                task=self,
                rejection=TransitionRejection(
                    current_stage=self.stage,
                    requested_stage=new_stage,
                    reason=(
                        f"переход {self.stage.value} → {new_stage.value} "
                        "не входит в lifecycle"
                    ),
                    missing_condition="допустимое ребро графа состояний",
                    expected_action=self._suggest_next_action(),
                ),
            )
        guard_fail = self._guard_for(new_stage)
        if guard_fail is not None:
            return TransitionResult(success=False, task=self, rejection=guard_fail)
        return TransitionResult(success=True, task=self, rejection=None)

    def transition_to(self, new_stage: TaskStage) -> TransitionResult:
        """
        Sole stage mutation API: rules → guards → apply or reject.
        Never assign self.stage elsewhere.
        """
        result = self.evaluate_transition(new_stage)
        if not result.success:
            return result

        previous = self.stage
        self.stage = new_stage

        # Side-effects on successful forward / fix-loop moves.
        if previous == TaskStage.PLANNING and new_stage == TaskStage.EXECUTION:
            self.current_step = self.current_step or "реализовать по утверждённому плану"
            self.expected_action = "завершить execution (execution_completed=true)"
        elif previous == TaskStage.EXECUTION and new_stage == TaskStage.VALIDATION:
            self.validation_passed = False
            self.current_step = "провести validation"
            self.expected_action = "отметить validation_passed=true при успехе"
        elif previous == TaskStage.VALIDATION and new_stage == TaskStage.EXECUTION:
            # Fix loop: re-open execution; validation must pass again later.
            self.execution_completed = False
            self.validation_passed = False
            self.current_step = "исправить замечания validation"
            self.expected_action = "завершить execution заново"
        elif new_stage == TaskStage.DONE:
            self.current_step = "задача завершена"
            self.expected_action = "(нет)"

        return TransitionResult(success=True, task=self, rejection=None)

    def approve_plan(self) -> None:
        if self.stage != TaskStage.PLANNING:
            raise TaskStateError(
                "Утверждать план можно только в stage=planning."
            )
        self.plan_approved = True
        self.expected_action = "перейти planning → execution"

    def mark_execution_completed(self) -> None:
        if self.stage != TaskStage.EXECUTION:
            raise TaskStateError(
                "execution_completed можно выставить только в stage=execution."
            )
        self.execution_completed = True
        self.expected_action = "перейти execution → validation"

    def mark_validation_passed(self, passed: bool = True) -> None:
        if self.stage != TaskStage.VALIDATION:
            raise TaskStateError(
                "validation_passed можно выставить только в stage=validation."
            )
        self.validation_passed = passed
        if passed:
            self.expected_action = "перейти validation → done"
        else:
            self.expected_action = "вернуться в execution или исправить и повторить"

    def pause(self) -> None:
        if self.stage == TaskStage.DONE:
            raise TaskStateError("Нельзя поставить на pause задачу в done.")
        if self.status == TaskStatus.PAUSED:
            raise TaskStateError("Задача уже на pause.")
        self.status = TaskStatus.PAUSED

    def resume(self) -> None:
        if self.status != TaskStatus.PAUSED:
            raise TaskStateError("Задача не на pause — resume невозможен.")
        if self.stage == TaskStage.DONE:
            raise TaskStateError("Нельзя resume задачу в done.")
        self.status = TaskStatus.ACTIVE

    def _suggest_next_action(self) -> str:
        if self.stage == TaskStage.PLANNING:
            if not self.plan_approved:
                return "утвердить план"
            return "перейти в execution"
        if self.stage == TaskStage.EXECUTION:
            if not self.execution_completed:
                return "завершить execution"
            return "перейти в validation"
        if self.stage == TaskStage.VALIDATION:
            if not self.validation_passed:
                return "пройти validation или вернуться в execution"
            return "перейти в done"
        return "задача завершена"

    def guards_summary(self) -> dict[str, bool]:
        return {
            "plan_approved": self.plan_approved,
            "execution_completed": self.execution_completed,
            "validation_passed": self.validation_passed,
        }

    def format_prompt_block(self) -> str:
        g = self.guards_summary()
        return "\n".join(
            [
                "CURRENT TASK",
                f"Goal: {self.goal}",
                f"Stage: {self.stage.value}",
                f"Status: {self.status.value}",
                f"Current step: {self.current_step or '(not set)'}",
                f"Expected action: {self.expected_action or '(not set)'}",
                "",
                "TRANSITION GUARDS",
                f"- plan_approved: {g['plan_approved']}",
                f"- execution_completed: {g['execution_completed']}",
                f"- validation_passed: {g['validation_passed']}",
                "",
                "TaskState is the source of truth for stage. "
                "You MUST NOT change stage yourself. "
                "Transitions go only through the controlled API "
                "(rules + guards). Skipping stages is forbidden. "
                "If Status is paused, wait for resume. "
                "If the user says «Продолжай», continue from Current step "
                "without inventing a stage change.",
            ]
        )

    def pipeline_labels(self) -> list[tuple[str, bool]]:
        return [(stage.value, stage == self.stage) for stage in STAGE_ORDER]
