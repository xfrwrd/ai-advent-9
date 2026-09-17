"""Day 13: Task State Machine — controlled stages + pause/resume."""

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


# Controlled transitions only — stage is never changed ad hoc.
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
class TaskState:
    """Finite-state representation of the current task (Working Memory)."""

    goal: str
    stage: TaskStage = TaskStage.PLANNING
    status: TaskStatus = TaskStatus.ACTIVE
    current_step: str = ""
    expected_action: str = ""

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "stage": self.stage.value,
            "status": self.status.value,
            "current_step": self.current_step,
            "expected_action": self.expected_action,
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
            expected_action="утвердить план и перейти к execution",
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
        return new_stage in ALLOWED_TRANSITIONS.get(self.stage, set())

    def transition_to(self, new_stage: TaskStage) -> None:
        if self.status == TaskStatus.PAUSED:
            raise TaskStateError(
                "Нельзя менять stage, пока задача на pause. Сначала resume."
            )
        if self.stage == TaskStage.DONE:
            raise TaskStateError("Задача уже в done — переходы запрещены.")
        if not self.can_transition_to(new_stage):
            raise TaskStateError(
                f"Недопустимый переход: {self.stage.value} → {new_stage.value}."
            )
        self.stage = new_stage

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

    def format_prompt_block(self) -> str:
        """Structured CURRENT TASK block for the LLM system prompt."""
        return "\n".join(
            [
                "CURRENT TASK",
                f"Goal: {self.goal}",
                f"Stage: {self.stage.value}",
                f"Status: {self.status.value}",
                f"Current step: {self.current_step or '(not set)'}",
                f"Expected action: {self.expected_action or '(not set)'}",
                "",
                "TaskState is the source of truth for stage/step/action.",
                "Do not invent a different stage. If Status is paused, wait for resume.",
                "If the user says «Продолжай», continue from Current step / Expected action "
                "without asking them to re-explain the goal and without returning to planning "
                "unless Stage is still planning.",
            ]
        )

    def pipeline_labels(self) -> list[tuple[str, bool]]:
        """For UI: (stage_name, is_current)."""
        return [(stage.value, stage == self.stage) for stage in STAGE_ORDER]
