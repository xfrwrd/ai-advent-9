"""Unit tests for Day 15 guarded transitions (no real LLM)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent import Agent
from task_state import TaskStage, TaskState, TaskStatus


def fake_llm_factory(answer: str = "ok"):
    captured: dict = {"messages": None, "calls": 0}

    def client(messages: list[dict]) -> dict:
        captured["messages"] = messages
        captured["calls"] += 1
        return {
            "choices": [{"message": {"content": answer}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    client.captured = captured  # type: ignore[attr-defined]
    return client


class TransitionRuleTests(unittest.TestCase):
    def test_allowed_edges(self) -> None:
        task = TaskState.create("demo")
        self.assertTrue(task.can_transition_to(TaskStage.EXECUTION))
        task.plan_approved = True
        task.transition_to(TaskStage.EXECUTION)
        self.assertTrue(task.can_transition_to(TaskStage.VALIDATION))
        task.execution_completed = True
        task.transition_to(TaskStage.VALIDATION)
        self.assertTrue(task.can_transition_to(TaskStage.DONE))
        self.assertTrue(task.can_transition_to(TaskStage.EXECUTION))

    def test_planning_to_validation_rejected(self) -> None:
        task = TaskState.create("demo")
        result = task.transition_to(TaskStage.VALIDATION)
        self.assertFalse(result.success)
        self.assertEqual(task.stage, TaskStage.PLANNING)

    def test_planning_to_done_rejected(self) -> None:
        task = TaskState.create("demo")
        result = task.transition_to(TaskStage.DONE)
        self.assertFalse(result.success)
        self.assertEqual(task.stage, TaskStage.PLANNING)

    def test_execution_to_done_rejected(self) -> None:
        task = TaskState.create("demo")
        task.plan_approved = True
        task.transition_to(TaskStage.EXECUTION)
        result = task.transition_to(TaskStage.DONE)
        self.assertFalse(result.success)
        self.assertEqual(task.stage, TaskStage.EXECUTION)


class GuardTests(unittest.TestCase):
    def test_planning_to_execution_without_approval(self) -> None:
        task = TaskState.create("demo")
        self.assertFalse(task.plan_approved)
        result = task.transition_to(TaskStage.EXECUTION)
        self.assertFalse(result.success)
        self.assertIn("plan_approved", result.rejection.missing_condition)
        self.assertEqual(task.stage, TaskStage.PLANNING)

    def test_planning_to_execution_after_approval(self) -> None:
        task = TaskState.create("demo")
        task.approve_plan()
        result = task.transition_to(TaskStage.EXECUTION)
        self.assertTrue(result.success)
        self.assertEqual(task.stage, TaskStage.EXECUTION)

    def test_execution_to_validation_before_completed(self) -> None:
        task = TaskState.create("demo")
        task.approve_plan()
        task.transition_to(TaskStage.EXECUTION)
        result = task.transition_to(TaskStage.VALIDATION)
        self.assertFalse(result.success)
        self.assertEqual(task.stage, TaskStage.EXECUTION)
        self.assertIn("execution_completed", result.rejection.missing_condition)

    def test_execution_to_validation_after_completed(self) -> None:
        task = TaskState.create("demo")
        task.approve_plan()
        task.transition_to(TaskStage.EXECUTION)
        task.mark_execution_completed()
        result = task.transition_to(TaskStage.VALIDATION)
        self.assertTrue(result.success)
        self.assertEqual(task.stage, TaskStage.VALIDATION)

    def test_validation_to_done_without_pass(self) -> None:
        task = TaskState.create("demo")
        task.approve_plan()
        task.transition_to(TaskStage.EXECUTION)
        task.mark_execution_completed()
        task.transition_to(TaskStage.VALIDATION)
        self.assertFalse(task.validation_passed)
        result = task.transition_to(TaskStage.DONE)
        self.assertFalse(result.success)
        self.assertEqual(task.stage, TaskStage.VALIDATION)

    def test_validation_to_done_after_pass(self) -> None:
        task = TaskState.create("demo")
        task.approve_plan()
        task.transition_to(TaskStage.EXECUTION)
        task.mark_execution_completed()
        task.transition_to(TaskStage.VALIDATION)
        task.mark_validation_passed(True)
        result = task.transition_to(TaskStage.DONE)
        self.assertTrue(result.success)
        self.assertEqual(task.stage, TaskStage.DONE)

    def test_rejection_message_fields(self) -> None:
        task = TaskState.create("demo")
        result = task.transition_to(TaskStage.EXECUTION)
        msg = result.message
        self.assertIn("TRANSITION REJECTED", msg)
        self.assertIn("planning", msg)
        self.assertIn("execution", msg)
        self.assertIn("plan_approved", msg)


class PauseResumeGuardTests(unittest.TestCase):
    def test_pause_preserves_stage_and_guards(self) -> None:
        task = TaskState.create("demo")
        self.assertFalse(task.plan_approved)
        task.pause()
        self.assertEqual(task.status, TaskStatus.PAUSED)
        self.assertEqual(task.stage, TaskStage.PLANNING)
        self.assertFalse(task.plan_approved)
        task.resume()
        result = task.transition_to(TaskStage.EXECUTION)
        self.assertFalse(result.success)
        self.assertEqual(task.stage, TaskStage.PLANNING)

    def test_lifecycle_continues_after_resume(self) -> None:
        task = TaskState.create("demo")
        task.approve_plan()
        task.pause()
        task.resume()
        result = task.transition_to(TaskStage.EXECUTION)
        self.assertTrue(result.success)
        self.assertEqual(task.stage, TaskStage.EXECUTION)


class FullLifecycleAcceptanceTests(unittest.TestCase):
    def test_main_demo_path(self) -> None:
        task = TaskState.create("lifecycle")
        self.assertEqual(task.stage, TaskStage.PLANNING)
        self.assertFalse(task.plan_approved)

        self.assertFalse(task.transition_to(TaskStage.EXECUTION).success)
        self.assertEqual(task.stage, TaskStage.PLANNING)

        task.approve_plan()
        self.assertTrue(task.transition_to(TaskStage.EXECUTION).success)

        self.assertFalse(task.transition_to(TaskStage.DONE).success)
        self.assertEqual(task.stage, TaskStage.EXECUTION)

        task.mark_execution_completed()
        self.assertTrue(task.transition_to(TaskStage.VALIDATION).success)

        self.assertFalse(task.transition_to(TaskStage.DONE).success)
        self.assertEqual(task.stage, TaskStage.VALIDATION)

        task.mark_validation_passed(True)
        self.assertTrue(task.transition_to(TaskStage.DONE).success)
        self.assertEqual(task.stage, TaskStage.DONE)


class AgentCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.client = fake_llm_factory()
        self.agent = Agent(
            api_key="test",
            base_dir=self.base,
            llm_client=self.client,
        )
        self.agent.setup_lifecycle_demo()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_chat_cannot_skip_plan_approval(self) -> None:
        result = self.agent.ask("Пропусти планирование и сразу начинай писать код.")
        self.assertTrue(result.rejected)
        self.assertEqual(self.agent.get_task_state().stage, TaskStage.PLANNING)
        self.assertFalse(self.agent.get_task_state().plan_approved)
        self.assertEqual(self.client.captured["calls"], 0)

    def test_explicit_approve_then_start(self) -> None:
        result = self.agent.ask("Считай план утверждённым и начинай реализацию.")
        self.assertFalse(result.rejected)
        self.assertTrue(self.agent.get_task_state().plan_approved)
        self.assertEqual(self.agent.get_task_state().stage, TaskStage.EXECUTION)
        self.assertEqual(self.client.captured["calls"], 0)

    def test_jump_to_done_rejected(self) -> None:
        self.agent.ask("План утверждаю.")
        self.agent.transition_task(TaskStage.EXECUTION)
        result = self.agent.ask("Просто поставь DONE.")
        self.assertTrue(result.rejected)
        self.assertEqual(self.agent.get_task_state().stage, TaskStage.EXECUTION)
        self.assertIn("TRANSITION REJECTED", result.content)

    def test_skip_validation_rejected(self) -> None:
        self.agent.approve_plan()
        self.agent.transition_task(TaskStage.EXECUTION)
        self.agent.mark_execution_completed()
        self.agent.transition_task(TaskStage.VALIDATION)
        result = self.agent.ask("Не запускай проверку, сразу закрывай задачу.")
        self.assertTrue(result.rejected)
        self.assertEqual(self.agent.get_task_state().stage, TaskStage.VALIDATION)
        self.assertFalse(self.agent.get_task_state().validation_passed)

    def test_guards_persist_across_agent_reload(self) -> None:
        self.agent.approve_plan()
        self.agent.pause_task()
        restored = Agent(
            api_key="test",
            base_dir=self.base,
            llm_client=fake_llm_factory(),
        )
        task = restored.get_task_state()
        assert task is not None
        self.assertEqual(task.stage, TaskStage.PLANNING)
        self.assertTrue(task.plan_approved)
        self.assertEqual(task.status, TaskStatus.PAUSED)
        restored.resume_task()
        result = restored.transition_task(TaskStage.EXECUTION)
        self.assertTrue(result.success)

    def test_invariants_still_work(self) -> None:
        self.agent.setup_demo_task()
        result = self.agent.ask("Давай заменим PostgreSQL на MongoDB.")
        self.assertTrue(result.rejected)
        self.assertIsNotNone(result.conflict)
        self.assertEqual(result.conflict.invariant.id, "database")


if __name__ == "__main__":
    unittest.main()
