"""Unit tests for Day 13 Task State Machine (no real LLM calls)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent import DEMO_GOAL, Agent
from memory import TASK_STATE_KEY, MemoryStore
from task_state import TaskStage, TaskState, TaskStateError, TaskStatus


def fake_llm_factory(answer: str = "fake answer"):
    captured: dict = {"messages": None}

    def client(messages: list[dict]) -> dict:
        captured["messages"] = messages
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


class TaskStateModelTests(unittest.TestCase):
    def test_create_starts_in_planning(self) -> None:
        task = TaskState.create("Сделать API")
        self.assertEqual(task.goal, "Сделать API")
        self.assertEqual(task.stage, TaskStage.PLANNING)
        self.assertEqual(task.status, TaskStatus.ACTIVE)
        self.assertTrue(task.current_step)
        self.assertTrue(task.expected_action)

    def test_create_rejects_empty_goal(self) -> None:
        with self.assertRaises(TaskStateError):
            TaskState.create("  ")

    def test_planning_to_execution(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.transition_to(TaskStage.EXECUTION)
        self.assertEqual(task.stage, TaskStage.EXECUTION)

    def test_execution_to_validation(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.transition_to(TaskStage.EXECUTION)
        task.transition_to(TaskStage.VALIDATION)
        self.assertEqual(task.stage, TaskStage.VALIDATION)

    def test_validation_to_done(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.transition_to(TaskStage.EXECUTION)
        task.transition_to(TaskStage.VALIDATION)
        task.transition_to(TaskStage.DONE)
        self.assertEqual(task.stage, TaskStage.DONE)

    def test_validation_back_to_execution(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.transition_to(TaskStage.EXECUTION)
        task.transition_to(TaskStage.VALIDATION)
        task.transition_to(TaskStage.EXECUTION)
        self.assertEqual(task.stage, TaskStage.EXECUTION)

    def test_invalid_transition_rejected(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        with self.assertRaises(TaskStateError):
            task.transition_to(TaskStage.VALIDATION)
        with self.assertRaises(TaskStateError):
            task.transition_to(TaskStage.DONE)
        self.assertEqual(task.stage, TaskStage.PLANNING)

    def test_done_has_no_transitions(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.transition_to(TaskStage.EXECUTION)
        task.transition_to(TaskStage.VALIDATION)
        task.transition_to(TaskStage.DONE)
        with self.assertRaises(TaskStateError):
            task.transition_to(TaskStage.EXECUTION)

    def test_save_current_step_and_expected_action(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.set_step("определить API endpoints", "описать request/response models")
        self.assertEqual(task.current_step, "определить API endpoints")
        self.assertEqual(task.expected_action, "описать request/response models")

    def test_serialization_roundtrip(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.transition_to(TaskStage.EXECUTION)
        task.set_step("step", "action")
        restored = TaskState.from_json(task.to_json())
        assert restored is not None
        self.assertEqual(restored.to_dict(), task.to_dict())

    def test_format_prompt_block(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        block = task.format_prompt_block()
        self.assertIn("CURRENT TASK", block)
        self.assertIn(DEMO_GOAL, block)
        self.assertIn("Stage: planning", block)
        self.assertIn("Status: active", block)


class PauseResumeTests(unittest.TestCase):
    def test_pause_and_resume(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.transition_to(TaskStage.EXECUTION)
        task.set_step("endpoints", "models")
        task.pause()
        self.assertEqual(task.status, TaskStatus.PAUSED)
        self.assertEqual(task.stage, TaskStage.EXECUTION)
        self.assertEqual(task.current_step, "endpoints")
        task.resume()
        self.assertEqual(task.status, TaskStatus.ACTIVE)
        self.assertEqual(task.stage, TaskStage.EXECUTION)

    def test_pause_resume_on_planning(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.pause()
        self.assertEqual(task.stage, TaskStage.PLANNING)
        self.assertEqual(task.status, TaskStatus.PAUSED)
        task.resume()
        self.assertEqual(task.status, TaskStatus.ACTIVE)
        self.assertEqual(task.stage, TaskStage.PLANNING)

    def test_pause_resume_on_execution(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.transition_to(TaskStage.EXECUTION)
        task.pause()
        self.assertEqual(task.stage, TaskStage.EXECUTION)
        task.resume()
        self.assertEqual(task.stage, TaskStage.EXECUTION)

    def test_pause_resume_on_validation(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.transition_to(TaskStage.EXECUTION)
        task.transition_to(TaskStage.VALIDATION)
        task.pause()
        self.assertEqual(task.stage, TaskStage.VALIDATION)
        self.assertEqual(task.status, TaskStatus.PAUSED)
        task.resume()
        self.assertEqual(task.status, TaskStatus.ACTIVE)
        self.assertEqual(task.stage, TaskStage.VALIDATION)

    def test_cannot_transition_while_paused(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.pause()
        with self.assertRaises(TaskStateError):
            task.transition_to(TaskStage.EXECUTION)

    def test_cannot_pause_done(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.transition_to(TaskStage.EXECUTION)
        task.transition_to(TaskStage.VALIDATION)
        task.transition_to(TaskStage.DONE)
        with self.assertRaises(TaskStateError):
            task.pause()

    def test_double_pause_rejected(self) -> None:
        task = TaskState.create(DEMO_GOAL)
        task.pause()
        with self.assertRaises(TaskStateError):
            task.pause()


class PersistenceAndPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.client = fake_llm_factory("продолжаю с endpoints")
        self.agent = Agent(
            api_key="test",
            base_dir=self.base,
            llm_client=self.client,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_task_persisted_in_working_memory(self) -> None:
        self.agent.create_task(DEMO_GOAL)
        self.agent.transition_task(TaskStage.EXECUTION)
        self.agent.update_task_step("endpoints", "models")
        self.assertIn(TASK_STATE_KEY, self.agent.memory.working)
        raw = json.loads((self.base / "working_memory.json").read_text())
        parsed = json.loads(raw[TASK_STATE_KEY])
        self.assertEqual(parsed["stage"], "execution")
        self.assertEqual(parsed["current_step"], "endpoints")

    def test_survives_clear_short_term(self) -> None:
        self.agent.create_task(DEMO_GOAL)
        self.agent.transition_task(TaskStage.EXECUTION)
        self.agent.update_task_step(
            "определить API endpoints",
            "описать request/response models",
        )
        self.agent.pause_task()
        self.agent.memory.append_message("user", "старый диалог")
        self.agent.clear_short_term()
        self.assertEqual(self.agent.memory.short_term, [])
        task = self.agent.get_task_state()
        assert task is not None
        self.assertEqual(task.stage, TaskStage.EXECUTION)
        self.assertEqual(task.status, TaskStatus.PAUSED)
        self.assertEqual(task.current_step, "определить API endpoints")

    def test_restore_after_new_agent_session(self) -> None:
        self.agent.create_task(DEMO_GOAL)
        self.agent.transition_task(TaskStage.EXECUTION)
        self.agent.update_task_step("endpoints", "models")
        self.agent.pause_task()
        restored = Agent(
            api_key="test",
            base_dir=self.base,
            llm_client=fake_llm_factory(),
        )
        task = restored.get_task_state()
        assert task is not None
        self.assertEqual(task.goal, DEMO_GOAL)
        self.assertEqual(task.stage, TaskStage.EXECUTION)
        self.assertEqual(task.status, TaskStatus.PAUSED)
        restored.resume_task()
        self.assertEqual(restored.get_task_state().status, TaskStatus.ACTIVE)

    def test_task_state_in_llm_context(self) -> None:
        self.agent.create_task(DEMO_GOAL)
        self.agent.transition_task(TaskStage.EXECUTION)
        self.agent.update_task_step(
            "определить API endpoints",
            "описать request/response models",
        )
        self.agent.ask("Продолжай")
        messages = self.client.captured["messages"]
        self.assertIsNotNone(messages)
        system = messages[0]["content"]
        self.assertIn("CURRENT TASK", system)
        self.assertIn(DEMO_GOAL, system)
        self.assertIn("Stage: execution", system)
        self.assertIn("определить API endpoints", system)
        self.assertIn("описать request/response models", system)
        self.assertEqual(messages[-1]["content"], "Продолжай")

    def test_task_state_excluded_from_generic_working_dump(self) -> None:
        self.agent.create_task(DEMO_GOAL)
        self.agent.set_working_memory("scratch", "note")
        system = self.agent.build_system_content()
        self.assertIn("CURRENT TASK", system)
        self.assertIn("scratch", system)
        working_section = system.split("Working memory:")[-1]
        self.assertNotIn(TASK_STATE_KEY, working_section)

    def test_acceptance_continue_after_pause_and_clear_stm(self) -> None:
        """Main Day 13 acceptance: pause → clear STM → continue."""
        self.agent.create_task(DEMO_GOAL)
        self.agent.transition_task(TaskStage.EXECUTION)
        self.agent.update_task_step(
            "определить API endpoints",
            "описать request/response models",
        )
        self.agent.pause_task()
        self.agent.memory.append_message("user", "раньше обсуждали план")
        self.agent.memory.append_message("assistant", "вот план...")
        self.agent.clear_short_term()
        self.agent.resume_task()

        result = self.agent.ask("Продолжай")
        self.assertEqual(result.content, "продолжаю с endpoints")
        system = self.client.captured["messages"][0]["content"]
        self.assertIn("CURRENT TASK", system)
        self.assertIn("Stage: execution", system)
        self.assertIn("Status: active", system)
        self.assertIn("определить API endpoints", system)
        # Dialogue history empty except current turn
        roles = [m["role"] for m in self.client.captured["messages"]]
        self.assertEqual(roles.count("user"), 1)


if __name__ == "__main__":
    unittest.main()
