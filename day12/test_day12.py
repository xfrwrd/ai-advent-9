"""Unit tests for Day 12 personalization (no real LLM calls)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent import Agent
from memory import PROFILE_KEY, MemoryStore
from profile import (
    UserProfile,
    profile_a_technical,
    profile_b_beginner,
)


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


class UserProfileTests(unittest.TestCase):
    def test_create_defaults(self) -> None:
        profile = UserProfile()
        self.assertIsNone(profile.name)
        self.assertEqual(profile.language, "ru")
        self.assertEqual(profile.style, "concise")
        self.assertEqual(profile.format, "structured")
        self.assertEqual(profile.expertise_level, "intermediate")
        self.assertEqual(profile.constraints, [])

    def test_update_preferences(self) -> None:
        profile = UserProfile(name="X")
        profile.update(
            style="detailed",
            format="step_by_step",
            constraints=["explain_terminology"],
        )
        self.assertEqual(profile.style, "detailed")
        self.assertEqual(profile.format, "step_by_step")
        self.assertEqual(profile.constraints, ["explain_terminology"])

    def test_serialization_roundtrip(self) -> None:
        original = profile_a_technical()
        restored = UserProfile.from_json(original.to_json())
        self.assertEqual(restored.to_dict(), original.to_dict())

    def test_from_dict_handles_empty(self) -> None:
        profile = UserProfile.from_dict(None)
        self.assertEqual(profile.style, "concise")

    def test_format_prompt_block_contains_preferences(self) -> None:
        block = profile_b_beginner().format_prompt_block()
        self.assertIn("USER PROFILE:", block)
        self.assertIn("Алекс", block)
        self.assertIn("detailed", block)
        self.assertIn("step_by_step", block)
        self.assertIn("explain_terminology", block)


class ProfilePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.memory = MemoryStore(base_dir=self.base)
        self.agent = Agent(
            api_key="test",
            memory=self.memory,
            base_dir=self.base,
            llm_client=fake_llm_factory(),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_and_load_profile(self) -> None:
        self.agent.save_profile(profile_a_technical())
        loaded = self.agent.get_profile()
        self.assertEqual(loaded.name, "Ксения")
        self.assertEqual(loaded.style, "concise")
        raw = json.loads((self.base / "long_term_memory.json").read_text())
        self.assertIn(PROFILE_KEY, raw)
        parsed = json.loads(raw[PROFILE_KEY])
        self.assertEqual(parsed["name"], "Ксения")

    def test_profile_in_long_term_memory(self) -> None:
        self.agent.save_profile(profile_b_beginner())
        self.assertIn(PROFILE_KEY, self.memory.long_term)
        self.assertNotIn(PROFILE_KEY, self.memory.working)

    def test_survives_clear_short_term(self) -> None:
        self.agent.save_profile(profile_a_technical())
        self.memory.append_message("user", "hi")
        self.agent.clear_short_term()
        self.assertEqual(self.memory.short_term, [])
        self.assertEqual(self.agent.get_profile().name, "Ксения")

    def test_survives_clear_working_memory(self) -> None:
        self.agent.save_profile(profile_a_technical())
        self.agent.set_working_memory("task", "demo")
        self.agent.clear_working_memory()
        self.assertEqual(self.memory.working, {})
        self.assertEqual(self.agent.get_profile().name, "Ксения")

    def test_update_profile_persists(self) -> None:
        self.agent.save_profile(UserProfile(name="X", style="concise"))
        self.agent.update_profile(style="friendly", expertise_level="beginner")
        reloaded = Agent(
            api_key="test",
            base_dir=self.base,
            llm_client=fake_llm_factory(),
        )
        self.assertEqual(reloaded.get_profile().style, "friendly")
        self.assertEqual(reloaded.get_profile().expertise_level, "beginner")


class PromptInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.client = fake_llm_factory("ok")
        self.agent = Agent(
            api_key="test",
            base_dir=self.base,
            llm_client=self.client,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_profile_in_every_llm_request(self) -> None:
        self.agent.save_profile(profile_a_technical())
        self.agent.ask("Что такое idempotency?")
        messages = self.client.captured["messages"]
        self.assertIsNotNone(messages)
        system = messages[0]["content"]
        self.assertIn("USER PROFILE:", system)
        self.assertIn("Ксения", system)
        self.assertIn("concise", system)
        self.assertIn("advanced", system)
        self.assertEqual(messages[-1]["content"], "Что такое idempotency?")
        # User must NOT need to repeat preferences in the request itself.
        self.assertNotIn("ответь коротко", messages[-1]["content"].lower())

    def test_different_profiles_change_system_prompt(self) -> None:
        self.agent.save_profile(profile_a_technical())
        sys_a = self.agent.build_system_content()
        self.agent.save_profile(profile_b_beginner())
        sys_b = self.agent.build_system_content()
        self.assertIn("Ксения", sys_a)
        self.assertIn("Алекс", sys_b)
        self.assertIn("concise", sys_a)
        self.assertIn("detailed", sys_b)
        self.assertNotEqual(sys_a, sys_b)

    def test_profile_excluded_from_generic_long_term_dump(self) -> None:
        self.agent.save_profile(profile_a_technical())
        self.agent.set_long_term_memory("preferred_stack", "Go")
        system = self.agent.build_system_content()
        # Dedicated block present
        self.assertIn("USER PROFILE:", system)
        self.assertIn("Ксения", system)
        # Other LT facts still present
        self.assertIn("preferred_stack", system)
        # Raw JSON blob should not be dumped as a plain LT line twice awkwardly —
        # profile key excluded from Long-term memory list
        lt_section = system.split("Long-term memory:")[-1].split("Working memory:")[0]
        self.assertNotIn(PROFILE_KEY, lt_section)


if __name__ == "__main__":
    unittest.main()
