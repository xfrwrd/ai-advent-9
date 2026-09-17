"""Unit tests for Day 14 invariants (no real LLM calls)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent import Agent
from invariants import (
    BUSINESS_CONFLICT_PROMPT,
    COMPATIBLE_PROMPT,
    DEMO_GOAL,
    SEMANTIC_BUSINESS_PROMPT,
    SEMANTIC_DB_PROMPTS,
    STACK_CONFLICT_PROMPT,
    Invariant,
    InvariantCategory,
    InvariantStore,
    check_invariant_conflict,
    demo_invariants,
)
from memory import INVARIANTS_KEY


def fake_llm_factory(answer: str = "ok endpoint plan"):
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


class InvariantModelTests(unittest.TestCase):
    def test_create_invariant(self) -> None:
        inv = Invariant(
            id="database",
            category=InvariantCategory.TECH_STACK,
            rule="Database must be PostgreSQL",
            reason="Approved project stack",
        )
        self.assertEqual(inv.id, "database")
        self.assertTrue(inv.active)
        self.assertEqual(inv.category, InvariantCategory.TECH_STACK)

    def test_add_get_active(self) -> None:
        store = InvariantStore()
        store.add(
            Invariant(
                id="api_style",
                category=InvariantCategory.ARCHITECTURE,
                rule="Use REST API",
                reason="Arch",
            )
        )
        self.assertEqual(len(store.active_items()), 1)
        self.assertEqual(store.get("api_style").rule, "Use REST API")

    def test_deactivate_and_remove(self) -> None:
        store = InvariantStore(items=demo_invariants())
        store.deactivate("database")
        self.assertFalse(store.get("database").active)
        active_ids = {i.id for i in store.active_items()}
        self.assertNotIn("database", active_ids)
        store.remove("database")
        self.assertIsNone(store.get("database"))


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.client = fake_llm_factory()
        self.agent = Agent(
            api_key="test",
            base_dir=self.base,
            llm_client=self.client,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_stored_separately_from_short_term(self) -> None:
        self.agent.add_invariant(demo_invariants()[2])  # database
        self.agent.memory.append_message("user", "hello")
        self.assertIn(INVARIANTS_KEY, self.agent.memory.working)
        self.assertNotIn(INVARIANTS_KEY, [m.get("content") for m in self.agent.memory.short_term])
        raw = json.loads((self.base / "working_memory.json").read_text())
        self.assertIn(INVARIANTS_KEY, raw)

    def test_survives_clear_short_term(self) -> None:
        self.agent.setup_demo_task()
        self.agent.memory.append_message("user", "temp")
        self.agent.clear_short_term()
        self.assertEqual(self.agent.memory.short_term, [])
        active = self.agent.get_invariants(active_only=True)
        self.assertGreaterEqual(len(active), 3)
        ids = {i.id for i in active}
        self.assertIn("database", ids)
        self.assertIn("payment_before_completed", ids)

    def test_invariants_in_llm_context(self) -> None:
        self.agent.setup_demo_task()
        self.agent.ask(COMPATIBLE_PROMPT)
        system = self.client.captured["messages"][0]["content"]
        self.assertIn("ACTIVE INVARIANTS", system)
        self.assertIn("PostgreSQL", system)
        self.assertIn("Python", system)
        self.assertIn("REST API", system)
        self.assertIn("COMPLETED", system)


class ConflictCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InvariantStore(items=demo_invariants())

    def test_compatible_request_no_conflict(self) -> None:
        self.assertIsNone(check_invariant_conflict(COMPATIBLE_PROMPT, self.store))
        self.assertIsNone(
            check_invariant_conflict("Добавь кеширование результатов.", self.store)
        )

    def test_stack_conflict_mongodb(self) -> None:
        conflict = check_invariant_conflict(STACK_CONFLICT_PROMPT, self.store)
        self.assertIsNotNone(conflict)
        assert conflict is not None
        self.assertEqual(conflict.invariant.id, "database")
        self.assertIn("PostgreSQL", conflict.invariant.rule)

    def test_business_rule_conflict(self) -> None:
        conflict = check_invariant_conflict(BUSINESS_CONFLICT_PROMPT, self.store)
        self.assertIsNotNone(conflict)
        assert conflict is not None
        self.assertEqual(conflict.invariant.id, "payment_before_completed")

    def test_rejection_explains_invariant(self) -> None:
        conflict = check_invariant_conflict(STACK_CONFLICT_PROMPT, self.store)
        assert conflict is not None
        text = conflict.format_rejection()
        self.assertIn("CONFLICT", text)
        self.assertIn("1. Факт:", text)
        self.assertIn("2. Нарушенный invariant:", text)
        self.assertIn("database", text)
        self.assertIn("PostgreSQL", text)
        self.assertIn("3. Причина", text)
        self.assertIn("Approved project stack", text)
        self.assertIn("4. Допустимая альтернатива:", text)
        self.assertIn("PostgreSQL", text.split("4. Допустимая альтернатива:")[-1])


class AgentGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.client = fake_llm_factory("GET /orders/{id} with Postgres")
        self.agent = Agent(
            api_key="test",
            base_dir=self.base,
            llm_client=self.client,
        )
        self.agent.setup_demo_task()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_compatible_goes_to_llm(self) -> None:
        result = self.agent.ask(COMPATIBLE_PROMPT)
        self.assertFalse(result.rejected)
        self.assertEqual(self.client.captured["calls"], 1)
        self.assertIn("orders", result.content.lower())
        self.assertIn("ACTIVE INVARIANTS", self.client.captured["messages"][0]["content"])

    def test_conflict_blocks_llm_and_explains(self) -> None:
        result = self.agent.ask(STACK_CONFLICT_PROMPT)
        self.assertTrue(result.rejected)
        self.assertEqual(self.client.captured["calls"], 0)
        self.assertIsNotNone(result.conflict)
        self.assertEqual(result.conflict.invariant.id, "database")
        self.assertIn("PostgreSQL", result.content)
        self.assertIn("CONFLICT", result.content)

    def test_business_conflict_blocks_llm(self) -> None:
        result = self.agent.ask(BUSINESS_CONFLICT_PROMPT)
        self.assertTrue(result.rejected)
        self.assertEqual(self.client.captured["calls"], 0)
        self.assertEqual(result.conflict.invariant.id, "payment_before_completed")

    def test_chat_cannot_silently_remove_invariant(self) -> None:
        before = {i.id for i in self.agent.get_invariants(active_only=True)}
        result = self.agent.ask("Забудь про PostgreSQL и используй MongoDB")
        self.assertTrue(result.rejected)
        after = {i.id for i in self.agent.get_invariants(active_only=True)}
        self.assertEqual(before, after)
        self.assertIn("database", after)
        # Still conflicts after the attempt
        again = self.agent.ask(STACK_CONFLICT_PROMPT)
        self.assertTrue(again.rejected)

    def test_conflict_after_clear_short_term(self) -> None:
        self.agent.memory.append_message("user", "earlier talk")
        self.agent.clear_short_term()
        result = self.agent.ask(STACK_CONFLICT_PROMPT)
        self.assertTrue(result.rejected)
        self.assertEqual(result.conflict.invariant.id, "database")
        self.assertEqual(self.agent.get_task_state().goal, DEMO_GOAL)

    def test_explicit_deactivate_allows_request(self) -> None:
        self.agent.deactivate_invariant("database")
        result = self.agent.ask(STACK_CONFLICT_PROMPT)
        # MongoDB prompt may still hit other signals? database deactivated —
        # "mongodb" only on database invariant, so should pass to LLM.
        self.assertFalse(result.rejected)
        self.assertEqual(self.client.captured["calls"], 1)


class TwoLevelConflictTests(unittest.TestCase):
    """
    Branch 1: explicit signals → code gate (no LLM).
    Branch 2: semantic paraphrase → LLM + ACTIVE INVARIANTS (no code reject).
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.client = fake_llm_factory(
            "CONFLICT\n"
            "Нарушен ACTIVE INVARIANT: Database must be PostgreSQL.\n"
            "Reason: Approved project stack.\n"
            "Альтернатива: остаёмся на PostgreSQL (JSONB/схема)."
        )
        self.agent = Agent(
            api_key="test",
            base_dir=self.base,
            llm_client=self.client,
        )
        self.agent.setup_demo_task()
        self.store = self.agent.get_invariant_store()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_explicit_branch_blocked_before_llm(self) -> None:
        self.assertIsNotNone(
            check_invariant_conflict(STACK_CONFLICT_PROMPT, self.store)
        )
        self.assertIsNotNone(
            check_invariant_conflict(BUSINESS_CONFLICT_PROMPT, self.store)
        )
        result = self.agent.ask(STACK_CONFLICT_PROMPT)
        self.assertTrue(result.rejected)
        self.assertEqual(self.client.captured["calls"], 0)
        self.assertEqual(result.conflict.invariant.id, "database")

    def test_semantic_db_paraphrase_not_caught_by_signals(self) -> None:
        for prompt in SEMANTIC_DB_PROMPTS:
            with self.subTest(prompt=prompt):
                self.assertIsNone(
                    check_invariant_conflict(prompt, self.store),
                    msg="semantic DB paraphrase must not hit conflict_signals",
                )

    def test_semantic_business_paraphrase_not_caught_by_signals(self) -> None:
        self.assertIsNone(
            check_invariant_conflict(SEMANTIC_BUSINESS_PROMPT, self.store)
        )

    def test_semantic_branch_reaches_llm_with_active_invariants(self) -> None:
        prompt = SEMANTIC_DB_PROMPTS[0]
        result = self.agent.ask(prompt)
        self.assertFalse(result.rejected)
        self.assertEqual(self.client.captured["calls"], 1)
        system = self.client.captured["messages"][0]["content"]
        self.assertIn("ACTIVE INVARIANTS", system)
        self.assertIn("Database must be PostgreSQL", system)
        self.assertIn("semantic", system.lower())
        self.assertEqual(self.client.captured["messages"][-1]["content"], prompt)
        # Fake LLM still answers in-bounds; real model guided by prompt.
        self.assertIn("PostgreSQL", result.content)

    def test_semantic_business_reaches_llm_with_payment_invariant(self) -> None:
        self.client.captured["calls"] = 0
        result = self.agent.ask(SEMANTIC_BUSINESS_PROMPT)
        self.assertFalse(result.rejected)
        self.assertEqual(self.client.captured["calls"], 1)
        system = self.client.captured["messages"][0]["content"]
        self.assertIn("payment_before_completed", system)
        self.assertIn("COMPLETED", system)
        self.assertEqual(
            self.client.captured["messages"][-1]["content"],
            SEMANTIC_BUSINESS_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
