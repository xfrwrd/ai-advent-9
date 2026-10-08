"""Day 29 settings stay distinct and share one retrieved context. No Ollama calls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day24.ground import GROUNDED_SYSTEM, UNKNOWN_ANSWER
from day29.answer import answer_with
from day29.config import BASELINE_CONFIG, OPTIMIZED_CONFIG, request_body
from day29.ollama import _tokens_per_sec
from day29.prompts import OPTIMIZED_SYSTEM, context_budget, render_prompt, worst_case_hits
from day29.retrieve import FrozenContext


class ConfigTests(unittest.TestCase):
    def test_baseline_matches_the_day28_request(self) -> None:
        body = request_body(BASELINE_CONFIG, "вопрос")
        self.assertEqual(body["model"], "qwen3:8b")
        self.assertEqual(body["stream"], False)
        self.assertEqual(body["think"], False)
        self.assertNotIn("options", body)

    def test_optimized_sends_the_experiment_options(self) -> None:
        body = request_body(OPTIMIZED_CONFIG, "вопрос")
        self.assertEqual(body["think"], False)
        self.assertEqual(
            body["options"],
            {"temperature": 0.1, "num_predict": 512, "num_ctx": 4096},
        )

    def test_both_prompts_share_the_context_and_keep_their_templates(self) -> None:
        hits = worst_case_hits()[:1]
        baseline = render_prompt("Чему равен CHUNK_SIZE_CHARS?", hits, BASELINE_CONFIG)
        optimized = render_prompt("Чему равен CHUNK_SIZE_CHARS?", hits, OPTIMIZED_CONFIG)
        self.assertIn(GROUNDED_SYSTEM, baseline)
        self.assertNotIn(GROUNDED_SYSTEM, optimized)
        self.assertIn(OPTIMIZED_SYSTEM, optimized)
        self.assertEqual(baseline.split("USER:\n", 1)[1], optimized.split("USER:\n", 1)[1])
        self.assertIn(hits[0].chunk.chunk_id, baseline)

    def test_optimized_window_is_tight_for_five_full_chunks(self) -> None:
        prompt = render_prompt("вопрос", worst_case_hits(), OPTIMIZED_CONFIG)
        budget = context_budget(prompt, OPTIMIZED_CONFIG.num_ctx)
        self.assertFalse(budget["fits_prompt_and_answer"])

    def test_empty_context_does_not_call_the_model(self) -> None:
        context = FrozenContext("телефон?", "телефон?", [], 0.01, 0.2, 9.5)
        called = {"count": 0}

        def boom(prompt: str, config):  # noqa: ANN001
            called["count"] += 1
            raise AssertionError("model should not be called")

        import day29.answer as answer_module

        original = answer_module.generate
        answer_module.generate = boom
        try:
            result = answer_with(context, OPTIMIZED_CONFIG)
        finally:
            answer_module.generate = original
        self.assertEqual(called["count"], 0)
        self.assertEqual(result.response.answer, UNKNOWN_ANSWER)
        self.assertFalse(result.called_model)
        self.assertEqual(result.generation_s, 0.0)

    def test_tokens_per_sec_uses_eval_count_and_duration(self) -> None:
        self.assertEqual(_tokens_per_sec(10, 2_000_000_000), 5.0)
        self.assertIsNone(_tokens_per_sec(10, 0))
        self.assertIsNone(_tokens_per_sec(None, 1))


if __name__ == "__main__":
    unittest.main()
