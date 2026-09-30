"""Day 23 filter, rewrite routing, and evaluation. The LLM is a fake function."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day21.chunking import Chunk
from day21.index_store import LoadedIndex
from day22.questions import QUESTIONS
from day22.retrieve import Hit
from day23.agent import NO_CONTEXT_SYSTEM, Day23Agent
from day23.evaluate import evaluate
from day23.filter import FILTERED_OUT, KEPT, apply_filter


def _chunk(source: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"id-{source}",
        text=text,
        source=source,
        title=source,
        section="body",
        chunking_strategy="structure",
        start=0,
        end=len(text),
        file_type="py",
    )


def _hit(source: str, score: float, text: str = "text") -> Hit:
    return Hit(_chunk(source, text), score)


class FilterTests(unittest.TestCase):
    def test_threshold_drops_lower_score_and_keeps_higher_score(self) -> None:
        result = apply_filter([_hit("low.py", 0.2), _hit("high.py", 0.9)], threshold=0.5, filtered_top_k=5)
        self.assertEqual([chunk.source for chunk in result.kept], ["high.py"])
        self.assertEqual(result.dropped[0].source, "low.py")
        self.assertEqual(result.dropped[0].status, FILTERED_OUT)
        self.assertEqual(result.dropped[0].reason, "below threshold")

    def test_distance_threshold_keeps_the_smaller_score(self) -> None:
        result = apply_filter(
            [_hit("far.py", 0.9), _hit("near.py", 0.2)],
            threshold=0.5,
            filtered_top_k=5,
            higher_is_better=False,
        )
        self.assertEqual([chunk.source for chunk in result.kept], ["near.py"])
        self.assertEqual(result.dropped[0].reason, "above threshold")

    def test_filtered_top_k_limits_passing_chunks_and_preserves_score_order(self) -> None:
        result = apply_filter(
            [_hit("a.py", 0.6), _hit("b.py", 0.95), _hit("c.py", 0.8)],
            threshold=0.5,
            filtered_top_k=2,
        )
        self.assertEqual([chunk.source for chunk in result.kept], ["b.py", "c.py"])
        ordered = apply_filter([_hit("a.py", 0.7), _hit("b.py", 0.9)], threshold=0.5, filtered_top_k=2)
        self.assertEqual([chunk.source for chunk in ordered.kept], ["b.py", "a.py"])
        self.assertEqual([chunk.rank for chunk in result.kept], [2, 3])
        overflow = [chunk for chunk in result.dropped if chunk.reason == "beyond filtered_top_k"]
        self.assertEqual([chunk.source for chunk in overflow], ["a.py"])
        self.assertEqual(result.kept[0].status, KEPT)

    def test_equal_scores_keep_the_earlier_retrieval_rank(self) -> None:
        result = apply_filter([_hit("first.py", 0.8), _hit("second.py", 0.8)], threshold=0.5, filtered_top_k=2)
        self.assertEqual([chunk.source for chunk in result.kept], ["first.py", "second.py"])


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.messages: list[list[dict[str, str]]] = []
        self.hits = [_hit("kept.py", 0.9, "kept fact"), _hit("weak.py", 0.2, "weak fact")]
        chunk = _chunk("unused.py", "")
        self.index = LoadedIndex("fake", 1, "structure", [chunk], [[0.0]])

    def _search(self, index, embedder, question: str, k: int) -> list[Hit]:
        self.calls.append((question, k))
        return self.hits[:k]

    def _chat(self, messages: list[dict[str, str]]) -> str:
        self.messages.append(messages)
        if "Перепиши" in messages[0]["content"]:
            return "rewritten search query"
        return "answer"

    def _agent(self, chat=None, threshold: float = 0.5) -> Day23Agent:
        return Day23Agent(
            self.index,
            chat_fn=chat or self._chat,
            search_fn=self._search,
            embedder=_Embedder(),
            retrieval_top_k=10,
            filtered_top_k=5,
            similarity_threshold=threshold,
            baseline_top_k=5,
        )

    def test_rewrite_is_the_search_query_and_the_answer_uses_the_original_question(self) -> None:
        result = self._agent().run("original question", "rewrite_filtered")
        self.assertEqual(self.calls, [("rewritten search query", 10)])
        self.assertEqual(result.rewritten_query, "rewritten search query")
        answer_call = self.messages[-1]
        self.assertIn("original question", answer_call[1]["content"])
        self.assertNotIn("rewritten search query", answer_call[1]["content"])
        self.assertIn("kept fact", answer_call[1]["content"])

    def test_rewrite_error_falls_back_to_the_original_query(self) -> None:
        def chat(messages: list[dict[str, str]]) -> str:
            self.messages.append(messages)
            if "Перепиши" in messages[0]["content"]:
                raise RuntimeError("rewrite down")
            return "answer"

        result = self._agent(chat).run("original question", "rewrite_filtered")
        self.assertEqual(self.calls, [("original question", 10)])
        self.assertEqual(result.rewritten_query, "original question")

    def test_empty_filter_tells_the_model_there_is_no_relevant_context(self) -> None:
        result = self._agent(threshold=5).run("original question", "filtered")
        self.assertTrue(result.no_relevant_context)
        self.assertEqual(result.sources, [])
        self.assertIn(NO_CONTEXT_SYSTEM, self.messages[-1][0]["content"])
        self.assertEqual(len(result.retrieved_before_filter), 2)
        self.assertEqual(result.retrieved_after_filter, [])

    def test_sources_come_only_from_kept_chunks(self) -> None:
        result = self._agent().run("original question", "filtered")
        self.assertEqual(result.sources, ["kept.py"])
        self.assertNotIn("weak.py", result.sources)

    def test_baseline_does_not_apply_the_threshold(self) -> None:
        result = self._agent(threshold=5).run("original question", "baseline")
        self.assertEqual(self.calls, [("original question", 5)])
        self.assertEqual(result.filtered_out, [])
        self.assertEqual(result.sources, ["kept.py", "weak.py"])
        self.assertIsNone(result.threshold)


class EvaluationTests(unittest.TestCase):
    def test_evaluation_uses_the_ten_day22_questions(self) -> None:
        chunk = _chunk("day20/task_server.py", "get_task")
        index = LoadedIndex("fake", 1, "structure", [chunk], [[1.0]])

        def chat(messages: list[dict[str, str]]) -> str:
            if "Перепиши" in messages[0]["content"]:
                return "same query"
            return "answer"

        def search_fn(index, embedder, question: str, k: int) -> list[Hit]:
            return [Hit(chunk, 1.0)]

        agent = Day23Agent(index, chat_fn=chat, search_fn=search_fn, embedder=_Embedder(), similarity_threshold=0.5)
        report = evaluate(agent)
        self.assertEqual(
            [row["question"] for row in report["questions"]],
            [item.question for item in QUESTIONS],
        )
        self.assertEqual(len(QUESTIONS), 10)
        self.assertEqual(report["summary"]["baseline_expected_source_found"], 1)


class _Embedder:
    name = "fake"
    dimension = 1

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


if __name__ == "__main__":
    unittest.main()
