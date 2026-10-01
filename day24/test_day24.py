"""Day 24 sources, quotes, and the empty-context guard. The LLM is fake."""

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
from day23.agent import Day23Agent
from day24.agent import Day24Agent
from day24.evaluate import evaluate
from day24.ground import GROUNDED_SYSTEM, UNKNOWN_ANSWER, quotes_are_valid, validate_quotes
from day24.response import Quote, RetrievedChunk


def _chunk(source: str, text: str, section: str = "body") -> Chunk:
    return Chunk(
        chunk_id=f"id-{source}-{section}",
        text=text,
        source=source,
        title=source,
        section=section,
        chunking_strategy="structure",
        start=0,
        end=len(text),
        file_type="py",
    )


def _hit(chunk: Chunk, score: float) -> Hit:
    return Hit(chunk, score)


class QuoteTests(unittest.TestCase):
    def test_invalid_quote_is_dropped_and_a_verbatim_quote_stays(self) -> None:
        chunk = _chunk("day18/scheduler.py", "interval default is 10 seconds")
        quotes = validate_quotes(
            [
                {"chunk_id": chunk.chunk_id, "text": "invented sentence"},
                {"chunk_id": chunk.chunk_id, "text": "default is 10"},
                {"chunk_id": "missing", "text": "default is 10"},
            ],
            [_hit(chunk, 12.0)],
        )
        self.assertEqual([quote.text for quote in quotes], ["default is 10"])
        self.assertEqual(quotes[0].chunk_id, chunk.chunk_id)
        self.assertIn(quotes[0].text, chunk.text)
        self.assertTrue(
            quotes_are_valid(
                quotes,
                [RetrievedChunk(1, chunk.chunk_id, chunk.source, chunk.section, 12.0, chunk.text)],
            )
        )


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kept = _chunk("day18/scheduler.py", "DAY18_INTERVAL_SECONDS default is 10")
        self.weak = _chunk("day99/other.py", "unrelated note", section="tail")
        self.searches: list[str] = []
        self.prompts: list[str] = []
        blob = _chunk("unused.py", "")
        index = LoadedIndex("fake", 1, "structure", [blob], [[0.0]])

        def search_fn(index, embedder, question: str, k: int) -> list[Hit]:
            self.searches.append(question)
            return [_hit(self.kept, 16.0), _hit(self.weak, 2.0)][:k]

        def chat(messages: list[dict[str, str]]) -> str:
            self.prompts.append(messages[0]["content"])
            if "Перепиши" in messages[0]["content"]:
                return "rewritten scheduler interval"
            if "Оцени" in messages[0]["content"]:
                return '{"supported": true, "reason": "цитата содержит 10"}'
            user = messages[1]["content"]
            self.prompts.append(user)
            return (
                '{"answer": "Значение 10.", "quotes": ['
                '{"chunk_id": "' + self.kept.chunk_id + '", "text": "invented"},'
                '{"chunk_id": "' + self.kept.chunk_id + '", "text": "default is 10"},'
                '{"chunk_id": "' + self.weak.chunk_id + '", "text": "unrelated note"}'
                "]}"
            )

        pipeline = Day23Agent(
            index,
            chat_fn=chat,
            search_fn=search_fn,
            embedder=_Embedder(),
            retrieval_top_k=10,
            filtered_top_k=5,
            similarity_threshold=9.5,
        )
        self.agent = Day24Agent(pipeline)

    def test_sources_and_quotes_come_only_from_kept_chunks(self) -> None:
        result = self.agent.ask("Какое значение по умолчанию?")
        self.assertEqual(self.searches, ["rewritten scheduler interval"])
        self.assertEqual(result.original_query, "Какое значение по умолчанию?")
        self.assertIn("ORIGINAL QUESTION:\nКакое значение по умолчанию?", self.prompts[-1])
        self.assertNotIn("rewritten scheduler interval", self.prompts[-1])
        self.assertEqual([(item.source, item.section, item.chunk_id) for item in result.sources], [
            (self.kept.source, self.kept.section, self.kept.chunk_id)
        ])
        self.assertEqual([quote.text for quote in result.quotes], ["default is 10"])
        self.assertTrue(result.quotes[0].text in self.kept.text)
        self.assertTrue(result.sources)
        self.assertTrue(result.quotes)
        self.assertTrue(result.is_grounded)
        self.assertFalse(result.insufficient_context)

    def test_empty_filter_does_not_call_grounded_generation(self) -> None:
        def search_fn(index, embedder, question: str, k: int) -> list[Hit]:
            return [_hit(self.weak, 1.0)]

        self.agent.pipeline.search_fn = search_fn
        result = self.agent.ask("Какой номер телефона дежурного?")
        self.assertTrue(result.insufficient_context)
        self.assertFalse(result.is_grounded)
        self.assertEqual(result.sources, [])
        self.assertEqual(result.quotes, [])
        self.assertEqual(result.answer, UNKNOWN_ANSWER)
        self.assertNotIn(GROUNDED_SYSTEM, self.prompts)
        self.assertEqual(result.original_query, "Какой номер телефона дежурного?")


class EvaluationTests(unittest.TestCase):
    def test_evaluation_uses_the_ten_day22_questions(self) -> None:
        chunk = _chunk("day20/task_server.py", "def get_task")
        index = LoadedIndex("fake", 1, "structure", [chunk], [[1.0]])

        def chat(messages: list[dict[str, str]]) -> str:
            if "Перепиши" in messages[0]["content"]:
                return "same query"
            if "Оцени" in messages[0]["content"]:
                return '{"supported": true, "reason": "ok"}'
            return '{"answer": "get_task", "quotes": [{"chunk_id": "' + chunk.chunk_id + '", "text": "get_task"}]}'

        def search_fn(index, embedder, question: str, k: int) -> list[Hit]:
            return [_hit(chunk, 12.0)]

        agent = Day24Agent(
            Day23Agent(index, chat_fn=chat, search_fn=search_fn, embedder=_Embedder(), similarity_threshold=9.5)
        )
        report = evaluate(agent, negatives=())
        self.assertEqual(
            [row["question"] for row in report["questions"]],
            [item.question for item in QUESTIONS],
        )
        self.assertEqual(report["summary"]["questions"], 10)
        self.assertEqual(report["summary"]["answers_with_sources"], 10)
        self.assertEqual(report["summary"]["valid_quotes"], 10)


class _Embedder:
    name = "fake"
    dimension = 1

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


if __name__ == "__main__":
    unittest.main()
