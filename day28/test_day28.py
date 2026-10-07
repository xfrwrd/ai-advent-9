"""Day 28 timings and provider isolation. The LLM and the search are fakes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day21.chunking import Chunk
from day21.index_store import LoadedIndex
from day22.retrieve import Hit
from day24.ground import UNKNOWN_ANSWER
from day28.agent import CLOUD, LOCAL, Day28RAG


def _chunk() -> Chunk:
    text = "CHUNK_SIZE_CHARS = 2000"
    return Chunk(
        chunk_id="id-config",
        text=text,
        source="day21/config.py",
        title="config",
        section="module",
        chunking_strategy="structure",
        start=0,
        end=len(text),
        file_type="py",
    )


class ProviderTests(unittest.TestCase):
    def test_local_does_not_call_the_cloud_model(self) -> None:
        cloud_calls: list[str] = []
        local_calls: list[str] = []
        rag = _rag(12.0, local_calls, cloud_calls)
        result = rag.ask("Чему равен CHUNK_SIZE_CHARS?", LOCAL)
        self.assertEqual(cloud_calls, [])
        self.assertTrue(local_calls)
        self.assertIn("2000", result.response.answer)
        self.assertEqual(result.response.sources[0].source, "day21/config.py")
        self.assertGreaterEqual(result.total_s, result.retrieval_s)
        self.assertFalse(result.response.insufficient_context)

    def test_low_score_skips_the_answer_call_and_says_unknown(self) -> None:
        local_calls: list[str] = []
        rag = _rag(1.0, local_calls, [])
        result = rag.ask("Какой номер телефона дежурного?", LOCAL)
        self.assertTrue(result.response.insufficient_context)
        self.assertEqual(result.response.answer, UNKNOWN_ANSWER)
        self.assertEqual(result.response.sources, [])
        self.assertEqual(len(local_calls), 1)
        self.assertIn("Перепиши", local_calls[0])

    def test_both_providers_search_the_same_rewritten_query(self) -> None:
        searches: list[str] = []
        rag = _rag(12.0, [], [], searches)
        rag.ask("Чему равен CHUNK_SIZE_CHARS?", LOCAL)
        rag.ask("Чему равен CHUNK_SIZE_CHARS?", CLOUD)
        self.assertEqual(searches, ["day21 CHUNK_SIZE_CHARS", "day21 CHUNK_SIZE_CHARS"])


def _rag(
    score: float,
    local_calls: list[str],
    cloud_calls: list[str],
    searches: list[str] | None = None,
) -> Day28RAG:
    chunk = _chunk()
    index = LoadedIndex("fake", 1, "structure", [chunk], [[1.0]])

    def chat(bucket: list[str], answer: str):
        def inner(messages: list[dict[str, str]]) -> str:
            system = messages[0]["content"]
            bucket.append(system)
            if "Перепиши" in system:
                return "day21 CHUNK_SIZE_CHARS"
            return answer

        return inner

    def search_fn(index, embedder, question: str, k: int) -> list[Hit]:
        if searches is not None:
            searches.append(question)
        return [Hit(chunk, score)]

    return Day28RAG(
        index,
        chat(local_calls, '{"answer": "2000 символов.", "quotes": [{"chunk_id": "id-config", "text": "CHUNK_SIZE_CHARS = 2000"}]}'),
        chat(cloud_calls, '{"answer": "облако: 2000", "quotes": [{"chunk_id": "id-config", "text": "CHUNK_SIZE_CHARS = 2000"}]}'),
        search_fn=search_fn,
    )


if __name__ == "__main__":
    unittest.main()
