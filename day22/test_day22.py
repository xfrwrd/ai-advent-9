"""Day 22 retrieval and prompt assembly. The LLM is a fake function."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day21.chunking import Chunk
from day21.index_store import LoadedIndex
from day22.agent import RagAgent, _messages
from day22.questions import QUESTIONS
from day22.retrieve import search


class FakeEmbedder:
    name = "fake"
    dimension = 2

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if "get_task" in lowered else 0.0, 1.0 if "summary" in lowered else 0.0])
        return vectors


def _chunk(source: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=source,
        text=text,
        source=source,
        title=source,
        section="body",
        chunking_strategy="structure",
        start=0,
        end=len(text),
        file_type="py",
    )


class Day22Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = FakeEmbedder()
        chunks = [
            _chunk("day20/task_server.py", "def get_task(task_id): return task"),
            _chunk("day20/history_server.py", "def get_task_summary(task_id): return summary"),
        ]
        vectors = self.embedder.embed_batch([chunk.text for chunk in chunks])
        self.index = LoadedIndex("fake", 2, "structure", chunks, vectors)

    def test_search_ranks_the_matching_chunk_first(self) -> None:
        hits = search(self.index, self.embedder, "what does get_task do", k=2)
        self.assertEqual(hits[0].chunk.source, "day20/task_server.py")
        self.assertGreaterEqual(hits[0].score, hits[1].score)

    def test_rag_prompt_contains_chunk_and_plain_prompt_does_not(self) -> None:
        hits = search(self.index, self.embedder, "get_task", k=1)
        rag = _messages("get_task?", hits)
        plain = _messages("get_task?", None)
        self.assertIn("day20/task_server.py", rag[1]["content"])
        self.assertIn("get_task?", rag[1]["content"])
        self.assertNotIn("day20/task_server.py", plain[1]["content"])

    def test_agent_calls_llm_once_per_mode_and_keeps_hits_only_for_rag(self) -> None:
        seen: list[list[dict[str, str]]] = []

        def fake_chat(messages: list[dict[str, str]]) -> str:
            seen.append(messages)
            return "ok"

        agent = RagAgent(self.index, chat_fn=fake_chat, embedder=self.embedder, top_k=1)
        plain = agent.ask("get_task?", use_rag=False)
        rag = agent.ask("get_task?", use_rag=True)
        self.assertEqual(len(seen), 2)
        self.assertEqual(plain.hits, [])
        self.assertEqual(rag.sources, ["day20/task_server.py"])
        self.assertIn("def get_task", seen[1][1]["content"])

    def test_ten_control_questions_have_expectations_and_sources(self) -> None:
        self.assertEqual(len(QUESTIONS), 10)
        ids = [item.id for item in QUESTIONS]
        self.assertEqual(ids, list(range(1, 11)))
        for item in QUESTIONS:
            self.assertTrue(item.question.strip())
            self.assertTrue(item.expected.strip())
            self.assertTrue(item.sources)
            self.assertTrue(all(source.endswith((".py", ".md")) for source in item.sources))


if __name__ == "__main__":
    unittest.main()
