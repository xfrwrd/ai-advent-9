"""Day 25 chat memory. The LLM and the search are fakes."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day21.chunking import Chunk
from day21.index_store import LoadedIndex
from day22.retrieve import Hit
from day23.agent import Day23Agent
from day24.agent import Day24Agent
from day25.agent import ChatAgent
from day25.config import RECENT_MESSAGES_LIMIT
from day25.models import ChatMessage, TaskState
from day25.state import merge_task_state, working_messages
from day25.store import ConversationStore


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


class MemoryTests(unittest.TestCase):
    def test_messages_keep_order_and_a_new_chat_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(Path(tmp))
            first = store.new()
            second = store.new()
            self.assertEqual(first.messages, [])
            self.assertEqual(first.task_state.goal, "")
            agent = _agent(store, score=12.0)
            agent.send(first, "первый вопрос про day19")
            agent.send(first, "второй вопрос")
            reloaded = store.load(first.conversation_id)
            other = store.load(second.conversation_id)
            self.assertEqual([item.role for item in reloaded.messages], ["user", "assistant", "user", "assistant"])
            self.assertEqual([item.content for item in reloaded.messages if item.role == "user"], ["первый вопрос про day19", "второй вопрос"])
            self.assertEqual(other.messages, [])
            self.assertNotEqual(first.conversation_id, second.conversation_id)

    def test_task_state_keeps_goal_until_the_user_changes_it(self) -> None:
        old = TaskState(goal="day19 pipeline", constraints=["только day19"], terms=["get_tasks"], clarified_facts=["это day19"])
        kept = merge_task_state(
            old,
            TaskState(goal="другая тема", constraints=["ещё одно"], terms=["save_summary"], clarified_facts=["файл json"]),
            "А куда сохраняется сводка?",
        )
        self.assertEqual(kept.goal, "day19 pipeline")
        self.assertEqual(kept.constraints, ["только day19", "ещё одно"])
        self.assertEqual(kept.terms, ["get_tasks", "save_summary"])
        self.assertEqual(kept.clarified_facts, ["это day19", "файл json"])
        changed = merge_task_state(old, TaskState(goal="transport day16"), "Меняю цель: теперь day16")
        self.assertEqual(changed.goal, "transport day16")

    def test_working_context_is_shorter_than_the_full_history(self) -> None:
        messages = [ChatMessage("user", f"message-{index}", "t") for index in range(10)]
        recent = working_messages(messages, RECENT_MESSAGES_LIMIT)
        self.assertEqual(len(recent), RECENT_MESSAGES_LIMIT)
        self.assertEqual(recent[0].content, "message-4")
        self.assertEqual(len(messages), 10)

    def test_follow_up_query_uses_the_task_and_retrieval_runs_every_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(Path(tmp))
            conversation = store.new()
            seen: list[str] = []
            searches: list[str] = []
            agent = _agent(store, score=12.0, seen=seen, searches=searches)
            agent.send(conversation, "Расскажи про day19 pipeline")
            agent.send(conversation, "А куда она сохраняется?")
            self.assertEqual(searches, ["day19 pipeline save path", "day19 pipeline save path"])
            self.assertIn("А куда она сохраняется?", seen[1])
            self.assertIn("day19 pipeline", seen[1])
            saved = store.load(conversation.conversation_id)
            self.assertTrue(saved.messages[-1].sources)
            self.assertEqual(saved.messages[-1].sources[0]["source"], "day19/pipeline.py")
            self.assertIn("chunk_id", saved.messages[-1].sources[0])

    def test_low_score_is_insufficient_context_without_invented_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(Path(tmp))
            conversation = store.new()
            agent = _agent(store, score=1.0)
            turn = agent.send(conversation, "Какой номер телефона дежурного?")
            self.assertTrue(turn.insufficient_context)
            self.assertEqual(turn.sources, [])
            self.assertIn("Не знаю", turn.answer)
            self.assertEqual(store.load(conversation.conversation_id).messages[-1].sources, [])


def _agent(store: ConversationStore, score: float, seen: list[str] | None = None, searches: list[str] | None = None) -> ChatAgent:
    chunk = _chunk("day19/pipeline.py", "get_tasks then summarize_tasks then save_summary")
    index = LoadedIndex("fake", 1, "structure", [chunk], [[1.0]])

    def chat(messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        if "Перепиши" in system and seen is not None:
            seen.append(messages[-1]["content"])
        if "TaskState" in system:
            return '{"goal": "day19 pipeline", "constraints": ["только day19"], "terms": ["pipeline"], "clarified_facts": ["day19"]}'
        if "Перепиши" in system:
            return "day19 pipeline save path"
        return '{"answer": "Сводка пишется после summarize_tasks.", "quotes": [{"chunk_id": "' + chunk.chunk_id + '", "text": "save_summary"}]}'

    def search_fn(index, embedder, question: str, k: int) -> list[Hit]:
        if searches is not None:
            searches.append(question)
        return [Hit(chunk, score)]

    rag = Day24Agent(
        Day23Agent(index, chat_fn=chat, search_fn=search_fn, embedder=_Embedder(), similarity_threshold=9.5)
    )
    return ChatAgent(rag, store, recent_limit=RECENT_MESSAGES_LIMIT)


class _Embedder:
    name = "fake"
    dimension = 1

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


if __name__ == "__main__":
    unittest.main()
