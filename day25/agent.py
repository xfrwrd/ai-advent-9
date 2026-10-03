"""One chat turn: remember the task, search with that context, answer from Day 24."""

from __future__ import annotations

from dataclasses import dataclass

from day22.llm import chat
from day24.agent import Day24Agent, load_day24
from day24.response import Quote, Source
from day25.config import RECENT_MESSAGES_LIMIT
from day25.models import ChatMessage, Conversation, TaskState, now_iso
from day25.query import context_aware_query
from day25.state import update_task_state, working_messages
from day25.store import ConversationStore


@dataclass
class TurnRecord:
    turn_number: int
    user_message: str
    task_state_before: TaskState
    task_state_after: TaskState
    search_query: str
    answer: str
    sources: list[Source]
    quotes: list[Quote]
    sources_present: bool
    insufficient_context: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "turn_number": self.turn_number,
            "user_message": self.user_message,
            "task_state_before": self.task_state_before.as_dict(),
            "task_state_after": self.task_state_after.as_dict(),
            "search_query": self.search_query,
            "answer": self.answer,
            "sources": [item.as_dict() for item in self.sources],
            "quotes": [item.as_dict() for item in self.quotes],
            "sources_present": self.sources_present,
            "insufficient_context": self.insufficient_context,
        }


class ChatAgent:
    def __init__(
        self,
        rag: Day24Agent,
        store: ConversationStore,
        recent_limit: int = RECENT_MESSAGES_LIMIT,
    ) -> None:
        self.rag = rag
        self.store = store
        self.recent_limit = recent_limit

    def send(self, conversation: Conversation, message: str) -> TurnRecord:
        before = TaskState.from_dict(conversation.task_state.as_dict())
        recent = working_messages(conversation.messages, self.recent_limit)
        conversation.task_state = update_task_state(
            conversation.task_state,
            message,
            recent,
            self.rag.pipeline.chat_fn,
        )
        search_query = context_aware_query(
            message,
            conversation.task_state,
            recent,
            self.rag.pipeline.chat_fn,
        )
        answer = self.rag.ask(message, search_query=search_query)
        sources_present = bool(answer.sources) and not answer.insufficient_context
        conversation.messages.append(
            ChatMessage(role="user", content=message, timestamp=now_iso())
        )
        conversation.messages.append(
            ChatMessage(
                role="assistant",
                content=answer.answer,
                timestamp=now_iso(),
                sources=[item.as_dict() for item in answer.sources],
                quotes=[item.as_dict() for item in answer.quotes],
                search_query=search_query,
                insufficient_context=answer.insufficient_context,
            )
        )
        self.store.save(conversation)
        return TurnRecord(
            turn_number=len(conversation.messages) // 2,
            user_message=message,
            task_state_before=before,
            task_state_after=TaskState.from_dict(conversation.task_state.as_dict()),
            search_query=search_query,
            answer=answer.answer,
            sources=list(answer.sources),
            quotes=list(answer.quotes),
            sources_present=sources_present,
            insufficient_context=answer.insufficient_context,
        )


def load_chat(store: ConversationStore | None = None, chat_fn=chat) -> ChatAgent:
    return ChatAgent(load_day24(chat_fn=chat_fn), store or ConversationStore())
