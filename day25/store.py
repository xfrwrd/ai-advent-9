"""One JSON file per conversation."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from day25.models import Conversation, TaskState, now_iso

CONVERSATIONS_DIR = Path(__file__).resolve().parent / "data" / "conversations"


class ConversationStore:
    def __init__(self, directory: Path = CONVERSATIONS_DIR) -> None:
        self.directory = directory

    def new(self) -> Conversation:
        stamp = now_iso()
        conversation = Conversation(
            conversation_id=uuid.uuid4().hex[:12],
            messages=[],
            task_state=TaskState(),
            created_at=stamp,
            updated_at=stamp,
        )
        self.save(conversation)
        return conversation

    def save(self, conversation: Conversation) -> None:
        conversation.updated_at = now_iso()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(conversation.conversation_id)
        path.write_text(json.dumps(conversation.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, conversation_id: str) -> Conversation:
        return Conversation.from_dict(json.loads(self._path(conversation_id).read_text(encoding="utf-8")))

    def list_ids(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(path.stem for path in self.directory.glob("*.json"))

    def _path(self, conversation_id: str) -> Path:
        return self.directory / f"{conversation_id}.json"
