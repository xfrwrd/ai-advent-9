"""Day 07: Agent with dialog history persisted to JSON."""

import json
from pathlib import Path

import requests

API_URL = "https://api.deepseek.com/chat/completions"
THINKING_DISABLED = {"type": "disabled"}
HISTORY_PATH = Path(__file__).resolve().parent / "history.json"


class AgentError(Exception):
    """Raised when the agent cannot get a valid answer from the API."""


class Agent:
    """Agent that keeps conversation history on disk between restarts."""

    def __init__(self, api_key: str, model: str = "deepseek-v4-flash") -> None:
        self.api_key = api_key
        self.model = model
        self.history_path = HISTORY_PATH
        self.messages = self._load_history()

    def _load_history(self) -> list[dict]:
        if not self.history_path.exists():
            return []

        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

        if not isinstance(data, list):
            return []

        return data

    def _save_history(self) -> None:
        self.history_path.write_text(
            json.dumps(self.messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_history(self) -> None:
        self.messages = []
        self._save_history()

    def ask(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": self.messages,
            "thinking": THINKING_DISABLED,
        }

        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
        except requests.Timeout as exc:
            self.messages.pop()
            raise AgentError("Таймаут при обращении к DeepSeek API.") from exc
        except requests.RequestException as exc:
            self.messages.pop()
            raise AgentError(
                f"Не удалось связаться с DeepSeek API: {exc.__class__.__name__}."
            ) from exc

        if not response.ok:
            self.messages.pop()
            raise AgentError(
                f"DeepSeek API вернул ошибку HTTP {response.status_code}."
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self.messages.pop()
            raise AgentError("Не удалось разобрать ответ DeepSeek API.") from exc

        answer = content.strip()
        if not answer:
            self.messages.pop()
            raise AgentError("DeepSeek API вернул пустой ответ.")

        self.messages.append({"role": "assistant", "content": answer})
        self._save_history()
        return answer
