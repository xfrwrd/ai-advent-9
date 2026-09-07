"""Day 06: Minimal Agent that talks to DeepSeek API."""

import requests

API_URL = "https://api.deepseek.com/chat/completions"
THINKING_DISABLED = {"type": "disabled"}


class AgentError(Exception):
    """Raised when the agent cannot get a valid answer from the API."""


class Agent:
    """Separate agent entity: encapsulates DeepSeek request/response logic."""

    def __init__(self, api_key: str, model: str = "deepseek-v4-flash") -> None:
        self.api_key = api_key
        self.model = model

    def ask(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
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
            raise AgentError("Таймаут при обращении к DeepSeek API.") from exc
        except requests.RequestException as exc:
            raise AgentError(
                f"Не удалось связаться с DeepSeek API: {exc.__class__.__name__}."
            ) from exc

        if not response.ok:
            raise AgentError(
                f"DeepSeek API вернул ошибку HTTP {response.status_code}."
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AgentError("Не удалось разобрать ответ DeepSeek API.") from exc

        answer = content.strip()
        if not answer:
            raise AgentError("DeepSeek API вернул пустой ответ.")

        return answer
