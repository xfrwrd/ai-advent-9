"""Day 26 client with JSON output and a wider context window for the agent loop."""

from __future__ import annotations

import requests

from day26.client import LocalLLMClient

NUM_CTX = 24576
READ_TIMEOUT = 300


class AgentLLM(LocalLLMClient):
    """Same endpoint and errors as Day 26. Adds format=json, num_ctx and low temperature."""

    temperature = 0.1

    def ask(self, prompt: str, *, think: bool | None = None) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"num_ctx": NUM_CTX, "temperature": self.temperature},
        }
        if think is not None:
            payload["think"] = think
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=(10, READ_TIMEOUT))
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Нет соединения с Ollama по адресу {self.base_url}. Запустите сервер командой: ollama serve"
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError(f"Ollama не ответила вовремя ({self.base_url}, модель {self.model}).") from exc
        if response.status_code >= 400:
            raise RuntimeError(f"Ollama вернула HTTP {response.status_code}: {response.text[:200]}")
        text = response.json().get("response")
        if not isinstance(text, str):
            raise RuntimeError("Ответ Ollama не содержит текстовое поле response.")
        return text
