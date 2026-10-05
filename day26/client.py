"""Python → HTTP → localhost:11434 → Ollama → qwen3:8b."""

from __future__ import annotations

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"


class LocalLLMClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def ask(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        try:
            response = requests.post(
                url,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=(10, 180),
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Нет соединения с Ollama по адресу {self.base_url}. "
                "Запустите сервер командой: ollama serve"
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Ollama не ответила вовремя ({self.base_url}, модель {self.model})."
            ) from exc
        response.raise_for_status()
        payload = response.json()
        text = payload.get("response")
        if not isinstance(text, str):
            raise RuntimeError("Ответ Ollama не содержит текстовое поле response.")
        return text


PROMPTS = (
    "Объясни одним предложением, что такое goroutine в Go.",
    "Объясни разницу между mutex и channel в Go. Приведи по одному примеру ситуации, когда лучше использовать каждый из них.",
    "Напиши на Go функцию, которая параллельно проверяет healthcheck 5 HTTP-сервисов. "
    "Для каждого запроса используй context.Context с timeout 2 секунды. "
    "Функция должна дождаться завершения всех проверок и вернуть результаты по каждому сервису. "
    "Объясни архитектурные решения и убедись, что goroutines не утекают.",
)


if __name__ == "__main__":
    client = LocalLLMClient()
    for number, prompt in enumerate(PROMPTS, start=1):
        print(f"=== Тест {number} ===")
        print(prompt)
        print()
        print(client.ask(prompt))
        print()
