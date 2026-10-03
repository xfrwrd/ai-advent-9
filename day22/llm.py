"""DeepSeek chat call. Tests pass their own function and never hit this."""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-pro"


def chat(messages: list[dict[str, str]]) -> str:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "messages": messages},
        timeout=(10, 45),
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"])
