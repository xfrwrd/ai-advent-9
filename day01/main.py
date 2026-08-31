"""Day 01: Minimal LLM API client via DeepSeek REST API."""

import os

import requests
from dotenv import load_dotenv

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"


def main() -> None:
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Объясни одним предложением, что такое LLM.",
                }
            ],
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    print(answer)


if __name__ == "__main__":
    main()
