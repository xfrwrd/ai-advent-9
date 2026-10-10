"""Ten lightweight calls against an already running Day 30 service. Does not start Ollama."""

from __future__ import annotations

import os
import time

import requests

BASE = os.environ.get("DAY30_URL", "http://127.0.0.1:8030")
KEY = os.environ.get("DAY30_API_KEY", "")


def main() -> None:
    if not KEY:
        raise SystemExit("Задайте DAY30_API_KEY. Скрипт не подставляет токен сам.")
    headers = {"Authorization": f"Bearer {KEY}"}
    started = time.perf_counter()
    ok = 0
    for index in range(1, 9):
        mark = time.perf_counter()
        if index % 2 == 0:
            response = requests.get(f"{BASE}/health", timeout=10)
        else:
            response = requests.post(
                f"{BASE}/chat",
                headers={**headers, "X-Session-Token": os.environ.get("DAY30_SESSION_TOKEN", "")},
                json={"session_id": "stability", "message": "Ответь одним словом: ок"},
                timeout=180,
            )
        elapsed = time.perf_counter() - mark
        print(f"{index} {response.status_code} {elapsed:.2f}s", flush=True)
        if response.ok:
            ok += 1
            token = response.json().get("session_token")
            if token:
                os.environ["DAY30_SESSION_TOKEN"] = token
                headers["X-Session-Token"] = token
    print(f"ok {ok}/8 total {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
