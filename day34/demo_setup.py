"""Recreate the demo module for scenario 2: code and a README that disagree on two facts.

    .venv/bin/python -m day34.demo_setup
"""

from __future__ import annotations

from day34.policy import PROJECT_ROOT

DEMO = PROJECT_ROOT / "day34" / "demo"

CODE = '''"""Demo HTTP fetcher used by the Day 34 documentation scenario."""

import time

import requests

DEFAULT_TIMEOUT = 180
MAX_RETRIES = 3


def fetch(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.get(url, timeout=timeout).text
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(attempt)
    return ""
'''

README = """# Demo fetcher

Модуль `fetcher.py` скачивает страницу по URL.

## Настройки

Таймаут запроса по умолчанию 60 секунд.

При ошибке сети запрос повторяется до 5 раз.

## Использование

```python
from day34.demo.fetcher import fetch

fetch("https://example.com")
```
"""


def main() -> int:
    DEMO.mkdir(parents=True, exist_ok=True)
    (DEMO / "__init__.py").write_text("", encoding="utf-8")
    (DEMO / "fetcher.py").write_text(CODE, encoding="utf-8")
    (DEMO / "README.md").write_text(README, encoding="utf-8")
    print(f"Демо-модуль готов: {DEMO.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
