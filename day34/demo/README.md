# Demo fetcher

Модуль `fetcher.py` скачивает страницу по URL.

## Настройки

Таймаут запроса по умолчанию 60 секунд.

При ошибке сети запрос повторяется до 5 раз.

## Использование

```python
from day34.demo.fetcher import fetch

fetch("https://example.com")
```
