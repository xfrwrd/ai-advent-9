# Day 16 — MCP: connect + list tools

Минимальное задание: подключиться к MCP-server и вывести список tools.
Без LLM, без tool calling, без интеграции с Agent/Memory.

## Выбор

Локальный MCP-server (`server.py`) + Python MCP client (`client.py`) через **stdio**.

Почему: официальный SDK, без токенов/сети, стабильно для демо и тестов.
Список tools приходит от сервера через `list_tools()`, не хардкодится в клиенте.

## Run

```bash
cd ~/Projects/ai-advent-9
.venv/bin/pip install -r requirements.txt
.venv/bin/python day16/client.py
.venv/bin/python -m unittest discover -s day16 -p 'test_*.py' -v
```

## Files

| File | Role |
|------|------|
| `server.py` | минимальный MCP server (add / echo / greet) |
| `client.py` | connect → initialize → list_tools → print |
| `test_day16.py` | проверка непустого списка tools |
