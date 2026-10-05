# Day 26 — запуск локальной LLM

Цель — вызвать локальную модель из Python по HTTP, без облачного API.

Используется Ollama и модель `qwen3:8b`. Inference выполняется на этой машине. Ollama поднимает HTTP API на `http://localhost:11434`, клиент ходит в `POST /api/generate`.

Запуск Ollama:

```bash
ollama serve
```

Запуск клиента. Программа отправляет 3 запроса разной сложности: одно предложение про goroutine, сравнение mutex и channel, параллельный healthcheck на Go.

```bash
cd ~/Projects/ai-advent-9
.venv/bin/python -m day26.client
```
