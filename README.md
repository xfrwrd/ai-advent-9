# AI Advent Challenge 9

Учебный репозиторий. Каждый день лежит в своей папке `dayNN`. Общего HTTP API и файла OpenAPI в корне нет.

Архитектура дней 16–31: `docs/architecture.md`. Подробности конкретного дня: `dayNN/README.md`, если этот файл есть.

Локальная модель — Ollama, `qwen3:8b`, адрес `http://127.0.0.1:11434`. Клиент: `day26/client.py`.

Приватный сервис Day 30 слушает `127.0.0.1:8030` и сам ходит в Ollama. Запуск: `.venv/bin/python -m day30` после `export DAY30_API_KEY`.

Ассистент Day 31 отвечает на `/help` по документации и по текущей Git-ветке. Запуск страницы: `.venv/bin/streamlit run day31/ui.py`.
