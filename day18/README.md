# Day 18 — MCP Background Agent

Scheduler собирает snapshot задачи в SQLite. Streamlit только показывает состояние. Сводка считается в MCP tool `get_task_summary`, не в UI.

```text
scheduler (backend thread, один на процесс)
    ↓
SQLite snapshots
    ↓
Streamlit читает статус и последние строки

кнопка «Получить сводку через MCP»
    ↓
Agent → MCP Client → get_task_summary → SQLite aggregation → Agent → Streamlit
```

## Run

```bash
cd ~/Projects/ai-advent-9
DAY18_INTERVAL_SECONDS=10 .venv/bin/streamlit run day18/app.py
```

Интервал по умолчанию — 10 секунд (`DAY18_INTERVAL_SECONDS`). База — `day18/data/snapshots.db`.

Проверка persistence: остановить Streamlit (Ctrl+C) и запустить ту же команду снова. Старые строки остаются, scheduler дописывает новые.

```bash
.venv/bin/python -m unittest discover -s day18 -p 'test_*.py' -v
```
