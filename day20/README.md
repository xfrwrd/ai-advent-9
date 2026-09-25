# Day 20 — Orchestration MCP

Два отдельных MCP-сервера. Agent выбирает tool по тексту запроса и вызывает его через registry.

```text
task_server     get_task              → Task API
history_server  get_task_summary      → Day 18 SQLite
                get_recent_snapshots  → Day 18 SQLite
```

## Run

```bash
cd ~/Projects/ai-advent-9
.venv/bin/streamlit run day20/app.py
.venv/bin/python day20/run_demo.py
.venv/bin/python -m unittest discover -s day20 -p 'test_*.py' -v
```
