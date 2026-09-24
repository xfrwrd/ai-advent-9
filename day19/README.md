# Day 19 — MCP pipeline

Один запуск выполняет цепочку через MCP Client:

```text
get_tasks  →  Task API GET /health, затем GET /tasks/{id}
    ↓
summarize_tasks(tasks)
    ↓
save_summary(summary)  →  day19/data/day19_summary.json
```

`summarize_tasks` считает только переданный список и не ходит в API повторно.

## Run

```bash
cd ~/Projects/ai-advent-9
.venv/bin/python day19/run_demo.py
.venv/bin/python -m unittest discover -s day19 -p 'test_*.py' -v
```

```bash
.venv/bin/streamlit run day19/app.py
```
