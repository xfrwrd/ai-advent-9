# Day 25 — мини-чат

История и TaskState лежат в `day25/data/conversations`. Поиск и ответ — Day 23 и Day 24. В запрос к модели попадают только последние 6 сообщений плюс память задачи.

```bash
cd ~/Projects/ai-advent-9
.venv/bin/python -m unittest discover -s day25 -p 'test_*.py' -v
.venv/bin/python -m day25.run_scenarios
.venv/bin/streamlit run day25/app.py
```
