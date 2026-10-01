# Day 24 — цитаты и «не знаю»

Поиск, rewrite и порог берутся из Day 23. Ответ — отдельные поля: текст, источники и дословные цитаты. Если после фильтра не осталось чанков, модель для ответа не вызывается.

```bash
cd ~/Projects/ai-advent-9
.venv/bin/python -m unittest discover -s day24 -p 'test_*.py' -v
.venv/bin/python -m day24.run_eval
.venv/bin/streamlit run day24/app.py
```

Отчёт пишется в `day24/data/evaluation.json`.
