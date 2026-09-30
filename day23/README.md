# Day 23 — фильтр и rewrite поверх Day 22

Индекс Day 21 и поиск Day 22 не меняются. Day 23 забирает больше кандидатов, отсекает score ниже порога и может переписать запрос перед поиском. Ответ модели всё равно идёт на исходный вопрос.

Score Day 22 — это BM25 плюс косинус эмбеддинга. Большее число ближе к вопросу.

```bash
cd ~/Projects/ai-advent-9
.venv/bin/python -m unittest discover -s day23 -p 'test_*.py' -v
.venv/bin/python -m day23.run_eval
.venv/bin/streamlit run day23/app.py
```

Порог, размеры top-K и сравнение 10 вопросов Day 22 пишутся в `day23/data/evaluation.json`.
