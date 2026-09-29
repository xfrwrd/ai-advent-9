# Day 22 — первый RAG-запрос

Вопрос ищется в индексе `day21/index/structure.sqlite`: BM25 по тексту и пути чанка, косинус эмбеддинга `local-token-hash-v1` решает ничью. Тот же вопрос уходит в DeepSeek дважды: без фрагментов и вместе с найденными чанками.

```bash
cd ~/Projects/ai-advent-9
.venv/bin/python -m unittest discover -s day22 -p 'test_*.py' -v
.venv/bin/python -m day22.run_eval
.venv/bin/streamlit run day22/app.py
```

Нужен `DEEPSEEK_API_KEY` в `.env`. Тесты LLM не вызывают.
