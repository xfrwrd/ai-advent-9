# Day 21 — индексация документов

Читает `.md`, `.txt` и `.py` проекта, режет двумя стратегиями, считает локальные embeddings и пишет SQLite-индекс.

Размер fixed-size задан в символах: 2000 символов ≈ 500 токенов, overlap 200 символов ≈ 50 токенов. Слова не режутся.

Structure-aware: Markdown по заголовкам, Python по `class` / `def` / методам, текст по абзацам.

Embedding model: `local-token-hash-v1`, размерность 256. Платный API не используется.

```bash
cd ~/Projects/ai-advent-9
python -m day21.index_documents
.venv/bin/python -m unittest discover -s day21 -p 'test_*.py' -v
```

Индексы: `day21/index/fixed.sqlite` и `day21/index/structure.sqlite`. Повторный запуск читает их с диска без пересчёта, функция `load_index`.
