# Day 28 — локальная LLM + RAG

Один и тот же RAG Days 21–24 отвечает либо локальной моделью, либо уже подключённым DeepSeek. Новый индекс и новый поиск не создаются.

```text
Documents
   ↓
Chunks                         day21/chunking.py, CHUNK_SIZE_CHARS = 2000
   ↓
Local Embeddings               local-token-hash-v1, размерность 256
   ↓
Local Vector Index             day21/index/structure.sqlite
   ↓
User Query
   ↓
Retrieval                      Day 22 BM25 + cosine, Day 23 threshold 9.5, top 10 → 5
   ↓
TOP-K Context
   ↓
Local Qwen3:8b / Cloud LLM     Ollama или DeepSeek, один и тот же контекст
   ↓
Answer + Sources
```

Эмбеддинги считает `LocalHashEmbedder` на этой машине, без API. Поиск читает готовый файл `day21/index/structure.sqlite`. При выборе Local rewrite и итоговый ответ идут в Ollama `qwen3:8b` на `http://localhost:11434`. В облако этот запрос не отправляется. При выборе Cloud та же выборка уходит в DeepSeek через `day22/llm.py`.

Если после порога 9.5 не осталось фрагментов, ответ фиксированный: «Не знаю: в базе знаний недостаточно релевантной информации». Цитаты остаются только если их текст есть в найденном chunk.

Запуск Ollama, если сервер ещё не слушает порт:

```bash
ollama serve
```

Запуск страницы:

```bash
cd ~/Projects/ai-advent-9
.venv/bin/streamlit run day28/app.py
```

Переключатель **Local / Cloud** стоит над вопросом. Один и тот же текст можно отправить дважды, сменив модель. Страница показывает ответ, sources, quotes и три времени: retrieval, generation, total. Рядом ручная оценка: качество 1–5 и отметка, держится ли ответ контекста. Автоматической оценки другой моделью нет.

Три вопроса для сравнения:

1. Факт из одного файла: `Чему равен CHUNK_SIZE_CHARS в day21 config?` Ожидание: 2000, источник `day21/config.py`.
2. Несколько chunks: `В каком порядке day19 pipeline вызывает get_tasks, summarize_tasks и save_summary, и в какой файл save_summary пишет сводку?` Ожидание: `get_tasks → summarize_tasks → save_summary` и `day19/data/day19_summary.json`.
3. В документах этого нет: `Какой номер телефона дежурного администратора указан в oncall-ротации?` Ожидание: «Не знаю», sources пустые.

Времена и оценки 1–5 снимаются на странице во время этих трёх пар. Local на `qwen3:8b` отвечает дольше Cloud: генерация идёт на машине, retrieval у обоих общий и локальный.
