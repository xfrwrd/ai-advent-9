# Day 27 — интеграция локальной LLM в приложение

Day 26 один раз прогоняет три готовых запроса. Day 27 — веб-чат: вы пишете свой вопрос, страница отправляет его в локальную модель и показывает ответ.

Запросы идут через `LocalLLMClient` из Day 26 в Ollama на `http://localhost:11434`, модель `qwen3:8b`. Облачные модели не вызываются.

Запуск Ollama:

```bash
ollama serve
```

Запуск страницы:

```bash
cd ~/Projects/ai-advent-9
.venv/bin/streamlit run day27/app.py
```

Тот же чат в терминале:

```bash
.venv/bin/python -m day27.cli
```
