# Day 12 — Персонализация ассистента

## Personalization

```text
Memory tells the agent WHAT to remember.
Profile tells the agent HOW to respond.
```

Day 11 дал три слоя памяти. Day 12 добавляет **UserProfile** поверх Long-Term Memory:
профиль автоматически попадает в каждый LLM request, без «ответь коротко / технически» в каждом сообщении.

```text
Long-Term Memory
        ↓
   User Profile  (ключ user_profile)
        ↓
 Personalization block в system prompt
        ↓
 Working + Short-Term
        ↓
       LLM
```

## Architecture (поверх Day 11)

Day 11 **не изменён**. Day 12 — отдельная папка `day12/` с той же моделью памяти:

| Слой | Назначение |
|------|------------|
| Short-Term | текущий диалог |
| Working | данные текущей задачи |
| Long-Term | устойчивые факты **и** `user_profile` |

Профиль хранится в Long-Term как JSON-строка под ключом `user_profile`, потому что это устойчивые предпочтения: переживают clear Short-Term, clear Working и перезапуск приложения.

## UserProfile preferences

| Поле | Примеры |
|------|---------|
| `name` | Ксения, Алекс |
| `language` | `ru`, `en` |
| `style` | concise, detailed, friendly, formal, technical, simple |
| `format` | plain_text, bullets, step_by_step, structured |
| `expertise_level` | beginner, intermediate, advanced, technical |
| `constraints` | no_basic_explanations, explain_terminology, no_emojis, … |

## Как профиль попадает в каждый LLM request

`Agent.build_system_content()` всегда собирает:

1. system instructions  
2. **USER PROFILE:** (из Long-Term)  
3. Long-term memory (остальные ключи, без дубля `user_profile`)  
4. Working memory  

Затем short-term messages + текущий user request.

Пользователю не нужно повторять предпочтения в prompt.

## Как изменить профиль

- Streamlit UI: форма **User Profile** → Save profile  
- Кнопки **Profile A / Profile B**  
- Код: `agent.save_profile(...)` / `agent.update_profile(style="detailed")`

## Demo experiment

Один и тот же prompt:

```text
Объясни, что такое circuit breaker в микросервисах.
```

### Profile A — Technical / Concise (Ксения)

style=concise, format=structured, expertise=advanced,  
constraints: no_basic_explanations, use_technical_terminology, …

**Ответ (сокращённо):** FSM Closed→Open→Half-Open, failureThreshold, Resilience4j/Envoy, без бытовых аналогий.

### Profile B — Beginner / Detailed (Алекс)

style=detailed, format=step_by_step, expertise=beginner,  
constraints: explain_terminology, use_simple_examples, …

**Ответ (сокращённо):** шаги 1–7, аналогия с пробкой в щитке, объяснение каскадного отказа простыми словами.

Полные ответы сохранены в `demo_results.json` (после локального прогона).

### Persistence

После clear Short-Term профиль **Алекс** остался; следующий вопрос про idempotency снова ответил подробно / step-by-step.  
После clear Working Memory профиль тоже остался.

## Run

```bash
cd day12
../.venv/bin/python -m unittest test_day12.py -v
../.venv/bin/streamlit run app.py
```

## Files

| File | Role |
|------|------|
| `memory.py` | 3 слоя памяти (как Day 11, injectable `base_dir`) |
| `profile.py` | `UserProfile` + presets A/B |
| `agent.py` | injection профиля в каждый request |
| `app.py` | Streamlit demo |
| `test_day12.py` | unit-тесты с fake LLM |
