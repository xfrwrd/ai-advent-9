# Day 14 — Инварианты и ограничения состояния

## Roles

```text
UserProfile  → HOW TO RESPOND
TaskState    → WHAT ARE WE DOING
Invariants   → WHAT MUST NOT BE VIOLATED
```

| | TaskState | Invariants |
|--|-----------|------------|
| Вопрос | Где мы? Что дальше? | Какие границы нельзя нарушать? |
| Примеры | stage=execution, current_step | PostgreSQL, Python, REST, payment→COMPLETED |
| Хранение | Working `task_state` | Working `task_invariants` |

Инварианты **не** смешаны с `stage` / `current_step` / `expected_action`.

## Почему не только в диалоге

Short-Term можно очистить — ограничения должны остаться.  
Поэтому `InvariantStore` живёт в **Working Memory**, рядом с TaskState.

## Как попадают в LLM

`Agent.build_system_content()` добавляет блок **ACTIVE INVARIANTS** в каждый system prompt
(вместе с CURRENT TASK и USER PROFILE).

## Как проверяется конфликт

Двухуровневая архитектура (без semantic rule engine):

```text
User Request
     ↓
check_invariant_conflict()   ← явные conflict_signals (код, до LLM)
     ↓
 conflict? ──YES──→ reject / explain / alternative  (без LLM)
     │                 ответ всегда: CONFLICT + rule + reason + alternative
     NO
     ↓
 LLM + ACTIVE INVARIANTS в system prompt
     ↓
 семантический конфликт (нет ключевых слов) → модель отказывает в рамках invariants
```

### Ограничение

Детерминированный gate ловит **явные** формулировки (`MongoDB`, `COMPLETED сразу`, …).  
Семантические перефразы вроде «документоориентированная БД» или «считать заказ
завершённым, оплату потом» **не** разбираются кодом: они уходят в LLM вместе с
блоком ACTIVE INVARIANTS. Embeddings / vector search / отдельный policy engine
для Day 14 не используются.

| Ветка | Пример | Кто решает |
|-------|--------|------------|
| Явная | «заменим PostgreSQL на MongoDB» | код до LLM |
| Семантическая | «перейдём на документоориентированную БД» | LLM + ACTIVE INVARIANTS |

Обычный чат **не** удаляет invariant. Только явный API: `add` / `deactivate` / `remove`.

## Demo

Задача: `Разработать backend для сервиса заказов.`

Invariants:
- architecture: Use REST API
- tech_stack: Backend language must be Python
- tech_stack: Database must be PostgreSQL
- business_rule: Order cannot become COMPLETED before payment is confirmed

### Results (live demo)

| Request | Result |
|---------|--------|
| `Добавь endpoint для получения заказа.` | **accepted** → `GET /api/v1/orders/{order_id}` с учётом REST/Python/Postgres |
| `Давай заменим PostgreSQL на MongoDB.` | **rejected** → `database` / «Database must be PostgreSQL» + alternative в Postgres |
| `COMPLETED сразу после создания…` | **rejected** → `payment_before_completed` |
| Clear Short-Term → снова MongoDB | **rejected** — invariant остался в Working Memory |

## Run

```bash
cd day14
../.venv/bin/python -m unittest test_day14.py -v
../.venv/bin/streamlit run app.py
../.venv/bin/python run_demo.py
```

## Files

| File | Role |
|------|------|
| `invariants.py` | `Invariant`, store, conflict check, demo presets |
| `memory.py` | 3 слоя + `INVARIANTS_KEY` |
| `task_state.py` / `profile.py` | как Day 13/12 |
| `agent.py` | gate + ACTIVE INVARIANTS в prompt |
| `app.py` | Streamlit |
| `test_day14.py` | unit-тесты (fake LLM) |
