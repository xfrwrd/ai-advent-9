# Day 13 — Состояние задачи (Task State Machine)

## Stages

```text
planning → execution → validation → done
```

Дополнительно: `validation → execution` (если проверка нашла проблему).

| Поле | Назначение |
|------|------------|
| `stage` | этап FSM (контролируемые transitions) |
| `current_step` | что делается сейчас |
| `expected_action` | что ожидается дальше |
| `status` | `active` / `paused` — **отдельно** от stage |

Pause не меняет `stage` / `current_step` / `expected_action`.

## Memory layers

```text
Short-Term  → conversation (диалог)
Working     → TaskState     (ключ task_state)
Long-Term   → UserProfile   (ключ user_profile) + устойчивые факты
```

`TaskState` — source of truth. LLM получает блок **CURRENT TASK** в system prompt,
но не выбирает stage сама: переходы только через `TaskState.transition_to(...)`.

## Allowed transitions

```text
planning   → execution
execution  → validation
validation → done | execution
done       → (нет)
```

Пока `status=paused`, transitions запрещены.

## Pause / Resume

```text
stage = execution, status = paused   # остановка
stage = execution, status = active   # продолжение с того же места
```

Сохраняются: goal, stage, current_step, expected_action.

## Demo acceptance

1. Create: `Подготовить технический план REST API`
2. `planning → execution`
3. `current_step = определить API endpoints`, `expected_action = описать request/response models`
4. Pause
5. Clear Short-Term Memory
6. Resume (или новый Agent с тем же `base_dir`)
7. User: `Продолжай`

Агент продолжает с сохранённого step/action **без** просьбы заново объяснить задачу
и без возврата к planning.

### Demo result (live LLM)

После pause → clear Short-Term → new session → resume → «Продолжай» агент сразу
описал request/response models для endpoints (Users/Auth), **не** спросил цель заново
и **не** вернулся к planning. Затем успешно: `execution → validation → done`.

## Run

```bash
cd day13
../.venv/bin/python -m unittest test_day13.py -v
../.venv/bin/streamlit run app.py
```

## Files

| File | Role |
|------|------|
| `task_state.py` | `TaskStage`, `TaskStatus`, `TaskState` FSM |
| `memory.py` | 3 слоя (+ `TASK_STATE_KEY`) |
| `profile.py` | UserProfile (Day 12) |
| `agent.py` | injection CURRENT TASK + TaskState API |
| `app.py` | Streamlit: pipeline, Pause/Resume, demo |
| `test_day13.py` | unit-тесты с fake LLM |
