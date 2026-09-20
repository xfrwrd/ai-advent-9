# Day 15 — Контролируемые переходы состояний

## Day 13 vs Day 15

```text
Day 13:
Task has a formal state.

Day 15:
Task can change state only through
controlled transitions with guards.
```

Переиспользована **та же** State Machine (`TaskStage`, `ALLOWED_TRANSITIONS`).  
Вторая FSM не создавалась.

## Lifecycle

```text
planning
   │ plan_approved = true
   ↓
execution
   │ execution_completed = true
   ↓
validation
   │ validation_passed = true
   ↓
done
```

Также сохранён fix-loop: `validation → execution`.

## Rules vs Guards

| | Transition rule | Transition guard |
|--|-----------------|------------------|
| Вопрос | Можно ли вообще такое ребро? | Можно ли **сейчас**? |
| Пример | planning → execution = YES | plan_approved? |

```text
current → target
     ↓
is transition allowed?   (rule)
     ↓
are guards satisfied?    (guard)
     ↓
YES → stage changes
NO  → REJECTED, stage unchanged
```

Единственный мутатор stage: `TaskState.transition_to()`.

## Rejected examples

```text
attempt: planning → execution
plan_approved = false
→ REJECTED, stage remains planning
```

```text
attempt: validation → done
validation_passed = false
→ REJECTED, stage remains validation
```

Прыжки `planning → validation/done`, `execution → done` — REJECTED по rule.

## Pause / Resume

Pause не меняет stage и **не сбрасывает** guards.  
После resume запрещённый ранее переход остаётся запрещённым, пока guard не выполнен.

## Связь с Day 14

```text
TaskState          → где задача
Transition rules   → куда можно
Transition guards  → выполнены ли условия
Invariants         → какие решения нельзя нарушать
```

Сначала lifecycle-команды, затем invariant gate, затем LLM.

## Run

```bash
cd day15
../.venv/bin/python -m unittest test_day15.py -v
../.venv/bin/streamlit run app.py
../.venv/bin/python run_demo.py
```

## Files

| File | Role |
|------|------|
| `task_state.py` | Day 13 FSM + guards + `TransitionResult` |
| `agent.py` | controlled API + chat lifecycle commands |
| `app.py` | lifecycle UI |
| `invariants.py` / `memory.py` / `profile.py` | Day 14/12 |
| `test_day15.py` | unit-тесты |
