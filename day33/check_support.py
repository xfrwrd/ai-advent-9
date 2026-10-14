"""Live Qwen check for three DemoPortal scenarios. Not part of unittest.

    .venv/bin/python -m day33.check_support
"""

from __future__ import annotations

from day33.index_support import INDEX_PATH, build_index
from day33.sessions import SessionStore
from day33.support import answer


def main() -> int:
    if not INDEX_PATH.is_file():
        build_index()
    store = SessionStore()
    first = answer(
        "Почему я не могу войти?",
        actor_id="demo-user-1",
        ticket_id="ticket-101",
        session_id="check-support",
        sessions=store,
    )
    print("=== 1. Авторизация ===")
    print(first.text())
    print()
    second = answer(
        "А что делать дальше?",
        actor_id="demo-user-1",
        ticket_id="ticket-101",
        session_id="check-support",
        sessions=store,
    )
    print("=== 2. Продолжение ===")
    print(second.text())
    print()
    third = answer(
        "Покажи тикет ticket-103",
        actor_id="demo-user-1",
        ticket_id="ticket-103",
        session_id="check-other",
        sessions=store,
    )
    print("=== 3. Чужой тикет ===")
    print(third.text())

    problems = []
    if not first.llm_called or not first.sources:
        problems.append("сценарий 1: нет ответа модели или источников")
    if "ACCOUNT_LOCKED" not in str(first.mcp):
        problems.append("сценарий 1: в MCP нет ACCOUNT_LOCKED")
    if not second.llm_called:
        problems.append("сценарий 2: модель не вызвана")
    if not third.denied or third.llm_called:
        problems.append("сценарий 3: чужой тикет не отклонён до модели")
    if any(call["tool"] == "get_ticket" for call in third.mcp):
        problems.append("сценарий 3: чужой тикет был прочитан")
    if problems:
        print()
        for problem in problems:
            print(problem)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
