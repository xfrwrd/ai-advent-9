"""One support question from the terminal.

    python -m day33 --user demo-user-1 --ticket ticket-101 "Почему я не могу войти?"
"""

from __future__ import annotations

import argparse

from day33.support import answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Вопрос в поддержку DemoPortal")
    parser.add_argument("--user", required=True)
    parser.add_argument("--ticket", required=True)
    parser.add_argument("--session", default="cli-session")
    parser.add_argument("--operator", action="store_true")
    parser.add_argument("question")
    args = parser.parse_args(argv)
    result = answer(
        args.question,
        actor_id=args.user,
        ticket_id=args.ticket,
        session_id=args.session,
        mode="operator" if args.operator else "user",
    )
    print(result.text())
    if result.errors and not result.llm_called and not result.denied:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
