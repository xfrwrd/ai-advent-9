"""Manual check against the real index, the real MCP server, and Qwen.

Not part of unittest. A run here is a real model call.

    .venv/bin/python -m day31.check_help
"""

from __future__ import annotations

from day31.help import answer

QUESTIONS = (
    "/help Как устроен локальный RAG?",
    "/help Какая текущая Git-ветка?",
    "/help Какой пароль от офисного Wi-Fi?",
)


def main() -> None:
    for question in QUESTIONS:
        print("=" * 60)
        print(question)
        print(answer(question).text())
        print()


if __name__ == "__main__":
    main()
