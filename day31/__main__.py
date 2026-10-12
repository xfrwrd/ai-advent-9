"""CLI: python -m day31 "/help Как устроен локальный RAG?" """

from __future__ import annotations

import sys

from day31.help import answer


def main() -> None:
    if len(sys.argv) < 2:
        print('Нужен вопрос: python -m day31 "/help Как устроен локальный RAG?"')
        raise SystemExit(2)
    print(answer(" ".join(sys.argv[1:])).text())


if __name__ == "__main__":
    main()
