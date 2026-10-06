"""CLI chat. Each line goes to the local model and the answer is printed back."""

from __future__ import annotations

from day26.client import LocalLLMClient

EXIT_WORDS = {"exit", "quit", "выход"}


def main() -> None:
    client = LocalLLMClient()
    print(f"Локальный чат. Модель {client.model} на {client.base_url}. Облако не используется.")
    print("Пустая строка пропускается. Выход: exit")
    print()
    while True:
        try:
            prompt = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.casefold() in EXIT_WORDS:
            break
        try:
            answer = client.ask(prompt)
        except RuntimeError as exc:
            print(exc)
            break
        print()
        print(answer)
        print()


if __name__ == "__main__":
    main()
