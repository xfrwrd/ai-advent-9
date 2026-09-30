"""Turn the user question into a search query. The answer still uses the original."""

from __future__ import annotations

from day22.agent import ChatFn

REWRITE_SYSTEM = (
    "Перепиши вопрос пользователя в один поисковый запрос. "
    "Сохрани исходный смысл. Сделай запрос конкретнее для поиска по коду и документам. "
    "Не отвечай на вопрос. Не добавляй факты, которых нет в исходном запросе. "
    "Верни только rewritten query, без кавычек и пояснений."
)


def rewrite_query(question: str, chat_fn: ChatFn) -> str:
    try:
        raw = chat_fn(
            [
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": question},
            ]
        )
    except Exception:
        return question
    line = raw.strip().splitlines()[0].strip() if raw and raw.strip() else ""
    line = line.strip("`").strip().strip('"').strip("'").strip()
    return line or question
