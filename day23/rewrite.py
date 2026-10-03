"""Turn the user question into a search query. The answer still uses the original."""

from __future__ import annotations

from day22.agent import ChatFn

REWRITE_SYSTEM = (
    "Перепиши вопрос пользователя в один поисковый запрос. "
    "Сохрани исходный смысл. Сделай запрос конкретнее для поиска по коду и документам. "
    "Не отвечай на вопрос. Не добавляй факты, которых нет в исходном запросе. "
    "Верни только rewritten query, без кавычек и пояснений."
)


def rewrite_query(question: str, chat_fn: ChatFn, context: str = "") -> str:
    system = REWRITE_SYSTEM
    user = question
    if context.strip():
        system += (
            " Если дан контекст диалога, раскрой местоимения и тему из него. "
            "Не добавляй факты, которых нет ни в вопросе, ни в этом контексте."
        )
        user = f"Контекст диалога:\n{context.strip()}\n\nТекущий вопрос:\n{question}"
    try:
        raw = chat_fn(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
    except Exception:
        return question
    line = raw.strip().splitlines()[0].strip() if raw and raw.strip() else ""
    line = line.strip("`").strip().strip('"').strip("'").strip()
    return line or question
