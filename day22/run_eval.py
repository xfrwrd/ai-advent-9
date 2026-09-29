"""Ask the 10 control questions with RAG and without it."""

from __future__ import annotations

import re

from day22.agent import load_agent
from day22.questions import QUESTIONS

_TERM = re.compile(r"[A-Za-z0-9_./-]+")


def expectation_met(answer: str, expected: str) -> bool:
    terms = [term.lower() for term in _TERM.findall(expected) if len(term) > 1]
    lowered = answer.lower()
    return bool(terms) and all(term in lowered for term in terms)


def main() -> None:
    agent = load_agent()
    rag_hits = 0
    plain_hits = 0
    for item in QUESTIONS:
        without = agent.ask(item.question, use_rag=False)
        with_rag = agent.ask(item.question, use_rag=True)
        rag_ok = expectation_met(with_rag.text, item.expected)
        plain_ok = expectation_met(without.text, item.expected)
        rag_hits += int(rag_ok)
        plain_hits += int(plain_ok)
        sources = ", ".join(dict.fromkeys(with_rag.sources)) or "—"
        print(f"Q{item.id}. {item.question}")
        print(f"  ожидание: {item.expected}")
        print(f"  источники: {', '.join(item.sources)}")
        print(f"  найдено: {sources}")
        print(f"  без RAG: {without.text.strip().replace(chr(10), ' ')}")
        print(f"  с RAG:   {with_rag.text.strip().replace(chr(10), ' ')}")
        print(f"  ожидание в ответе: без RAG={'да' if plain_ok else 'нет'}, с RAG={'да' if rag_ok else 'нет'}")
        print()
    print(f"Итого ожидание найдено: без RAG {plain_hits}/10, с RAG {rag_hits}/10")


if __name__ == "__main__":
    main()
