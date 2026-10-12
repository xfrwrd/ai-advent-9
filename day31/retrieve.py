"""Documentation search on top of the Day 22 ranker.

A hit counts only when the question and the chunk share a content word.
Hash cosine alone is not treated as a source.
"""

from __future__ import annotations

import re

from day21.embeddings import LocalHashEmbedder
from day21.index_store import LoadedIndex
from day22.retrieve import Hit, search

_TOKEN = re.compile(r"\w+")
_STOP = {
    "какой", "какая", "какие", "какое", "что", "где", "как", "чем", "чему",
    "это", "эта", "этот", "этой", "для", "при", "или", "если", "есть",
    "в", "из", "за", "и", "у", "на", "по", "а", "не", "ли", "о", "об",
    "от", "до", "со", "же", "бы", "то", "сейчас", "текущая", "текущий",
    "help", "меня", "мне", "про", "ли",
}


def relevant(index: LoadedIndex, question: str, k: int = 4) -> list[Hit]:
    if not index.chunks:
        return []
    hits = search(index, LocalHashEmbedder(), question, k=len(index.chunks))
    terms = _terms(question)
    ranked: list[tuple[int, float, Hit]] = []
    for hit in hits:
        shared = _shared(terms, _terms(f"{hit.chunk.source} {hit.chunk.section} {hit.chunk.text}"))
        if not shared:
            continue
        ranked.append((shared, hit.score + _day_bonus(question, hit.chunk.source), hit))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked[:k]]


def _shared(left: set[str], right: set[str]) -> int:
    left = _expand(left)
    right = _expand(right)
    return sum(1 for token in left if any(_close(token, other) for other in right))


def _expand(terms: set[str]) -> set[str]:
    extra = set(terms)
    for token in terms:
        match = re.fullmatch(r"day(\d+)", token)
        if match:
            extra.add("day")
            extra.add(match.group(1))
    return extra


def _close(left: str, right: str) -> bool:
    if left == right:
        return True
    return len(left) >= 5 and len(right) >= 5 and left[:5] == right[:5]


def _day_bonus(question: str, source: str) -> float:
    bonus = 0.0
    folded = source.lower()
    for number in re.findall(r"day\s*(\d+)", question.lower()):
        if f"day{number}" in folded:
            bonus += 5
    return bonus


def _terms(text: str) -> set[str]:
    found: set[str] = set()
    for token in _TOKEN.findall(text.lower()):
        if token in _STOP:
            continue
        if len(token) > 2 or any(char.isdigit() for char in token):
            found.add(token)
    return found
