"""Run the Day 22 questions and the negative questions through Day 24."""

from __future__ import annotations

import json
from pathlib import Path

from day22.agent import ChatFn
from day22.questions import QUESTIONS, ControlQuestion
from day23.evaluate import source_found
from day24.agent import Day24Agent
from day24.ground import quotes_are_valid
from day24.negative import NEGATIVE_QUESTIONS
from day24.response import RAGResponse

REPORT_PATH = Path(__file__).resolve().parent / "data" / "evaluation.json"

JUDGE_SYSTEM = (
    "Оцени, следует ли ответ из цитат. "
    "supported=true только если смысл ответа есть в цитатах и ответ не добавляет фактов вне цитат. "
    'Верни один JSON: {"supported": true, "reason": "..."}'
)


def evaluate(
    agent: Day24Agent,
    questions: tuple[ControlQuestion, ...] = QUESTIONS,
    negatives: tuple[ControlQuestion, ...] = NEGATIVE_QUESTIONS,
    judge_fn: ChatFn | None = None,
) -> dict[str, object]:
    judge = judge_fn or agent.pipeline.chat_fn
    rows = [_row(agent.ask(item.question), item, judge) for item in questions]
    negative_rows = [_negative_row(agent.ask(item.question), item) for item in negatives]
    return {
        "questions": rows,
        "negatives": negative_rows,
        "summary": summarize(rows, negative_rows),
        "similarity_threshold": agent.pipeline.similarity_threshold,
    }


def summarize(rows: list[dict[str, object]], negatives: list[dict[str, object]]) -> dict[str, object]:
    grounded = [row for row in rows if not row["insufficient_context"]]
    return {
        "questions": len(rows),
        "grounded_answers": len(grounded),
        "insufficient_on_control": [
            row["question"] for row in rows if row["insufficient_context"]
        ],
        "answers_with_sources": sum(1 for row in grounded if row["sources_present"]),
        "answers_with_quotes": sum(1 for row in grounded if row["quotes_present"]),
        "valid_quotes": sum(1 for row in grounded if row["quotes_valid"]),
        "expected_sources_found": sum(1 for row in rows if row["expected_source_found"]),
        "answers_supported_by_quotes": sum(1 for row in grounded if row["answer_supported_by_quotes"]),
        "negative_questions": len(negatives),
        "correct_i_dont_know": sum(1 for row in negatives if row["correct_i_dont_know"]),
    }


def save_report(report: dict[str, object], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_report(path: Path = REPORT_PATH) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def judge_support(question: str, answer: str, quotes: list[str], chat_fn: ChatFn) -> dict[str, object]:
    quoted = "\n".join(f"- {item}" for item in quotes) or "(нет цитат)"
    raw = chat_fn(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nQUOTES:\n{quoted}",
            },
        ]
    )
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("supported"), bool):
            return {"supported": data["supported"], "reason": str(data.get("reason") or "")}
    return {"supported": False, "reason": "judge не вернул JSON"}


def _row(result: RAGResponse, item: ControlQuestion, judge: ChatFn) -> dict[str, object]:
    kept_ids = {chunk.chunk_id for chunk in result.retrieved_chunks}
    sources_present = bool(result.sources) and all(source.chunk_id in kept_ids for source in result.sources)
    quotes_present = bool(result.quotes)
    quotes_valid = quotes_are_valid(result.quotes, result.retrieved_chunks)
    support = {"supported": False, "reason": "insufficient context"}
    if result.is_grounded:
        support = judge_support(
            result.original_query,
            result.answer,
            [quote.text for quote in result.quotes],
            judge,
        )
    return {
        "id": item.id,
        "question": item.question,
        "expected": item.expected,
        "expected_sources": list(item.sources),
        "answer": result.answer,
        "sources": [source.as_dict() for source in result.sources],
        "quotes": [quote.as_dict() for quote in result.quotes],
        "retrieved_chunks": [
            {key: value for key, value in chunk.as_dict().items() if key != "text"}
            for chunk in result.retrieved_chunks
        ],
        "sources_present": sources_present,
        "quotes_present": quotes_present,
        "quotes_valid": quotes_valid,
        "expected_source_found": source_found([source.source for source in result.sources], item.sources),
        "insufficient_context": result.insufficient_context,
        "is_grounded": result.is_grounded,
        "answer_supported_by_quotes": bool(support["supported"]),
        "support_reason": support["reason"],
        "original_query": result.original_query,
        "rewritten_query": result.rewritten_query,
        "top_score": result.top_score,
    }


def _negative_row(result: RAGResponse, item: ControlQuestion) -> dict[str, object]:
    refused = "не знаю" in result.answer.lower() and "уточн" in result.answer.lower()
    correct = result.insufficient_context and not result.sources and not result.quotes and refused
    return {
        "id": item.id,
        "question": item.question,
        "answer": result.answer,
        "sources": [source.as_dict() for source in result.sources],
        "quotes": [quote.as_dict() for quote in result.quotes],
        "top_score": result.top_score,
        "insufficient_context": result.insufficient_context,
        "correct_i_dont_know": correct,
        "rewritten_query": result.rewritten_query,
    }
