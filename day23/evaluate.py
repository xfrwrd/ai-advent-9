"""Compare baseline, filtered, and rewrite-filtered retrieval on the Day 22 questions."""

from __future__ import annotations

import json
from pathlib import Path

from day21.embeddings import Embedder
from day21.index_store import LoadedIndex
from day22.questions import QUESTIONS, ControlQuestion
from day22.retrieve import search
from day23.agent import BASELINE, FILTERED, REWRITE_FILTERED, Day23Agent
from day23.config import FILTERED_TOP_K, RETRIEVAL_TOP_K
from day23.filter import apply_filter

REPORT_PATH = Path(__file__).resolve().parent / "data" / "evaluation.json"
SWEEP_THRESHOLDS = (6.0, 8.0, 9.0, 9.5, 10.0, 12.0, 14.0, 16.0)


def source_found(sources: list[str], expected: tuple[str, ...]) -> bool:
    return any(any(item in source for item in expected) for source in sources)


def evaluate(agent: Day23Agent, questions: tuple[ControlQuestion, ...] = QUESTIONS) -> dict[str, object]:
    rows = []
    for item in questions:
        baseline = agent.run(item.question, BASELINE)
        filtered = agent.run(item.question, FILTERED)
        rewritten = agent.run(item.question, REWRITE_FILTERED)
        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "expected": item.expected,
                "expected_sources": list(item.sources),
                "baseline": _view(baseline, item.sources),
                "filtered": _view(filtered, item.sources),
                "rewrite_filtered": _view(rewritten, item.sources),
            }
        )
    return {"questions": rows, "summary": summarize(rows)}


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    def count(mode: str, field: str) -> int:
        return sum(1 for row in rows if row[mode][field])  # type: ignore[index]

    def average(mode: str, field: str) -> float:
        if not rows:
            return 0.0
        return sum(row[mode][field] for row in rows) / len(rows)  # type: ignore[index, operator]

    lost = sum(
        1
        for row in rows
        if row["filtered"]["expected_source_found_before_filter"]  # type: ignore[index]
        and not row["filtered"]["expected_source_found_after_filter"]  # type: ignore[index]
    )
    gained = sum(
        1
        for row in rows
        if not row["filtered"]["expected_source_found_after_filter"]  # type: ignore[index]
        and row["rewrite_filtered"]["expected_source_found_after_filter"]  # type: ignore[index]
    )
    return {
        "questions": len(rows),
        "baseline_expected_source_found": count("baseline", "expected_source_found_after_filter"),
        "filtered_expected_source_found_before": count("filtered", "expected_source_found_before_filter"),
        "filtered_expected_source_found_after": count("filtered", "expected_source_found_after_filter"),
        "rewrite_expected_source_found_before": count("rewrite_filtered", "expected_source_found_before_filter"),
        "rewrite_expected_source_found_after": count("rewrite_filtered", "expected_source_found_after_filter"),
        "filtered_average_chunks_before": average("filtered", "chunks_before"),
        "filtered_average_chunks_after": average("filtered", "chunks_after"),
        "rewrite_average_chunks_before": average("rewrite_filtered", "chunks_before"),
        "rewrite_average_chunks_after": average("rewrite_filtered", "chunks_after"),
        "filtered_total_filtered_chunks": sum(row["filtered"]["filtered_count"] for row in rows),  # type: ignore[index, operator]
        "questions_lost_expected_source_after_filter": lost,
        "questions_gained_expected_source_by_rewrite": gained,
    }


def threshold_sweep(
    index: LoadedIndex,
    embedder: Embedder,
    thresholds: tuple[float, ...] = SWEEP_THRESHOLDS,
    retrieval_top_k: int = RETRIEVAL_TOP_K,
    filtered_top_k: int = FILTERED_TOP_K,
    questions: tuple[ControlQuestion, ...] = QUESTIONS,
) -> list[dict[str, object]]:
    rows = []
    for threshold in thresholds:
        retained = 0
        irrelevant = 0
        after_counts: list[int] = []
        for item in questions:
            hits = search(index, embedder, item.question, retrieval_top_k)
            result = apply_filter(hits, threshold, filtered_top_k, higher_is_better=True)
            after = [chunk.source for chunk in result.kept]
            if source_found(after, item.sources):
                retained += 1
            for chunk in result.dropped:
                if chunk.reason == "below threshold" and not source_found([chunk.source], item.sources):
                    irrelevant += 1
            after_counts.append(len(result.kept))
        rows.append(
            {
                "threshold": threshold,
                "expected_sources_retained": retained,
                "irrelevant_chunks_filtered": irrelevant,
                "average_chunks_after": sum(after_counts) / len(after_counts) if after_counts else 0.0,
            }
        )
    return rows


def save_report(report: dict[str, object], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def load_report(path: Path = REPORT_PATH) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _view(result, expected: tuple[str, ...]) -> dict[str, object]:
    before = [chunk.source for chunk in result.retrieved_before_filter]
    after = [chunk.source for chunk in result.retrieved_after_filter]
    return {
        "original_query": result.original_query,
        "rewritten_query": result.rewritten_query,
        "retrieved_before_filter": [chunk.as_dict() for chunk in result.retrieved_before_filter],
        "filtered_out": [chunk.as_dict() for chunk in result.filtered_out],
        "retrieved_after_filter": [chunk.as_dict() for chunk in result.retrieved_after_filter],
        "answer": result.answer,
        "sources": result.sources,
        "no_relevant_context": result.no_relevant_context,
        "chunks_before": len(result.retrieved_before_filter),
        "chunks_after": len(result.retrieved_after_filter),
        "filtered_count": len(result.filtered_out),
        "expected_source_found_before_filter": source_found(before, expected),
        "expected_source_found_after_filter": source_found(after, expected),
    }
