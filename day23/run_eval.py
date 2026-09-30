"""Run the Day 22 questions through baseline, filter, and rewrite + filter."""

from __future__ import annotations

from day23.agent import load_agent
from day23.config import FILTERED_TOP_K, RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD
from day23.evaluate import REPORT_PATH, evaluate, save_report, threshold_sweep


def main() -> None:
    agent = load_agent()
    sweep = threshold_sweep(agent.index, agent.embedder)
    report = evaluate(agent)
    report["retrieval_top_k"] = RETRIEVAL_TOP_K
    report["filtered_top_k"] = FILTERED_TOP_K
    report["similarity_threshold"] = SIMILARITY_THRESHOLD
    report["score"] = "Day 22 score = BM25 + cosine. Larger score is a better match."
    report["threshold_sweep"] = sweep
    save_report(report)
    summary = report["summary"]
    print(f"retrieval_top_k={RETRIEVAL_TOP_K} filtered_top_k={FILTERED_TOP_K} threshold={SIMILARITY_THRESHOLD}")
    print("threshold sweep:")
    for row in sweep:
        print(
            f"  {row['threshold']}: retained {row['expected_sources_retained']}/10, "
            f"irrelevant filtered {row['irrelevant_chunks_filtered']}, "
            f"avg after {row['average_chunks_after']:.1f}"
        )
    print(
        "expected source: "
        f"baseline {summary['baseline_expected_source_found']}/10, "
        f"filtered {summary['filtered_expected_source_found_after']}/10, "
        f"rewrite {summary['rewrite_expected_source_found_after']}/10"
    )
    print(
        f"filtered avg chunks {summary['filtered_average_chunks_before']:.1f} -> "
        f"{summary['filtered_average_chunks_after']:.1f}, "
        f"lost {summary['questions_lost_expected_source_after_filter']}, "
        f"gained by rewrite {summary['questions_gained_expected_source_by_rewrite']}"
    )
    print(f"saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
