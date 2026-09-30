"""Day 23 playground: four modes on the Day 22 index and questions."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from day22.questions import QUESTIONS
from day23.agent import MODES, Day23Agent, load_agent
from day23.config import FILTERED_TOP_K, RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD
from day23.evaluate import evaluate, load_report, save_report

MODE_LABELS = {
    "without_rag": "WITHOUT RAG",
    "baseline": "BASELINE RAG",
    "filtered": "FILTERED RAG",
    "rewrite_filtered": "REWRITE + FILTER",
}

st.set_page_config(page_title="Day 23 — RAG filter", layout="wide")
st.title("Day 23 — реранкинг и фильтрация")
st.caption("Тот же индекс Day 21 и тот же агент Day 22. Фильтр и rewrite включаются выбранным режимом.")


@st.cache_resource
def get_agent() -> Day23Agent:
    return load_agent()


labels = [f"Q{item.id}. {item.question}" for item in QUESTIONS]
picked = st.selectbox("Контрольный вопрос", labels, index=5)
custom = st.text_input("Или свой вопрос", value="")
question = custom.strip() or QUESTIONS[labels.index(picked)].question
mode = st.radio("Режим", MODES, format_func=lambda item: MODE_LABELS[item], horizontal=True)

if st.button("Спросить"):
    with st.spinner("Запрос…"):
        try:
            result = get_agent().run(question, mode)
        except Exception as exc:  # noqa: BLE001 — show the API or index error on the page
            st.session_state["day23_error"] = str(exc)
            st.session_state["day23_result"] = None
        else:
            st.session_state["day23_error"] = None
            st.session_state["day23_result"] = result

error = st.session_state.get("day23_error")
result = st.session_state.get("day23_result")
if error:
    st.error(error)
elif result:
    st.markdown(f"**Original query:** {result.original_query}")
    if result.rewritten_query:
        st.markdown(f"**Rewritten query:** {result.rewritten_query}")
    if result.mode == "without_rag":
        st.caption("Поиск не выполнялся.")
    else:
        before = len(result.retrieved_before_filter)
        after = len(result.retrieved_after_filter)
        st.markdown(f"**Retrieved before filtering:** {before}")
        st.markdown(f"**Retrieved after filtering:** {after}")
        if result.threshold is None:
            st.markdown("**Threshold:** не применяется (BASELINE совпадает с Day 22)")
        else:
            st.markdown(
                f"**Threshold:** {result.threshold} "
                f"(retrieval_top_k={RETRIEVAL_TOP_K}, filtered_top_k={FILTERED_TOP_K})"
            )
        rows = result.retrieved_before_filter or result.retrieved_after_filter
        if rows:
            st.dataframe(
                [
                    {
                        "rank": chunk.rank,
                        "score": round(chunk.score, 3),
                        "source": chunk.source,
                        "section": chunk.section,
                        "status": chunk.status,
                    }
                    for chunk in rows
                ],
                hide_index=True,
                width="stretch",
            )
        if result.no_relevant_context:
            st.warning("Релевантный контекст не найден: все чанки ниже порога.")
    st.subheader("Final answer")
    st.write(result.answer)
    if result.sources:
        st.markdown("**Sources**")
        for source in result.sources:
            st.markdown(f"- `{source}`")
else:
    st.caption("Запрос ещё не запускался.")

st.divider()
st.subheader("Day 23 Evaluation")
st.caption("Сравнение трёх RAG-режимов на 10 вопросах Day 22. Само не запускается.")

report = load_report()
if st.button("Запустить evaluation"):
    with st.spinner("Три режима на 10 вопросах…"):
        try:
            agent = get_agent()
            report = evaluate(agent)
            report["retrieval_top_k"] = RETRIEVAL_TOP_K
            report["filtered_top_k"] = FILTERED_TOP_K
            report["similarity_threshold"] = SIMILARITY_THRESHOLD
            save_report(report)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            report = load_report()

if report and "summary" in report:
    summary = report["summary"]
    one, two, three = st.columns(3)
    one.metric("BASELINE expected source", f"{summary['baseline_expected_source_found']}/10")
    two.metric("FILTERED expected source", f"{summary['filtered_expected_source_found_after']}/10")
    two.caption(f"average chunks {summary['filtered_average_chunks_after']:.1f}")
    three.metric("REWRITE + FILTER expected source", f"{summary['rewrite_expected_source_found_after']}/10")
    three.caption(f"average chunks {summary['rewrite_average_chunks_after']:.1f}")
    st.caption(
        f"Порог {report.get('similarity_threshold', SIMILARITY_THRESHOLD)}. "
        f"После фильтра потеряли ожидаемый source: {summary['questions_lost_expected_source_after_filter']}. "
        f"Rewrite вернул ожидаемый source: {summary['questions_gained_expected_source_by_rewrite']}."
    )
else:
    st.caption("Готового отчёта нет. Запустите `.venv/bin/python -m day23.run_eval`.")
