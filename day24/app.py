"""Day 24 playground: grounded answer, verified quotes, and "не знаю"."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from day22.questions import QUESTIONS
from day24.agent import Day24Agent, load_day24
from day24.evaluate import evaluate, load_report, save_report
from day24.negative import NEGATIVE_QUESTIONS

st.set_page_config(page_title="Day 24 — citations", layout="wide")
st.title("Day 24 — цитаты и анти-галлюцинации")
st.caption("Пайплайн Day 23. Источники и цитаты берутся из чанков, которые прошли фильтр.")


@st.cache_resource
def get_agent() -> Day24Agent:
    return load_day24()


labels = [f"Q{item.id}. {item.question}" for item in QUESTIONS]
labels += [f"NEG{item.id}. {item.question}" for item in NEGATIVE_QUESTIONS]
picked = st.selectbox("Вопрос", labels, index=5)
if picked.startswith("NEG"):
    question = NEGATIVE_QUESTIONS[int(picked[3]) - 1].question
else:
    question = QUESTIONS[labels.index(picked)].question
custom = st.text_input("Или свой вопрос", value="")
question = custom.strip() or question

if st.button("Спросить"):
    with st.spinner("Поиск и ответ…"):
        try:
            result = get_agent().ask(question)
        except Exception as exc:  # noqa: BLE001 — show the API or index error on the page
            st.session_state["day24_error"] = str(exc)
            st.session_state["day24_result"] = None
        else:
            st.session_state["day24_error"] = None
            st.session_state["day24_result"] = result

error = st.session_state.get("day24_error")
result = st.session_state.get("day24_result")
if error:
    st.error(error)
elif result:
    st.markdown(f"**Original query:** {result.original_query}")
    if result.rewritten_query:
        st.markdown(f"**Rewritten query:** {result.rewritten_query}")
    score = "—" if result.top_score is None else f"{result.top_score:.3f}"
    st.markdown(f"**Retrieval confidence / similarity:** {score}")
    st.markdown(f"**Threshold:** {result.threshold}")
    st.markdown(f"**Grounded:** {'YES' if result.is_grounded else 'NO'}")
    if result.insufficient_context:
        st.warning(
            "Недостаточно релевантной информации в базе знаний.\n\n"
            "Не знаю точного ответа. Уточните вопрос."
        )
        st.subheader("ANSWER")
        st.write(result.answer)
    else:
        st.subheader("ANSWER")
        st.write(result.answer)
        st.subheader("SOURCES")
        for source in result.sources:
            st.markdown(f"- `{source.source}` / {source.section} / `{source.chunk_id}`")
        st.subheader("SUPPORTING QUOTES")
        for quote in result.quotes:
            st.markdown(f"> {quote.text}")
            st.caption(f"{quote.source} / {quote.section} / {quote.chunk_id}")
else:
    st.caption("Запрос ещё не запускался.")

st.divider()
st.subheader("Day 24 Evaluation")
st.caption("10 вопросов Day 22 и 3 отрицательных. Само не запускается.")

report = load_report()
if st.button("Запустить evaluation"):
    with st.spinner("Прогон вопросов…"):
        try:
            report = evaluate(get_agent())
            save_report(report)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            report = load_report()

if report and "summary" in report:
    summary = report["summary"]
    left, right = st.columns(2)
    left.metric("Answers with sources", f"{summary['answers_with_sources']}/{summary['grounded_answers']}")
    left.metric("Valid quotes", f"{summary['valid_quotes']}/{summary['grounded_answers']}")
    left.metric("Supported by quotes", f"{summary['answers_supported_by_quotes']}/{summary['grounded_answers']}")
    right.metric("Answers with quotes", f"{summary['answers_with_quotes']}/{summary['grounded_answers']}")
    right.metric("Expected source found", f"{summary['expected_sources_found']}/{summary['questions']}")
    right.metric("Correct «не знаю»", f"{summary['correct_i_dont_know']}/{summary['negative_questions']}")
    st.dataframe(
        [
            {
                "Question": row["id"],
                "Sources present": row["sources_present"],
                "Quotes present": row["quotes_present"],
                "Quotes valid": row["quotes_valid"],
                "Expected source found": row["expected_source_found"],
                "Supported by quotes": row["answer_supported_by_quotes"],
                "Insufficient context": row["insufficient_context"],
            }
            for row in report["questions"]
        ],
        hide_index=True,
        width="stretch",
    )
    for row in report["questions"]:
        with st.expander(f"Q{row['id']}. {row['question']}"):
            st.markdown("**Answer**")
            st.write(row["answer"])
            st.markdown("**Sources**")
            st.json(row["sources"])
            st.markdown("**Quotes**")
            st.json(row["quotes"])
            st.markdown("**Retrieved chunks**")
            st.json(row["retrieved_chunks"])
else:
    st.caption("Готового отчёта нет. Запустите `.venv/bin/python -m day24.run_eval`.")
