"""Same RAG question, answered by local Ollama or by the existing cloud model."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from day28.agent import CLOUD, LOCAL, TimedRAG, load_day28
from day28.questions import DEMO_PHRASES

st.set_page_config(page_title="Day 28 — локальный RAG", layout="wide")
st.title("Day 28 — локальный RAG")
st.caption(
    "Один и тот же поиск Day 21–24. Local отвечает через Ollama qwen3:8b. "
    "Cloud отвечает через DeepSeek. Эмбеддинги и индекс остаются на этой машине."
)


@st.cache_resource
def get_rag():
    return load_day28()


LABELS = {LOCAL: "Local", CLOUD: "Cloud"}


def _show(result: TimedRAG) -> None:
    response = result.response
    st.write(response.answer)
    if response.insufficient_context:
        st.warning("Релевантных фрагментов нет. Модель не достраивает ответ.")
    st.markdown(
        f"**Retrieval** {result.retrieval_s:.2f} s · "
        f"**Generation** {result.generation_s:.2f} s · "
        f"**Total** {result.total_s:.2f} s"
    )
    st.caption(f"Поисковый запрос: {response.rewritten_query or '—'}")
    if response.sources:
        st.markdown("**Sources**")
        for source in response.sources:
            st.markdown(f"- `{source.source}` / {source.section} / `{source.chunk_id}`")
    if response.quotes:
        st.markdown("**Quotes**")
        for quote in response.quotes:
            st.markdown(f"> {quote.text}")
    st.slider(
        "Качество ответа",
        min_value=1,
        max_value=5,
        value=3,
        key=f"day28_quality_{result.provider}",
    )
    st.checkbox(
        "Ответ держится retrieved context и не выдумывает лишнее",
        key=f"day28_stable_{result.provider}",
    )


if "day28_runs" not in st.session_state:
    st.session_state["day28_runs"] = {}

provider_label = st.radio("Модель", [LABELS[LOCAL], LABELS[CLOUD]], horizontal=True)
provider = LOCAL if provider_label == LABELS[LOCAL] else CLOUD
phrase = st.selectbox("Фраза", DEMO_PHRASES)
queued = phrase.split(". ", 1)[1] if st.button("Спросить") else None
typed = st.chat_input("Вопрос")
prompt = queued or typed
if prompt:
    with st.spinner(f"{LABELS[provider]} ищет фрагменты и отвечает…"):
        try:
            result = get_rag().ask(prompt, provider)
        except Exception as exc:  # noqa: BLE001 — show Ollama or DeepSeek errors on the page
            st.error(str(exc))
        else:
            st.session_state["day28_runs"][provider] = result
            st.session_state["day28_question"] = prompt

runs: dict[str, TimedRAG] = st.session_state["day28_runs"]
if runs:
    columns = st.columns(2)
    for column, name in zip(columns, (LOCAL, CLOUD)):
        with column:
            st.subheader(LABELS[name])
            result = runs.get(name)
            if result is None:
                st.write("Этот вариант ещё не запускали.")
                continue
            _show(result)
