"""Day 22 Streamlit: one question, two answers."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from day22.agent import load_agent
from day22.questions import QUESTIONS

st.set_page_config(page_title="Day 22 — RAG", layout="wide")
st.title("Day 22 — первый RAG-запрос")
st.caption("Один вопрос. Слева ответ без документов, справа — с найденными чанками из индекса Day 21.")

labels = [f"Q{item.id}. {item.question}" for item in QUESTIONS]
picked = st.selectbox("Контрольный вопрос", labels, index=2)
custom = st.text_input("Или свой вопрос", value="")
question = custom.strip() or QUESTIONS[labels.index(picked)].question

if st.button("Сравнить"):
    with st.spinner("Два запроса к модели…"):
        try:
            agent = load_agent()
            without = agent.ask(question, use_rag=False)
            with_rag = agent.ask(question, use_rag=True)
        except Exception as exc:  # noqa: BLE001 — show API or index failure on the page
            st.session_state["day22_error"] = str(exc)
            st.session_state["day22_pair"] = None
        else:
            st.session_state["day22_error"] = None
            st.session_state["day22_pair"] = (without, with_rag)

error = st.session_state.get("day22_error")
pair = st.session_state.get("day22_pair")
if error:
    st.error(error)
elif pair:
    without, with_rag = pair
    left, right = st.columns(2)
    with left:
        st.subheader("Без RAG")
        st.write(without.text)
    with right:
        st.subheader("С RAG")
        st.write(with_rag.text)
        st.markdown("**Источники**")
        for hit in with_rag.hits:
            st.markdown(f"- `{hit.chunk.source}` / {hit.chunk.section} ({hit.score:.3f})")
else:
    st.caption("Сравнение ещё не запускалось.")
