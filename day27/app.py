"""Web chat. Each message goes to the local Ollama model from Day 26."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from day26.client import LocalLLMClient

DEMO_PHRASES = (
    "1. Объясни одним предложением, что такое goroutine в Go.",
    "2. Объясни разницу между mutex и channel в Go. По одному примеру, когда что лучше.",
)

st.set_page_config(page_title="Day 27 — локальная LLM", layout="wide")
st.title("Day 27 — локальная LLM")
st.caption("Вопрос уходит в Ollama, модель qwen3:8b, http://localhost:11434. Облако не используется.")


@st.cache_resource
def get_client() -> LocalLLMClient:
    return LocalLLMClient()


if "day27_messages" not in st.session_state:
    st.session_state["day27_messages"] = []

for message in st.session_state["day27_messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

phrase = st.selectbox("Фраза", DEMO_PHRASES)
queued = phrase.split(". ", 1)[1] if st.button("Отправить фразу") else None
typed = st.chat_input("Сообщение")
prompt = queued or typed
if prompt:
    st.session_state["day27_messages"].append({"role": "user", "content": prompt})
    with st.spinner("Локальная модель отвечает…"):
        try:
            answer = get_client().ask(prompt)
        except RuntimeError as exc:
            st.session_state["day27_messages"].pop()
            st.error(str(exc))
        else:
            st.session_state["day27_messages"].append({"role": "assistant", "content": answer})
            st.rerun()
