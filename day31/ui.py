"""Page for /help. The button calls the Day 31 assistant, not Ollama directly."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from day31.help import answer

DEMO = (
    "/help Как устроен RAG в этом проекте?",
    "/help Что находится в папке day28?",
    "/help Как запустить Day 30?",
    "/help Какие компоненты используются для работы с Ollama?",
    "/help Какая текущая Git-ветка?",
    "/help Какой пароль от офисного Wi-Fi?",
)

st.set_page_config(page_title="Day 31 — /help", layout="wide")
st.title("Day 31 — ассистент разработчика")
st.caption("Документация ищется локально. Git-ветка запрашивается через MCP. Ответ пишет локальная Qwen. Облако не используется.")

if "day31_log" not in st.session_state:
    st.session_state["day31_log"] = []

phrase = st.selectbox("Фраза", DEMO)
queued = phrase if st.button("Спросить") else None
typed = st.chat_input("Вопрос, можно начать с /help")
prompt = queued or typed
if prompt:
    with st.spinner("Ищу в документации и, если нужно, спрашиваю MCP…"):
        result = answer(prompt)
    st.session_state["day31_log"].append(result)

for item in st.session_state["day31_log"]:
    with st.chat_message("user"):
        st.write(item.question)
    with st.chat_message("assistant"):
        st.write(item.answer)
        st.markdown("**Документы**")
        if item.sources:
            for source in item.sources:
                st.write(f"{source['source']} · {source['section']} · `{source['chunk_id']}`")
        else:
            st.write("Совпадений в индексе документации нет.")
        st.markdown("**MCP**")
        if item.mcp:
            for call in item.mcp:
                st.write(call["tool"])
                st.json(call["result"])
        else:
            st.write("MCP не вызывался.")
        if item.errors:
            for error in item.errors:
                st.warning(error)
