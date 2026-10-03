"""RAG chat: history on disk, task memory beside it, Day 24 answers."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from day25.agent import ChatAgent, load_chat
from day25.models import Conversation, TaskState
from day25.store import ConversationStore

st.set_page_config(page_title="Day 25 — RAG chat", layout="wide", initial_sidebar_state="expanded")
st.title("Day 25 — мини-чат с RAG")
st.caption("История диалога и память задачи хранятся отдельно. Каждый вопрос идёт в поиск Day 23 и ответ Day 24.")


@st.cache_resource
def get_agent() -> ChatAgent:
    return load_chat()


store = ConversationStore()
agent = get_agent()

DEMO_PHRASES = (
    "1. Моя цель — разобраться, как day19 pipeline вызывает MCP tools.",
    "2. В каком они идут порядке?",
    "3. Ограничение: смотри только day19 pipeline, не day20.",
    "4. А куда сохраняется сводка?",
)

if "day25_conversation_id" not in st.session_state:
    st.session_state["day25_conversation_id"] = None

current_id = st.session_state["day25_conversation_id"]
if current_id and (store.directory / f"{current_id}.json").exists():
    conversation = store.load(current_id)
else:
    st.session_state["day25_conversation_id"] = None
    conversation = Conversation("", [], TaskState(), "", "")

with st.sidebar:
    st.subheader("Диалог")
    if st.button("New conversation"):
        created = store.new()
        st.session_state["day25_conversation_id"] = created.conversation_id
        st.session_state["day25_open"] = created.conversation_id
        st.rerun()
    saved = store.list_ids()
    if saved:
        if st.session_state.get("day25_open") not in saved:
            st.session_state["day25_open"] = current_id if current_id in saved else saved[0]
        chosen = st.selectbox("Открыть", saved, key="day25_open")
        if chosen != current_id:
            st.session_state["day25_conversation_id"] = chosen
            st.rerun()
    phrase = st.selectbox("Фраза", DEMO_PHRASES)
    queued = phrase.split(". ", 1)[1] if st.button("Отправить фразу") else None

for message in conversation.messages:
    with st.chat_message(message.role):
        st.write(message.content)
        if message.role == "assistant" and message.insufficient_context:
            st.warning("Недостаточно релевантной информации. Не знаю точного ответа. Уточните вопрос.")
        if message.sources:
            st.markdown("**Sources**")
            for source in message.sources:
                st.markdown(f"- `{source['source']}` / {source['section']} / `{source['chunk_id']}`")
        if message.quotes:
            st.markdown("**Quotes**")
            for quote in message.quotes:
                st.markdown(f"> {quote['text']}")

typed = st.chat_input("Сообщение")
prompt = queued or typed
if prompt:
    with st.spinner("Память, поиск, ответ…"):
        try:
            if not conversation.conversation_id:
                conversation = store.new()
                st.session_state["day25_conversation_id"] = conversation.conversation_id
                st.session_state["day25_open"] = conversation.conversation_id
            agent.send(conversation, prompt)
        except Exception as exc:  # noqa: BLE001 — show the API error in the chat
            st.error(str(exc))
        else:
            st.rerun()

with st.expander("Current Task State"):
    state = conversation.task_state
    st.markdown(f"**Goal:** {state.goal or '—'}")
    st.markdown("**Constraints:**")
    for item in state.constraints or ["—"]:
        st.markdown(f"- {item}")
    st.markdown("**Terms:**")
    for item in state.terms or ["—"]:
        st.markdown(f"- {item}")
    st.markdown("**Clarified facts:**")
    for item in state.clarified_facts or ["—"]:
        st.markdown(f"- {item}")
    last_user = next((item for item in reversed(conversation.messages) if item.role == "user"), None)
    last_assistant = next((item for item in reversed(conversation.messages) if item.role == "assistant"), None)
    st.markdown(f"**Original user message:** {last_user.content if last_user else '—'}")
    st.markdown(f"**Context-aware search query:** {last_assistant.search_query if last_assistant else '—'}")
