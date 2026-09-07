"""Day 06: Streamlit UI for the first agent."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import Agent, AgentError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(page_title="Day 6 — Первый агент", layout="centered")
st.title("Day 6 — Первый агент")
st.caption(
    "UI → Agent → DeepSeek API. "
    "Вся переписка сохраняется и каждый раз отправляется в модель как контекст."
)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

agent = Agent(api_key=api_key, model="deepseek-v4-flash")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_message = st.chat_input("Введите запрос...")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        try:
            answer = agent.ask(st.session_state.messages)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except AgentError as exc:
            st.session_state.messages.pop()
            st.error(str(exc))
