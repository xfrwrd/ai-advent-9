"""Day 07: Streamlit UI for an agent with persistent dialog history."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import Agent, AgentError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(page_title="Day 7 — Память агента", layout="centered")
st.title("Day 7 — Память агента")
st.caption(
    "UI → Agent → DeepSeek API. "
    "История диалога хранится в history.json и переживает перезапуск Streamlit."
)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
    st.stop()

agent = Agent(api_key=api_key, model="deepseek-v4-flash")

if st.button("Очистить историю"):
    agent.clear_history()
    st.rerun()

for message in agent.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_message = st.chat_input("Введите запрос...")

if user_message:
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        try:
            answer = agent.ask(user_message)
            st.markdown(answer)
        except AgentError as exc:
            st.error(str(exc))
