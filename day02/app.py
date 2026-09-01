"""Day 02: Compare DeepSeek responses with and without output constraints."""

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
DEFAULT_PROMPT = (
    "Объясни, что такое идемпотентность и зачем она нужна при проектировании API."
)
SYSTEM_PROMPT = """Ответь строго в двух частях:

Определение: ...
Пример: ...

Весь ответ — не более 70 слов. После "Пример" на новой строке выведи <<<END>>>. После <<<END>>> ничего не пиши."""
THINKING_DISABLED = {"type": "disabled"}

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def call_deepseek(api_key: str, payload: dict) -> requests.Response:
    return requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )


def fetch_result(api_key: str, payload: dict, label: str) -> dict | None:
    try:
        response = call_deepseek(api_key, payload)
    except requests.RequestException as exc:
        st.error(f"Запрос «{label}» не выполнен")
        st.code(str(exc))
        return None

    if not response.ok:
        st.error(f"Запрос «{label}» не выполнен")
        st.code(f"HTTP {response.status_code}\n{response.text}")
        return None

    return response.json()


def show_result(data: dict) -> None:
    choice = data["choices"][0]
    content = (choice["message"].get("content") or "").strip()
    st.markdown(content or "—")
    st.caption(f"output tokens: {data['usage']['completion_tokens']}")
    st.caption(f"finish_reason: {choice['finish_reason']}")


st.title("Day 2 — контроль ответа LLM")

user_prompt = st.text_area("USER_PROMPT", value=DEFAULT_PROMPT, height=120)

if st.button("Отправить"):
    if not user_prompt.strip():
        st.error("Введите текст запроса.")
        st.stop()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
        st.stop()

    unrestricted_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_prompt}],
        "thinking": THINKING_DISABLED,
    }

    restricted_payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 200,
        "stop": ["<<<END>>>"],
        "thinking": THINKING_DISABLED,
    }

    with st.spinner("Выполняются два запроса..."):
        free_data = fetch_result(api_key, unrestricted_payload, "Без ограничений")
        restricted_data = fetch_result(api_key, restricted_payload, "С ограничениями")

    if not free_data and not restricted_data:
        st.stop()

    col_free, col_restricted = st.columns(2)

    with col_free:
        st.subheader("Без ограничений")
        st.caption("Без формата · без max_tokens · без stop")
        if free_data:
            show_result(free_data)
        else:
            st.warning("Ответ не получен.")

    with col_restricted:
        st.subheader("С ограничениями")
        st.caption("Определение + Пример · ≤ 70 слов · max_tokens=200 · stop=<<<END>>>")
        if restricted_data:
            show_result(restricted_data)
        else:
            st.warning("Ответ не получен.")
