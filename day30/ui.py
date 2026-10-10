"""Chat page for the Day 30 service. Messages go to FastAPI, not to Ollama directly."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import requests
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEMO_PHRASES = (
    "1. Объясни goroutine простыми словами",
    "2. Чем это отличается от обычного потока? Ответь одним предложением.",
)
DEFAULT_URL = "http://127.0.0.1:8030"

st.set_page_config(page_title="Day 30 — приватный чат", layout="wide")
st.title("Day 30 — приватный чат")
st.caption("Сообщение уходит в FastAPI на этом Mac. Сервис сам спрашивает локальную модель. Облако не используется.")


def _service_url() -> str:
    return st.session_state.get("day30_url", DEFAULT_URL).rstrip("/")


def _api_key() -> str:
    typed = st.session_state.get("day30_key_input", "").strip()
    if typed:
        return typed
    return os.environ.get("DAY30_API_KEY", "").strip()


def _health(url: str) -> str:
    try:
        response = requests.get(f"{url}/health", timeout=(3, 5))
    except requests.RequestException:
        return "сервис не отвечает"
    if response.status_code == 200 and response.json().get("status") == "ok":
        return "сервис отвечает"
    return f"сервис вернул {response.status_code}"


def _send(url: str, api_key: str, session_id: str, token: str | None, message: str) -> requests.Response:
    headers = {"Authorization": f"Bearer {api_key}"}
    if token:
        headers["X-Session-Token"] = token
    return requests.post(
        f"{url}/chat",
        headers=headers,
        json={"session_id": session_id, "message": message},
        timeout=(10, 190),
    )


def _new_session() -> None:
    st.session_state["day30_messages"] = []
    st.session_state["day30_session_id"] = f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state["day30_session_token"] = ""


if "day30_messages" not in st.session_state:
    _new_session()

with st.sidebar:
    st.text_input("Адрес сервиса", value=DEFAULT_URL, key="day30_url")
    st.text_input("API key", type="password", key="day30_key_input")
    if os.environ.get("DAY30_API_KEY", "").strip() and not st.session_state.get("day30_key_input", "").strip():
        st.caption("Ключ взят из окружения этого процесса.")
    st.caption(_health(_service_url()))
    st.caption(f"Сессия: {st.session_state['day30_session_id']}")
    if st.button("Новый разговор"):
        _new_session()
        st.rerun()

for message in st.session_state["day30_messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

phrase = st.selectbox("Фраза", DEMO_PHRASES)
queued = phrase.split(". ", 1)[1] if st.button("Отправить фразу") else None
typed = st.chat_input("Сообщение")
prompt = queued or typed
if prompt:
    api_key = _api_key()
    if not api_key:
        st.error("Нет API key. Впишите его в боковой панели или задайте DAY30_API_KEY перед запуском страницы.")
    else:
        st.session_state["day30_messages"].append({"role": "user", "content": prompt})
        with st.spinner("Локальная модель отвечает…"):
            try:
                response = _send(
                    _service_url(),
                    api_key,
                    st.session_state["day30_session_id"],
                    st.session_state["day30_session_token"] or None,
                    prompt,
                )
            except requests.RequestException:
                st.session_state["day30_messages"].pop()
                st.error("Сервис недоступен. Запустите его и оставьте терминал открытым.")
            else:
                payload = response.json() if response.content else {}
                if response.status_code != 200:
                    st.session_state["day30_messages"].pop()
                    st.error(payload.get("detail") or f"HTTP {response.status_code}")
                else:
                    st.session_state["day30_session_token"] = payload.get("session_token", "")
                    st.session_state["day30_messages"].append(
                        {"role": "assistant", "content": payload.get("answer", "")}
                    )
                    st.rerun()
