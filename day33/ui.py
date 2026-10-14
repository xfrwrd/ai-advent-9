"""DemoPortal support page. The question box does not choose the user."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from day33.mcp_client import call_many_sync
from day33.support import answer

ACTORS = ("demo-user-1", "demo-user-2", "demo-user-3")
DEMO = (
    "Почему я не могу войти?",
    "А что делать дальше?",
    "Что означает код ошибки в моём тикете?",
)

st.set_page_config(page_title="Day 33 — поддержка DemoPortal", layout="wide")
st.title("Day 33 — поддержка DemoPortal")
st.caption(
    "Общие инструкции ищутся в FAQ. Данные тикета читаются через MCP "
    "только для выбранного demo-пользователя. Ответ пишет локальная Qwen."
)

if "day33_log" not in st.session_state:
    st.session_state["day33_log"] = []
if "day33_sid" not in st.session_state:
    st.session_state["day33_sid"] = uuid.uuid4().hex

mode_label = st.radio("Режим", ("Пользователь", "Оператор"), horizontal=True)
mode = "operator" if mode_label == "Оператор" else "user"

profiles: dict[str, dict] = {}
tickets_by_user: dict[str, list[dict]] = {}
mcp_error = ""
try:
    calls = [("get_user", {"user_id": actor}) for actor in ACTORS]
    calls.extend(("list_user_tickets", {"user_id": actor}) for actor in ACTORS)
    payloads = call_many_sync(calls)
    for actor, profile, listing in zip(ACTORS, payloads[: len(ACTORS)], payloads[len(ACTORS) :]):
        profiles[actor] = profile
        rows = listing.get("tickets") if isinstance(listing, dict) else None
        tickets_by_user[actor] = rows if isinstance(rows, list) else []
        if isinstance(profile, dict) and profile.get("error"):
            mcp_error = str(profile.get("message") or profile["error"])
except Exception as exc:
    mcp_error = f"MCP-сервер поддержки недоступен: {exc}"
    tickets_by_user = {actor: [] for actor in ACTORS}

if mcp_error:
    st.error(mcp_error)


def _user_label(actor: str) -> str:
    profile = profiles.get(actor) or {}
    name = profile.get("display_name") or actor
    status = profile.get("account_status") or "нет данных"
    return f"{name} · {actor} · {status}"


actor_id = st.selectbox("Demo-пользователь", ACTORS, format_func=_user_label)
own_tickets = [item for item in tickets_by_user.get(actor_id, []) if isinstance(item, dict)]

if mode == "user":
    if not own_tickets:
        st.info("У выбранного пользователя нет тикетов.")
        selected = ""
    else:
        ticket_ids = [str(item.get("ticket_id") or "") for item in own_tickets]

        def _ticket_label(ticket: str) -> str:
            item = next(row for row in own_tickets if str(row.get("ticket_id") or "") == ticket)
            return (
                f"{item.get('ticket_id')} · {item.get('error_code')} · "
                f"{item.get('status')} · {item.get('created_at')}"
            )

        selected = st.selectbox("Тикет пользователя", ticket_ids, format_func=_ticket_label)
else:
    st.warning("Режим оператора читает любой тикет. Это не пользовательский вход.")
    selected = st.text_input("ticket_id", value="ticket-101").strip()

probe = ""
foreign = False
if mode == "user":
    probe = st.text_input(
        "Проверка доступа: другой ticket_id",
        value="",
        help="Если указать чужой тикет, отказ приходит до вызова модели.",
    ).strip()
    foreign = st.button("Запросить чужой тикет ticket-103")

ticket_id = "ticket-103" if foreign else (probe or selected)
identity = f"{mode}:{actor_id}"
if st.session_state.get("day33_pick") != identity:
    st.session_state["day33_pick"] = identity
    st.session_state["day33_sid"] = uuid.uuid4().hex
    st.session_state["day33_log"] = []

phrase = st.selectbox("Фраза", DEMO)
asked = st.button("Спросить")
typed = st.chat_input("Свой вопрос")
prompt = typed or (phrase if asked or foreign else None)
if prompt:
    with st.spinner("Читаю тикет через MCP и ищу инструкцию в FAQ…"):
        result = answer(
            prompt,
            actor_id=actor_id,
            ticket_id=ticket_id,
            session_id=st.session_state["day33_sid"],
            mode=mode,
        )
    st.session_state["day33_log"].append(result)

for item in st.session_state["day33_log"]:
    with st.chat_message("user"):
        st.write(item.question)
    with st.chat_message("assistant"):
        if item.denied:
            st.error(item.answer)
        else:
            st.write(item.answer)
        if item.operator_recommended:
            st.info("Рекомендация: обратиться к оператору.")
        left, right = st.columns(2)
        with left:
            st.markdown("**Источники RAG**")
            if item.sources:
                for source in item.sources:
                    st.write(f"{source['source']} · {source['section']} · `{source['chunk_id']}`")
            else:
                st.write("Совпадений в базе знаний нет.")
        with right:
            st.markdown("**Данные MCP**")
            if item.mcp:
                for call in item.mcp:
                    st.write(f"{call['tool']} {call['arguments']}")
                    st.json(call["result"])
            else:
                st.write("MCP не вызывался.")
        if item.errors:
            for error in item.errors:
                st.warning(error)
