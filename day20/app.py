"""Day 20 Streamlit page: one request, tools from two MCP servers."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from day20.agent import Agent

DEFAULT_REQUEST = "Получи текущий статус TASK-123 и подготовь сводку по истории"

st.set_page_config(page_title="MCP Orchestration", layout="wide")
st.title("MCP Orchestration")
st.caption("Agent выбирает tool и отправляет его на task_server или history_server.")

if "day20_runs" not in st.session_state:
    st.session_state["day20_runs"] = []

request = st.text_input("Запрос", value=DEFAULT_REQUEST)
if st.button("Run Agent"):
    with st.spinner("Agent выполняет flow…"):
        try:
            result = asyncio.run(Agent().run(request))
        except Exception as exc:  # noqa: BLE001 — show the failure on the page
            st.session_state["day20_error"] = str(exc)
        else:
            st.session_state["day20_error"] = None
            st.session_state["day20_runs"].append(result)

error = st.session_state.get("day20_error")
if error:
    st.error(error)

runs = st.session_state["day20_runs"]
if not runs:
    st.caption("Запрос ещё не запускался.")
else:
    for number, result in enumerate(reversed(runs), start=1):
        launch = len(runs) - number + 1
        st.divider()
        st.subheader(f"Запуск {launch}")
        st.markdown(result.request)
        st.markdown("**Trace**")
        st.code("\n".join(result.trace))
        st.markdown("**Calls**")
        st.dataframe(
            [
                {
                    "Step": index,
                    "Server": call.server,
                    "Tool": call.tool,
                    "Result": str(call.result),
                }
                for index, call in enumerate(result.calls, start=1)
            ],
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Final Answer**")
        st.code(result.answer or "—")
