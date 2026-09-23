"""Day 18 Streamlit page: show scheduler state, SQLite rows, and MCP summary."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import streamlit as st

from agent import Agent
from scheduler import SchedulerView, get_scheduler
from store import clock_from_stored, default_db_path

RECENT_LIMIT = 20
REFRESH_SECONDS = 2

st.set_page_config(page_title="MCP Background Agent", layout="wide")
st.title("MCP Background Agent")
st.caption(
    "Scheduler пишет snapshots в SQLite сам. "
    "Эта страница только показывает состояние и запрашивает сводку через MCP."
)

# Idempotent: the same thread survives Streamlit reruns and fragment refreshes.
get_scheduler()
agent = Agent(default_db_path())


def _clock(stored: str | None) -> str:
    if not stored:
        return "—"
    return clock_from_stored(stored)


def _next_run(view: SchedulerView) -> str:
    if view.next_run_at is None:
        return "—"
    return view.next_run_at.strftime("%H:%M:%S")


def _interval_label(seconds: float) -> str:
    if seconds == int(seconds):
        return f"{int(seconds)} s"
    return f"{seconds:g} s"


@st.fragment(run_every=timedelta(seconds=REFRESH_SECONDS))
def render_live() -> None:
    scheduler = get_scheduler()
    view = scheduler.view()

    st.subheader("Scheduler status")
    state, interval, last_ok, saved, nxt = st.columns(5)
    state.metric("State", "Running" if view.running else "Stopped")
    interval.metric("Interval", _interval_label(view.interval_seconds))
    last_ok.metric("Last collection", _clock(view.last_success_at))
    saved.metric("Snapshots", view.snapshot_count)
    nxt.metric("Next run", _next_run(view))
    st.caption(
        "Сбор идёт на backend и не привязан к кнопкам или к обновлению экрана. "
        f"Таблица ниже перечитывает SQLite каждые {REFRESH_SECONDS} с."
    )

    st.subheader("Stored data")
    st.caption(f"Последние {RECENT_LIMIT} записей из SQLite, новые сверху.")
    rows = scheduler.store.recent(RECENT_LIMIT)
    if not rows:
        st.info("SQLite пока пуст.")
        return
    st.dataframe(
        [
            {
                "task_id": row.task_id,
                "status": row.status,
                "collected_at": _clock(row.collected_at),
            }
            for row in rows
        ],
        hide_index=True,
        width="stretch",
    )


render_live()

st.divider()
st.subheader("MCP Summary")
st.caption(
    "Streamlit → Agent → MCP Client → get_task_summary → SQLite → MCP result → Agent → Streamlit"
)

if st.button("Получить сводку через MCP"):
    with st.spinner("Agent вызывает MCP tool get_task_summary…"):
        try:
            result = asyncio.run(agent.fetch_summary())
        except Exception as exc:  # noqa: BLE001 — show MCP/client failure in the page
            st.session_state["mcp_output"] = None
            st.session_state["mcp_payload"] = None
            st.session_state["mcp_error"] = str(exc)
        else:
            st.session_state["mcp_output"] = result.output
            st.session_state["mcp_payload"] = result.payload
            st.session_state["mcp_error"] = result.error

error = st.session_state.get("mcp_error")
output = st.session_state.get("mcp_output")
payload = st.session_state.get("mcp_payload")

if error:
    st.error(error)
elif output:
    st.code(output)
    if payload is not None:
        with st.expander("MCP tool result"):
            st.json(payload)
else:
    st.caption("Сводка ещё не запрашивалась.")
