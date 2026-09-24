"""Day 19 Streamlit page: one button runs the MCP pipeline."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from day19.agent import Agent

st.set_page_config(page_title="Day 19 — MCP Pipeline", layout="wide")
st.title("Day 19 — MCP Pipeline")
st.caption("Одна кнопка запускает Agent: get_tasks → summarize_tasks → save_summary через MCP.")

agent = Agent()

run_col, clear_col = st.columns(2)
if run_col.button("Запустить MCP pipeline"):
    with st.spinner("Agent выполняет MCP pipeline…"):
        try:
            pipeline_result = asyncio.run(agent.run_tasks_pipeline())
        except Exception as exc:  # noqa: BLE001 — show pipeline failure in the page
            st.session_state["day19_error"] = str(exc)
            st.session_state["day19_result"] = None
        else:
            st.session_state["day19_error"] = None
            st.session_state["day19_result"] = pipeline_result
            st.session_state["day19_cleared"] = False

if clear_col.button("Очистить"):
    agent.summary_path.unlink(missing_ok=True)
    st.session_state["day19_error"] = None
    st.session_state["day19_result"] = None
    st.session_state["day19_cleared"] = True
    st.rerun()

day19_error = st.session_state.get("day19_error")
day19_result = st.session_state.get("day19_result")
if day19_error:
    st.error(day19_error)
elif day19_result is not None:
    labels = {
        "get_tasks": "GET TASKS",
        "summarize_tasks": "SUMMARIZE",
        "save_summary": "SAVE",
    }
    done_steps = {step.name: step for step in day19_result.steps}
    for name, label in labels.items():
        step = done_steps.get(name)
        if step is None:
            st.markdown(f"**{label}** — not run")
        else:
            st.markdown(f"**{label}** — {step.status.lower()}")
    summary = day19_result.summary or {}
    saved = day19_result.saved or {}
    st.markdown(f"Задач: **{len(day19_result.tasks)}**")
    st.markdown(f"Summary: `{summary.get('summary', '—')}`")
    st.markdown(f"Файл: `{saved.get('path', '—')}`")
    st.markdown(f"Pipeline: **{day19_result.status}**")
    st.code(day19_result.report)
elif st.session_state.get("day19_cleared"):
    st.caption("Результат убран с экрана, файл day19_summary.json удалён.")
else:
    st.caption("Pipeline ещё не запускался.")
