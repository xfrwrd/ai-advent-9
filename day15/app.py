"""Day 15: Streamlit — controlled transitions with guards."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import (
    INPUT_PRICE_PER_MILLION,
    OUTPUT_PRICE_PER_MILLION,
    Agent,
    AgentError,
)
from memory import INVARIANTS_KEY, PROFILE_KEY, TASK_STATE_KEY
from task_state import STAGE_ORDER, TaskStage, TaskStateError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(page_title="Day 15 — Controlled Transitions", layout="wide")
st.title("Day 15 — Контролируемые переходы состояний")
st.caption(
    "Day 13: у задачи есть состояние · "
    "Day 15: переход только если rules + guards выполнены"
)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
    st.stop()

agent = Agent(api_key=api_key)
task = agent.get_task_state()

left, right = st.columns([2, 1])

with right:
    st.subheader("TASK LIFECYCLE")
    if task is None:
        st.info("Нет активной задачи.")
        goal = st.text_input("Goal", value="Контролируемый lifecycle REST API")
        if st.button("Create task", type="primary"):
            agent.create_task(goal)
            st.rerun()
    else:
        parts = []
        for stage in STAGE_ORDER:
            label = stage.value.capitalize()
            parts.append(f"**[{label}]**" if stage == task.stage else label)
        st.markdown(" → ".join(parts))
        st.markdown(f"**Current stage:** `{task.stage.value.upper()}`")
        st.markdown(f"**Status:** `{task.status.value}`")
        st.markdown(f"**Goal:** {task.goal}")

        st.markdown("**Transition conditions:**")
        for key, val in task.guards_summary().items():
            mark = "✅" if val else "❌"
            st.markdown(f"- {key}: {mark}")
        st.markdown(f"**Expected action:** {task.expected_action or '—'}")

        st.divider()
        st.subheader("Guards / transitions")
        g1, g2, g3 = st.columns(3)
        with g1:
            if st.button("Approve plan"):
                try:
                    agent.approve_plan()
                    st.rerun()
                except TaskStateError as exc:
                    st.error(str(exc))
        with g2:
            if st.button("Exec completed"):
                try:
                    agent.mark_execution_completed()
                    st.rerun()
                except TaskStateError as exc:
                    st.error(str(exc))
        with g3:
            if st.button("Validation passed"):
                try:
                    agent.mark_validation_passed(True)
                    st.rerun()
                except TaskStateError as exc:
                    st.error(str(exc))

        targets = {
            TaskStage.PLANNING: TaskStage.EXECUTION,
            TaskStage.EXECUTION: TaskStage.VALIDATION,
            TaskStage.VALIDATION: TaskStage.DONE,
        }
        nxt = targets.get(task.stage)
        c_fwd, c_fix = st.columns(2)
        with c_fwd:
            label = f"→ {nxt.value}" if nxt else "done"
            if st.button(label, disabled=nxt is None):
                result = agent.transition_task(nxt)
                st.session_state.last_transition = result.message
                if not result.success:
                    st.session_state.last_result_note = result.message
                st.rerun()
        with c_fix:
            if task.stage == TaskStage.VALIDATION:
                if st.button("→ execution (fix)"):
                    result = agent.transition_task(TaskStage.EXECUTION)
                    st.session_state.last_transition = result.message
                    st.rerun()

        p1, p2 = st.columns(2)
        with p1:
            if st.button("Pause"):
                try:
                    agent.pause_task()
                    st.rerun()
                except TaskStateError as exc:
                    st.error(str(exc))
        with p2:
            if st.button("Resume"):
                try:
                    agent.resume_task()
                    st.rerun()
                except TaskStateError as exc:
                    st.error(str(exc))

        if "last_transition" in st.session_state:
            st.code(st.session_state.last_transition, language="text")

    st.divider()
    st.subheader("Demo")
    if st.button("Setup lifecycle demo (planning)", type="primary"):
        agent.setup_lifecycle_demo()
        st.session_state.pop("last_transition", None)
        st.rerun()
    if st.button("Setup invariants demo (execution)"):
        agent.setup_demo_task()
        st.rerun()

    st.divider()
    st.subheader("Memory")
    if st.button("Clear Short-Term"):
        agent.clear_short_term()
        st.rerun()
    if st.button("Очистить всё"):
        agent.clear_all()
        st.session_state.clear()
        st.rerun()

with left:
    st.subheader("Диалог")
    for message in agent.memory.short_term:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input(
        "Напр. «Начинай реализацию» / «План утверждаю» / «Просто поставь DONE»"
    )
    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)
        with st.chat_message("assistant"):
            try:
                result = agent.ask(user_message)
                if result.rejected:
                    st.warning(result.content)
                else:
                    st.markdown(result.content)
                st.session_state.last_result = result
            except AgentError as exc:
                st.error(str(exc))

st.divider()
with st.expander("Prompt preview"):
    st.code(agent.preview_prompt(), language="text")

with st.expander("Working Memory"):
    st.json(agent.memory.working or {"(empty)": True})

if "last_result" in st.session_state and st.session_state.last_result is not None:
    result = st.session_state.last_result
    stats = result.token_stats
    st.subheader("Last response meta")
    st.markdown(
        f"- rejected: **{result.rejected}** · "
        f"tokens: **{stats.total_tokens}** · "
        f"cost: **${stats.estimated_cost:.6f}** "
        f"(${INPUT_PRICE_PER_MILLION}/1M in · ${OUTPUT_PRICE_PER_MILLION}/1M out)"
    )

st.caption(f"Keys: `{TASK_STATE_KEY}` · `{INVARIANTS_KEY}` · `{PROFILE_KEY}`")
