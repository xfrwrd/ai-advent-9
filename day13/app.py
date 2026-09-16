"""Day 13: Streamlit UI — Task State Machine on Day 11/12 architecture."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import (
    DEMO_GOAL,
    INPUT_PRICE_PER_MILLION,
    OUTPUT_PRICE_PER_MILLION,
    Agent,
    AgentError,
)
from memory import PROFILE_KEY, TASK_STATE_KEY
from task_state import STAGE_ORDER, TaskStage, TaskStateError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(page_title="Day 13 — Task State Machine", layout="wide")
st.title("Day 13 — Состояние задачи (Task State Machine)")
st.caption(
    "Short-Term → диалог · Working → TaskState · Long-Term → UserProfile. "
    "Pause/resume не меняет stage."
)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
    st.stop()

agent = Agent(api_key=api_key)
task = agent.get_task_state()

left, right = st.columns([2, 1])

with right:
    st.subheader("Current Task")
    if task is None:
        st.info("Нет активной задачи.")
        goal = st.text_input("Goal", value=DEMO_GOAL)
        if st.button("Create task", type="primary"):
            try:
                agent.create_task(goal)
                st.success("Задача создана (stage=planning).")
                st.rerun()
            except TaskStateError as exc:
                st.error(str(exc))
    else:
        st.markdown(f"**Goal:** {task.goal}")
        st.markdown(f"**Stage:** `{task.stage.value.upper()}`")
        st.markdown(f"**Status:** `{task.status.value}`")
        st.markdown(f"**Current step:** {task.current_step or '—'}")
        st.markdown(f"**Expected action:** {task.expected_action or '—'}")

        # Pipeline visualization
        parts = []
        for stage in STAGE_ORDER:
            label = stage.value.capitalize()
            if stage == task.stage:
                parts.append(f"**[{label}]**")
            else:
                parts.append(label)
        st.markdown(" → ".join(parts))

        st.divider()
        st.subheader("Steps")
        step = st.text_input("current_step", value=task.current_step)
        action = st.text_input("expected_action", value=task.expected_action)
        if st.button("Update step/action"):
            try:
                agent.update_task_step(step, action)
                st.rerun()
            except TaskStateError as exc:
                st.error(str(exc))

        st.divider()
        st.subheader("Transitions")
        allowed = {
            TaskStage.PLANNING: TaskStage.EXECUTION,
            TaskStage.EXECUTION: TaskStage.VALIDATION,
            TaskStage.VALIDATION: TaskStage.DONE,
        }
        col_fwd, col_back = st.columns(2)
        with col_fwd:
            nxt = allowed.get(task.stage)
            label = f"→ {nxt.value}" if nxt else "done"
            if st.button(label, disabled=nxt is None or task.status.value == "paused"):
                try:
                    agent.transition_task(nxt)
                    st.rerun()
                except TaskStateError as exc:
                    st.error(str(exc))
        with col_back:
            if task.stage == TaskStage.VALIDATION:
                if st.button("→ execution (fix)", disabled=task.status.value == "paused"):
                    try:
                        agent.transition_task(TaskStage.EXECUTION)
                        st.rerun()
                    except TaskStateError as exc:
                        st.error(str(exc))

        st.divider()
        p1, p2 = st.columns(2)
        with p1:
            if st.button("Pause", disabled=task.status.value == "paused"):
                try:
                    agent.pause_task()
                    st.rerun()
                except TaskStateError as exc:
                    st.error(str(exc))
        with p2:
            if st.button("Resume", disabled=task.status.value != "paused"):
                try:
                    agent.resume_task()
                    st.rerun()
                except TaskStateError as exc:
                    st.error(str(exc))

        if st.button("Clear task"):
            agent.clear_task_state()
            st.rerun()

    st.divider()
    st.subheader("Demo scenario")
    st.caption(
        "Create REST API task → execution + step → Pause → "
        "Clear Short-Term → Resume → ask «Продолжай»"
    )
    if st.button("Setup demo (create + execution + step)"):
        try:
            agent.clear_short_term()
            agent.create_task(DEMO_GOAL)
            agent.transition_task(TaskStage.EXECUTION)
            agent.update_task_step(
                "определить API endpoints",
                "описать request/response models",
            )
            st.success("Demo state готов (execution, active).")
            st.rerun()
        except TaskStateError as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Memory controls")
    if st.button("Clear Short-Term"):
        agent.clear_short_term()
        st.info("Short-Term очищен — TaskState в Working остаётся.")
        st.rerun()
    if st.button("Clear Working Memory"):
        agent.clear_working_memory()
        st.warning("Working очищен — TaskState удалён.")
        st.rerun()
    if st.button("Очистить всё"):
        agent.clear_all()
        st.session_state.pop("last_result", None)
        st.rerun()

with left:
    st.subheader("Диалог (Short-Term)")
    for message in agent.memory.short_term:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input("Сообщение… (после pause+clear попробуйте «Продолжай»)")
    if user_message:
        with st.chat_message("user"):
            st.markdown(user_message)
        with st.chat_message("assistant"):
            try:
                result = agent.ask(user_message)
                st.markdown(result.content)
                st.session_state.last_result = result
            except AgentError as exc:
                st.error(str(exc))

st.divider()
st.subheader("Что сейчас попадёт в prompt")
st.code(agent.preview_prompt(), language="text")

with st.expander("Working Memory (включая task_state)"):
    st.json(agent.memory.working or {"(empty)": True})

with st.expander("Long-Term Memory"):
    st.json(agent.memory.long_term or {"(empty)": True})

if "last_result" in st.session_state and st.session_state.last_result is not None:
    stats = st.session_state.last_result.token_stats
    st.divider()
    st.subheader("Token statistics")
    st.markdown(
        f"""
- prompt_tokens: **{stats.prompt_tokens}**
- completion_tokens: **{stats.completion_tokens}**
- total_tokens: **{stats.total_tokens}**
- Estimated cost: **${stats.estimated_cost:.6f}**
  (input ${INPUT_PRICE_PER_MILLION}/1M · output ${OUTPUT_PRICE_PER_MILLION}/1M)
"""
    )

st.caption(f"TaskState key: `{TASK_STATE_KEY}` · Profile key: `{PROFILE_KEY}`")
