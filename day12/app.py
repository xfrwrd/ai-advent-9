"""Day 12: Streamlit UI — personalization on top of memory layers."""

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
from memory import PROFILE_KEY
from profile import (
    CONSTRAINT_OPTIONS,
    DEMO_PROMPT,
    EXPERTISE_OPTIONS,
    FORMAT_OPTIONS,
    IDEMPOTENCY_PROMPT,
    STYLE_OPTIONS,
    UserProfile,
    profile_a_technical,
    profile_b_beginner,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(page_title="Day 12 — Personalization", layout="wide")
st.title("Day 12 — Персонализация ассистента")
st.caption(
    "Memory = WHAT to remember · Profile = HOW to respond. "
    "Профиль в Long-Term Memory, автоматически в каждом LLM request."
)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
    st.stop()

agent = Agent(api_key=api_key)
profile = agent.get_profile()

left, right = st.columns([2, 1])

with right:
    st.subheader("Active profile")
    for line in profile.summary_lines():
        st.markdown(f"- {line}")

    st.divider()
    st.subheader("User Profile")
    name = st.text_input("Name", value=profile.name or "")
    language = st.selectbox(
        "Language",
        options=["ru", "en"],
        index=0 if profile.language == "ru" else 1,
    )
    style = st.selectbox(
        "Style",
        options=list(STYLE_OPTIONS),
        index=list(STYLE_OPTIONS).index(profile.style)
        if profile.style in STYLE_OPTIONS
        else 0,
    )
    fmt = st.selectbox(
        "Format",
        options=list(FORMAT_OPTIONS),
        index=list(FORMAT_OPTIONS).index(profile.format)
        if profile.format in FORMAT_OPTIONS
        else 0,
    )
    expertise = st.selectbox(
        "Expertise",
        options=list(EXPERTISE_OPTIONS),
        index=list(EXPERTISE_OPTIONS).index(profile.expertise_level)
        if profile.expertise_level in EXPERTISE_OPTIONS
        else 0,
    )
    constraints = st.multiselect(
        "Constraints",
        options=list(CONSTRAINT_OPTIONS),
        default=[c for c in profile.constraints if c in CONSTRAINT_OPTIONS],
    )

    if st.button("Save profile", type="primary"):
        agent.save_profile(
            UserProfile(
                name=name.strip() or None,
                language=language,
                style=style,
                format=fmt,
                expertise_level=expertise,
                constraints=list(constraints),
            )
        )
        st.success("Профиль сохранён в Long-Term Memory.")
        st.rerun()

    st.divider()
    st.subheader("Demo profiles")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load Profile A (Technical)"):
            agent.save_profile(profile_a_technical())
            agent.clear_short_term()
            st.session_state.pop("compare_a", None)
            st.session_state.pop("compare_b", None)
            st.rerun()
    with c2:
        if st.button("Load Profile B (Beginner)"):
            agent.save_profile(profile_b_beginner())
            agent.clear_short_term()
            st.rerun()

    st.divider()
    st.subheader("Same prompt experiment")
    st.code(DEMO_PROMPT, language="text")
    if st.button("Ask DEMO_PROMPT (current profile)"):
        try:
            result = agent.ask(DEMO_PROMPT)
            st.session_state.last_result = result
            active = agent.get_profile()
            if active.name == "Ксения":
                st.session_state.compare_a = result.content
            elif active.name == "Алекс":
                st.session_state.compare_b = result.content
        except AgentError as exc:
            st.error(str(exc))
        st.rerun()

    if st.button("Ask idempotency (auto personalization)"):
        try:
            result = agent.ask(IDEMPOTENCY_PROMPT)
            st.session_state.last_result = result
        except AgentError as exc:
            st.error(str(exc))
        st.rerun()

    st.divider()
    st.subheader("Persistence check")
    if st.button("Clear Short-Term"):
        agent.clear_short_term()
        st.info("Short-Term очищен — профиль в Long-Term остаётся.")
        st.rerun()
    if st.button("Clear Working Memory"):
        agent.clear_working_memory()
        st.info("Working очищен — профиль в Long-Term остаётся.")
        st.rerun()
    if st.button("Очистить всё"):
        agent.clear_all()
        st.session_state.pop("last_result", None)
        st.session_state.pop("compare_a", None)
        st.session_state.pop("compare_b", None)
        st.rerun()

with left:
    st.subheader("Диалог (Short-Term)")
    for message in agent.memory.short_term:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input("Введите запрос (без «ответь коротко» — профиль сам)...")
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
st.subheader("Сравнение Profile A vs B")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Profile A (Ксения / technical)**")
    st.write(st.session_state.get("compare_a") or "— загрузите A и нажмите DEMO_PROMPT")
with col_b:
    st.markdown("**Profile B (Алекс / beginner)**")
    st.write(st.session_state.get("compare_b") or "— загрузите B и нажмите DEMO_PROMPT")

st.divider()
st.subheader("Что сейчас попадёт в prompt")
st.code(agent.preview_prompt(), language="text")

with st.expander("Long-Term raw (включая user_profile)"):
    st.json(agent.memory.long_term)

with st.expander("Working Memory"):
    st.json(agent.memory.working or {"(empty)": True})

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

st.caption(f"Profile key in Long-Term: `{PROFILE_KEY}`")
