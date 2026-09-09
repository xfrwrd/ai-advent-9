"""Day 08: Streamlit UI — history persistence + token/cost visibility."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import (
    HISTORY_PATH,
    INPUT_PRICE_PER_MILLION,
    MAX_CONTEXT_TOKENS,
    OUTPUT_PRICE_PER_MILLION,
    Agent,
    AgentError,
    estimate_messages_tokens,
    estimate_tokens,
)
from demo_helpers import inflate_history_file

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Low limit only for safe overflow demo — not the real model window.
DEMO_OVERFLOW_LIMIT = 1_500

st.set_page_config(page_title="Day 8 — Токены и контекст", layout="centered")
st.title("Day 8 — Токены и контекст")
st.caption(
    "UI → Agent → DeepSeek API. "
    "История в history.json. Показываем рост токенов, стоимость и лимит контекста."
)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
    st.stop()

if "demo_max_context" not in st.session_state:
    st.session_state.demo_max_context = MAX_CONTEXT_TOKENS

agent = Agent(
    api_key=api_key,
    model="deepseek-v4-flash",
    max_context_tokens=st.session_state.demo_max_context,
)

col_clear, col_info = st.columns([1, 2])
with col_clear:
    if st.button("Очистить историю"):
        agent.clear_history()
        st.session_state.demo_max_context = MAX_CONTEXT_TOKENS
        st.session_state.pop("last_token_stats", None)
        st.rerun()

with col_info:
    st.caption(
        f"Лимит контекста агента: {agent.max_context_tokens:,} tokens · "
        f"официальное окно модели: {MAX_CONTEXT_TOKENS:,}"
    )

with st.expander("Тесты Day 8 (A / B / C)"):
    st.markdown(
        """
**A. Short dialogue** — просто напиши 2–3 коротких сообщения в чат.

**B. Long dialogue** — кнопка ниже раздувает `history.json` синтетическими
парами сообщений без вызова API.

**C. Context overflow** — включает демо-лимит и раздувает историю так,
чтобы локальная проверка сработала **без** дорогого запроса в API.
"""
    )

    if st.button("B: Раздуть историю (~20 пар)"):
        estimated = inflate_history_file(HISTORY_PATH, pairs=20, chars_per_message=400)
        st.success(f"История раздута. Оценка: ~{estimated} tokens (estimate).")
        st.rerun()

    if st.button("C: Демо overflow (лимит 1500, без API)"):
        st.session_state.demo_max_context = DEMO_OVERFLOW_LIMIT
        inflate_history_file(HISTORY_PATH, pairs=30, chars_per_message=500)
        st.warning(
            f"Демо-лимит = {DEMO_OVERFLOW_LIMIT}. История раздута. "
            "Отправь любое сообщение — ожидается AgentError без API-вызова."
        )
        st.rerun()

    st.caption(
        "Если убрать локальную защиту и отправить слишком большой prompt в API, "
        "DeepSeek обычно вернёт HTTP 400 с ошибкой про превышение context length "
        "(maximum context length / context window exceeded)."
    )

st.subheader("Диалог")
for message in agent.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_message = st.chat_input("Введите запрос...")

if user_message:
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        try:
            result = agent.ask(user_message)
            st.markdown(result.content)
            st.session_state.last_token_stats = result.token_stats
        except AgentError as exc:
            st.error(str(exc))

if "last_token_stats" in st.session_state:
    stats = st.session_state.last_token_stats
    st.divider()
    st.subheader("Токены последнего запроса")
    st.caption(
        "current message / history — приблизительная оценка (`estimate_tokens`). "
        "Input API / ответ / всего — фактический `usage` из ответа DeepSeek."
    )
    st.markdown(
        f"""
- Текущий запрос: **{stats.current_message_tokens}** tokens *(estimate)*
- История: **{stats.history_tokens}** tokens *(estimate)*
- Input API: **{stats.prompt_tokens}** tokens *(usage)*
- Ответ: **{stats.completion_tokens}** tokens *(usage)*
- Всего: **{stats.total_tokens}** tokens *(usage)*
- Estimated cost (cache miss, off-peak): **${stats.estimated_cost:.6f}**
  (input ${INPUT_PRICE_PER_MILLION}/1M · output ${OUTPUT_PRICE_PER_MILLION}/1M)
"""
    )
    st.caption("Оценка стоимости по off-peak cache-miss тарифу")

st.divider()
st.subheader("Накопительная статистика диалога")
st.markdown(
    f"""
- Cumulative tokens: **{agent.cumulative.cumulative_tokens}**
- Cumulative cost: **${agent.cumulative.cumulative_cost:.6f}**
- Текущая история (estimate): **~{estimate_messages_tokens(agent.messages)}** tokens
"""
)

if user_message:
    st.caption(
        f"Оценка текущего user-сообщения: ~{estimate_tokens(user_message)} tokens"
    )
