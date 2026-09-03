"""Day 04: Compare DeepSeek responses at different temperatures."""

import os
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
THINKING_DISABLED = {"type": "disabled"}
TEMPERATURES = (0, 0.7, 1.2)
MAX_ANALYSIS_TOKENS = 1500

DEFAULT_PROMPT = """Объясни junior-разработчику, что такое retry storm.
Дай технически точное определение, одну аналогию из обычной жизни
и один практический пример."""

COLUMN_CAPTIONS = {
    0: "Более предсказуемые и сфокусированные ответы",
    0.7: "Баланс предсказуемости и разнообразия",
    1.2: "Больше вариативности и неожиданных формулировок",
}

COMPARISON_REQUEST = """Тебе дан один и тот же запрос и три ответа,
полученные с разной temperature у одной модели.

Запрос:
{prompt}

---

## Temperature = 0
{answer_0}

---

## Temperature = 0.7
{answer_07}

---

## Temperature = 1.2
{answer_12}

---

Сравни ответы по трём критериям:
1. Точность — насколько определение и пример технически верны.
2. Креативность — насколько живая аналогия и формулировки.
3. Разнообразие — чем ответы отличаются друг от друга по стилю и деталям.

Ответь структурированно, не более 400 слов.
В конце на отдельных строках явно укажи:
Наиболее точный: <0 | 0.7 | 1.2>
Наиболее креативный: <0 | 0.7 | 1.2>
"""

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


def extract_content(data: dict | None) -> str:
    if not data:
        return "—"
    content = (data["choices"][0]["message"].get("content") or "").strip()
    return content or "—"


def show_answer(content: str) -> None:
    with st.container(height=420, border=True):
        st.markdown(content or "—")


def show_metrics(data: dict, temperature: float | None = None, elapsed: float | None = None) -> None:
    choice = data["choices"][0]
    if temperature is not None:
        st.caption(f"temperature: {temperature}")
    if elapsed is not None:
        st.caption(f"время: {elapsed:.1f} с")
    st.caption(f"output tokens: {data['usage']['completion_tokens']}")
    finish_reason = choice["finish_reason"]
    st.caption(f"finish_reason: {finish_reason}")
    if finish_reason == "length":
        st.warning("Ответ обрезан лимитом max_tokens.")


def show_result(data: dict, temperature: float, elapsed: float | None = None) -> None:
    show_answer(extract_content(data))
    show_metrics(data, temperature=temperature, elapsed=elapsed)


def build_payload(prompt: str, temperature: float) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "thinking": THINKING_DISABLED,
    }


def run_analysis(
    api_key: str,
    prompt: str,
    answers: dict[float, str],
) -> tuple[dict | None, float | None]:
    if all(answer == "—" for answer in answers.values()):
        return None, None

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": COMPARISON_REQUEST.format(
                    prompt=prompt,
                    answer_0=answers[0],
                    answer_07=answers[0.7],
                    answer_12=answers[1.2],
                ),
            }
        ],
        "max_tokens": MAX_ANALYSIS_TOKENS,
        "thinking": THINKING_DISABLED,
    }

    started_at = time.perf_counter()
    data = fetch_result(api_key, payload, "Сравнительный анализ")
    elapsed = time.perf_counter() - started_at
    return data, elapsed


st.set_page_config(page_title="Day 4 — влияние температуры", layout="wide")
st.title("Day 4 — влияние температуры")
st.caption(
    "Один и тот же prompt отправляется три раза. "
    "Отличается только temperature; thinking отключён."
)

user_prompt = st.text_area("PROMPT", value=DEFAULT_PROMPT, height=150)

if st.button("Запустить сравнение"):
    if not user_prompt.strip():
        st.error("Введите текст запроса.")
        st.stop()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
        st.stop()

    results: dict[float, dict | None] = {}
    timings: dict[float | str, float] = {}
    started_at = time.perf_counter()

    with st.spinner("Выполняются 4 запроса (3 с разной temperature, затем сравнительный анализ)..."):
        for temperature in TEMPERATURES:
            request_started = time.perf_counter()
            results[temperature] = fetch_result(
                api_key,
                build_payload(user_prompt, temperature),
                f"Temperature = {temperature}",
            )
            timings[temperature] = time.perf_counter() - request_started

        analysis_data, analysis_elapsed = run_analysis(
            api_key,
            user_prompt,
            {temperature: extract_content(results[temperature]) for temperature in TEMPERATURES},
        )
        if analysis_elapsed is not None:
            timings["Сравнительный анализ"] = analysis_elapsed

    total_elapsed = time.perf_counter() - started_at
    st.caption(f"Общее время: {total_elapsed:.1f} с")

    columns = st.columns(3)
    for column, temperature in zip(columns, TEMPERATURES):
        with column:
            st.subheader(f"Temperature = {temperature}")
            st.caption(COLUMN_CAPTIONS[temperature])
            data = results[temperature]
            if data:
                show_result(data, temperature, timings.get(temperature))
            else:
                st.warning("Ответ не получен.")

    st.divider()
    st.subheader("Сравнительный анализ")
    st.caption("DeepSeek сравнивает три ответа по точности, креативности и разнообразию")
    if analysis_data:
        show_answer(extract_content(analysis_data))
        show_metrics(analysis_data, elapsed=timings.get("Сравнительный анализ"))
    else:
        st.warning("Сравнительный анализ не получен.")

    st.divider()
    st.subheader("Выводы")
    st.caption(
        "Общие свойства настройки temperature. "
        "Один запуск не даёт статистического доказательства."
    )
    st.markdown(
        """
| Temperature | Свойства |
|---|---|
| **0** | Более предсказуемые и сфокусированные ответы. Подходит для задач, где важнее стабильность и точность. |
| **0.7** | Баланс предсказуемости и разнообразия. Подходит для большинства обычных текстовых задач. |
| **1.2** | Больше вариативности и неожиданных формулировок. Подходит для brainstorming и творческих задач; выше риск менее точного или лишнего ответа. |
"""
    )
