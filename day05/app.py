"""Day 05: Compare DeepSeek V4 models of different tiers."""

import os
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

API_URL = "https://api.deepseek.com/chat/completions"
TEMPERATURE = 0.3
TIMEOUT_SECONDS = 60
THINKING_DISABLED = {"type": "disabled"}

# OFF-PEAK prices: USD per 1M tokens
PRICING = {
    "deepseek-v4-flash": {
        "cache_hit": 0.007,
        "cache_miss": 0.22,
        "output": 0.66,
    },
    "deepseek-v4-flash-vision-exp": {
        "cache_hit": 0.007,
        "cache_miss": 0.22,
        "output": 0.66,
    },
    "deepseek-v4-pro": {
        "cache_hit": 0.022,
        "cache_miss": 0.66,
        "output": 1.98,
    },
}

MODELS = (
    {
        "key": "flash",
        "label": "Базовая — DeepSeek V4 Flash",
        "level": "Базовая",
        "id": "deepseek-v4-flash",
    },
    {
        "key": "vision",
        "label": "Экспериментальная — V4 Flash Vision Exp",
        "level": "Экспериментальная",
        "id": "deepseek-v4-flash-vision-exp",
    },
    {
        "key": "pro",
        "label": "Сильная — DeepSeek V4 Pro",
        "level": "Сильная",
        "id": "deepseek-v4-pro",
    },
)

DEFAULT_PROMPT = """Сервис обрабатывает заказы. После сохранения заказа в PostgreSQL
необходимо отправить событие в Kafka. Иногда база успешно сохраняет
заказ, но отправка события завершается ошибкой.
Предложи надёжное решение проблемы и объясни его основные компромиссы."""

ANALYSIS_MODEL = "deepseek-v4-flash"

COMPARISON_REQUEST = """Тебе дан один и тот же запрос и три ответа от разных моделей DeepSeek.

Запрос:
{prompt}

---

## Базовая — DeepSeek V4 Flash
{flash}

---

## Экспериментальная — V4 Flash Vision Exp
{vision}

---

## Сильная — DeepSeek V4 Pro
{pro}

---

Сравни ответы по качеству решения задачи:
1. Насколько верно и полно описано решение (например outbox / transactional messaging).
2. Насколько ясно объяснены компромиссы.
3. Практическая полезность для инженера.
4. Чем ответы отличаются по структуре, глубине и детализации.

Учти: Vision Exp — экспериментальная мультимодальная версия, и на чисто текстовых
задачах её возможности могут быть сопоставимы с Flash. Не предполагай заранее,
что Pro всегда лучше, а Vision всегда слабее — опирайся только на сами тексты.

Ответь структурированно.
В конце на отдельных строках явно укажи:
Наиболее качественный: <Flash | Vision Exp | Pro>
Лучший баланс полноты и ясности: <Flash | Vision Exp | Pro>
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
        timeout=TIMEOUT_SECONDS,
    )


def fetch_result(api_key: str, payload: dict, label: str) -> tuple[dict | None, float]:
    started_at = time.perf_counter()
    try:
        response = call_deepseek(api_key, payload)
    except requests.RequestException as exc:
        elapsed = time.perf_counter() - started_at
        st.error(f"Запрос «{label}» не выполнен")
        st.code(str(exc))
        return None, elapsed

    elapsed = time.perf_counter() - started_at
    if not response.ok:
        st.error(f"Запрос «{label}» не выполнен")
        st.code(f"HTTP {response.status_code}\n{response.text}")
        return None, elapsed

    return response.json(), elapsed


def build_payload(prompt: str, model_id: str) -> dict:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "thinking": THINKING_DISABLED,
    }


def calculate_cost(model_id: str, usage: dict) -> float | None:
    prices = PRICING.get(model_id)
    if not prices:
        return None

    completion_tokens = usage.get("completion_tokens") or 0

    # Prefer explicit cache breakdown when present.
    # If the API omits hit/miss fields, treat all prompt_tokens as cache miss.
    if "prompt_cache_hit_tokens" in usage or "prompt_cache_miss_tokens" in usage:
        cache_hit_tokens = usage.get("prompt_cache_hit_tokens") or 0
        cache_miss_tokens = usage.get("prompt_cache_miss_tokens") or 0
    else:
        cache_hit_tokens = 0
        cache_miss_tokens = usage.get("prompt_tokens") or 0

    return (
        (cache_hit_tokens * prices["cache_hit"] / 1_000_000)
        + (cache_miss_tokens * prices["cache_miss"] / 1_000_000)
        + (completion_tokens * prices["output"] / 1_000_000)
    )


def format_cost(cost: float | None) -> str:
    if cost is None:
        return "unavailable"
    return f"${cost:.6f}"


def extract_content(data: dict | None) -> str:
    if not data:
        return "—"
    content = (data["choices"][0]["message"].get("content") or "").strip()
    return content or "—"


def show_answer(content: str) -> None:
    with st.container(height=420, border=True):
        st.markdown(content or "—")


def show_result(data: dict, model_id: str, elapsed: float) -> None:
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    cost = calculate_cost(model_id, usage)

    show_answer(extract_content(data))
    st.caption(f"model: {model_id}")
    st.caption(f"Время ответа API: {elapsed:.2f} с")
    st.caption(f"prompt_tokens: {usage.get('prompt_tokens', '—')}")
    st.caption(f"completion_tokens: {usage.get('completion_tokens', '—')}")
    st.caption(f"total_tokens: {usage.get('total_tokens', '—')}")
    st.caption(f"prompt_cache_hit_tokens: {usage.get('prompt_cache_hit_tokens', '—')}")
    st.caption(f"prompt_cache_miss_tokens: {usage.get('prompt_cache_miss_tokens', '—')}")
    st.caption(f"Расчётная стоимость по тарифу OFF-PEAK: {format_cost(cost)}")
    st.caption(f"finish_reason: {choice.get('finish_reason', '—')}")


def run_analysis(
    api_key: str,
    prompt: str,
    answers: dict[str, str],
) -> tuple[dict | None, float | None]:
    if all(answer == "—" for answer in answers.values()):
        return None, None

    payload = {
        "model": ANALYSIS_MODEL,
        "messages": [
            {
                "role": "user",
                "content": COMPARISON_REQUEST.format(
                    prompt=prompt,
                    flash=answers["flash"],
                    vision=answers["vision"],
                    pro=answers["pro"],
                ),
            }
        ],
        "temperature": TEMPERATURE,
        "thinking": THINKING_DISABLED,
    }
    return fetch_result(api_key, payload, "Сравнительный анализ")


st.set_page_config(page_title="Day 5 — сравнение моделей", layout="wide")
st.title("Day 5 — сравнение моделей")
st.caption(
    "Один и тот же текстовый prompt на трёх моделях DeepSeek. "
    f"Одинаковые temperature={TEMPERATURE}, thinking=disabled; "
    "отличается только model. Без max_tokens."
)

user_prompt = st.text_area("PROMPT", value=DEFAULT_PROMPT, height=150)

if st.button("Сравнить модели"):
    if not user_prompt.strip():
        st.error("Введите текст запроса.")
        st.stop()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
        st.stop()

    results: dict[str, dict | None] = {}
    timings: dict[str, float] = {}

    with st.spinner("Выполняются 4 запроса к DeepSeek (3 модели + сравнительный анализ)..."):
        for model in MODELS:
            data, elapsed = fetch_result(
                api_key,
                build_payload(user_prompt, model["id"]),
                model["label"],
            )
            results[model["key"]] = data
            timings[model["key"]] = elapsed

        analysis_data, analysis_elapsed = run_analysis(
            api_key,
            user_prompt,
            {
                model["key"]: extract_content(results[model["key"]])
                for model in MODELS
            },
        )
        if analysis_elapsed is not None:
            timings["analysis"] = analysis_elapsed

    if not any(results.values()):
        st.stop()

    columns = st.columns(3)
    for column, model in zip(columns, MODELS):
        with column:
            st.subheader(model["label"])
            data = results[model["key"]]
            if data:
                show_result(data, model["id"], timings[model["key"]])
            else:
                st.warning("Ответ не получен.")
                st.caption(f"Время ответа API: {timings[model['key']]:.2f} с")

    comparison_rows = []
    for model in MODELS:
        data = results[model["key"]]
        usage = (data or {}).get("usage") or {}
        cost = calculate_cost(model["id"], usage) if data else None
        comparison_rows.append(
            {
                "уровень": model["level"],
                "модель": model["id"],
                "время ответа": f"{timings[model['key']]:.2f} с",
                "input tokens": usage.get("prompt_tokens", "—"),
                "output tokens": usage.get("completion_tokens", "—"),
                "total tokens": usage.get("total_tokens", "—"),
                "estimated cost": format_cost(cost),
            }
        )

    st.divider()
    st.subheader("Сравнение")
    st.caption("Расчётная стоимость по тарифу OFF-PEAK")
    st.dataframe(comparison_rows, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Сравнительный анализ")
    st.caption(f"DeepSeek ({ANALYSIS_MODEL}) сравнивает три ответа по качеству решения")
    if analysis_data:
        show_answer(extract_content(analysis_data))
        st.caption(f"model: {ANALYSIS_MODEL}")
        st.caption(f"Время ответа API: {timings.get('analysis', 0):.2f} с")
        usage = analysis_data.get("usage") or {}
        st.caption(f"prompt_tokens: {usage.get('prompt_tokens', '—')}")
        st.caption(f"completion_tokens: {usage.get('completion_tokens', '—')}")
        st.caption(f"total_tokens: {usage.get('total_tokens', '—')}")
        st.caption(
            f"Расчётная стоимость по тарифу OFF-PEAK: "
            f"{format_cost(calculate_cost(ANALYSIS_MODEL, usage))}"
        )
        st.caption(f"finish_reason: {analysis_data['choices'][0].get('finish_reason', '—')}")
    else:
        st.warning("Сравнительный анализ не получен.")

    st.divider()
    st.subheader("Вывод")
    st.markdown(
        """
- Базовая Flash обычно быстрее и дешевле — может хватить для простых задач.
- Vision Exp — экспериментальная мультимодальная версия; на чисто текстовых
  задачах её возможности примерно сопоставимы с V4 Flash, поэтому заранее
  не стоит считать её сильнее.
- Pro потенциально лучше справляется со сложным анализом, но размер/тариф
  сами по себе не гарантируют лучший ответ.
- Скорость и стоимость зависят от модели, кэша prompt и загрузки API.

После запуска скорректируй этот вывод по реальным результатам и
сравнительному анализу выше: какой ответ оказался полезнее,
где быстрее API, где дешевле запрос.
"""
    )
