"""Day 03: Compare four reasoning strategies via DeepSeek API."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
THINKING_DISABLED = {"type": "disabled"}

DEFAULT_TASK = """Сервис способен стабильно обрабатывать 1000 запросов в секунду.
В течение 30 секунд на него приходит 5000 запросов в секунду.
Клиенты при таймауте делают до 3 повторных попыток без backoff.
Какие проблемы могут возникнуть и какие меры нужно принять,
чтобы система не ушла в каскадный отказ?"""

META_PROMPT_REQUEST = """Создай самодостаточный промпт для решения следующей аналитической задачи.

Промпт должен:
- содержать полный исходный текст задачи;
- содержать инструкции, необходимые для качественного анализа;
- просить лаконичный структурированный ответ (не более 300 слов);
- быть готовым к отправке другой LLM без дополнительного контекста.

Верни только готовый промпт, без вводных фраз и пояснений.

Задача:
{task}"""

STEP_BY_STEP_SUFFIX = "\n\nРеши задачу пошагово."

CONCISE_SUFFIX = "\n\nОтветь структурированно, не более 300 слов."

MAX_OUTPUT_TOKENS = 600
MAX_ANALYSIS_TOKENS = 800

EXPERTS_SYSTEM_PROMPT = """Ты проводишь экспертный консилиум из трёх независимых ролей.

Аналитик:
- выявляет причинно-следственные связи;
- описывает развитие проблемы;
- выявляет риски.

Инженер:
- предлагает конкретные технические меры;
- объясняет, где именно их применять.

Критик:
- проверяет выводы аналитика и инженера;
- ищет пропущенные сценарии;
- указывает слабые места предложенного решения.

Ответь строго в трёх отдельных разделах с заголовками:

## Аналитик
...

## Инженер
...

## Критик
...

Не объединяй роли в один общий ответ. Весь ответ — не более 300 слов."""

COMPARISON_REQUEST = """Тебе дана аналитическая задача и четыре решения, полученные разными способами рассуждения.

Задача:
{task}

---

## Direct
{direct}

---

## Step-by-step
{step}

---

## Meta-prompting
{meta}

---

## Experts
{experts}

---

Сравни ответы:
1. Чем они отличаются по структуре, глубине и полноте?
2. Какой ответ наиболее точный и полезный с технической точки зрения?
3. Какой способ рассуждения дал лучший результат и почему?

Ответь структурированно, не более 400 слов.
В конце на отдельной строке явно укажи:
Наиболее точный: <Direct|Step-by-step|Meta-prompting|Experts>"""

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


def request_deepseek(
    api_key: str, payload: dict
) -> tuple[dict | None, str | None, float]:
    started_at = time.perf_counter()
    try:
        response = call_deepseek(api_key, payload)
    except requests.RequestException as exc:
        return None, str(exc), time.perf_counter() - started_at

    elapsed = time.perf_counter() - started_at
    if not response.ok:
        return None, f"HTTP {response.status_code}\n{response.text}", elapsed

    return response.json(), None, elapsed


def report_error(label: str, error: str) -> None:
    st.error(f"Запрос «{label}» не выполнен")
    st.code(error)


def fetch_result(api_key: str, payload: dict, label: str) -> dict | None:
    data, error, _elapsed = request_deepseek(api_key, payload)
    if error:
        report_error(label, error)
        return None
    return data


def extract_content(data: dict | None) -> str:
    if not data:
        return "—"
    content = (data["choices"][0]["message"].get("content") or "").strip()
    return content or "—"


def extract_meta_answer(meta: dict | None) -> str:
    if not meta:
        return "—"
    solution_data = meta.get("solution_data")
    return extract_content(solution_data)


def show_metrics(data: dict, elapsed: float | None = None) -> None:
    choice = data["choices"][0]
    if elapsed is not None:
        st.caption(f"время: {elapsed:.1f} с")
    st.caption(f"output tokens: {data['usage']['completion_tokens']}")
    st.caption(f"finish_reason: {choice['finish_reason']}")


def show_answer(content: str) -> None:
    with st.container(height=420, border=True):
        st.markdown(content or "—")


def show_result(data: dict, elapsed: float | None = None) -> None:
    content = extract_content(data)
    show_answer(content)
    show_metrics(data, elapsed)


def base_payload(messages: list[dict]) -> dict:
    return {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "thinking": THINKING_DISABLED,
    }


def task_with_suffix(task: str, extra: str = "") -> str:
    return task + extra + CONCISE_SUFFIX


def finish_meta_prompting(
    api_key: str, prompt_data: dict
) -> tuple[dict | None, str | None, float | None]:
    generated_prompt = (prompt_data["choices"][0]["message"].get("content") or "").strip()
    if not generated_prompt:
        return None, "модель вернула пустой prompt", None

    solution_data, error, elapsed = request_deepseek(
        api_key,
        base_payload([{"role": "user", "content": generated_prompt}]),
    )
    if error:
        return {
            "prompt_data": prompt_data,
            "generated_prompt": generated_prompt,
        }, error, elapsed

    return {
        "prompt_data": prompt_data,
        "generated_prompt": generated_prompt,
        "solution_data": solution_data,
    }, None, elapsed


def run_analysis(
    api_key: str,
    task: str,
    direct_data: dict | None,
    step_data: dict | None,
    meta_data: dict | None,
    experts_data: dict | None,
) -> tuple[dict | None, float | None]:
    answers = [
        extract_content(direct_data),
        extract_content(step_data),
        extract_meta_answer(meta_data),
        extract_content(experts_data),
    ]
    if all(answer == "—" for answer in answers):
        return None, None

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": COMPARISON_REQUEST.format(
                    task=task,
                    direct=answers[0],
                    step=answers[1],
                    meta=answers[2],
                    experts=answers[3],
                ),
            }
        ],
        "max_tokens": MAX_ANALYSIS_TOKENS,
        "thinking": THINKING_DISABLED,
    }
    data, error, elapsed = request_deepseek(api_key, payload)
    if error:
        report_error("Сравнительный анализ", error)
        return None, elapsed
    return data, elapsed


def run_comparison(api_key: str, task: str) -> dict:
    parallel_jobs = {
        "Direct": base_payload([{"role": "user", "content": task_with_suffix(task)}]),
        "Step-by-step": base_payload(
            [{"role": "user", "content": task_with_suffix(task, STEP_BY_STEP_SUFFIX)}]
        ),
        "Meta-prompting — генерация prompt": base_payload(
            [{"role": "user", "content": META_PROMPT_REQUEST.format(task=task)}]
        ),
        "Experts": base_payload(
            [
                {"role": "system", "content": EXPERTS_SYSTEM_PROMPT},
                {"role": "user", "content": task_with_suffix(task)},
            ]
        ),
    }

    started_at = time.perf_counter()
    parallel_results: dict[str, dict] = {}
    timings: dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            label: executor.submit(request_deepseek, api_key, payload)
            for label, payload in parallel_jobs.items()
        }
        for label, future in futures.items():
            data, error, elapsed = future.result()
            timings[label] = elapsed
            if error:
                report_error(label, error)
            elif data:
                parallel_results[label] = data

    meta_data = None
    prompt_data = parallel_results.get("Meta-prompting — генерация prompt")
    if prompt_data:
        meta_data, error, meta_solution_elapsed = finish_meta_prompting(api_key, prompt_data)
        if meta_solution_elapsed is not None:
            timings["Meta-prompting — решение"] = meta_solution_elapsed
        if error:
            if meta_data:
                report_error("Meta-prompting — решение", error)
            else:
                st.error(f"Meta-prompting: {error}")

    direct_data = parallel_results.get("Direct")
    step_data = parallel_results.get("Step-by-step")
    experts_data = parallel_results.get("Experts")

    analysis_data, analysis_elapsed = run_analysis(
        api_key, task, direct_data, step_data, meta_data, experts_data
    )
    if analysis_elapsed is not None:
        timings["Сравнительный анализ"] = analysis_elapsed

    return {
        "direct_data": direct_data,
        "step_data": step_data,
        "meta_data": meta_data,
        "experts_data": experts_data,
        "analysis_data": analysis_data,
        "timings": timings,
        "total_elapsed": time.perf_counter() - started_at,
    }


def show_meta_result(meta: dict, prompt_elapsed: float | None, solution_elapsed: float | None) -> None:
    with st.expander("Сгенерированный prompt"):
        st.code(meta["generated_prompt"])

    prompt_tokens = meta["prompt_data"]["usage"]["completion_tokens"]
    if prompt_elapsed is not None:
        st.caption(f"время (генерация prompt): {prompt_elapsed:.1f} с")
    st.caption(f"output tokens (генерация prompt): {prompt_tokens}")

    solution_data = meta.get("solution_data")
    if not solution_data:
        st.warning("Финальное решение не получено.")
        return

    content = extract_content(solution_data)
    show_answer(content)

    solution_tokens = solution_data["usage"]["completion_tokens"]
    if solution_elapsed is not None:
        st.caption(f"время (решение): {solution_elapsed:.1f} с")
    st.caption(f"output tokens (решение): {solution_tokens}")
    st.caption(f"total output tokens: {prompt_tokens + solution_tokens}")
    st.caption(f"finish_reason: {solution_data['choices'][0]['finish_reason']}")


st.set_page_config(page_title="Day 3 — способы рассуждения", layout="wide")

st.title("Day 3 — способы рассуждения")
st.caption("Все стратегии получают одинаковое ограничение: не более 300 слов, max_tokens=600.")

task = st.text_area("Задача", value=DEFAULT_TASK, height=150)

if st.button("Запустить сравнение"):
    if not task.strip():
        st.error("Введите текст задачи.")
        st.stop()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("DEEPSEEK_API_KEY не задан. Добавьте ключ в .env в корне проекта.")
        st.stop()

    with st.spinner("Выполняются 6 запросов (4 параллельно, Meta-решение, сравнительный анализ)..."):
        results = run_comparison(api_key, task)

    direct_data = results["direct_data"]
    step_data = results["step_data"]
    meta_data = results["meta_data"]
    experts_data = results["experts_data"]
    analysis_data = results["analysis_data"]
    timings = results["timings"]
    elapsed = results["total_elapsed"]

    st.caption(f"Общее время: {elapsed:.1f} с")

    row1_left, row1_right = st.columns(2)
    row2_left, row2_right = st.columns(2)

    with row1_left:
        st.subheader("Direct")
        st.caption("Только исходная задача")
        if direct_data:
            show_result(direct_data, timings.get("Direct"))
        else:
            st.warning("Ответ не получен.")

    with row1_right:
        st.subheader("Step-by-step")
        st.caption("Исходная задача + инструкция решать пошагово")
        if step_data:
            show_result(step_data, timings.get("Step-by-step"))
        else:
            st.warning("Ответ не получен.")

    with row2_left:
        st.subheader("Meta-prompting")
        st.caption("LLM сначала создаёт prompt, затем решает задачу")
        if meta_data:
            show_meta_result(
                meta_data,
                timings.get("Meta-prompting — генерация prompt"),
                timings.get("Meta-prompting — решение"),
            )
        else:
            st.warning("Ответ не получен.")

    with row2_right:
        st.subheader("Experts")
        st.caption("Одна LLM рассматривает задачу в трёх экспертных ролях")
        if experts_data:
            show_result(experts_data, timings.get("Experts"))
        else:
            st.warning("Ответ не получен.")

    st.divider()
    st.subheader("Сравнительный анализ")
    st.caption("DeepSeek сравнивает четыре ответа и выбирает наиболее точный")
    if analysis_data:
        show_answer(extract_content(analysis_data))
        show_metrics(analysis_data, timings.get("Сравнительный анализ"))
    else:
        st.warning("Сравнительный анализ не получен.")
