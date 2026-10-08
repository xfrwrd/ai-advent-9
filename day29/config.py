"""Baseline is the Day 28 Ollama call. Optimized is a separate experiment."""

from __future__ import annotations

from dataclasses import dataclass

BASELINE = "baseline"
OPTIMIZED = "optimized"
MODES = (BASELINE, OPTIMIZED)

MODEL = "qwen3:8b"
BASE_URL = "http://localhost:11434"

# Read from Ollama 0.35.1 on this machine, 2026-10-08. Not guessed.
# Modelfile parameters: temperature 0.6, top_k 20, top_p 0.95, repeat_penalty 1.
# num_ctx and num_predict are absent there. Model thinking default is true.
# Day 28 local_chat sends think=false and no options, so those Modelfile
# values stay in force and the server fills the missing options.
MODEL_FACTS = {
    "name": "qwen3:8b",
    "size_bytes": 5225388164,
    "format": "gguf",
    "family": "qwen3",
    "parameter_size": "8.2B",
    "quantization_level": "Q4_K_M",
    "context_length": 40960,
    "thinking_default": True,
    "thinking_values": [False, True],
    "modelfile_temperature": 0.6,
    "modelfile_top_k": 20,
    "modelfile_top_p": 0.95,
    "modelfile_repeat_penalty": 1,
    "modelfile_sets_num_ctx": False,
    "modelfile_sets_num_predict": False,
    "ollama_version": "0.35.1",
    "ollama_context_length_env": None,
    "installed_variants": ["qwen3:8b"],
}


@dataclass(frozen=True)
class GenerateConfig:
    name: str
    model: str
    think: bool
    temperature: float | None
    num_predict: int | None
    num_ctx: int | None
    prompt_name: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": self.model,
            "think": self.think,
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            "prompt_name": self.prompt_name,
        }

    def options(self) -> dict[str, object]:
        options: dict[str, object] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        return options


# Day 28 local_chat: think=False, options omitted.
BASELINE_CONFIG = GenerateConfig(
    name=BASELINE,
    model=MODEL,
    think=False,
    temperature=None,
    num_predict=None,
    num_ctx=None,
    prompt_name="day24_grounded",
)

# Starting point for the experiment, not a measured optimum.
OPTIMIZED_CONFIG = GenerateConfig(
    name=OPTIMIZED,
    model=MODEL,
    think=False,
    temperature=0.1,
    num_predict=512,
    num_ctx=4096,
    prompt_name="day29_rag",
)

CONFIGS = {BASELINE: BASELINE_CONFIG, OPTIMIZED: OPTIMIZED_CONFIG}


def request_body(config: GenerateConfig, prompt: str) -> dict[str, object]:
    body: dict[str, object] = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "think": config.think,
    }
    options = config.options()
    if options:
        body["options"] = options
    return body
