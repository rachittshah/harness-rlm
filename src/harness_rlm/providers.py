"""Multi-provider LM clients.

Three providers, each callable like the base `LM`:
  - `OpenAILM`    — direct openai SDK (gpt-5, gpt-5-mini, o-series)
  - `LiteLLMLM`   — LiteLLM shim → 100+ providers (OpenRouter, Bedrock,
                    Vertex, Groq, Fireworks, Mistral, local Ollama, ...)
  - `ClaudeCLILM` — `claude -p` shell-out (already in claude_cli_lm.py)

All three share the `LM` call signature so modules can swap providers
without code changes. They also expose `total_calls` / `total_cost_usd`
counters for orchestrator telemetry parity.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from harness_rlm.llm import LMResult


# ---------------------------------------------------------------------------
# OpenAILM — direct openai SDK
# ---------------------------------------------------------------------------
# Approximate pricing per 1M tokens (May 2026 list prices).
OPENAI_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.2": {"input": 5.0, "output": 25.0},
    "gpt-5.2-mini": {"input": 1.0, "output": 4.0},
    "gpt-5.2-nano": {"input": 0.3, "output": 1.2},
    "gpt-5": {"input": 5.0, "output": 25.0},
    "gpt-5-mini": {"input": 1.0, "output": 4.0},
    "o3": {"input": 15.0, "output": 60.0},
    "o3-mini": {"input": 3.0, "output": 12.0},
}
_OPENAI_FALLBACK = OPENAI_PRICING["gpt-5-mini"]


def _openai_cost(model: str, in_tok: int, out_tok: int) -> float:
    rates = _OPENAI_FALLBACK
    # Match by longest prefix so "gpt-5-mini" picks up the mini rate before "gpt-5".
    for prefix in sorted(OPENAI_PRICING.keys(), key=len, reverse=True):
        if model.startswith(prefix):
            rates = OPENAI_PRICING[prefix]
            break
    return (in_tok * rates["input"] + out_tok * rates["output"]) / 1_000_000.0


@dataclass
class OpenAILM:
    """OpenAI SDK provider. Drop-in for LM.

    Args:
        model:             OpenAI model ID (e.g. "gpt-5", "gpt-5-mini", "o3").
        max_tokens:        Default `max_output_tokens`. Some o-series models
                           ignore this — they have hidden reasoning budgets.
        api_key:           Override for OPENAI_API_KEY.
        reasoning_effort:  For o-series + gpt-5: "minimal" | "low" | "medium" | "high".
        base_url:          Optional override (Azure, custom proxies, local routers).
    """

    model: str = "gpt-5-mini"
    max_tokens: int = 1024
    api_key: str | None = None
    reasoning_effort: str | None = None
    base_url: str | None = None
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai SDK not installed. `uv add openai`.") from e

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set and api_key not provided.")
        kwargs: dict[str, Any] = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        self._lock = threading.Lock()

    def __call__(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LMResult:
        used_model = model or self.model
        used_max = max_tokens or self.max_tokens
        used_effort = reasoning_effort or self.reasoning_effort

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": used_model,
            "messages": messages,
            "max_completion_tokens": used_max,
        }
        if used_effort and used_model.startswith(("o", "gpt-5")):
            kwargs["reasoning_effort"] = used_effort

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0

        text = (resp.choices[0].message.content or "").strip()
        in_tok = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens
        cost = _openai_cost(used_model, in_tok, out_tok)

        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += in_tok
            self.total_output_tokens += out_tok
            self.total_cost_usd += cost

        return LMResult(
            text=text,
            model=used_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            latency_s=latency,
            stop_reason=resp.choices[0].finish_reason or "",
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.total_calls,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "cost_usd": round(self.total_cost_usd, 6),
            }


# ---------------------------------------------------------------------------
# LiteLLMLM — universal multi-provider shim
# ---------------------------------------------------------------------------
@dataclass
class LiteLLMLM:
    """LiteLLM provider — same call surface, 100+ backends.

    Set `model` to a LiteLLM-formatted ID:
      - openai/gpt-5.2
      - anthropic/claude-haiku-4-5-20251001
      - openrouter/qwen/qwen-3-72b
      - bedrock/anthropic.claude-haiku-4-5
      - ollama/llama3.3:70b   (local)
      - groq/llama-3.3-70b
      - fireworks_ai/llama-v3p3-70b-instruct

    Auth is via env vars per LiteLLM's docs (OPENAI_API_KEY, OPENROUTER_API_KEY,
    AWS_REGION + AWS profile, OLLAMA_BASE_URL, etc.).
    """

    model: str = "openai/gpt-5-mini"
    max_tokens: int = 1024
    num_retries: int = 0  # rule: don't retry — surface failures clearly
    api_base: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise RuntimeError("litellm not installed. `uv add litellm`.") from e
        self._lock = threading.Lock()

    def __call__(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LMResult:
        import litellm

        used_model = model or self.model
        used_max = max_tokens or self.max_tokens

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": used_model,
            "messages": messages,
            "max_tokens": used_max,
            "num_retries": self.num_retries,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        kwargs.update(self.extra_kwargs)

        t0 = time.perf_counter()
        resp = litellm.completion(**kwargs)
        latency = time.perf_counter() - t0

        text = (resp.choices[0].message.content or "").strip()
        in_tok = getattr(resp.usage, "prompt_tokens", 0) or 0
        out_tok = getattr(resp.usage, "completion_tokens", 0) or 0
        # LiteLLM exposes a cost helper.
        try:
            cost = float(litellm.completion_cost(completion_response=resp))
        except Exception:  # noqa: BLE001
            cost = 0.0

        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += in_tok
            self.total_output_tokens += out_tok
            self.total_cost_usd += cost

        return LMResult(
            text=text,
            model=used_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            latency_s=latency,
            stop_reason=resp.choices[0].finish_reason or "",
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.total_calls,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "cost_usd": round(self.total_cost_usd, 6),
            }


__all__ = ["OpenAILM", "LiteLLMLM"]
