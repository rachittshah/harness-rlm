"""Direct LLM client used by harness modules.

This wraps the Anthropic SDK with the same pricing/audit-log surface as the MCP
server, but is callable in-process — modules don't need to be inside an MCP
client to use it. The MCP server (`mcp_server.py`) and this client share
`compute_cost` and the JSONL audit log path so a single session can mix the two
without losing trace.

Configuration is two-tier:
  1. A global default LM via `configure(lm=...)` (DSPy-style ergonomic).
  2. Per-module overrides via `Module(..., lm=...)`.

Sub-LLM calls (cheap, parallel-friendly) and root-LLM calls (expensive, single)
both go through this class — the cost difference is in the `model` field.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from harness_rlm.models import (
    DEFAULT_MODEL,
    LLMQueryRequest,
    LLMQueryResponse,
    SubCallLog,
)

# Mirror the MCP server's pricing table — these are the source of truth.
PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
}
_FALLBACK_PRICING = PRICING["claude-haiku-4-5"]

SUB_CALLS_LOG = Path("/tmp/rlm/sub_calls.jsonl")


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for one completion. Unknown models fall back to Haiku rates."""
    rates = _FALLBACK_PRICING
    for prefix, r in PRICING.items():
        if model.startswith(prefix):
            rates = r
            break
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# LM client
# ---------------------------------------------------------------------------
@dataclass
class LMResult:
    """Result of one LM call. The `text` field is the parsed string output."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    raw: Any = None  # Hold the SDK Message for debugging if callers want it.
    thinking: str = ""  # extended-thinking content (Claude 4 thinking blocks)
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str = ""


class LM:
    """Anthropic LM client. Stateless — safe to share across threads.

    Args:
        model:      Anthropic model ID (default: Haiku 4.5 — cheap default).
        max_tokens: Default ceiling on output tokens per call.
        api_key:    Override for ANTHROPIC_API_KEY (mostly for tests).
        log_path:   Where to append a SubCallLog jsonl line per call. Set to
                    None to disable disk logging entirely.

    Cost and latency are tracked per call AND in cumulative counters on the
    instance so an orchestrator can report run-level totals without scraping
    the audit log.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        api_key: str | None = None,
        log_path: Path | None = SUB_CALLS_LOG,
        *,
        enable_caching: bool = False,
        thinking_budget: int | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.log_path = log_path
        # SOTA features.
        self.enable_caching = enable_caching
        self.thinking_budget = thinking_budget  # None = no thinking; int = budget tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set and no api_key passed. "
                "Export it or pass api_key=... to LM()."
            )
        self._client = anthropic.Anthropic(api_key=self._api_key)
        # Cumulative counters (thread-safe — calls may be parallel).
        self._lock = threading.Lock()
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0

    def __call__(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        enable_caching: bool | None = None,
        thinking_budget: int | None = None,
    ) -> LMResult:
        """One round-trip. Logs cost. Updates cumulative counters.

        Args (in addition to prompt/system/model/max_tokens):
            enable_caching:   Per-call override for prompt caching. When True
                              and `system` is set, the system block is sent as
                              a cached content block (saves ~90% on hits).
            thinking_budget:  Per-call override for extended-thinking budget
                              (Claude 4 only). Token count for thinking; the
                              thinking content lands in LMResult.thinking.
        """
        used_model = model or self.model
        used_max = max_tokens or self.max_tokens
        use_cache = enable_caching if enable_caching is not None else self.enable_caching
        use_thinking = thinking_budget if thinking_budget is not None else self.thinking_budget

        kwargs: dict[str, Any] = {
            "model": used_model,
            "max_tokens": used_max,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            if use_cache:
                from harness_rlm.caching import cached_system_block

                kwargs["system"] = cached_system_block(system)
            else:
                kwargs["system"] = system
        if use_thinking is not None and use_thinking > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": int(use_thinking)}
            # Thinking requires temperature=1 + max_tokens > budget.
            kwargs["max_tokens"] = max(used_max, int(use_thinking) + 256)

        t0 = time.perf_counter()
        msg = self._client.messages.create(**kwargs)
        latency = time.perf_counter() - t0

        # Pull text + thinking blocks separately.
        parts: list[str] = []
        thinking_parts: list[str] = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                tt = getattr(block, "thinking", None) or getattr(block, "text", None)
                if tt:
                    thinking_parts.append(tt)
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        text = "".join(parts)
        thinking_text = "".join(thinking_parts)

        in_tok = msg.usage.input_tokens
        out_tok = msg.usage.output_tokens
        cache_read = getattr(msg.usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(msg.usage, "cache_creation_input_tokens", 0) or 0
        cost = compute_cost(used_model, in_tok, out_tok)

        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += in_tok
            self.total_output_tokens += out_tok
            self.total_cost_usd += cost
            self.total_cache_read_tokens += cache_read
            self.total_cache_write_tokens += cache_write

        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                line = SubCallLog(
                    timestamp=_now_iso(),
                    prompt_preview=prompt[:200],
                    response_chars=len(text),
                    model=used_model,
                    cost_usd=cost,
                ).model_dump_json()
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                # Don't fail a call because the log dir is read-only.
                pass

        return LMResult(
            text=text,
            model=used_model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            latency_s=latency,
            raw=msg,
            thinking=thinking_text,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            stop_reason=getattr(msg, "stop_reason", "") or "",
        )

    # ---- Convenience: structured request via existing pydantic model -------
    def query(self, req: LLMQueryRequest) -> LLMQueryResponse:
        """Run an LLMQueryRequest and return the existing LLMQueryResponse shape.

        Useful for callers that already speak the MCP-server's request/response
        pydantic models — gives them an in-process path with the same contract.
        """
        result = self(
            req.prompt,
            system=req.system,
            model=req.model,
            max_tokens=req.max_tokens,
        )
        return LLMQueryResponse(
            content=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            model=result.model,
            cost_usd=result.cost_usd,
        )

    def stats(self) -> dict[str, Any]:
        """Cumulative usage snapshot (for orchestrator telemetry)."""
        with self._lock:
            return {
                "calls": self.total_calls,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "cost_usd": round(self.total_cost_usd, 6),
                "cache_read_tokens": self.total_cache_read_tokens,
                "cache_write_tokens": self.total_cache_write_tokens,
            }


# ---------------------------------------------------------------------------
# Global default LM (DSPy-style configure/get).
# ---------------------------------------------------------------------------
@dataclass
class _Settings:
    lm: LM | None = field(default=None)


_settings = _Settings()


def configure(*, lm: LM | None = None) -> None:
    """Set the global default LM that modules will use if none is passed."""
    if lm is not None:
        _settings.lm = lm


def get_lm() -> LM:
    """Return the global LM, lazily constructing a Haiku default if unset."""
    if _settings.lm is None:
        _settings.lm = LM()
    return _settings.lm


__all__ = [
    "LM",
    "LMResult",
    "configure",
    "get_lm",
    "compute_cost",
    "SUB_CALLS_LOG",
]
