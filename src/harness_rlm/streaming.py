"""Streaming + async LM support.

Two pieces:

  stream(...)       — sync generator yielding StreamEvent objects as the LM
                      produces output. Useful for CLIs, live logging,
                      progressive UI.

  AsyncLM           — asyncio LM client. Same call surface as `LM` but
                      `await lm(prompt)` instead of `lm(prompt)`. Internally
                      uses anthropic.AsyncAnthropic.

Event shapes:
    StreamEvent(kind="text_delta", text="...")     — partial text
    StreamEvent(kind="thinking_delta", text="...") — extended-thinking content
    StreamEvent(kind="done", result=LMResult)      — final result + totals

The sync `stream()` works against the regular `LM` instance (no separate
streaming client needed). It collects deltas internally and ALSO updates the
LM's cumulative counters, so cost reporting stays consistent.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Generator

import anthropic

from harness_rlm.llm import LM, LMResult, compute_cost
from harness_rlm.models import DEFAULT_MODEL


@dataclass
class StreamEvent:
    """One event emitted during a streaming call."""

    kind: str  # "text_delta" | "thinking_delta" | "tool_use_start" | "done"
    text: str = ""
    result: LMResult | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def stream(
    lm: LM,
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> Generator[StreamEvent, None, None]:
    """Stream an LM call, yielding StreamEvents as the model produces output.

    Updates the LM's cumulative cost/token counters once the stream finishes,
    so the underlying `lm.stats()` stays accurate.
    """
    used_model = model or lm.model
    used_max = max_tokens or lm.max_tokens

    kwargs: dict[str, Any] = {
        "model": used_model,
        "max_tokens": used_max,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system is not None:
        kwargs["system"] = system

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    t0 = time.perf_counter()
    with lm._client.messages.stream(**kwargs) as response:
        for event in response:
            etype = getattr(event, "type", "")
            if etype == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta is None:
                    continue
                dtype = getattr(delta, "type", "")
                if dtype == "text_delta":
                    txt = getattr(delta, "text", "")
                    text_parts.append(txt)
                    yield StreamEvent(kind="text_delta", text=txt)
                elif dtype == "thinking_delta":
                    txt = getattr(delta, "thinking", "")
                    thinking_parts.append(txt)
                    yield StreamEvent(kind="thinking_delta", text=txt)
            elif etype == "message_stop":
                # Final message available on response.get_final_message().
                final = response.get_final_message()
                latency = time.perf_counter() - t0
                in_tok = final.usage.input_tokens
                out_tok = final.usage.output_tokens
                cost = compute_cost(used_model, in_tok, out_tok)
                # Update LM counters atomically.
                with lm._lock:
                    lm.total_calls += 1
                    lm.total_input_tokens += in_tok
                    lm.total_output_tokens += out_tok
                    lm.total_cost_usd += cost
                result = LMResult(
                    text="".join(text_parts),
                    model=used_model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=cost,
                    latency_s=latency,
                    raw=final,
                    thinking="".join(thinking_parts),
                    stop_reason=getattr(final, "stop_reason", "") or "",
                )
                yield StreamEvent(kind="done", result=result)
                return


# ---------------------------------------------------------------------------
# AsyncLM
# ---------------------------------------------------------------------------
class AsyncLM:
    """Asyncio LM client. Same surface as `LM` but `await lm(prompt)`.

    Use for high-fan-out workloads where blocking the event loop is expensive
    — e.g. orchestrating dozens of sub-LM calls per second. For single-shot
    use, the sync `LM` is simpler.

    NOTE: pricing + audit log are mirrored from the sync client; we don't
    duplicate the implementation here, we share `compute_cost`.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(api_key=key)
        self._lock = asyncio.Lock()
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    async def __call__(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LMResult:
        used_model = model or self.model
        used_max = max_tokens or self.max_tokens

        kwargs: dict[str, Any] = {
            "model": used_model,
            "max_tokens": used_max,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system

        t0 = time.perf_counter()
        msg = await self._client.messages.create(**kwargs)
        latency = time.perf_counter() - t0

        parts: list[str] = []
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        text = "".join(parts)

        in_tok = msg.usage.input_tokens
        out_tok = msg.usage.output_tokens
        cost = compute_cost(used_model, in_tok, out_tok)

        async with self._lock:
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
            raw=msg,
            stop_reason=getattr(msg, "stop_reason", "") or "",
        )

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming generator. Use with `async for evt in lm.stream(...)`."""
        used_model = model or self.model
        used_max = max_tokens or self.max_tokens

        kwargs: dict[str, Any] = {
            "model": used_model,
            "max_tokens": used_max,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        t0 = time.perf_counter()
        async with self._client.messages.stream(**kwargs) as response:
            async for event in response:
                etype = getattr(event, "type", "")
                if etype == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    dtype = getattr(delta, "type", "")
                    if dtype == "text_delta":
                        txt = getattr(delta, "text", "")
                        text_parts.append(txt)
                        yield StreamEvent(kind="text_delta", text=txt)
                    elif dtype == "thinking_delta":
                        txt = getattr(delta, "thinking", "")
                        thinking_parts.append(txt)
                        yield StreamEvent(kind="thinking_delta", text=txt)
                elif etype == "message_stop":
                    final = await response.get_final_message()
                    latency = time.perf_counter() - t0
                    in_tok = final.usage.input_tokens
                    out_tok = final.usage.output_tokens
                    cost = compute_cost(used_model, in_tok, out_tok)
                    async with self._lock:
                        self.total_calls += 1
                        self.total_input_tokens += in_tok
                        self.total_output_tokens += out_tok
                        self.total_cost_usd += cost
                    result = LMResult(
                        text="".join(text_parts),
                        model=used_model,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cost_usd=cost,
                        latency_s=latency,
                        raw=final,
                        thinking="".join(thinking_parts),
                        stop_reason=getattr(final, "stop_reason", "") or "",
                    )
                    yield StreamEvent(kind="done", result=result)
                    return

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self.total_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
        }


__all__ = ["stream", "AsyncLM", "StreamEvent"]
