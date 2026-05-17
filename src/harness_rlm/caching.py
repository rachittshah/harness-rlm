"""Anthropic prompt caching helpers.

Prompt caching cuts repeated-context cost by ~90% when the same prefix is
re-sent. Critical for RLM (the same system + tools + long context flow
through many sub-calls) and for tool-using agent loops (the same tool schemas
flow through every turn).

Mechanics (Anthropic SDK 0.40+):
  - Add `cache_control={"type": "ephemeral"}` to a content block.
  - Up to 4 cache breakpoints per request.
  - Cache TTL is 5 minutes by default (1h with the `1h-cache` beta header).
  - Pricing: writes cost 1.25× input, hits cost 0.1× input.

This module exposes:
  - `cached_text_block(text)`     — wrap text as a cached block
  - `cached_system_block(text)`   — for system prompts (top of every call)
  - `cached_tools(tools)`         — mark the last tool with cache_control
  - `CACHE_TTL_SECONDS`           — 300 (5 min, default)
"""

from __future__ import annotations

from typing import Any

CACHE_TTL_SECONDS = 300


def cached_text_block(text: str) -> dict[str, Any]:
    """Wrap a string as a content block with ephemeral cache_control set."""
    return {
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral"},
    }


def cached_system_block(text: str) -> list[dict[str, Any]]:
    """Build a system prompt list with one cached block.

    Anthropic accepts `system` as either a string or a list of content blocks.
    Pass the list form to the SDK to get caching.
    """
    return [cached_text_block(text)]


def cached_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the LAST tool with cache_control.

    Cache breakpoints are positional — marking the last tool caches the entire
    tool schema block (which is what we want — tools rarely change mid-session).
    Idempotent: re-calling does not stack breakpoints.
    """
    if not tools:
        return tools
    out: list[dict[str, Any]] = []
    for t in tools[:-1]:
        # Strip any existing cache_control so we never exceed 4 breakpoints.
        t = {k: v for k, v in t.items() if k != "cache_control"}
        out.append(t)
    last = dict(tools[-1])
    last["cache_control"] = {"type": "ephemeral"}
    out.append(last)
    return out


def estimate_cache_savings(
    cached_tokens: int,
    uncached_tokens: int,
    *,
    base_rate_per_million: float = 5.0,
) -> dict[str, float]:
    """Return a quick savings estimate.

    Returns a dict with `uncached_cost`, `cached_cost`, `savings_usd`,
    `savings_pct`. Uses Anthropic's published 0.1× hit rate.
    """
    uncached = uncached_tokens / 1_000_000.0 * base_rate_per_million
    # Cache hits cost 0.1× the base input rate.
    cached = cached_tokens / 1_000_000.0 * (base_rate_per_million * 0.1)
    total_with_cache = uncached + cached
    total_without_cache = (uncached_tokens + cached_tokens) / 1_000_000.0 * base_rate_per_million
    savings = total_without_cache - total_with_cache
    pct = (savings / total_without_cache * 100.0) if total_without_cache > 0 else 0.0
    return {
        "uncached_cost": round(uncached, 6),
        "cached_cost": round(cached, 6),
        "total_with_cache": round(total_with_cache, 6),
        "total_without_cache": round(total_without_cache, 6),
        "savings_usd": round(savings, 6),
        "savings_pct": round(pct, 2),
    }


__all__ = [
    "CACHE_TTL_SECONDS",
    "cached_text_block",
    "cached_system_block",
    "cached_tools",
    "estimate_cache_savings",
]
