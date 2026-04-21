"""RLM MCP server — exposes a single `llm_query` tool over stdio.

Design motivation (R6 §5.1):
    Claude Code's `Task` tool re-injects CLAUDE.md + skills + MCP descriptions on
    every spawn (~50K tokens). For an RLM loop with 40 sub-calls, that's ~$1.73
    per run just in re-injection tax. Going direct to the Anthropic API via this
    MCP server drops it to ~$0.79, while still letting the root Claude Code call
    us as a tool (so the skill-side budget guard + trajectory logger still work).

This server intentionally does NOT spawn sub-agents, has NO tool access of its
own, and holds NO conversation state. It is a stateless text-in / text-out
proxy with pricing + audit logging bolted on.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import ValidationError

from harness_rlm.models import DEFAULT_MODEL, LLMQueryRequest, LLMQueryResponse, SubCallLog

# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens) — R3 §4, Apr 2026.
# Keys use prefix match so dated model IDs (e.g. claude-haiku-4-5-20251001)
# resolve to the family rate.
# ---------------------------------------------------------------------------
PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
}
_FALLBACK_PRICING = PRICING["claude-haiku-4-5"]

SUB_CALLS_LOG = Path("/tmp/rlm/sub_calls.jsonl")


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for a completion. Unknown models fall back to Haiku rates."""
    rates = _FALLBACK_PRICING
    for prefix, r in PRICING.items():
        if model.startswith(prefix):
            rates = r
            break
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_sub_call_log(log_entry: SubCallLog, path: Path = SUB_CALLS_LOG) -> None:
    """Append a SubCallLog line to sub_calls.jsonl. Creates parent dir if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(log_entry.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# Core sub-LLM call — shared by the MCP tool handler and --selftest.
# ---------------------------------------------------------------------------
def run_llm_query(req: LLMQueryRequest, api_key: str) -> LLMQueryResponse:
    """Execute one Anthropic completion and log it. Raises on API failure."""
    client = anthropic.Anthropic(api_key=api_key)

    kwargs: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "messages": [{"role": "user", "content": req.prompt}],
    }
    if req.system is not None:
        kwargs["system"] = req.system

    msg = client.messages.create(**kwargs)

    # Concatenate any text blocks in the response. Haiku/Sonnet/Opus return
    # a list of content blocks; we only care about `text` blocks here.
    content_parts: list[str] = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            content_parts.append(text)
    content = "".join(content_parts)

    input_tokens = msg.usage.input_tokens
    output_tokens = msg.usage.output_tokens
    cost_usd = compute_cost(req.model, input_tokens, output_tokens)

    _append_sub_call_log(
        SubCallLog(
            timestamp=_now_iso(),
            prompt_preview=req.prompt[:200],
            response_chars=len(content),
            model=req.model,
            cost_usd=cost_usd,
        )
    )

    return LLMQueryResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=req.model,
        cost_usd=cost_usd,
    )


def _require_api_key() -> str:
    """Pull ANTHROPIC_API_KEY from env or raise a clear RuntimeError."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in the shell that launches "
            "the rlm-mcp-server (or in your Claude Code MCP config env block)."
        )
    return api_key


# ---------------------------------------------------------------------------
# MCP server plumbing.
# ---------------------------------------------------------------------------
server: Server = Server("rlm")

LLM_QUERY_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "User prompt to send to the sub-LLM.",
        },
        "model": {
            "type": "string",
            "description": "Anthropic model ID. Defaults to Haiku 4.5 for cheap sub-calls.",
            "default": DEFAULT_MODEL,
        },
        "max_tokens": {
            "type": "integer",
            "description": "Maximum output tokens.",
            "default": 1024,
            "minimum": 1,
            "maximum": 64000,
        },
        "system": {
            "type": ["string", "null"],
            "description": "Optional system prompt.",
            "default": None,
        },
    },
    "required": ["prompt"],
    "additionalProperties": False,
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise the single `llm_query` tool to the MCP client."""
    return [
        Tool(
            name="llm_query",
            description=(
                "Call an Anthropic model directly (bypassing Claude Code's 50K-token "
                "Task-tool re-injection tax). Returns the model's text response as a "
                "string. Use this for all text-only sub-LLM queries inside an RLM loop."
            ),
            inputSchema=LLM_QUERY_INPUT_SCHEMA,
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Dispatch `llm_query` (the only tool we expose)."""
    if name != "llm_query":
        return [TextContent(type="text", text=f"ERROR: unknown tool '{name}'")]

    try:
        req = LLMQueryRequest(**arguments)
    except ValidationError as e:
        return [TextContent(type="text", text=f"ERROR: invalid arguments — {e}")]

    try:
        api_key = _require_api_key()
    except RuntimeError as e:
        return [TextContent(type="text", text=f"ERROR: {e}")]

    # Run the blocking Anthropic sync client off the event loop.
    try:
        resp = await asyncio.to_thread(run_llm_query, req, api_key)
    except anthropic.APIError as e:
        return [TextContent(type="text", text=f"ERROR: Anthropic API — {e}")]
    except Exception as e:  # noqa: BLE001 — surface everything with context
        return [TextContent(type="text", text=f"ERROR: {type(e).__name__} — {e}")]

    return [TextContent(type="text", text=resp.content)]


# ---------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------
async def _run_stdio() -> None:
    """Run the MCP server over stdio (the transport Claude Code uses)."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def _run_selftest() -> int:
    """Call the tool end-to-end with a canary prompt. Exits 0 on success, 1 on failure."""
    try:
        api_key = _require_api_key()
    except RuntimeError as e:
        print(f"[selftest] FAIL: {e}", file=sys.stderr)
        return 1

    req = LLMQueryRequest(
        prompt="Say exactly: RLM-SELFTEST-OK",
        model=DEFAULT_MODEL,
        max_tokens=32,
        system=None,
    )
    try:
        resp = run_llm_query(req, api_key)
    except Exception as e:  # noqa: BLE001
        print(f"[selftest] FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("[selftest] response:", resp.content)
    print(
        "[selftest] tokens in/out:",
        resp.input_tokens,
        "/",
        resp.output_tokens,
    )
    print(f"[selftest] cost_usd: {resp.cost_usd:.6f}")
    print(f"[selftest] model: {resp.model}")
    print("[selftest] sub_call log appended to:", SUB_CALLS_LOG)

    if "RLM-SELFTEST-OK" in resp.content:
        print("[selftest] PASS")
        return 0
    print("[selftest] WARN: canary string not present in response")
    return 0  # don't fail — model may paraphrase. API round-trip succeeded.


def main() -> None:
    """Console-script entry point. Parses `--selftest` or runs stdio server."""
    parser = argparse.ArgumentParser(
        prog="rlm-mcp-server",
        description="RLM MCP server exposing a single llm_query tool over stdio.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run a one-shot llm_query call and exit (for verifying setup).",
    )
    args = parser.parse_args()

    if args.selftest:
        sys.exit(_run_selftest())

    try:
        asyncio.run(_run_stdio())
    except KeyboardInterrupt:
        # Clean exit when the parent harness closes stdin.
        return


if __name__ == "__main__":
    main()
