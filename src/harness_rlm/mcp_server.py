"""harness-rlm MCP server — exposes the harness as MCP tools over stdio.

Anything that speaks MCP (Claude Desktop, Claude Code, Cursor, Codex, OpenCode,
Goose, custom clients) can use the harness through this server. Ten tools:

  llm_query           single LLM call (text in → text out)
  rlm_run             RLM over a long document → answer + cost
  predict             render a signature → call LM → parse outputs
  chain_of_thought    Predict with explicit reasoning slot
  best_of_n           N samples + majority vote
  compress_text       LM-summarised compaction
  chunk_text          deterministic overlap-based chunking (no LM)
  dispatch_subagent   run a Codex-style TOML subagent on a task
  list_subagents      discover available subagent specs
  estimate_cost       compute USD cost for token counts (no LM)

Design:
- Stateless. The server holds no conversation state — each call is independent.
- Audit logging. Every LLM-touching tool appends to /tmp/rlm/sub_calls.jsonl.
- Backward-compat. `llm_query` keeps its original API verbatim.
- Stdio transport. Standard MCP — works with every MCP client.

Run:
    rlm-mcp-server                # starts stdio server
    rlm-mcp-server --selftest     # canary round-trip + exit
    rlm-mcp-server --list-tools   # print tool catalog and exit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import anthropic
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import ValidationError

from harness_rlm.models import DEFAULT_MODEL, LLMQueryRequest, LLMQueryResponse, SubCallLog

# ---------------------------------------------------------------------------
# Pricing (USD per 1M tokens) — May 2026.
# ---------------------------------------------------------------------------
PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
}
_FALLBACK_PRICING = PRICING["claude-haiku-4-5"]

SUB_CALLS_LOG = Path("/tmp/rlm/sub_calls.jsonl")


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a completion. Unknown models fall back to Haiku rates."""
    rates = _FALLBACK_PRICING
    # Longest-prefix match so "claude-haiku-4-5-20251001" beats "claude-haiku".
    for prefix in sorted(PRICING.keys(), key=len, reverse=True):
        if model.startswith(prefix):
            rates = PRICING[prefix]
            break
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_sub_call_log(log_entry: SubCallLog, path: Path = SUB_CALLS_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(log_entry.model_dump_json() + "\n")


def _require_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it in the shell that launches "
            "the rlm-mcp-server (or in your MCP client's env block)."
        )
    return api_key


# ---------------------------------------------------------------------------
# Core helpers — shared across tools.
# ---------------------------------------------------------------------------
def run_llm_query(req: LLMQueryRequest, api_key: str) -> LLMQueryResponse:
    """Execute one Anthropic completion and log it. Raises on API failure.

    Backwards-compat: kept verbatim for callers using the original llm_query path.
    """
    client = anthropic.Anthropic(api_key=api_key)

    kwargs: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "messages": [{"role": "user", "content": req.prompt}],
    }
    if req.system is not None:
        kwargs["system"] = req.system

    msg = client.messages.create(**kwargs)
    content = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "text", None))
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


# ---------------------------------------------------------------------------
# Tool registry — each entry is a dict describing one tool.
# ---------------------------------------------------------------------------
# Tool handlers return a JSON-serialisable dict; the dispatcher wraps that in
# a TextContent block. Errors return {"error": "...message..."}.
ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]

server: Server = Server("harness-rlm")

_TOOLS: dict[str, dict[str, Any]] = {}


def _register(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    handler: ToolHandler,
) -> None:
    _TOOLS[name] = {
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "handler": handler,
    }


# ---------------------------------------------------------------------------
# 1. llm_query — single LLM call (backwards-compat).
# ---------------------------------------------------------------------------
LLM_QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string", "description": "User prompt to send to the LLM."},
        "model": {
            "type": "string",
            "description": "Anthropic model ID. Defaults to Haiku 4.5.",
            "default": DEFAULT_MODEL,
        },
        "max_tokens": {
            "type": "integer",
            "default": 1024,
            "minimum": 1,
            "maximum": 64000,
        },
        "system": {"type": ["string", "null"], "default": None},
    },
    "required": ["prompt"],
    "additionalProperties": False,
}


def _handle_llm_query(args: dict[str, Any]) -> dict[str, Any]:
    try:
        req = LLMQueryRequest(**args)
    except ValidationError as e:
        return {"error": f"invalid arguments: {e}"}
    api_key = _require_api_key()
    resp = run_llm_query(req, api_key)
    return resp.model_dump()


_register(
    "llm_query",
    "Single LLM completion. Returns content + tokens + cost. Use for cheap sub-LLM calls inside an RLM loop.",
    LLM_QUERY_SCHEMA,
    _handle_llm_query,
)


# ---------------------------------------------------------------------------
# 2. rlm_run — RLM over a long document.
# ---------------------------------------------------------------------------
RLM_RUN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "The question to answer."},
        "document": {
            "type": "string",
            "description": "The long context to decompose.",
        },
        "strategy": {
            "type": "string",
            "enum": ["map_reduce", "tree", "filter_then_query"],
            "default": "map_reduce",
        },
        "root_model": {"type": "string", "default": "claude-opus-4-7"},
        "sub_model": {"type": "string", "default": "claude-haiku-4-5-20251001"},
        "chunk_size": {"type": "integer", "default": 20000, "minimum": 1000},
        "max_parallel": {"type": "integer", "default": 4, "minimum": 1, "maximum": 16},
        "max_llm_calls": {"type": "integer", "default": 20, "minimum": 1, "maximum": 200},
        "filter_pattern": {
            "type": ["string", "null"],
            "default": None,
            "description": "Only for strategy=filter_then_query — regex applied line-wise.",
        },
    },
    "required": ["question", "document"],
    "additionalProperties": False,
}


def _handle_rlm_run(args: dict[str, Any]) -> dict[str, Any]:
    from harness_rlm.llm import LM
    from harness_rlm.rlm import RLM, RLMConfig

    cfg = RLMConfig(
        strategy=args.get("strategy", "map_reduce"),
        chunk_size=int(args.get("chunk_size", 20000)),
        max_parallel=int(args.get("max_parallel", 4)),
        max_llm_calls=int(args.get("max_llm_calls", 20)),
        filter_pattern=args.get("filter_pattern"),
    )
    root_lm = LM(model=args.get("root_model", "claude-opus-4-7"))
    sub_lm = LM(model=args.get("sub_model", "claude-haiku-4-5-20251001"))
    rlm = RLM(
        "question, document -> answer",
        long_context_field="document",
        config=cfg,
        root_lm=root_lm,
        sub_lm=sub_lm,
    )
    pred = rlm(question=args["question"], document=args["document"])
    return {
        "answer": pred.fields.get("answer", ""),
        "calls": pred.trace.calls if pred.trace else 0,
        "cost_usd": round(pred.trace.cost_usd, 6) if pred.trace else 0.0,
        "latency_s": round(pred.trace.latency_s, 2) if pred.trace else 0.0,
        "strategy": cfg.strategy,
        "input_tokens": pred.trace.input_tokens if pred.trace else 0,
        "output_tokens": pred.trace.output_tokens if pred.trace else 0,
    }


_register(
    "rlm_run",
    "Run a Recursive Language Model over a long document. Decomposes into chunks, dispatches a cheap sub-LM in parallel, synthesizes the answer with the root LM. Strategies: map_reduce (default), tree (hierarchical), filter_then_query (regex-first). Returns the answer plus cost/latency/call counts.",
    RLM_RUN_SCHEMA,
    _handle_rlm_run,
)


# ---------------------------------------------------------------------------
# 3. predict — typed I/O signature → one LM call → parsed fields.
# ---------------------------------------------------------------------------
PREDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "signature": {
            "type": "string",
            "description": "DSPy-style shorthand: 'question, context -> answer'.",
        },
        "inputs": {
            "type": "object",
            "description": "Map of input field name → value.",
            "additionalProperties": True,
        },
        "instruction": {
            "type": ["string", "null"],
            "default": None,
            "description": "Optional natural-language instruction prepended to the prompt.",
        },
        "model": {"type": "string", "default": DEFAULT_MODEL},
        "max_tokens": {"type": "integer", "default": 1024, "minimum": 1, "maximum": 64000},
    },
    "required": ["signature", "inputs"],
    "additionalProperties": False,
}


def _handle_predict(args: dict[str, Any]) -> dict[str, Any]:
    from harness_rlm.llm import LM
    from harness_rlm.modules import Predict
    from harness_rlm.signatures import Signature

    sig = Signature(args["signature"])
    if args.get("instruction"):
        sig = sig.with_instruction(args["instruction"])
    lm = LM(
        model=args.get("model", DEFAULT_MODEL),
        max_tokens=int(args.get("max_tokens", 1024)),
    )
    pred = Predict(sig, lm=lm)(**args["inputs"])
    return {
        "fields": pred.fields,
        "cost_usd": round(pred.trace.cost_usd, 6) if pred.trace else 0.0,
        "input_tokens": pred.trace.input_tokens if pred.trace else 0,
        "output_tokens": pred.trace.output_tokens if pred.trace else 0,
    }


_register(
    "predict",
    "Render a typed signature ('q, ctx -> answer'), call the LLM once, parse the response into named output fields. Returns {fields: {...}, cost_usd, tokens}.",
    PREDICT_SCHEMA,
    _handle_predict,
)


# ---------------------------------------------------------------------------
# 4. chain_of_thought — Predict + reasoning field.
# ---------------------------------------------------------------------------
COT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "signature": {"type": "string"},
        "inputs": {"type": "object", "additionalProperties": True},
        "instruction": {"type": ["string", "null"], "default": None},
        "model": {"type": "string", "default": DEFAULT_MODEL},
        "max_tokens": {"type": "integer", "default": 1024, "minimum": 1, "maximum": 64000},
    },
    "required": ["signature", "inputs"],
    "additionalProperties": False,
}


def _handle_chain_of_thought(args: dict[str, Any]) -> dict[str, Any]:
    from harness_rlm.llm import LM
    from harness_rlm.modules import ChainOfThought
    from harness_rlm.signatures import Signature

    sig = Signature(args["signature"])
    if args.get("instruction"):
        sig = sig.with_instruction(args["instruction"])
    lm = LM(
        model=args.get("model", DEFAULT_MODEL),
        max_tokens=int(args.get("max_tokens", 1024)),
    )
    pred = ChainOfThought(sig, lm=lm)(**args["inputs"])
    return {
        "fields": pred.fields,
        "cost_usd": round(pred.trace.cost_usd, 6) if pred.trace else 0.0,
        "input_tokens": pred.trace.input_tokens if pred.trace else 0,
        "output_tokens": pred.trace.output_tokens if pred.trace else 0,
    }


_register(
    "chain_of_thought",
    "Same as predict, but injects a 'reasoning' output field that the LM fills before the answer. Use for non-trivial reasoning tasks.",
    COT_SCHEMA,
    _handle_chain_of_thought,
)


# ---------------------------------------------------------------------------
# 5. best_of_n — sample N times, majority vote.
# ---------------------------------------------------------------------------
BEST_OF_N_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "signature": {"type": "string"},
        "inputs": {"type": "object", "additionalProperties": True},
        "n": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
        "chain_of_thought": {
            "type": "boolean",
            "default": True,
            "description": "If true, child module is ChainOfThought (vote on `answer`); else Predict.",
        },
        "answer_field": {"type": "string", "default": "answer"},
        "model": {"type": "string", "default": DEFAULT_MODEL},
        "max_parallel": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
    },
    "required": ["signature", "inputs"],
    "additionalProperties": False,
}


def _handle_best_of_n(args: dict[str, Any]) -> dict[str, Any]:
    from harness_rlm.ensemble import BestOfN, SelfConsistency
    from harness_rlm.llm import LM
    from harness_rlm.modules import ChainOfThought, Predict
    from harness_rlm.signatures import Signature

    sig = Signature(args["signature"])
    lm = LM(model=args.get("model", DEFAULT_MODEL))
    if args.get("chain_of_thought", True):
        child = ChainOfThought(sig, lm=lm)
        m = SelfConsistency(
            child,
            n=int(args.get("n", 5)),
            answer_field=args.get("answer_field", "answer"),
            max_parallel=int(args.get("max_parallel", 5)),
        )
    else:
        child = Predict(sig, lm=lm)
        m = BestOfN(child, n=int(args.get("n", 5)), max_parallel=int(args.get("max_parallel", 5)))
    pred = m(**args["inputs"])
    # Extract vote distribution from the trace.
    vote_dist: list[list[Any]] = []
    for evt in pred.trace.events if pred.trace else []:
        if evt.get("label") == "best_of_n" and "vote_distribution" in evt:
            vote_dist = evt["vote_distribution"]
            break
    return {
        "fields": pred.fields,
        "vote_distribution": vote_dist,
        "calls": pred.trace.calls if pred.trace else 0,
        "cost_usd": round(pred.trace.cost_usd, 6) if pred.trace else 0.0,
    }


_register(
    "best_of_n",
    "Sample the child module N times in parallel, majority-vote on the answer field. Improves reasoning task accuracy 5-15% (Wei et al., arXiv:2203.11171). Set chain_of_thought=true for reasoning paths.",
    BEST_OF_N_SCHEMA,
    _handle_best_of_n,
)


# ---------------------------------------------------------------------------
# 6. compress_text — LM-summarised compaction.
# ---------------------------------------------------------------------------
COMPRESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "target_chars": {"type": "integer", "default": 4000, "minimum": 100},
        "model": {"type": "string", "default": DEFAULT_MODEL},
    },
    "required": ["text"],
    "additionalProperties": False,
}


def _handle_compress_text(args: dict[str, Any]) -> dict[str, Any]:
    from harness_rlm.llm import LM
    from harness_rlm.orchestrator import compress

    lm = LM(model=args.get("model", DEFAULT_MODEL))
    summary = compress(
        args["text"],
        target_chars=int(args.get("target_chars", 4000)),
        lm=lm,
    )
    return {
        "summary": summary,
        "original_chars": len(args["text"]),
        "summary_chars": len(summary),
        "compression_ratio": (
            round(len(args["text"]) / max(len(summary), 1), 2) if summary else 0.0
        ),
        "cost_usd": round(lm.total_cost_usd, 6),
    }


_register(
    "compress_text",
    "LM-summarised compaction. Preserves concrete facts, drops repetition. Targets the given char count. Useful for rolling conversation history.",
    COMPRESS_SCHEMA,
    _handle_compress_text,
)


# ---------------------------------------------------------------------------
# 7. chunk_text — overlap-based chunking (no LM).
# ---------------------------------------------------------------------------
CHUNK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "chunk_size": {"type": "integer", "default": 5000, "minimum": 100},
        "overlap": {"type": "integer", "default": 200, "minimum": 0},
    },
    "required": ["text"],
    "additionalProperties": False,
}


def _handle_chunk_text(args: dict[str, Any]) -> dict[str, Any]:
    from harness_rlm.core import chunk_context

    chunks = chunk_context(
        args["text"],
        chunk_size=int(args.get("chunk_size", 5000)),
        overlap=int(args.get("overlap", 200)),
    )
    return {
        "chunks": chunks,
        "count": len(chunks),
        "total_chars": sum(len(c) for c in chunks),
    }


_register(
    "chunk_text",
    "Split text into overlapping chunks. Deterministic, no LM. Returns a list of chunk strings plus metadata.",
    CHUNK_SCHEMA,
    _handle_chunk_text,
)


# ---------------------------------------------------------------------------
# 8. dispatch_subagent — run a TOML-declared subagent on a task.
# ---------------------------------------------------------------------------
DISPATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spec_name": {
            "type": "string",
            "description": "Name from a TOML in .harness-rlm/agents/ or ~/.harness-rlm/agents/.",
        },
        "task": {"type": "string", "description": "Task message for the subagent."},
        "parent_sandbox": {
            "type": "string",
            "enum": ["read-only", "workspace-write", "danger"],
            "default": "read-only",
            "description": "Subagent can only downgrade from this — never escalate.",
        },
        "max_turns": {"type": "integer", "default": 12, "minimum": 1, "maximum": 50},
    },
    "required": ["spec_name", "task"],
    "additionalProperties": False,
}


def _handle_dispatch_subagent(args: dict[str, Any]) -> dict[str, Any]:
    from harness_rlm.subagents import discover, dispatch

    specs = discover()
    name = args["spec_name"]
    if name not in specs:
        return {
            "error": (
                f"Unknown subagent {name!r}. Run list_subagents to see what's available. "
                f"Discovered: {sorted(specs.keys())}"
            )
        }
    result = dispatch(
        specs[name],
        args["task"],
        parent_sandbox=args.get("parent_sandbox", "read-only"),
        max_turns=int(args.get("max_turns", 12)),
    )
    return {
        "final_text": result.final_text,
        "turns": result.turns,
        "tool_call_count": result.tool_call_count,
        "terminated_by_tool": result.terminated_by_tool,
        "cost_usd": round(result.cost_usd, 6),
    }


_register(
    "dispatch_subagent",
    "Run a Codex-style declarative subagent (TOML in .harness-rlm/agents/) on a task. Subagent has its own model, instructions, sandbox tier, and tool set. Returns the final text + cost/turns.",
    DISPATCH_SCHEMA,
    _handle_dispatch_subagent,
)


# ---------------------------------------------------------------------------
# 9. list_subagents — discover available subagent specs.
# ---------------------------------------------------------------------------
LIST_SUBAGENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def _handle_list_subagents(args: dict[str, Any]) -> dict[str, Any]:
    from harness_rlm.subagents import discover

    specs = discover()
    return {
        "subagents": [
            {
                "name": s.name,
                "description": s.description,
                "model": s.model,
                "sandbox_mode": s.sandbox_mode,
                "tools": list(s.tools),
                "source_path": str(s.source_path) if s.source_path else None,
            }
            for s in specs.values()
        ],
        "count": len(specs),
    }


_register(
    "list_subagents",
    "List all discoverable subagent specs (.harness-rlm/agents/*.toml). Use before dispatch_subagent to find available roles.",
    LIST_SUBAGENTS_SCHEMA,
    _handle_list_subagents,
)


# ---------------------------------------------------------------------------
# 10. estimate_cost — pure calculator, no LM.
# ---------------------------------------------------------------------------
COST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "input_tokens": {"type": "integer", "minimum": 0},
        "output_tokens": {"type": "integer", "minimum": 0},
        "cached_input_tokens": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": "Tokens served from prompt cache (0.1× input rate).",
        },
    },
    "required": ["model", "input_tokens", "output_tokens"],
    "additionalProperties": False,
}


def _handle_estimate_cost(args: dict[str, Any]) -> dict[str, Any]:
    model = args["model"]
    in_tok = int(args["input_tokens"])
    out_tok = int(args["output_tokens"])
    cached = int(args.get("cached_input_tokens", 0))
    # Standard cost.
    base_cost = compute_cost(model, in_tok, out_tok)
    # Adjust for cache hits: cached portion costs 0.1× the input rate.
    rates = _FALLBACK_PRICING
    for prefix in sorted(PRICING.keys(), key=len, reverse=True):
        if model.startswith(prefix):
            rates = PRICING[prefix]
            break
    full_input_cost = in_tok * rates["input"] / 1_000_000.0
    cached_savings = cached * rates["input"] * 0.9 / 1_000_000.0
    net_cost = base_cost - cached_savings
    return {
        "model": model,
        "rate_input_per_million": rates["input"],
        "rate_output_per_million": rates["output"],
        "cost_input_usd": round(full_input_cost - cached_savings, 6),
        "cost_output_usd": round(out_tok * rates["output"] / 1_000_000.0, 6),
        "cost_total_usd": round(net_cost, 6),
        "cache_savings_usd": round(cached_savings, 6),
    }


_register(
    "estimate_cost",
    "Compute the USD cost of a token-count budget. No LM call. Accepts cached_input_tokens for prompt-cache projections (0.1× rate on hits).",
    COST_SCHEMA,
    _handle_estimate_cost,
)


# ---------------------------------------------------------------------------
# MCP plumbing — dispatch + listing.
# ---------------------------------------------------------------------------
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name=spec["name"],
            description=spec["description"],
            inputSchema=spec["input_schema"],
        )
        for spec in _TOOLS.values()
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    spec = _TOOLS.get(name)
    if spec is None:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": f"unknown tool {name!r}",
                        "available": sorted(_TOOLS.keys()),
                    }
                ),
            )
        ]
    try:
        result = await asyncio.to_thread(spec["handler"], arguments)
    except anthropic.APIError as e:
        result = {"error": f"Anthropic API: {e}"}
    except RuntimeError as e:
        result = {"error": str(e)}
    except Exception as e:  # noqa: BLE001 — surface everything with context
        result = {"error": f"{type(e).__name__}: {e}"}

    # For llm_query specifically, keep backwards-compat: return raw text (the
    # response content), not the JSON wrapper. This matches the prior contract
    # so existing skills + adapters don't break.
    if name == "llm_query" and "content" in result and "error" not in result:
        return [TextContent(type="text", text=result["content"])]
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# ---------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------
async def _run_stdio() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def _run_selftest() -> int:
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
    print(f"[selftest] tokens in/out: {resp.input_tokens}/{resp.output_tokens}")
    print(f"[selftest] cost_usd: {resp.cost_usd:.6f}")
    print(f"[selftest] model: {resp.model}")
    print(f"[selftest] sub_call log appended to: {SUB_CALLS_LOG}")
    print(f"[selftest] tools registered: {sorted(_TOOLS.keys())}")
    if "RLM-SELFTEST-OK" in resp.content:
        print("[selftest] PASS")
    else:
        print("[selftest] WARN: canary string not present (paraphrased)")
    return 0


def _run_list_tools() -> int:
    """Print the tool catalog as JSON and exit. Useful for debugging."""
    catalog = [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["input_schema"],
        }
        for s in _TOOLS.values()
    ]
    print(json.dumps(catalog, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rlm-mcp-server",
        description="harness-rlm MCP server — exposes 10 RLM/agent tools over stdio.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="One-shot llm_query canary + exit. Verifies API access + audit log.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the tool catalog as JSON and exit (no API call).",
    )
    args = parser.parse_args()

    if args.list_tools:
        sys.exit(_run_list_tools())
    if args.selftest:
        sys.exit(_run_selftest())

    try:
        asyncio.run(_run_stdio())
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
