# MCP tools reference

The `harness-rlm` MCP server exposes **10 tools** over stdio. Any MCP client (Claude Desktop, Claude Code, Cursor, Codex CLI, Goose, OpenCode, Cline, custom clients) can use them.

The full API surface — and decision tree for picking the right tool — is in [skill/SKILL.md](../skill/SKILL.md). This doc is the concise schema reference.

## Quick reference

| # | Tool | Inputs | Returns | LM cost |
|---|---|---|---|---|
| 1 | [`llm_query`](#1-llm_query) | prompt, model?, max_tokens?, system? | text (backwards-compat: raw string) | 1 call |
| 2 | [`rlm_run`](#2-rlm_run) | question, document, strategy?, root_model?, sub_model?, max_llm_calls? | answer + cost + calls | many |
| 3 | [`predict`](#3-predict) | signature, inputs, instruction?, model? | fields + cost | 1 call |
| 4 | [`chain_of_thought`](#4-chain_of_thought) | signature, inputs, instruction?, model? | fields incl. reasoning | 1 call |
| 5 | [`best_of_n`](#5-best_of_n) | signature, inputs, n, chain_of_thought?, answer_field?, model? | fields + vote_distribution | N calls |
| 6 | [`compress_text`](#6-compress_text) | text, target_chars?, model? | summary + ratio | 1 call |
| 7 | [`chunk_text`](#7-chunk_text) | text, chunk_size?, overlap? | chunks + count | 0 |
| 8 | [`dispatch_subagent`](#8-dispatch_subagent) | spec_name, task, parent_sandbox?, max_turns? | final_text + cost | many |
| 9 | [`list_subagents`](#9-list_subagents) | (no inputs) | subagents list | 0 |
| 10 | [`estimate_cost`](#10-estimate_cost) | model, input_tokens, output_tokens, cached_input_tokens? | cost breakdown | 0 |

Run `rlm-mcp-server --list-tools` to print the full schema as JSON.

---

## 1. `llm_query`

Single LLM completion. Returns content + tokens + cost. Use for cheap sub-LLM calls inside an RLM loop.

**Input**
```json
{
  "prompt": "string (required)",
  "model": "string (default: claude-haiku-4-5-20251001)",
  "max_tokens": "integer 1-64000 (default: 1024)",
  "system": "string | null (default: null)"
}
```

**Output** (backwards-compat: returns the raw response text, not JSON wrapper)
```
The model's response text.
```

---

## 2. `rlm_run`

Recursive Language Model over a long document. Decomposes into chunks, dispatches a cheap sub-LM in parallel, synthesizes the answer with the root LM.

**Input**
```json
{
  "question": "string (required)",
  "document": "string (required, long context)",
  "strategy": "map_reduce | tree | filter_then_query (default: map_reduce)",
  "root_model": "string (default: claude-opus-4-7)",
  "sub_model": "string (default: claude-haiku-4-5-20251001)",
  "chunk_size": "integer >= 1000 (default: 20000)",
  "max_parallel": "integer 1-16 (default: 4)",
  "max_llm_calls": "integer 1-200 (default: 20)",
  "filter_pattern": "string | null — only for strategy=filter_then_query"
}
```

**Output**
```json
{
  "answer": "string",
  "calls": "integer",
  "cost_usd": "float",
  "latency_s": "float",
  "strategy": "string",
  "input_tokens": "integer",
  "output_tokens": "integer"
}
```

---

## 3. `predict`

Render a typed signature, call the LLM once, parse the response into named output fields.

**Input**
```json
{
  "signature": "string (DSPy-style: 'q, ctx -> answer')",
  "inputs": "object (map of input field → value)",
  "instruction": "string | null",
  "model": "string",
  "max_tokens": "integer 1-64000"
}
```

**Output**
```json
{
  "fields": {"answer": "...", "...": "..."},
  "cost_usd": "float",
  "input_tokens": "integer",
  "output_tokens": "integer"
}
```

---

## 4. `chain_of_thought`

Same as `predict` but injects a `reasoning` output field that the LM fills before the answer. Use for non-trivial reasoning.

Input/output schema identical to `predict`. The `fields` always include `reasoning`.

---

## 5. `best_of_n`

Sample the child module N times in parallel, majority-vote on the answer field. Lifts reasoning accuracy 5-15% (Wei et al., arXiv:2203.11171).

**Input**
```json
{
  "signature": "string",
  "inputs": "object",
  "n": "integer 1-20 (default: 5)",
  "chain_of_thought": "boolean (default: true)",
  "answer_field": "string (default: answer)",
  "model": "string",
  "max_parallel": "integer 1-10 (default: 5)"
}
```

**Output**
```json
{
  "fields": {"answer": "...", "reasoning": "..."},
  "vote_distribution": [["answer-text", count], ...],
  "calls": "integer",
  "cost_usd": "float"
}
```

---

## 6. `compress_text`

LM-summarised compaction. Preserves concrete facts, drops repetition. Targets the given char count.

**Input**
```json
{
  "text": "string",
  "target_chars": "integer >= 100 (default: 4000)",
  "model": "string"
}
```

**Output**
```json
{
  "summary": "string",
  "original_chars": "integer",
  "summary_chars": "integer",
  "compression_ratio": "float",
  "cost_usd": "float"
}
```

If `len(text) <= target_chars`, returns the original verbatim (no LM call).

---

## 7. `chunk_text`

Deterministic overlap-based chunking. No LM call. Useful for pre-processing before manual sub-LM dispatch.

**Input**
```json
{
  "text": "string",
  "chunk_size": "integer >= 100 (default: 5000)",
  "overlap": "integer >= 0 (default: 200)"
}
```

**Output**
```json
{
  "chunks": ["string", ...],
  "count": "integer",
  "total_chars": "integer"
}
```

---

## 8. `dispatch_subagent`

Run a Codex-style declarative subagent (TOML spec in `.harness-rlm/agents/`) on a task. The subagent runs in its own AgentLoop with its declared model, instructions, sandbox tier, and tool set.

**Input**
```json
{
  "spec_name": "string (required, must match a TOML 'name' field)",
  "task": "string (required, the task to delegate)",
  "parent_sandbox": "read-only | workspace-write | danger (default: read-only)",
  "max_turns": "integer 1-50 (default: 12)"
}
```

**Output**
```json
{
  "final_text": "string",
  "turns": "integer",
  "tool_call_count": "integer",
  "terminated_by_tool": "boolean",
  "cost_usd": "float"
}
```

Sandbox enforcement: the subagent's effective sandbox is the more restrictive of (`parent_sandbox`, spec.sandbox_mode). Children can downgrade but never escalate.

---

## 9. `list_subagents`

Discover available subagent specs. Run before `dispatch_subagent` to find what's available.

**Input** — no parameters.

**Output**
```json
{
  "subagents": [
    {
      "name": "string",
      "description": "string",
      "model": "string",
      "sandbox_mode": "read-only | workspace-write | danger",
      "tools": ["read", "bash", "finish_task", ...],
      "source_path": "string | null"
    }
  ],
  "count": "integer"
}
```

---

## 10. `estimate_cost`

Compute the USD cost of a token-count budget. Pure calculator — no LM call.

**Input**
```json
{
  "model": "string (required)",
  "input_tokens": "integer >= 0",
  "output_tokens": "integer >= 0",
  "cached_input_tokens": "integer >= 0 (default: 0)"
}
```

**Output**
```json
{
  "model": "string",
  "rate_input_per_million": "float",
  "rate_output_per_million": "float",
  "cost_input_usd": "float",
  "cost_output_usd": "float",
  "cost_total_usd": "float",
  "cache_savings_usd": "float"
}
```

`cached_input_tokens` model: cached portions cost 0.1× the standard input rate (Anthropic prompt caching).

---

## Server modes

```bash
rlm-mcp-server                # stdio server (the normal mode)
rlm-mcp-server --list-tools   # print tool catalog JSON and exit
rlm-mcp-server --selftest     # one-shot llm_query canary + exit
```

## Error handling

All tool handlers return an error in this shape when something goes wrong:

```json
{"error": "<class>: <message>"}
```

Examples:
- `{"error": "ANTHROPIC_API_KEY is not set..."}`
- `{"error": "Unknown subagent 'foo'. ..."}`
- `{"error": "Anthropic API: ..."}` (network/auth failure)
- `{"error": "ValueError: chunk_size must be > 0"}`

The dispatcher catches all exceptions and surfaces them as JSON — no traceback to the client. Look in stderr of the server process for tracebacks.

## Audit log

Every LM-touching tool appends one JSON line to `/tmp/rlm/sub_calls.jsonl`:

```json
{"timestamp": "2026-05-17T08:15:00Z", "prompt_preview": "first 200 chars...", "response_chars": 142, "model": "claude-haiku-4-5-20251001", "cost_usd": 0.000054}
```

Tail it during development:

```bash
tail -f /tmp/rlm/sub_calls.jsonl | jq .
```

## Performance notes

- **Stateless**: each call is independent — no conversation state.
- **Backwards-compat**: `llm_query` returns raw text (not JSON-wrapped) so existing skills/adapters don't break.
- **Concurrent calls**: each tool handler runs in `asyncio.to_thread` so multiple in-flight calls don't block the event loop.
- **Prompt caching**: the MCP server does NOT currently auto-cache. Use the Python `LM(enable_caching=True)` path when caching matters.
