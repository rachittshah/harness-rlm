---
name: harness-rlm
description: |
  Universal harness-rlm skill — invoke recursive long-context LLM decomposition,
  prompt evolution, ensemble voting, and declarative subagent dispatch from any
  agent that speaks MCP. Use when the user needs to (a) answer over a >100K-token
  context, (b) sample-and-vote on hard reasoning, (c) run an isolated subagent
  role, (d) batch-evaluate a Module over a CSV. Implements Zhang/Kraska/Khattab
  arXiv:2512.24601 plus GEPA (arXiv:2507.19457) and Pi-mono's minimal tool loop.
---

# harness-rlm — universal skill

You are an agent that has been given access to the **harness-rlm MCP server**. The server exposes **10 tools** covering the full surface of harness-rlm: Recursive Language Model decomposition (long-context Q&A), prompt-evolution-optimized modules, ensemble voting, declarative subagents. This skill teaches you which tool to pick.

The skill is **harness-agnostic**: it works inside Claude Code, Claude Desktop, Cursor, Codex, Goose, OpenCode, Cline, or any custom MCP client. The tool names are the same everywhere. Adapter-specific config (where to drop the MCP block) lives under `adapters/<harness>/mcp-config.md`.

---

## The 10 tools at a glance

| Tool | Use when | LM cost |
|---|---|---|
| `llm_query` | You need a single cheap LLM call (text in → text out). | 1 call |
| `rlm_run` | You have a long context (>100K tokens) and a question over it. | 1+chunks |
| `predict` | You want typed inputs/outputs from one LLM call. | 1 call |
| `chain_of_thought` | Same as predict but with explicit reasoning. | 1 call |
| `best_of_n` | The answer is non-trivial and you can sample 3–5 times. | N calls |
| `compress_text` | A rolling history is too long; you need a faithful summary. | 1 call |
| `chunk_text` | You need to split text deterministically — no LM. | 0 calls |
| `dispatch_subagent` | You want to delegate to a role (read-only explorer, etc.). | many |
| `list_subagents` | You're not sure what subagent roles are available. | 0 calls |
| `estimate_cost` | You want a cost projection before running. | 0 calls |

---

## Decision tree

```
Is the input >100K chars / >25K tokens?
├── Yes → rlm_run (with strategy=map_reduce by default)
└── No
    ├── Does the question need reasoning? (math, multi-step inference)
    │   ├── Yes, and accuracy is critical → best_of_n with chain_of_thought=true
    │   ├── Yes, single sample is fine → chain_of_thought
    │   └── No (lookup, classification) → predict
    │
    ├── Do you have a specific subagent role for this? (e.g. "code explorer")
    │   ├── Yes → dispatch_subagent (after list_subagents to find the name)
    │   └── No → fall through to predict / chain_of_thought
    │
    └── Just need raw text-in/text-out? → llm_query
```

---

## Step-by-step: long-context Q&A

When the user has a long document (>100K chars) and a question:

1. **Decide the strategy**:
   - `map_reduce` (default) — uniform document, no structure.
   - `tree` — hierarchical document (book chapters, multi-file codebase).
   - `filter_then_query` — sparse signal; provide a `filter_pattern` regex.
2. **Call `rlm_run`**:
   ```json
   {
     "question": "<the question>",
     "document": "<the long text>",
     "strategy": "map_reduce",
     "root_model": "claude-opus-4-7",
     "sub_model": "claude-haiku-4-5-20251001",
     "max_llm_calls": 20
   }
   ```
3. **Return the answer** plus cost + call count. The tool already enforces a budget.

**When the document is `<100K chars` but `>10K chars`**: `rlm_run` will short-circuit to a flat call. Cheap and fine.

**Budget envelope**: by default 20 calls max. Adjust via `max_llm_calls`. Each call's prompt preview and cost are appended to `/tmp/rlm/sub_calls.jsonl`.

---

## Step-by-step: hard reasoning

When the user has a reasoning question (math, multi-step inference) and you want better accuracy:

1. **Call `best_of_n`** with `chain_of_thought=true`:
   ```json
   {
     "signature": "question -> answer",
     "inputs": {"question": "<the reasoning task>"},
     "n": 5,
     "chain_of_thought": true,
     "answer_field": "answer"
   }
   ```
2. The tool samples 5 times in parallel and majority-votes on the `answer` field. Reasoning paths may differ; the answer aggregates.
3. **Returns** `{fields, vote_distribution, calls, cost_usd}`. A unanimous vote (5/5) is high-confidence; a 3/2 split is worth flagging.

Wei et al. (arXiv:2203.11171) show this lifts GSM8K / MATH accuracy by 5–15% over a single sample.

---

## Step-by-step: typed predict (signature-driven)

When you want named outputs with a typed contract:

```json
{
  "signature": "question, context -> answer",
  "inputs": {"question": "...", "context": "..."},
  "instruction": "Be brief. One sentence."
}
```

Returns `{fields: {"answer": "..."}, cost_usd, tokens}`. The signature shorthand `"a, b -> c, d"` parses to inputs `[a, b]` and outputs `[c, d]`.

For step-by-step reasoning, use `chain_of_thought` instead — it adds a `reasoning` output field that the LM fills before the answer.

---

## Step-by-step: dispatch a declarative subagent

When the user wants to delegate to a specialized role (e.g., a read-only codebase explorer):

1. **First, discover available roles**:
   ```json
   {}
   ```
   (call `list_subagents`)
2. **Pick a spec** by name from the returned list.
3. **Dispatch**:
   ```json
   {
     "spec_name": "explorer",
     "task": "<the task to delegate>",
     "parent_sandbox": "read-only",
     "max_turns": 12
   }
   ```
4. The subagent runs in its own AgentLoop with its declared model, instructions, and tool set. Returns the final text + cost.

Subagents are defined as TOML files in `.harness-rlm/agents/<name>.toml` (project) or `~/.harness-rlm/agents/<name>.toml` (personal). The sandbox tier (`read-only` / `workspace-write` / `danger`) caps the subagent at the more restrictive of (parent, spec) — a child can downgrade but never escalate.

---

## Step-by-step: cost projection

Before running an expensive RLM:

```json
{
  "model": "claude-opus-4-7",
  "input_tokens": 50000,
  "output_tokens": 2000,
  "cached_input_tokens": 30000
}
```

Returns:
```json
{
  "rate_input_per_million": 5.0,
  "rate_output_per_million": 25.0,
  "cost_input_usd": 0.115,
  "cost_output_usd": 0.05,
  "cost_total_usd": 0.165,
  "cache_savings_usd": 0.135
}
```

Useful for: deciding whether to scale up an eval batch, choosing between Haiku and Opus for a task, justifying caching to a user.

---

## Anti-patterns

1. **Don't pass full long context inline to `llm_query`.** `llm_query` is for single short calls. For long documents, use `rlm_run`.
2. **Don't call `rlm_run` with `<10K chars` of context.** The flat path is faster and equivalent. The tool will short-circuit but you waste a round trip.
3. **Don't use `best_of_n` for deterministic tasks.** If the answer is uniquely determined (lookup, simple formula), one call suffices.
4. **Don't use the same model for root and sub in `rlm_run`.** Defeats the cost win. Use Opus root + Haiku sub (default).
5. **Don't ignore `cost_usd` in tool results.** A long RLM session can easily reach $0.50–$1. Surface costs to the user proactively.
6. **Don't store secrets in `dispatch_subagent` tasks.** The task message goes to the subagent's LM in plaintext.

---

## Where the tools live

The MCP server is `rlm-mcp-server` (installed by `pip install -e .` or `uv sync`). Configure your harness to launch it:

**Claude Code / Claude Desktop** (`.mcp.json`):
```json
{
  "mcpServers": {
    "harness-rlm": {
      "command": "rlm-mcp-server",
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

**Codex** (`config.toml`):
```toml
[mcp_servers.harness-rlm]
command = "rlm-mcp-server"

[mcp_servers.harness-rlm.env]
ANTHROPIC_API_KEY = "sk-ant-..."
```

**Goose** (`~/.config/goose/config.yaml`):
```yaml
extensions:
  harness-rlm:
    type: stdio
    command: rlm-mcp-server
    envs:
      ANTHROPIC_API_KEY: sk-ant-...
```

**OpenCode** (`~/.config/opencode/config.json`):
```json
{
  "mcpServers": {
    "harness-rlm": {
      "command": "rlm-mcp-server",
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

Per-harness setup details: `adapters/<harness>/mcp-config.md`.

---

## Session storage contract

For tools that touch an RLM session, the following files are written/read under `/tmp/rlm/` (override via env in future versions):

| File | Written by | Read by |
|---|---|---|
| `sub_calls.jsonl` | every LLM-touching tool | trace tooling, audit |
| `trajectory.jsonl` | adapter post-tool hook (optional) | trace tooling |
| `state.json` | adapter pre-call (optional) | budget guard |
| `context.pkl` | (legacy) skill-driven RLM flavour | skill-driven RLM flavour |

The programmatic flavour (the MCP tools above) doesn't need `context.pkl` — `rlm_run` takes the document inline.

---

## References

- RLM paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab, Dec 2025)
- GEPA: [arXiv:2507.19457](https://arxiv.org/abs/2507.19457) (Agrawal, Khattab et al., ICLR 2026)
- Self-Consistency: [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) (Wang, Wei et al., 2022)
- Pi-mono harness: [badlogic/pi-mono](https://github.com/badlogic/pi-mono)
- OpenAI Codex subagents: [developers.openai.com/codex/subagents](https://developers.openai.com/codex/subagents)
- MCP spec: [modelcontextprotocol.io](https://modelcontextprotocol.io)
- Open Agent Skills Standard: [agentskills.io](https://agentskills.io/specification)

---

## Quick selftest

After installing, verify the server works:

```bash
$ rlm-mcp-server --list-tools | head -5
[
  {
    "name": "llm_query",
    ...

$ rlm-mcp-server --selftest
[selftest] response: RLM-SELFTEST-OK
[selftest] tokens in/out: 14/8
[selftest] cost_usd: 0.000054
[selftest] tools registered: ['best_of_n', 'chain_of_thought', 'chunk_text', 'compress_text', 'dispatch_subagent', 'estimate_cost', 'list_subagents', 'llm_query', 'predict', 'rlm_run']
[selftest] PASS
```

If the server starts cleanly and `--selftest` round-trips, all 10 tools are reachable.
