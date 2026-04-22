# RLM loop mechanics (Codex CLI adapter)

> Progressive-disclosure reference. `SKILL.md` stays short (<500 lines, per
> https://agentskills.io/specification). This file is loaded only when the
> root LM needs deeper guidance on the loop.

## 1. Why RLM at all

A flat LLM call over a 500K-token context pays the full context cost on every
attempt, produces a single one-shot answer, and has no mechanism for the model
to say "I don't know — let me look closer." RLM inverts the surface area: the
root LM gets programmatic access to the context (as a variable) plus a cheap
sub-LLM (as a function), and iterates until confident.

Paper: Zhang, Kraska, Khattab, *"Recursive Language Models"* (arXiv:2512.24601).

## 2. The eight primitives (R2 rubric, re-stated)

A harness qualifies as an RLM substrate if it exposes:

| # | Primitive | Minimum acceptance test |
|---|-----------|------------------------|
| 1 | Code execution | Root LM can run arbitrary code that mutates a named `context` value |
| 2 | `llm_query(prompt)` | Sync sub-LLM dispatch that returns a string |
| 3 | Parallel sub-LLM (opt.) | Batched / concurrent sub-LLM calls |
| 4 | Context-as-variable | Root LM references context by name without re-sending |
| 5 | FINAL / SUBMIT | Sentinel to halt the loop and commit an answer |
| 6 | Trajectory recording | Append-only log of (reasoning, code, output) |
| 7 | Sub-LM model routing | Root and sub-LM independently configurable |
| 8 | Budget enforcement | Hard caps on iterations, LLM calls, output size |

Codex CLI (per R6 §2.5) covers 4 NATIVE + 2 SHIM-OK + 2 SHIM-EXPENSIVE. The
SHIM-EXPENSIVE cells (`llm_query`, parallel sub-LLM) are what the `rlm` MCP
server fixes.

## 3. Codex-specific mapping (detailed)

### 3.1 Code execution — NATIVE

Codex's built-in shell is sandboxed (`--sandbox workspace-write` default, or
`danger-full-access` for unrestricted). Python is invoked via `uv run python`
or `python3` directly. Each shell call is a fresh process — **state must
persist to disk**. This is identical to the Claude Code Bash tool constraint.

### 3.2 `llm_query` — SHIM-EXPENSIVE without MCP

**Preferred: MCP server.** With the `rlm` MCP server registered in
`~/.codex/config.toml`, the root LM calls `llm_query` as a first-class tool
with no process boot cost. This is the adapter's happy path.

**Fallback: shell-out-to-self.** If no MCP is available, the root LM can
invoke:

```bash
codex exec -p "Answer using ONLY the chunk below: $CHUNK\n\nQuestion: $Q" \
    --model "o4-mini" \
    --json
```

Every such call boots a fresh Codex process. On a 40-subcall RLM run this is
40× full init = minutes of wall clock + re-auth + MCP re-initialize. R6 §2.5
benchmark: ~$3-5 overhead per run on an OpenAI-hosted root. **Use the MCP
server.**

### 3.3 Parallel sub-LLM — NATIVE through MCP

The `rlm` MCP server block in `config.toml` includes
`supports_parallel_tool_calls = true`. When the root LM emits N `llm_query`
tool calls in one turn, Codex dispatches them concurrently. Cap at 10 parallel
to avoid API rate-limit burn.

### 3.4 Context-as-variable — SHIM-OK via pickle

Same pattern as all file-backed RLM adapters. `/tmp/rlm/context.pkl` is the
canonical location. Every shell cell loads at start, dumps at end.

### 3.5 FINAL / SUBMIT — NATIVE via `--output-schema`

Codex's killer feature for RLM. When the root LM is invoked via:

```bash
codex exec --output-schema /absolute/path/to/final_schema.json \
    "/rlm summarize this file /file README.md"
```

...Codex enforces that the final assistant message is valid JSON matching the
schema. Any prose response gets rejected and retried. This is the *tightest*
FINAL sentinel in the cohort.

Use this schema (also written by `install-codex.sh` to
`~/.codex/skills/rlm/references/final_schema.json`):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RLMFinalAnswer",
  "type": "object",
  "required": ["final"],
  "additionalProperties": false,
  "properties": {
    "final": {
      "type": "string",
      "description": "The answer to the user's question, synthesized from sub-LLM calls."
    },
    "iterations": {
      "type": "integer",
      "minimum": 0,
      "description": "Number of Python/shell cells executed."
    },
    "llm_calls": {
      "type": "integer",
      "minimum": 0,
      "description": "Number of sub-LLM calls dispatched."
    },
    "trajectory_path": {
      "type": "string",
      "description": "Absolute path to the JSONL trajectory file."
    }
  }
}
```

For interactive sessions without `--output-schema`, use the file-based
sentinel `/tmp/rlm/FINAL.txt` instead.

### 3.6 Trajectory log — NATIVE via `--json`

In headless mode (`codex exec --json`), Codex emits JSONL events on stdout
covering every tool call, model turn, and error. For interactive sessions
(which don't emit the JSONL stream), the orchestrator script writes to
`/tmp/rlm/trajectory.jsonl` when invoked with `log <tool> <preview>`.

### 3.7 Sub-LM model routing — NATIVE

The `llm_query` MCP tool takes a `model` argument. The root LM chooses the
sub-model per call. Default: `claude-haiku-4-5-20251001`. The root LM is
Codex's own model (typically `gpt-5` or configured via `--model`).

### 3.8 Budget enforcement — SHIM-OK

Codex has no native PreToolUse hook. `scripts/rlm_orchestrator.py` is the
workaround:

```bash
# Before every llm_query call, the root LM runs:
uv run python /absolute/path/to/rlm/scripts/rlm_orchestrator.py check
# Exit code 2 = over budget, halt.
```

The MCP server **also** logs every completion to `/tmp/rlm/sub_calls.jsonl`
with cost attribution (see `src/harness_rlm/mcp_server.py`). That log is the
authoritative cost ledger — if the root LM forgets to call the orchestrator,
the sub_calls log still captures every invocation.

## 4. Invocation patterns

### 4.1 Interactive

```bash
codex
> $rlm summarize this file /file /path/to/100k_doc.md
```

Codex's interactive `$skill-name` mention loads the skill into context.

### 4.2 Headless with structured FINAL

```bash
codex exec \
    --output-schema /Users/you/.codex/skills/rlm/references/final_schema.json \
    --json \
    --sandbox workspace-write \
    "[paste SKILL.md body inline] User query: summarize /path/to/doc.md"
```

Codex has **no `--skill` flag** (confirmed via developers.openai.com/codex/cli/reference, 2026-04-21). Headless skill invocation requires pasting the body inline into the prompt.

### 4.3 Pipeline-friendly stdin

```bash
cat context.txt | codex exec -
```

Stdin becomes additional context. Combine with `--output-schema` for JSON-in,
JSON-out RLM pipelines.

## 5. Cost model (from R6 §3)

On a Sonnet-root + Haiku-sub mix with 10 sub-LLM calls per run:

| Path | Per-run cost | Notes |
|---|---|---|
| MCP server (direct Anthropic) | ~$0.08 | No process boot; just API tokens |
| Shell-out-to-self (`codex exec -p`) | ~$3-5 | 10× full Codex init |

**Amortization threshold** from R6: harness overhead falls below 20% of total
cost at ~200K tokens of total task input (Codex CLI's per-call overhead is
lower than Claude Code's per-spawn overhead, so the break-even is earlier).

## 6. What this adapter does NOT do

- **Doesn't install the MCP server itself.** Points at the canonical
  `src/harness_rlm/mcp_server.py` in the monorepo. `install-codex.sh` prints
  the `~/.codex/config.toml` block for the user to paste.
- **Doesn't support recursion depth > 1.** Sub-LLM is a stateless text-in /
  text-out proxy (same as all other harness-rlm adapters). This matches the
  paper's experimental setup.
- **Doesn't bypass Codex's sandbox.** All shell calls honor `--sandbox`.
  `danger-full-access` is user-opt-in, not adapter default.

## 7. Related

- Claude Code adapter: `adapters/claude_code/` (reference implementation)
- R6 harness landscape: `/Users/rshah/rlm-research/R6_harness_landscape.md`
- Open Agent Skills Standard: https://agentskills.io/specification
- Codex skills docs: https://developers.openai.com/codex/skills
- Codex MCP docs: https://developers.openai.com/codex/mcp
