# harness-rlm — OpenCode adapter

OpenCode adapter for the [Recursive Language Model](https://arxiv.org/abs/2512.24601)
pattern. Turns OpenCode ([opencode.ai](https://opencode.ai), open-source TS
agent harness by Anomaly) into an RLM substrate via one plugin + one subagent,
with the MCP server reused from the root repo.

## What this adapter ships

| File | Purpose |
|---|---|
| `plugins/rlm.ts` | OpenCode TS plugin. Registers `rlm_run(context_path, query, budgets?)` tool. Dispatches sub-LLM queries via the `rlm-mcp-server` MCP tool. Enforces budgets via `tool.execute.before` hook. Mirrors every tool call to `/tmp/rlm/<session>.trajectory.jsonl` via `tool.execute.after`. |
| `agents/rlm-subquery.md` | Custom subagent. Takes (chunk + question) and returns relevant facts. Fallback path when the MCP server is not registered. Locked to `tools: {write:false, edit:false, bash:false, task:false}`. |
| `mcp-config.md` | Documentation for the `rlm` MCP server block that goes in `opencode.json`. |
| `install-opencode.sh` | Idempotent installer. Copies the plugin + subagent into `~/.config/opencode/{plugins,agents}/` and prints the MCP config block to splice into `opencode.json`. |

## Primitive mapping (R6 §2.3 — OpenCode scored 21/24)

| RLM primitive (paper) | OpenCode primitive | Implementation in this adapter |
|---|---|---|
| Sandboxed REPL | Built-in `bash` tool | Used indirectly via `tool.execute.before` hook budgets |
| Sub-LLM dispatch | MCP `rlm_llm_query` tool (preferred); `task` subagent (fallback) | `plugins/rlm.ts::subLlmQuery` — MCP path is default, saves ~50K re-injection tokens per call |
| Model routing | OpenCode provider config + `model` arg on `rlm_run` | Plugin exposes `model` tool arg; defaults to `claude-haiku-4-5` |
| Trajectory logging | `tool.execute.after` hook + JSONL file | `/tmp/rlm/<session>.trajectory.jsonl` |
| Budget enforcement | `tool.execute.before` hook + in-tool counter | Throws to abort (only documented abort mechanism); mirrors `DEFAULT_BUDGETS` from `src/harness_rlm/core.py` |
| Context ingestion | `readFile` inside tool; chunked via `chunkContext` | Mirrors `harness_rlm.core.chunk_context` — overlap-based |
| Session state | In-memory `Map<sessionId, SessionState>` + trajectory file | Plugin state API not documented (verified 2026-04-21); file-based is the portable path |
| Headless execution | `opencode run "..."` | Documented below |

R6's score rationale: third-highest primitive coverage (behind Claude Code and Codex), strongest plugin ergonomics for TS teams, first-class MCP support. The 3-point gap vs Claude Code was session-state persistence (no documented API) and `task` permissioning granularity — both worked around here via file-based trajectory + per-subagent `permission.task` config.

## Installation

```bash
# 1. Clone and install the MCP server side (Python)
cd /path/to/harness-rlm
uv sync

# 2. Run the OpenCode-side installer
./adapters/opencode/install-opencode.sh
# Prints the MCP config block for step 3.

# 3. Splice the printed "mcp" block into ~/.config/opencode/opencode.json
#    (see mcp-config.md for the exact JSON)

# 4. Export your key and restart OpenCode
export ANTHROPIC_API_KEY=sk-ant-...
```

See `mcp-config.md` for the exact config block and a standalone smoke-test
command for the MCP server.

## Usage

### Headless

```bash
# Long-doc summarization via the rlm_run tool
opencode run "use rlm_run with context_path=./long_doc.md query='summarize the key findings'"

# Or with an explicit budget
opencode run "use rlm_run with context_path=./contract.md query='find all termination clauses' max_llm_calls=20"
```

### Interactive (TUI)

```
> Use the rlm_run tool to summarize ./long_doc.md
```

The agent will call `rlm_run`, which:
1. Reads the file
2. Chunks into ~5K-char overlapping segments
3. Dispatches one `rlm_llm_query` MCP call per chunk (Haiku)
4. Synthesizes a final answer from the per-chunk notes
5. Returns `{answer, trajectory, counters, session_id}`

### Reading the trajectory

```bash
# Live-tail the current run
tail -f /tmp/rlm/*.trajectory.jsonl

# Inspect all sub-LLM calls the MCP server made
jq . < /tmp/rlm/sub_calls.jsonl
```

## Why a plugin (not just a skill / prompt)

1. **Budget enforcement needs a hook.** A markdown skill can *ask* the model
   to stop after N calls, but can't enforce it. OpenCode's
   `tool.execute.before` hook lets us throw on cap-cross — the only
   documented abort mechanism (verified 2026-04-21). That's the difference
   between "the agent usually stays within budget" and "the agent cannot
   exceed budget".
2. **Trajectory capture needs a hook.** `tool.execute.after` fires for every
   tool the root agent runs while an RLM session is open, not just the ones
   the skill author remembered to log. This mirrors the Claude-Code
   adapter's `PostToolUse` hook path.
3. **Typed tool args.** The `rlm_run` tool takes structured budgets
   (`max_iterations`, `max_llm_calls`, etc). With Zod schemas the plugin
   rejects malformed calls at the tool boundary; a free-text skill has to
   re-parse the prompt on every turn.
4. **State per-session.** In-memory `Map<sessionId, SessionState>` lets
   parallel `rlm_run` invocations in the same OpenCode process not
   clobber each other's counters. A skill has no scoped state.

## Known gaps / honest limitations

- **No documented OpenCode state-persistence API** (verified against
  `https://opencode.ai/docs/plugins` on 2026-04-21). Trajectory is written
  to `/tmp/rlm/<session>.trajectory.jsonl` from the plugin; on crash,
  in-memory counters are lost and a replay would re-charge them. The MCP
  server's own `/tmp/rlm/sub_calls.jsonl` is the source of truth for cost.
- **Session ID synthesis.** The plugin context does not expose a first-party
  session ID at tool-registration time; `rlm_run` generates its own. If you
  need to correlate with an `opencode export <sessionID>` dump, use the
  `session_id` field returned by `rlm_run`, not OpenCode's session ID.
- **MCP tool namespacing.** OpenCode's exact tool-name convention for MCP
  tools (`rlm_llm_query` vs `mcp__rlm__llm_query`) depends on the build;
  the plugin dispatches by the name the SDK client accepts. If dispatch
  fails, check `opencode serve` logs for the registered name and adjust
  `subLlmQuery` in `plugins/rlm.ts`.
- **Fallback path has a cost regression.** If the MCP server is not
  registered, the plugin falls back to OpenCode's built-in `task` tool with
  the `rlm-subquery` subagent. This works, but pays the ~50K-token
  re-injection tax per sub-call that the MCP path was designed to avoid.
  Register the MCP server.

## References

- RLM paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- OpenCode plugins: [opencode.ai/docs/plugins](https://opencode.ai/docs/plugins)
- OpenCode agents: [opencode.ai/docs/agents](https://opencode.ai/docs/agents)
- OpenCode MCP: [opencode.ai/docs/mcp-servers](https://opencode.ai/docs/mcp-servers)
- Harness landscape score: `R6 §2.3` — 21/24
