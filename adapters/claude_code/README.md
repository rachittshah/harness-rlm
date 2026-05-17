# harness-rlm — Claude Code adapter

Turns Claude Code into an RLM substrate per arXiv:2512.24601.

## What this adapter provides

- `SKILL.md` — the `/rlm` skill (self-contained; mirrors `skill/SKILL.md` with Claude-Code-specific frontmatter + primitive notes)
- `agents/rlm-subquery-haiku.md` — fallback sub-LLM subagent (Haiku, 3-turn cap, read-only)
- `hooks/budget_guard.py` — PreToolUse hook. Blocks Bash/Task with exit 2 once `/tmp/rlm/state.json::llm_calls` > 50
- `hooks/trajectory_log.py` — PostToolUse hook. Appends `{timestamp, tool, input, output_chars}` to `/tmp/rlm/trajectory.jsonl`

Both hooks are stdlib-only and are no-ops outside an RLM session (they gate on the existence of `/tmp/rlm/state.json`, which the skill's Step 0 creates).

## Install

From the repo root:

```bash
./install.sh --harness claude-code
```

This copies:
- `SKILL.md` → `~/.claude/skills/rlm/SKILL.md`
- `agents/rlm-subquery-haiku.md` → `~/.claude/agents/rlm-subquery-haiku.md`
- `hooks/budget_guard.py` → `~/.claude/hooks/harness-rlm/budget_guard.py` (chmod +x)
- `hooks/trajectory_log.py` → `~/.claude/hooks/harness-rlm/trajectory_log.py` (chmod +x)

The installer does NOT edit `~/.claude/settings.json` — it prints the exact JSON blocks to paste for the MCP server and the hook registrations.

## Manual steps

1. Register the `harness-rlm` MCP server under `mcpServers` in `~/.claude/settings.json`. The server exposes **10 tools** (`llm_query`, `rlm_run`, `predict`, `chain_of_thought`, `best_of_n`, `compress_text`, `chunk_text`, `dispatch_subagent`, `list_subagents`, `estimate_cost`). Full schema: [../../docs/MCP_TOOLS.md](../../docs/MCP_TOOLS.md).

   ```json
   {
     "mcpServers": {
       "harness-rlm": {
         "command": "rlm-mcp-server",
         "env": {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}
       }
     }
   }
   ```

   If `rlm-mcp-server` isn't on PATH for the Claude Code launcher, use the explicit-path form:

   ```json
   {
     "mcpServers": {
       "harness-rlm": {
         "command": "uv",
         "args": ["run", "--with", "anthropic", "--with", "mcp", "python",
                  "/path/to/harness-rlm/src/harness_rlm/mcp_server.py"],
         "env": {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}
       }
     }
   }
   ```

2. Register the hooks under `hooks`:

   ```json
   {
     "hooks": {
       "PreToolUse": [
         {"matcher": "Bash|Task",
          "hooks": [{"type": "command", "command": "/Users/YOU/.claude/hooks/harness-rlm/budget_guard.py"}]}
       ],
       "PostToolUse": [
         {"matcher": "Bash|Task",
          "hooks": [{"type": "command", "command": "/Users/YOU/.claude/hooks/harness-rlm/trajectory_log.py"}]}
       ]
     }
   }
   ```

3. Set `ANTHROPIC_API_KEY` in the shell that launches Claude Code.

4. Restart Claude Code so skills, subagents, hooks, and the MCP server are loaded.

## Example

```
claude -p '/rlm summarize this file /file README.md'
```

The `/rlm` skill:
1. Ingests via `/file` and pickles to `/tmp/rlm/context.pkl`
2. Forms a decomposition plan (map-reduce / targeted search / progressive refine)
3. Fans out `mcp__rlm__llm_query` calls against chunks (preferred) or `rlm-subquery-haiku` subagent (fallback)
4. Writes the synthesized answer to `/tmp/rlm/FINAL.txt` and echoes it
5. Reports iterations, sub-call count, and the `/tmp/rlm/trajectory.jsonl` path

## Budget caps (enforced)

- `max_iterations`: 20 (skill `maxTurns: 25` + manual check)
- `max_llm_calls`: 50 (PreToolUse hook exits 2 at 51+)
- `max_output_chars`: 10000 per Python cell (manual truncate)

## Why MCP over Task tool

`mcp__rlm__llm_query` calls the Anthropic API directly via the `rlm` MCP server, bypassing Claude Code's ~50K-token Task-tool re-injection overhead (system prompt + CLAUDE.md + skills + MCP tool definitions, re-injected into every subagent spawn). On a 40-sub-call RLM run, that's ~$1.73 vs. ~$0.79 — a 55% cost reduction. See `/Users/rshah/rlm-research/R6_harness_landscape.md` §5 for the economics.

The `rlm-subquery-haiku` subagent remains as a fallback for environments where the MCP server isn't registered.
