# Registering the `harness-rlm` MCP server in Codex CLI

The MCP server exposes **10 tools** — `llm_query`, `rlm_run`, `predict`,
`chain_of_thought`, `best_of_n`, `compress_text`, `chunk_text`,
`dispatch_subagent`, `list_subagents`, `estimate_cost`. Full schema:
[../../docs/MCP_TOOLS.md](../../docs/MCP_TOOLS.md). Selection decision tree:
[../../skill/SKILL.md](../../skill/SKILL.md).

The MCP server is shipped in the monorepo at `src/harness_rlm/mcp_server.py`
— **this adapter does NOT bundle it.**

## Why MCP matters

Codex CLI has **no native sub-agent primitive** (R6 §2.5). Every sub-LLM call
must go through one of two paths:

1. **MCP server (preferred)** — direct text-in / text-out proxy. No process
   boot on each call. This is the happy path.
2. **Shell-out-to-self** — `codex exec -p "<sub-prompt>" --json` per call.
   Each call pays full process init + auth + MCP re-init. R6 §3 measured this
   at 10-100× the MCP path for a 10-sub-LLM-call run.

The `rlm` MCP server is load-bearing. Without it, RLM on Codex CLI is
prohibitively expensive.

## Config file location

**Global**: `~/.codex/config.toml`
**Per-project** (overrides global): `.codex/config.toml` in the project root.

Source: https://developers.openai.com/codex/mcp (verified 2026-04-21).

## The config block to add

Open `~/.codex/config.toml` (or create it) and append:

```toml
[mcp_servers.harness-rlm]
command = "rlm-mcp-server"
startup_timeout_sec = 15
tool_timeout_sec = 120
supports_parallel_tool_calls = true
enabled = true

[mcp_servers.harness-rlm.env]
# ANTHROPIC_API_KEY is required. Codex substitutes ${...} from the parent env.
ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"
PYTHONUNBUFFERED = "1"
```

`rlm-mcp-server` is installed by `pip install -e .` / `uv sync` (it's the
project's console script). If your Codex CLI environment doesn't have it on
PATH, use the absolute-path form:

```toml
[mcp_servers.harness-rlm]
command = "uv"
args = ["run", "--with", "anthropic", "--with", "mcp",
        "python", "/ABSOLUTE/PATH/TO/harness-rlm/src/harness_rlm/mcp_server.py"]
cwd = "/ABSOLUTE/PATH/TO/harness-rlm"
```

Replace `/ABSOLUTE/PATH/TO/harness-rlm` with the actual clone path. The
installer (`install-codex.sh`) prints this block with your path already
substituted.

## Field reference

| Field | Purpose | Source |
|---|---|---|
| `command` | Launcher binary. `uv` ensures dependencies. | [codex-config](https://developers.openai.com/codex/mcp) |
| `args` | Arguments to `command`. Spawns the stdio MCP server. | [codex-config](https://developers.openai.com/codex/mcp) |
| `cwd` | Working directory for the MCP subprocess. | [codex-config](https://developers.openai.com/codex/config-reference) |
| `env` | Env map. `${FOO}` substitutes from parent env. | [codex-config](https://developers.openai.com/codex/config-reference) |
| `startup_timeout_sec` | Max wait for handshake. 10s default is tight for `uv`. | [codex-mcp](https://developers.openai.com/codex/mcp) |
| `tool_timeout_sec` | Max wait for one tool call. Haiku sometimes hits 60s. | [codex-mcp](https://developers.openai.com/codex/mcp) |
| `supports_parallel_tool_calls` | Enables batched `llm_query` calls in one turn. | [codex-mcp](https://developers.openai.com/codex/mcp) |
| `enabled` | Gate. `false` disables without deleting the block. | [codex-mcp](https://developers.openai.com/codex/mcp) |

## Validation

After editing, run:

```bash
codex mcp list
```

You should see `rlm` in the list. Then confirm the tool is discovered:

```bash
codex exec --sandbox workspace-write "List all tools available from the 'rlm' MCP server."
```

Expected output includes the 10 tools: `llm_query`, `rlm_run`, `predict`,
`chain_of_thought`, `best_of_n`, `compress_text`, `chunk_text`,
`dispatch_subagent`, `list_subagents`, `estimate_cost`.

## Selftest

The MCP server also supports a direct selftest (no Codex needed):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python /ABSOLUTE/PATH/TO/harness-rlm/src/harness_rlm/mcp_server.py --selftest
```

Expected: `[selftest] PASS` with a sub-$0.001 cost line.

## Troubleshooting

- **`ANTHROPIC_API_KEY is not set`**: export it in the shell where you run
  Codex. Codex's `env` sub-table substitutes from the parent env.
- **`uv: command not found`**: Install `uv` (`brew install uv`), or change
  `command = "uv"` to `command = "/absolute/path/to/python"` and drop the
  `--with` args.
- **Tool timeout at 60s**: Increase `tool_timeout_sec`. Haiku max_tokens=1024
  can take up to 45s on a cold API path.
- **`Codex does not use MCP servers defined in config.toml`**
  ([issue #3441](https://github.com/openai/codex/issues/3441)): resolved in
  Codex CLI >=0.42. Upgrade with `npm install -g @openai/codex@latest`.
