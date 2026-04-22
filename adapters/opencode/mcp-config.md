# Registering `rlm-mcp-server` in OpenCode

OpenCode's MCP config lives in the top-level `opencode.json` (or
`opencode.jsonc`). Global config: `~/.config/opencode/opencode.json`.
Project override: `./opencode.json` at the project root.

Verified against https://opencode.ai/docs/mcp-servers on 2026-04-21:
local MCP servers use `"type": "local"` and an array-form `command`.

## Minimal block (add under the top-level `mcp` key)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "rlm": {
      "type": "local",
      "command": [
        "uv",
        "run",
        "--with", "anthropic",
        "--with", "mcp",
        "python",
        "/Users/rshah/harness-rlm/src/harness_rlm/mcp_server.py"
      ],
      "enabled": true,
      "environment": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

Replace `/Users/rshah/harness-rlm` with the absolute path to your clone.

## What this exposes

The MCP server in `src/harness_rlm/mcp_server.py` advertises a single tool:
`llm_query(prompt, model, max_tokens?, system?) -> str`. Under OpenCode's
MCP-tool-namespacing convention the tool becomes addressable as
`mcp__rlm__llm_query` (or `rlm_llm_query`, depending on the client build
— the plugin tries both by name in `plugins/rlm.ts::subLlmQuery`).

## Merging with existing config

If you already have an `mcp` section, splice the `"rlm"` entry alongside
your other servers. OpenCode merges keys, not objects:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "other-server": { ... },
    "rlm": { ... }
  }
}
```

## Verifying

```bash
# 1. Confirm your API key is set
echo "$ANTHROPIC_API_KEY" | head -c 10   # expect: sk-ant-...

# 2. Confirm the server starts standalone (Ctrl-C after "initialized")
uv run --with anthropic --with mcp python \
  /Users/rshah/harness-rlm/src/harness_rlm/mcp_server.py --selftest

# 3. Restart OpenCode (MCP servers are connected at startup)
# 4. Inside an OpenCode session, verify the tool is listed:
#    Ask the agent: "list your available MCP tools"
#    Expect to see: rlm_llm_query (or mcp__rlm__llm_query)
```

## Why this path beats OpenCode's built-in `task` tool

Every `task` invocation re-injects the system prompt + tool catalog +
agent memory (~50K tokens against Claude Code; OpenCode's TS plugin
equivalent is in the same order of magnitude because the subagent boot
sequence hydrates CLAUDE.md / AGENTS.md + MCP descriptions + hook state).
For a 40-sub-call RLM run that is ~$1.73 at Haiku rates just in
re-injection tax. The MCP path is a raw Anthropic API round-trip at
~$0.79 for the same workload (R6 §5.1).

This is the "non-negotiable design constraint" from the root README.md.
The plugin (`plugins/rlm.ts`) will fall back to OpenCode's `task` tool
with the `rlm-subquery` agent if `rlm_llm_query` is not registered, but
that path is a cost regression by design.
