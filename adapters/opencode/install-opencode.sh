#!/usr/bin/env bash
# install-opencode.sh — install the harness-rlm OpenCode plugin, subagent, and
# print the MCP config block for manual splicing into opencode.json.
#
# Idempotent: safe to re-run. Overwrites:
#   ~/.config/opencode/plugins/rlm.ts
#   ~/.config/opencode/agents/rlm-subquery.md
#
# Does NOT modify opencode.json — prints instructions instead, since JSON
# editing in bash is fragile (and OpenCode supports multiple overlapping
# config sources whose precedence the installer cannot reason about safely).
#
# Verified against https://opencode.ai/docs/plugins and
# https://opencode.ai/docs/agents (2026-04-21):
#   - plugins in ~/.config/opencode/plugins/*.ts are auto-loaded at startup
#   - agents in  ~/.config/opencode/agents/*.md   are auto-loaded at startup
#   - opencode.json lives at ~/.config/opencode/opencode.json

set -euo pipefail

# --- Resolve repo root (directory containing adapters/opencode/) --------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Source files -------------------------------------------------------------
PLUGIN_SRC="${SCRIPT_DIR}/plugins/rlm.ts"
AGENT_SRC="${SCRIPT_DIR}/agents/rlm-subquery.md"
MCP_SERVER_SRC="${REPO_ROOT}/src/harness_rlm/mcp_server.py"

for f in "$PLUGIN_SRC" "$AGENT_SRC" "$MCP_SERVER_SRC"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing source file: $f" >&2
        echo "Are you running install-opencode.sh from adapters/opencode/?" >&2
        exit 1
    fi
done

# --- Target paths -------------------------------------------------------------
OC_CONFIG_DIR="${HOME}/.config/opencode"
PLUGIN_DIR="${OC_CONFIG_DIR}/plugins"
AGENT_DIR="${OC_CONFIG_DIR}/agents"
CONFIG_FILE="${OC_CONFIG_DIR}/opencode.json"

# --- Detect OpenCode install --------------------------------------------------
if [[ ! -d "$OC_CONFIG_DIR" ]]; then
    cat >&2 <<EOF
ERROR: ${OC_CONFIG_DIR} does not exist.

OpenCode stores its config in ~/.config/opencode/. If you have OpenCode
installed but have never launched it, run 'opencode' once to create the
directory, then re-run this installer.

If OpenCode is not installed:
    npm i -g opencode-ai      # or: curl -fsSL https://opencode.ai/install | bash

See https://opencode.ai/docs for the authoritative install path.
EOF
    exit 1
fi

if ! command -v opencode >/dev/null 2>&1; then
    echo "WARN: 'opencode' binary not on PATH. Config dir exists, so we will proceed." >&2
fi

# --- Create target dirs -------------------------------------------------------
mkdir -p "$PLUGIN_DIR" "$AGENT_DIR"

# --- Copy plugin --------------------------------------------------------------
cp "$PLUGIN_SRC" "${PLUGIN_DIR}/rlm.ts"
echo "installed plugin:  ${PLUGIN_DIR}/rlm.ts"

# --- Copy subagent ------------------------------------------------------------
cp "$AGENT_SRC" "${AGENT_DIR}/rlm-subquery.md"
echo "installed subagent: ${AGENT_DIR}/rlm-subquery.md"

# --- Check for @opencode-ai/plugin dep ----------------------------------------
# OpenCode auto-loads TS plugins, but the TS types require the SDK package to
# be importable. If the user runs their own bun/node project per-plugin this
# doesn't matter — if not, we flag it.
if ! (cd "$PLUGIN_DIR" && node -e "require.resolve('@opencode-ai/plugin')" >/dev/null 2>&1); then
    cat >&2 <<EOF

NOTE: @opencode-ai/plugin is not resolvable from ${PLUGIN_DIR}.
      OpenCode bundles Bun and resolves plugin imports itself at startup,
      so this is usually fine. If imports fail at runtime, initialise the
      plugin dir as a package:

          cd ${PLUGIN_DIR} && bun init -y && bun add @opencode-ai/plugin zod

EOF
fi

# --- MCP config instructions --------------------------------------------------
cat <<EOF

===============================================================================
  MANUAL STEP 1 — register rlm-mcp-server in ${CONFIG_FILE}
===============================================================================

Splice the block below into your opencode.json under the top-level "mcp" key
(create the key if missing; merge with existing servers otherwise):

{
  "\$schema": "https://opencode.ai/config.json",
  "mcp": {
    "rlm": {
      "type": "local",
      "command": [
        "uv", "run",
        "--with", "anthropic",
        "--with", "mcp",
        "python",
        "${MCP_SERVER_SRC}"
      ],
      "enabled": true,
      "environment": {
        "ANTHROPIC_API_KEY": "\${ANTHROPIC_API_KEY}"
      }
    }
  }
}

This exposes the tool named  rlm_llm_query  (or mcp__rlm__llm_query,
depending on your OpenCode build). The plugin dispatches sub-LLM calls to
this tool, bypassing OpenCode's \`task\` subagent-spawn re-injection tax.

===============================================================================
  MANUAL STEP 2 — verify installation
===============================================================================

  1. Set your API key:     export ANTHROPIC_API_KEY=sk-ant-...
  2. Smoke-test the MCP server standalone:
         uv run --with anthropic --with mcp python \\
             ${MCP_SERVER_SRC} --selftest
     Expect: "[selftest] PASS"
  3. Restart OpenCode (plugins + MCP servers load at startup).
  4. From an OpenCode session:
         opencode run "use the rlm_run tool to summarize /file README.md"
     Or from the TUI prompt:
         > rlm_run context_path=README.md query="summarize"

Trajectory logs: /tmp/rlm/<session>.trajectory.jsonl
Sub-call logs:   /tmp/rlm/sub_calls.jsonl  (written by rlm-mcp-server)

For design docs, see:  ${SCRIPT_DIR}/README.md
===============================================================================
EOF
