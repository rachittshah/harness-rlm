#!/usr/bin/env bash
# install.sh — install the harness-rlm /rlm skill, sub-agent, and hooks into
# ~/.claude/, and print MCP server registration instructions.
#
# Idempotent: safe to re-run. Overwrites existing files in ~/.claude/skills/rlm/,
# ~/.claude/agents/rlm-subquery-haiku.md, ~/.claude/hooks/harness-rlm/*.
#
# Does NOT modify ~/.claude/settings.json — printed instructions instead, since
# JSON editing in bash is fragile.

set -euo pipefail

# --- Resolve repo root (directory containing this script) -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# --- Target paths --------------------------------------------------------------
CLAUDE_DIR="${HOME}/.claude"
SKILL_DIR="${CLAUDE_DIR}/skills/rlm"
AGENTS_DIR="${CLAUDE_DIR}/agents"
HOOKS_DIR="${CLAUDE_DIR}/hooks/harness-rlm"

# --- Sanity: source files must exist -------------------------------------------
SKILL_SRC="${REPO_ROOT}/claude_skill/rlm/SKILL.md"
AGENT_SRC="${REPO_ROOT}/claude_skill/agents/rlm-subquery-haiku.md"
HOOKS_SRC_DIR="${REPO_ROOT}/claude_skill/hooks"

for f in "$SKILL_SRC" "$AGENT_SRC" "${HOOKS_SRC_DIR}/budget_guard.py" "${HOOKS_SRC_DIR}/trajectory_log.py"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing source file: $f" >&2
        echo "Are you running install.sh from the harness-rlm repo root?" >&2
        exit 1
    fi
done

# --- Create target dirs --------------------------------------------------------
mkdir -p "$SKILL_DIR" "$AGENTS_DIR" "$HOOKS_DIR"

# --- Copy skill ----------------------------------------------------------------
cp "$SKILL_SRC" "${SKILL_DIR}/SKILL.md"
echo "installed skill:    ${SKILL_DIR}/SKILL.md"

# --- Copy sub-agent ------------------------------------------------------------
cp "$AGENT_SRC" "${AGENTS_DIR}/rlm-subquery-haiku.md"
echo "installed subagent: ${AGENTS_DIR}/rlm-subquery-haiku.md"

# --- Copy hooks (make executable) ----------------------------------------------
cp "${HOOKS_SRC_DIR}/budget_guard.py"    "${HOOKS_DIR}/budget_guard.py"
cp "${HOOKS_SRC_DIR}/trajectory_log.py"  "${HOOKS_DIR}/trajectory_log.py"
chmod +x "${HOOKS_DIR}/budget_guard.py" "${HOOKS_DIR}/trajectory_log.py"
echo "installed hooks:    ${HOOKS_DIR}/{budget_guard,trajectory_log}.py"

# ------------------------------------------------------------------------------
# MCP server + hooks: print manual instructions (we don't edit settings.json).
# ------------------------------------------------------------------------------
cat <<'EOF'

===============================================================================
  MANUAL STEP 1 — register the MCP server in ~/.claude/settings.json
===============================================================================

Add the block below under the "mcpServers" key (create the key if missing):

{
  "mcpServers": {
    "rlm": {
      "command": "uv",
      "args": ["run", "--with", "anthropic", "--with", "mcp", "python", "/Users/rshah/harness-rlm/src/harness_rlm/mcp_server.py"],
      "env": {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}
    }
  }
}

This exposes the tool  mcp__rlm__llm_query  which bypasses Claude Code's 50K-token
Task-tool overhead by calling the Anthropic API directly. It is the preferred
sub-LLM dispatch path for the /rlm skill.

===============================================================================
  MANUAL STEP 2 — register the hooks in ~/.claude/settings.json
===============================================================================

Add the block below under the "hooks" key (merge with existing hooks if any):

{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Task",
        "hooks": [
          {"type": "command", "command": "/Users/rshah/.claude/hooks/harness-rlm/budget_guard.py"}
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash|Task",
        "hooks": [
          {"type": "command", "command": "/Users/rshah/.claude/hooks/harness-rlm/trajectory_log.py"}
        ]
      }
    ]
  }
}

Hooks are no-ops outside an RLM session (gated on /tmp/rlm/state.json existence).

===============================================================================
  NEXT STEPS
===============================================================================

  1. Set your API key:    export ANTHROPIC_API_KEY=sk-ant-...
  2. Restart Claude Code if it is already running (skills, subagents, hooks,
     and MCP servers are loaded at startup).
  3. Smoke test:          claude -p '/rlm hello'
  4. Real test:           claude -p '/rlm summarize this file /file README.md'

For design docs, see: /Users/rshah/rlm-research/R2_primitives_claude_code.md
===============================================================================
EOF
