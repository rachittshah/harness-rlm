#!/usr/bin/env bash
# install.sh — install harness-rlm adapters into the target harness's config dir.
#
# Usage:
#   ./install.sh                         # installs all available adapters (default)
#   ./install.sh --harness claude-code   # Claude Code only
#   ./install.sh --harness goose         # Goose only
#   ./install.sh --harness codex         # Codex only
#   ./install.sh --harness opencode      # OpenCode only
#   ./install.sh --harness all           # same as no flag
#
# Idempotent: safe to re-run. Overwrites existing files in target dirs.
# Does NOT edit settings.json files — prints the JSON blocks to paste instead.
#
# Adapter directories that don't yet exist (because their adapter isn't built)
# are skipped with a warning, not a fatal error.

set -euo pipefail

# --- Resolve repo root (directory containing this script) -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# --- Parse args ----------------------------------------------------------------
HARNESS="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --harness)
            HARNESS="${2:-}"
            if [[ -z "$HARNESS" ]]; then
                echo "ERROR: --harness requires a value" >&2
                exit 1
            fi
            shift 2
            ;;
        --harness=*)
            HARNESS="${1#--harness=}"
            shift
            ;;
        -h|--help)
            sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown flag: $1" >&2
            echo "Use --help for usage." >&2
            exit 1
            ;;
    esac
done

case "$HARNESS" in
    claude-code|goose|codex|opencode|all) ;;
    *)
        echo "ERROR: --harness must be one of: claude-code, goose, codex, opencode, all (got: $HARNESS)" >&2
        exit 1
        ;;
esac

# --- Adapter installers --------------------------------------------------------
install_claude_code() {
    local adapter_dir="${REPO_ROOT}/adapters/claude_code"
    if [[ ! -d "$adapter_dir" ]]; then
        echo "[claude-code] SKIP — adapter dir missing: $adapter_dir"
        return 0
    fi

    local CLAUDE_DIR="${HOME}/.claude"
    local SKILL_DIR="${CLAUDE_DIR}/skills/rlm"
    local AGENTS_DIR="${CLAUDE_DIR}/agents"
    local HOOKS_DIR="${CLAUDE_DIR}/hooks/harness-rlm"

    local SKILL_SRC="${adapter_dir}/SKILL.md"
    local AGENT_SRC="${adapter_dir}/agents/rlm-subquery-haiku.md"
    local HOOKS_SRC_DIR="${adapter_dir}/hooks"

    local missing=0
    for f in "$SKILL_SRC" "$AGENT_SRC" "${HOOKS_SRC_DIR}/budget_guard.py" "${HOOKS_SRC_DIR}/trajectory_log.py"; do
        if [[ ! -f "$f" ]]; then
            echo "[claude-code] ERROR: missing source file: $f" >&2
            missing=1
        fi
    done
    if [[ $missing -ne 0 ]]; then
        return 1
    fi

    mkdir -p "$SKILL_DIR" "$AGENTS_DIR" "$HOOKS_DIR"
    cp "$SKILL_SRC" "${SKILL_DIR}/SKILL.md"
    echo "[claude-code] installed skill:    ${SKILL_DIR}/SKILL.md"
    cp "$AGENT_SRC" "${AGENTS_DIR}/rlm-subquery-haiku.md"
    echo "[claude-code] installed subagent: ${AGENTS_DIR}/rlm-subquery-haiku.md"
    cp "${HOOKS_SRC_DIR}/budget_guard.py"    "${HOOKS_DIR}/budget_guard.py"
    cp "${HOOKS_SRC_DIR}/trajectory_log.py"  "${HOOKS_DIR}/trajectory_log.py"
    chmod +x "${HOOKS_DIR}/budget_guard.py" "${HOOKS_DIR}/trajectory_log.py"
    echo "[claude-code] installed hooks:    ${HOOKS_DIR}/{budget_guard,trajectory_log}.py"

    cat <<EOF

===============================================================================
  Claude Code — MANUAL STEP 1: register the MCP server in ~/.claude/settings.json
===============================================================================

Add the block below under the "mcpServers" key (create the key if missing):

{
  "mcpServers": {
    "rlm": {
      "command": "uv",
      "args": ["run", "--with", "anthropic", "--with", "mcp", "python", "${REPO_ROOT}/src/harness_rlm/mcp_server.py"],
      "env": {"ANTHROPIC_API_KEY": "\${ANTHROPIC_API_KEY}"}
    }
  }
}

This exposes mcp__rlm__llm_query, which bypasses Claude Code's 50K-token
Task-tool overhead by calling the Anthropic API directly. It is the preferred
sub-LLM dispatch path for the /rlm skill.

===============================================================================
  Claude Code — MANUAL STEP 2: register the hooks in ~/.claude/settings.json
===============================================================================

Add the block below under the "hooks" key (merge with existing hooks if any):

{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Task",
        "hooks": [
          {"type": "command", "command": "${HOOKS_DIR}/budget_guard.py"}
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash|Task",
        "hooks": [
          {"type": "command", "command": "${HOOKS_DIR}/trajectory_log.py"}
        ]
      }
    ]
  }
}

Hooks are no-ops outside an RLM session (gated on /tmp/rlm/state.json existence).

EOF
}

install_goose() {
    local adapter_dir="${REPO_ROOT}/adapters/goose"
    if [[ ! -d "$adapter_dir" ]] || [[ -z "$(ls -A "$adapter_dir" 2>/dev/null)" ]]; then
        echo "[goose] SKIP — adapter dir missing or empty: $adapter_dir"
        return 0
    fi

    # Goose config dir: $XDG_CONFIG_HOME/goose or ~/.config/goose.
    local target="${XDG_CONFIG_HOME:-$HOME/.config}/goose"
    mkdir -p "$target"
    cp -R "$adapter_dir"/. "$target"/
    echo "[goose] installed adapter files -> $target"
    cat <<EOF

===============================================================================
  Goose — MANUAL STEP: register the rlm MCP server as a Goose extension
===============================================================================

Edit ~/.config/goose/config.yaml (or goose-config.yaml) and add under 'extensions':

  extensions:
    rlm:
      type: stdio
      command: uv
      args: ["run", "--with", "anthropic", "--with", "mcp", "python",
             "${REPO_ROOT}/src/harness_rlm/mcp_server.py"]
      env:
        ANTHROPIC_API_KEY: \${ANTHROPIC_API_KEY}
      enabled: true

Then: goose session start  (or your usual invocation). The adapter's recipe/
subrecipe files should be picked up on next session.

EOF
}

install_codex() {
    local adapter_dir="${REPO_ROOT}/adapters/codex"
    if [[ ! -d "$adapter_dir" ]] || [[ -z "$(ls -A "$adapter_dir" 2>/dev/null)" ]]; then
        echo "[codex] SKIP — adapter dir missing or empty: $adapter_dir"
        return 0
    fi

    # Codex skills convention: ~/.codex/skills/<name>/
    local target="${HOME}/.codex/skills/rlm"
    mkdir -p "$target"
    cp -R "$adapter_dir"/. "$target"/
    echo "[codex] installed adapter files -> $target"
    cat <<EOF

===============================================================================
  Codex — MANUAL STEP: register the rlm MCP server in ~/.codex/config.toml
===============================================================================

Under [mcp_servers.rlm] add:

  [mcp_servers.rlm]
  command = "uv"
  args = ["run", "--with", "anthropic", "--with", "mcp", "python",
          "${REPO_ROOT}/src/harness_rlm/mcp_server.py"]

  [mcp_servers.rlm.env]
  ANTHROPIC_API_KEY = "\${ANTHROPIC_API_KEY}"

Codex follows the Open Agent Skills standard — the SKILL.md at $target/SKILL.md
is discovered automatically.

EOF
}

install_opencode() {
    local adapter_dir="${REPO_ROOT}/adapters/opencode"
    if [[ ! -d "$adapter_dir" ]] || [[ -z "$(ls -A "$adapter_dir" 2>/dev/null)" ]]; then
        echo "[opencode] SKIP — adapter dir missing or empty: $adapter_dir"
        return 0
    fi

    # OpenCode plugin convention: ~/.config/opencode/plugins/<name>/
    local target="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/plugins/rlm"
    mkdir -p "$target"
    cp -R "$adapter_dir"/. "$target"/
    echo "[opencode] installed adapter files -> $target"
    cat <<EOF

===============================================================================
  OpenCode — MANUAL STEP: register the rlm MCP server
===============================================================================

Edit ~/.config/opencode/opencode.json and add under 'mcp':

  "mcp": {
    "rlm": {
      "type": "local",
      "command": ["uv", "run", "--with", "anthropic", "--with", "mcp",
                  "python", "${REPO_ROOT}/src/harness_rlm/mcp_server.py"],
      "environment": {"ANTHROPIC_API_KEY": "\${ANTHROPIC_API_KEY}"}
    }
  }

Then restart OpenCode. The plugin at $target is loaded on next session.

EOF
}

# --- Dispatch ------------------------------------------------------------------
echo "harness-rlm installer — target: $HARNESS"
echo "repo root: $REPO_ROOT"
echo

case "$HARNESS" in
    claude-code) install_claude_code ;;
    goose)       install_goose ;;
    codex)       install_codex ;;
    opencode)    install_opencode ;;
    all)
        install_claude_code
        install_goose
        install_codex
        install_opencode
        ;;
esac

cat <<'EOF'

===============================================================================
  NEXT STEPS (common to all harnesses)
===============================================================================

  1. Set your API key:    export ANTHROPIC_API_KEY=sk-ant-...
  2. Restart the harness  so skills / subagents / hooks / MCP servers are loaded.
  3. Smoke test:          invoke /rlm in the harness.
  4. Real test:           /rlm summarize this file /file README.md

For design docs, see:
  - /Users/rshah/rlm-research/R2_primitives_claude_code.md  (Claude Code)
  - /Users/rshah/rlm-research/R6_harness_landscape.md         (all harnesses)
===============================================================================
EOF
