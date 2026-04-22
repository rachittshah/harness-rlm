#!/usr/bin/env bash
# install-codex.sh — install the harness-rlm /rlm skill into ~/.codex/skills/
# and print the MCP config block for the user to paste into ~/.codex/config.toml.
#
# Idempotent: safe to re-run. Overwrites existing files in ~/.codex/skills/rlm/.
# Does NOT modify ~/.codex/config.toml — printed instructions instead, since
# TOML editing in bash is fragile and may conflict with other MCP servers.
#
# References:
#   - https://developers.openai.com/codex/skills
#   - https://developers.openai.com/codex/mcp
#   - https://agentskills.io/specification

set -euo pipefail

# --- Resolve repo root (two levels up from this script) ------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Target paths --------------------------------------------------------------
CODEX_DIR="${HOME}/.codex"
SKILL_DIR="${CODEX_DIR}/skills/rlm"
MCP_SERVER_PATH="${REPO_ROOT}/src/harness_rlm/mcp_server.py"

# --- Sanity: source files must exist -------------------------------------------
SKILL_SRC="${ADAPTER_DIR}/rlm"

for f in \
    "${SKILL_SRC}/SKILL.md" \
    "${SKILL_SRC}/scripts/rlm_orchestrator.py" \
    "${SKILL_SRC}/references/loop.md" \
    "${SKILL_SRC}/references/final_schema.json" \
    "${SKILL_SRC}/agents/openai.yaml" \
    "$MCP_SERVER_PATH"
do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing source file: $f" >&2
        echo "Are you running install-codex.sh from adapters/codex/?" >&2
        exit 1
    fi
done

# --- Detect ~/.codex/ ----------------------------------------------------------
if [[ ! -d "$CODEX_DIR" ]]; then
    echo "Creating $CODEX_DIR (Codex CLI data dir did not exist)." >&2
    mkdir -p "$CODEX_DIR"
fi

# --- Copy skill tree -----------------------------------------------------------
mkdir -p "${SKILL_DIR}/scripts" "${SKILL_DIR}/references" "${SKILL_DIR}/agents"

cp "${SKILL_SRC}/SKILL.md"                         "${SKILL_DIR}/SKILL.md"
cp "${SKILL_SRC}/scripts/rlm_orchestrator.py"      "${SKILL_DIR}/scripts/rlm_orchestrator.py"
cp "${SKILL_SRC}/references/loop.md"               "${SKILL_DIR}/references/loop.md"
cp "${SKILL_SRC}/references/final_schema.json"     "${SKILL_DIR}/references/final_schema.json"
cp "${SKILL_SRC}/agents/openai.yaml"               "${SKILL_DIR}/agents/openai.yaml"

chmod +x "${SKILL_DIR}/scripts/rlm_orchestrator.py"

echo "installed skill:   ${SKILL_DIR}/SKILL.md"
echo "installed script:  ${SKILL_DIR}/scripts/rlm_orchestrator.py"
echo "installed refs:    ${SKILL_DIR}/references/{loop.md,final_schema.json}"
echo "installed agents:  ${SKILL_DIR}/agents/openai.yaml"

# ------------------------------------------------------------------------------
# MCP server: print config.toml block with paths already substituted.
# ------------------------------------------------------------------------------
cat <<EOF

===============================================================================
  MANUAL STEP 1 — register the MCP server in ~/.codex/config.toml
===============================================================================

Open (or create) ~/.codex/config.toml and append the block below. If you
already have [mcp_servers.*] entries, just add this one alongside them.

-------------------------------------------------------------------------------
[mcp_servers.rlm]
command = "uv"
args = [
    "run",
    "--with", "anthropic",
    "--with", "mcp",
    "python",
    "${MCP_SERVER_PATH}",
]
cwd = "${REPO_ROOT}"
startup_timeout_sec = 15
tool_timeout_sec = 120
supports_parallel_tool_calls = true
enabled = true

[mcp_servers.rlm.env]
ANTHROPIC_API_KEY = "\${ANTHROPIC_API_KEY}"
PYTHONUNBUFFERED = "1"
-------------------------------------------------------------------------------

This exposes the tool  llm_query  (namespaced as rlm.llm_query in Codex) which
bypasses Codex's shell-out-to-self tax by calling the Anthropic API directly.
It is the REQUIRED sub-LLM dispatch path for the rlm skill — without it, every
sub-LLM call falls back to expensive shell-out-to-self via 'codex exec -p'.

===============================================================================
  MANUAL STEP 2 — set your API key
===============================================================================

  export ANTHROPIC_API_KEY=sk-ant-...

Codex's env sub-table substitutes \${ANTHROPIC_API_KEY} from your shell.

===============================================================================
  MANUAL STEP 3 — verify
===============================================================================

  # 1. Confirm MCP server is discovered:
  codex mcp list
  # Expected: see 'rlm' in the list.

  # 2. Run the MCP server's own selftest (no Codex needed):
  uv run python ${MCP_SERVER_PATH} --selftest
  # Expected: '[selftest] PASS'

  # 3. Smoke test the skill (interactive):
  codex
  > \$rlm hello — just confirm you loaded
  # Expected: Codex mentions the rlm skill is active.

  # 4. Smoke test headless with structured FINAL:
  codex exec \\
      --output-schema ${SKILL_DIR}/references/final_schema.json \\
      --sandbox workspace-write \\
      --json \\
      "Act as the rlm skill from ~/.codex/skills/rlm/SKILL.md. User query: summarize /path/to/doc.md"

===============================================================================
  LIMITATIONS (honest accounting, from R6 §2.5)
===============================================================================

  1. Codex has NO --skill CLI flag (verified 2026-04-21 on
     developers.openai.com/codex/cli/reference). Headless skill invocation
     requires pasting the skill body inline or instructing Codex to activate
     it. Interactive sessions support '\$rlm' mention syntax.

  2. Codex has NO native PreToolUse hook. Budget enforcement relies on the
     root LM calling 'rlm_orchestrator.py check' before each sub-LLM call.
     The MCP server's /tmp/rlm/sub_calls.jsonl log is the authoritative
     cost ledger.

  3. Without the 'rlm' MCP server, the skill falls back to shell-out-to-self
     via 'codex exec -p', which costs 10-100x more per sub-LLM call. The MCP
     server is load-bearing — install it.

For adapter docs see:   ${ADAPTER_DIR}/README.md
For MCP config details: ${ADAPTER_DIR}/mcp-config.md
===============================================================================
EOF
