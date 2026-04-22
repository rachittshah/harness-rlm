#!/usr/bin/env bash
# install-goose.sh — install the harness-rlm Goose adapter recipes into the
# user's Goose config directory and print the MCP-server registration block.
#
# Idempotent. Safe to re-run. Does NOT edit config.yaml — that's fragile in
# bash; instead, we print the exact YAML block to paste.
#
# Verified against:
#   https://goose-docs.ai/docs/guides/config-files/   (config path)
#   https://goose-docs.ai/docs/guides/recipes/recipe-reference   (recipe schema)
#   https://goose-docs.ai/docs/guides/goose-cli-commands  (run flags)

set -euo pipefail

# --- Resolve adapter root (dir containing this script) -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_ROOT="$SCRIPT_DIR"
REPO_ROOT="$(cd "${ADAPTER_ROOT}/../.." && pwd)"

# --- Target paths ------------------------------------------------------------
# Goose config lives at ~/.config/goose on macOS & Linux (verified from docs).
# Windows isn't supported by this script.
GOOSE_CONFIG_DIR="${HOME}/.config/goose"
GOOSE_RECIPES_DIR="${GOOSE_CONFIG_DIR}/recipes"
GOOSE_CONFIG_FILE="${GOOSE_CONFIG_DIR}/config.yaml"

# --- Sanity: source recipes must exist ---------------------------------------
RLM_RECIPE_SRC="${ADAPTER_ROOT}/recipes/rlm.yaml"
SUBQUERY_RECIPE_SRC="${ADAPTER_ROOT}/recipes/rlm-subquery.yaml"
MCP_SERVER_SRC="${REPO_ROOT}/src/harness_rlm/mcp_server.py"

for f in "$RLM_RECIPE_SRC" "$SUBQUERY_RECIPE_SRC" "$MCP_SERVER_SRC"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing source file: $f" >&2
        echo "Are you running install-goose.sh from the adapters/goose/ dir of a checked-out harness-rlm repo?" >&2
        exit 1
    fi
done

# --- Check goose is installed (warn only; do not exit) -----------------------
if ! command -v goose >/dev/null 2>&1; then
    echo "WARN: 'goose' not found on PATH."
    echo "      Install from https://goose-docs.ai/docs/getting-started/installation"
    echo "      before running the recipes. Continuing with file copy anyway."
    echo
fi

# --- Check uv is installed (hard dependency of the MCP server) ---------------
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found on PATH. The rlm-mcp-server launches via 'uv run'." >&2
    echo "       Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# --- Create target dirs ------------------------------------------------------
mkdir -p "$GOOSE_RECIPES_DIR"
echo "goose config dir:   ${GOOSE_CONFIG_DIR}"
echo "goose recipes dir:  ${GOOSE_RECIPES_DIR}"

# --- Copy recipes ------------------------------------------------------------
cp "$RLM_RECIPE_SRC"      "${GOOSE_RECIPES_DIR}/rlm.yaml"
cp "$SUBQUERY_RECIPE_SRC" "${GOOSE_RECIPES_DIR}/rlm-subquery.yaml"
echo "installed recipe:   ${GOOSE_RECIPES_DIR}/rlm.yaml"
echo "installed recipe:   ${GOOSE_RECIPES_DIR}/rlm-subquery.yaml"

# --- Print MCP-server registration block -------------------------------------
# We do NOT edit config.yaml automatically — YAML-in-bash is fragile and the
# user may already have custom extensions. Paste-in is safer.

cat <<EOF

===============================================================================
  MANUAL STEP 1 — register the rlm-mcp-server extension
===============================================================================

Open ${GOOSE_CONFIG_FILE} and add the block below under the top-level
'extensions:' key. Create the key if it's absent.

Goose's extensions field is a **map**, keyed by extension name (not a list).
Source: https://goose-docs.ai/docs/guides/config-files/

-------------------------------------------------------------------------------
extensions:
  rlm-mcp-server:
    type: stdio
    name: rlm-mcp-server
    enabled: true
    bundled: false
    cmd: uv
    args:
      - run
      - --with
      - anthropic
      - --with
      - mcp
      - python
      - -m
      - harness_rlm.mcp_server
    env_keys:
      - ANTHROPIC_API_KEY
    timeout: 300
    description: "RLM sub-LLM dispatcher. Exposes llm_query() tool; proxies direct to Anthropic API."
-------------------------------------------------------------------------------

Alternative (if you want to point at the repo directly without installing the
package first), replace the last two args lines with:

      - ${MCP_SERVER_SRC}

===============================================================================
  MANUAL STEP 2 — install the rlm-mcp-server Python package
===============================================================================

The MCP server is imported as 'harness_rlm.mcp_server'. Install the package so
Goose can launch it via 'python -m':

  cd ${REPO_ROOT}
  uv sync          # creates .venv and installs harness_rlm + deps

Or if you prefer a system-wide install:

  uv pip install -e ${REPO_ROOT}

===============================================================================
  MANUAL STEP 3 — export your Anthropic API key
===============================================================================

  export ANTHROPIC_API_KEY="sk-ant-..."

Add to ~/.zshrc or ~/.bashrc to persist. Goose inherits env from the shell
that launches it.

===============================================================================
  SMOKE-TEST
===============================================================================

  1. Test the MCP server in isolation:

       ANTHROPIC_API_KEY="sk-ant-..." \\
         uv run --with anthropic --with mcp \\
         python -m harness_rlm.mcp_server --selftest

     Expected: '[selftest] PASS' and a cost line.

  2. Render the recipe (no LLM calls):

       goose run \\
         --recipe ${GOOSE_RECIPES_DIR}/rlm.yaml \\
         --params user_query="hello" \\
         --render-recipe

     Expected: the fully-templated YAML prints to stdout with no errors.

  3. Full end-to-end headless run:

       goose run \\
         --recipe ${GOOSE_RECIPES_DIR}/rlm.yaml \\
         --params user_query="Summarise the document in 3 bullets" \\
         --params context_path="/absolute/path/to/long_doc.md" \\
         --no-session \\
         --output-format json

     Expected: JSON on stdout; final answer also written to /tmp/rlm/FINAL.txt.

===============================================================================

Install complete.
EOF
