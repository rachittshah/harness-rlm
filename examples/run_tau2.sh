#!/usr/bin/env bash
# Run the tau2 airline domain against the harness-rlm claude-headless agent.
#
# This script prefers the programmatic Python entry point in
# examples/run_tau2_py.py (more reliable than `tau2 run --agent ...` because
# our custom agent has to be registered at import time). It still uses
# tau2's own uv-managed env under /Users/rshah/tau2-bench.

set -euo pipefail

: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be set (export it before running)}"

REPO_ROOT="/Users/rshah/harness-rlm"
TAU2_ROOT="/Users/rshah/tau2-bench"
RESULTS_DIR="${REPO_ROOT}/results"
OUT_FILE="${RESULTS_DIR}/tau2_airline.json"

mkdir -p "${RESULTS_DIR}"

echo "========== TAU2 AIRLINE RUN START =========="
echo "repo_root=${REPO_ROOT}"
echo "tau2_root=${TAU2_ROOT}"
echo "out_file=${OUT_FILE}"
echo "agent=harness-rlm/claude-headless"
echo "num_tasks=3"
echo "============================================"

cd "${TAU2_ROOT}"

# Run via the programmatic entrypoint so our agent factories are registered
# before any orchestrator is constructed. We use tau2's uv env but point it
# at harness-rlm's script, and rely on harness_rlm being installed in that
# environment (`uv pip install -e /Users/rshah/harness-rlm`). If it isn't,
# we fall back to running with harness-rlm's own uv env + PYTHONPATH.
set +e
uv run --project "${TAU2_ROOT}" python "${REPO_ROOT}/examples/run_tau2_py.py" \
    --domain airline \
    --agent harness-rlm/claude-headless \
    --user-llm openai/gpt-4.1 \
    --num-trials 1 \
    --num-tasks 3 \
    --out "${OUT_FILE}"
RUN_RC=$?
set -e

if [[ "${RUN_RC}" -ne 0 ]]; then
    echo "[run_tau2.sh] tau2 uv env run failed (rc=${RUN_RC}); retrying via harness-rlm env..."
    cd "${REPO_ROOT}"
    PYTHONPATH="${TAU2_ROOT}/src:${REPO_ROOT}/src:${REPO_ROOT}" \
        uv run python "${REPO_ROOT}/examples/run_tau2_py.py" \
        --domain airline \
        --agent harness-rlm/claude-headless \
        --user-llm openai/gpt-4.1 \
        --num-trials 1 \
        --num-tasks 3 \
        --out "${OUT_FILE}"
    RUN_RC=$?
fi

echo "========== TAU2 AIRLINE RUN END (rc=${RUN_RC}) =========="

if [[ -f "${OUT_FILE}" ]]; then
    echo "Summary: ${OUT_FILE}"
    python3 - <<PY
import json, sys
from pathlib import Path
p = Path("${OUT_FILE}")
data = json.loads(p.read_text())
results = data.get("results", [])
rewards = [r.get("reward") for r in results if isinstance(r.get("reward"), (int, float))]
print(f"  tasks_run:   {len(results)}")
print(f"  mean_reward: {sum(rewards)/len(rewards):.3f}" if rewards else "  mean_reward: n/a (no numeric rewards)")
print(f"  elapsed_sec: {data.get('elapsed_sec')}")
PY
else
    echo "No summary file at ${OUT_FILE} — the run produced no output."
fi

exit "${RUN_RC}"
