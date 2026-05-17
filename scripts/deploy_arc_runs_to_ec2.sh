#!/usr/bin/env bash
# Deploy + run full ARC-AGI evals (1, 2, 3) on an EC2 VM.
#
# Why a VM:
#   Max-effort Opus 4.7 1M context per claude -p call uses ~250–400 MB of RAM
#   while the SDK + harness-rlm itself sits at ~500 MB. At parallel=50 you need
#   roughly 25-30 GB RAM just for the worker pool. A laptop with 48GB will
#   swap-thrash at parallel=20; a VM with 64-128 GB runs at parallel=50+
#   cleanly and shaves wall-clock 3-5x.
#
# Recommended instance: c7i.4xlarge (16 vCPU, 32 GB) for parallel=30, or
# r7i.4xlarge (16 vCPU, 128 GB) for parallel=80+.
#
# Prerequisites on the VM (one-time):
#   1. Ubuntu 24.04 (or any glibc 2.39+ Linux).
#   2. `claude` CLI (Claude Code) installed + logged in (`claude` once to OAuth).
#      Subscription tier with no message cap (Enterprise/Max).
#   3. `git`, `curl`, `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
#   4. ARC_API_KEY available (or use anonymous).
#
# Usage on the VM:
#   git clone https://github.com/rachittshah/harness-rlm
#   cd harness-rlm
#   export ARC_API_KEY=944e972e-80ba-45c8-aabe-1fe1b5acb227
#   bash scripts/deploy_arc_runs_to_ec2.sh

set -euo pipefail

if ! command -v uv >/dev/null; then
  echo "ERROR: uv not installed. curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
if ! command -v claude >/dev/null; then
  echo "ERROR: claude CLI not installed. https://docs.anthropic.com/claude-code/install" >&2
  exit 1
fi
if [ -z "${ARC_API_KEY:-}" ]; then
  echo "WARN: ARC_API_KEY not set — using anonymous (scorecards won't attach to your account)" >&2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Pull the eval datasets.
mkdir -p /tmp
if [ ! -d /tmp/ARC-AGI ]; then
  echo "[deploy] cloning ARC-AGI-1..."
  git clone --depth 1 https://github.com/fchollet/ARC-AGI.git /tmp/ARC-AGI
fi
if [ ! -d /tmp/ARC-AGI-2 ]; then
  echo "[deploy] cloning ARC-AGI-2..."
  git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git /tmp/ARC-AGI-2
fi

# Sync deps.
echo "[deploy] uv sync..."
uv sync --extra dev 2>&1 | tail -3

# Detect available memory + suggest parallelism.
TOTAL_MEM_GB=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo 2>/dev/null || \
  sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}' || echo 16)
SUGGEST_P1=$(( TOTAL_MEM_GB * 1 ))   # 1 GB/worker rough estimate
if [ "$SUGGEST_P1" -gt 80 ]; then SUGGEST_P1=80; fi
if [ "$SUGGEST_P1" -lt 10 ]; then SUGGEST_P1=10; fi
SUGGEST_P3=$(( SUGGEST_P1 / 4 ))  # ARC-3 is heavier (interactive games)
if [ "$SUGGEST_P3" -lt 3 ]; then SUGGEST_P3=3; fi

echo "[deploy] Detected ${TOTAL_MEM_GB} GB RAM. Suggested parallel: ARC-1/2 = ${SUGGEST_P1}, ARC-3 = ${SUGGEST_P3}."

mkdir -p results/arc logs

# Launch the three runs detached, with logs.
LAUNCH_TS=$(date -u +%Y%m%dT%H%M%SZ)

echo "[deploy] launching ARC-AGI-1 (400 tasks, parallel=${SUGGEST_P1})..."
nohup uv run python -m arc_integration.runner \
  --dataset /tmp/ARC-AGI/data/evaluation \
  --num-tasks 400 --k 2 --parallel "$SUGGEST_P1" \
  --timeout 600 --checkpoint-every 10 \
  --model 'claude-opus-4-7[1m]' --effort max \
  --mcp-config results/mcp_config_harness_rlm.json \
  --out "results/arc/arc1_full_400tasks_k2.json" \
  > "logs/arc1_${LAUNCH_TS}.log" 2>&1 &
ARC1_PID=$!
echo "  PID=$ARC1_PID  log=logs/arc1_${LAUNCH_TS}.log"

echo "[deploy] launching ARC-AGI-2 (120 tasks, parallel=${SUGGEST_P1})..."
nohup uv run python -m arc_integration.runner \
  --dataset /tmp/ARC-AGI-2/data/evaluation \
  --num-tasks 120 --k 2 --parallel "$SUGGEST_P1" \
  --timeout 600 --checkpoint-every 5 \
  --model 'claude-opus-4-7[1m]' --effort max \
  --mcp-config results/mcp_config_harness_rlm.json \
  --out "results/arc/arc2_full_120tasks_k2.json" \
  > "logs/arc2_${LAUNCH_TS}.log" 2>&1 &
ARC2_PID=$!
echo "  PID=$ARC2_PID  log=logs/arc2_${LAUNCH_TS}.log"

echo "[deploy] launching ARC-AGI-3 (25 games, parallel=${SUGGEST_P3})..."
nohup uv run python -m arc_integration.arc3_runner \
  --num-games 25 --max-turns 60 --parallel "$SUGGEST_P3" \
  --timeout 600 --checkpoint-every 2 \
  --model 'claude-opus-4-7[1m]' --effort max \
  --mcp-config results/mcp_config_harness_rlm.json \
  --out "results/arc/arc3_25games.json" \
  > "logs/arc3_${LAUNCH_TS}.log" 2>&1 &
ARC3_PID=$!
echo "  PID=$ARC3_PID  log=logs/arc3_${LAUNCH_TS}.log"

# Caffeinate is macOS-only; on Linux, nohup is enough — the processes survive
# logout and don't need wakelock since servers don't sleep.
if command -v caffeinate >/dev/null; then
  echo "[deploy] (macOS) attaching caffeinate guards..."
  caffeinate -i -s -w "$ARC1_PID" > /dev/null 2>&1 &
  caffeinate -i -s -w "$ARC2_PID" > /dev/null 2>&1 &
  caffeinate -i -s -w "$ARC3_PID" > /dev/null 2>&1 &
fi

# Save PID file for monitoring/teardown.
{
  echo "ARC1_PID=$ARC1_PID"
  echo "ARC2_PID=$ARC2_PID"
  echo "ARC3_PID=$ARC3_PID"
  echo "LAUNCH_TS=$LAUNCH_TS"
} > "logs/arc_runs_${LAUNCH_TS}.pids"

echo ""
echo "[deploy] all 3 runs launched. To monitor:"
echo "  tail -f logs/arc1_${LAUNCH_TS}.log"
echo "  tail -f logs/arc2_${LAUNCH_TS}.log"
echo "  tail -f logs/arc3_${LAUNCH_TS}.log"
echo ""
echo "  watch -n 30 'ls -la results/arc/ ; jq -r .scoring.pass_rate_pct results/arc/arc1_full_400tasks_k2.json 2>/dev/null'"
echo ""
echo "[deploy] When all done, aggregate with:"
echo "  uv run python -m arc_integration.aggregate \\"
echo "    --arc1 results/arc/arc1_full_400tasks_k2.json \\"
echo "    --arc2 results/arc/arc2_full_120tasks_k2.json \\"
echo "    --arc3 results/arc/arc3_25games.json \\"
echo "    --out  results/arc/REPORT.md"
