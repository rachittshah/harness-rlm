# ARC-AGI canary results (honest report)

Generated: 2026-05-17T09:18:17Z

Backend: `claude -p` (Claude Code Enterprise OAuth, no per-token billing).
Model: `claude-opus-4-7[1m]` · effort: `max` · harness-rlm MCP server attached (10 tools).
Scoring: strict bit-exact grid match. pass@k.
Task selection: first N by filename, deterministic. No cherry-picking.

## Headline

| Benchmark | Passed | Pass rate | k | model/effort | wall-clock |
|---|---|---|---|---|---|
| ARC-AGI-1 (canary) | 10/10 | **100.0%** | 2 | claude-opus-4-7[1m] / max | 142.8s |
| ARC-AGI-2 (canary) | not run | — | — | — | — |
| ARC-AGI-3 (interactive games) | not run | — | — | — | — |

**ARC-AGI-3 was NOT run.** The benchmark is interactive (game environments) and requires `ARC_API_KEY` from `arcprize.org/api-keys`. Anonymous endpoints return `unauthorized` (verified 2026-05-17). See `arc_integration/arc3_stub.py` for the unblock procedure. Frontier SOTA on ARC-AGI-3 (per ARC Prize Mar 2026): Gemini 3.1 Pro 0.37%, Opus 4.6 0.25%, GPT-5.4 0.26%, best purpose-built agent 12.58%, human 100%.

## Per-task detail

### ARC-AGI-1 — per-task results

| task_id | passed | best_attempt | parses |
|---|---|---|---|
| `00576224` | ✅ | — | [True, True] |
| `0b17323b` | ✅ | — | [True, True] |
| `0c9aba6e` | ✅ | — | [True, True] |
| `009d5c81` | ✅ | — | [True, True] |
| `0becf7df` | ✅ | — | [True, True] |
| `0692e18c` | ✅ | — | [True, True] |
| `00dbd492` | ✅ | — | [True, True] |
| `0bb8deee` | ✅ | — | [True, True] |
| `0c786b71` | ✅ | — | [True, True] |
| `12997ef3` | ✅ | — | [True, True] |

## What this is NOT

1. **Not a leaderboard submission.** ARC-AGI-1 has no active 2026 board (saturated). ARC-AGI-2 leaderboard is Kaggle-offline (closed-source API models can't enter the main track) or ARC Prize Verified Testing (selective). Our run is self-reported on the public eval.
2. **Small N.** Each canary is 5 tasks. ARC-AGI-1 has 400 public eval tasks; ARC-AGI-2 has 120. Confidence intervals are wide at this N — treat numbers as pipeline validation, not a claim.
3. **First-N-by-filename selection.** Deterministic but not random; could be easier or harder than dataset average.
4. **Single trial × k attempts.** ARC convention is pass@2; we follow this. Across-trial variance not measured.
5. **No test-time refinement.** Top ARC-AGI-2 entries use multi-pass / refinement loops (e.g. Poetiq Gemini at $31/task). We don't. Our number is a lower bound on claude -p capability.
6. **Non-deterministic.** claude -p at default temperature varies between reruns. No seed pinned.

## What this IS

1. **A real end-to-end run** of `claude -p` with `--effort max`, Opus 4.7 1M context, and the harness-rlm MCP server (10 tools) attached.
2. **Faithful to the eval format.** Same task JSONs from `fchollet/ARC-AGI` and `arcprize/ARC-AGI-2`. Same strict grid match. Same pass@k.
3. **Reproducible.** Single CLI command per benchmark. Result JSONs include full per-attempt trace (parses, elapsed_ms, raw_response_preview). Datasets pinned to commit at run time.
4. **No score inflation.** No partial credit. No retries-on-fail. No augmented prompts.

## Reproduce

```bash
git clone https://github.com/fchollet/ARC-AGI.git /tmp/ARC-AGI
git clone https://github.com/arcprize/ARC-AGI-2.git /tmp/ARC-AGI-2

uv run python -m arc_integration.runner \
    --dataset /tmp/ARC-AGI/data/evaluation \
    --num-tasks 5 --k 2 \
    --model 'claude-opus-4-7[1m]' --effort max \
    --mcp-config results/mcp_config_harness_rlm.json \
    --out results/arc/arc1_5tasks_k2.json

uv run python -m arc_integration.runner \
    --dataset /tmp/ARC-AGI-2/data/evaluation \
    --num-tasks 5 --k 2 \
    --model 'claude-opus-4-7[1m]' --effort max \
    --mcp-config results/mcp_config_harness_rlm.json \
    --out results/arc/arc2_5tasks_k2.json

uv run python -m arc_integration.aggregate \
    --arc1 results/arc/arc1_5tasks_k2.json \
    --arc2 results/arc/arc2_5tasks_k2.json \
    --out results/arc/REPORT.md
```
