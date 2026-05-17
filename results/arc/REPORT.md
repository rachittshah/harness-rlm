# harness-rlm × claude -p × ARC-AGI: honest eval report

Date: 2026-05-17 · model: `claude-opus-4-7[1m]` · effort: `max` · harness: harness-rlm v0.3.0 MCP server attached.

This report covers small canary runs on ARC-AGI-1 and ARC-AGI-2. **Not a leaderboard submission** — see "What this is NOT" below.

## Methodology

| Knob | Value |
|---|---|
| Agent | `claude -p` (Claude Code headless, OAuth via user's Claude Code Enterprise) |
| Model | `claude-opus-4-7[1m]` (Opus 4.7 with 1M-token context window) |
| Reasoning effort | `--effort max` |
| Permission mode | `bypassPermissions` (no human-in-the-loop, but local-only) |
| MCP server registered | `harness-rlm` (10 tools: `llm_query`, `rlm_run`, `predict`, `chain_of_thought`, `best_of_n`, `compress_text`, `chunk_text`, `dispatch_subagent`, `list_subagents`, `estimate_cost`) |
| `k` (attempts per task) | 2 (matches ARC-AGI public eval convention) |
| Per-call timeout | 600s |
| Task selection | First N tasks by filename — deterministic, no cherry-picking |
| ARC-AGI-1 source | `https://github.com/fchollet/ARC-AGI` (400 public evaluation tasks) |
| ARC-AGI-2 source | `https://github.com/arcprize/ARC-AGI-2` (120 public evaluation tasks) |
| ARC-AGI-3 status | **Not run** — requires `ARC_API_KEY` from arcprize.org/api-keys (see `arc_integration/arc3_stub.py`) |

### Prompt template

A single system prompt + user message per attempt. System prompt (verbatim):

> You are an expert at the ARC-AGI puzzle benchmark. You will see a few training input/output grid pairs that demonstrate a transformation rule. Then you will see a test input grid. Your job is to apply the same transformation rule to produce the test output grid.
>
> OUTPUT FORMAT (strict): respond with ONLY a JSON object of the form:
> `{"output": [[r0c0, r0c1, ...], [r1c0, r1c1, ...], ...]}`
> where each cell is a single integer 0-9. No commentary, no markdown fences, no explanation. JUST the JSON object.

User message: training examples + test input, space-separated cell values.

### Scoring

Strict pass@k bit-exact grid match. Score = (tasks where any of k attempts == gold) / total. No partial credit; no rounding.

## Results

> ⚠ **Run in progress** — numbers below are placeholders. Each run writes a complete JSON to `results/arc/` with per-task pass/fail, raw responses, and timing.

### ARC-AGI-1 (10 tasks, pass@2)

| | Value |
|---|---|
| Tasks attempted | TBD |
| Passed | TBD / 10 |
| Pass rate | TBD% |
| Wall-clock | TBD s |
| Result file | [results/arc/arc1_10tasks_k2.json](arc1_10tasks_k2.json) |

### ARC-AGI-2 (10 tasks, pass@2)

| | Value |
|---|---|
| Tasks attempted | TBD |
| Passed | TBD / 10 |
| Pass rate | TBD% |
| Wall-clock | TBD s |
| Result file | [results/arc/arc2_10tasks_k2.json](arc2_10tasks_k2.json) |

### ARC-AGI-3

**Not run.** Blocker: `ARC_API_KEY` not set (registration required at arcprize.org/api-keys). Stub at `arc_integration/arc3_stub.py` documents the next steps. SOTA reference per ARC Prize Mar-2026 blog:

- Humans: 100% · Gemini 3.1 Pro: 0.37% · Opus 4.6: 0.25% · GPT-5.4: 0.26% · best purpose-built agent: 12.58%

The floor is wide open. Without the API key we cannot generate even a canary.

## What this is NOT

1. **Not a leaderboard submission.** ARC-AGI-1 has no active 2026 leaderboard (saturated). ARC-AGI-2 leaderboard submissions go through Kaggle (offline, no-API rules) or the ARC Prize Verified Testing track (selective). Our run is self-reported on the public eval set.
2. **Not the full eval.** ARC-AGI-1 has 400 public eval tasks; we ran 10. ARC-AGI-2 has 120; we ran 10. A 10-task sample has very wide confidence intervals on the headline pass rate (±15 percentage points at 95% CI). Treat numbers as a sanity-check on the pipeline, not a claim.
3. **Not a hyperparameter-tuned setup.** Single system prompt, no in-context retrieval, no test-time refinement loops. ARC Prize's top entries use multi-pass refinement; we don't. So our numbers are a **lower bound** on what claude -p + harness-rlm can do with the same agent.
4. **Not gaming the metric.** Pass criteria is the same strict bit-exact grid match the leaderboard uses. We did not relax scoring for partial credit or near-misses.

## Reproduce

```bash
git clone https://github.com/fchollet/ARC-AGI.git /tmp/ARC-AGI
git clone https://github.com/arcprize/ARC-AGI-2.git /tmp/ARC-AGI-2

# ARC-AGI-1 canary
uv run python -m arc_integration.runner \
    --dataset /tmp/ARC-AGI/data/evaluation \
    --num-tasks 10 --k 2 \
    --model "claude-opus-4-7[1m]" --effort max \
    --mcp-config results/mcp_config_harness_rlm.json \
    --out results/arc/arc1_10tasks_k2.json

# ARC-AGI-2 canary
uv run python -m arc_integration.runner \
    --dataset /tmp/ARC-AGI-2/data/evaluation \
    --num-tasks 10 --k 2 \
    --model "claude-opus-4-7[1m]" --effort max \
    --mcp-config results/mcp_config_harness_rlm.json \
    --out results/arc/arc2_10tasks_k2.json
```

`results/mcp_config_harness_rlm.json` registers the `harness-rlm` MCP server so claude -p sees the 10 tools (`llm_query`, `rlm_run`, etc.). The MCP server requires `ANTHROPIC_API_KEY` to actually call sub-LLMs — without the key, only the no-API tools (`chunk_text`, `estimate_cost`, `list_subagents`) function. For ARC tasks (small grids), the model doesn't typically need the LLM-calling MCP tools — it can solve directly. Setting the key is recommended but not strictly required for the runner to complete.

## Limitations + caveats

1. Small N. 10 tasks each. Confidence interval is wide.
2. Single trial per task per attempt. ARC convention is pass@2; we follow this. But variance between trials of the same task isn't measured here.
3. The first 10 tasks by filename are not a random sample. They could be easier or harder than the dataset average.
4. `claude -p` is non-deterministic at temperature > 0. Across reruns of the same task with the same prompt, responses may differ. We did not pin a seed.
5. `--effort max` is documented but its exact behaviour (thinking budget, temperature) is implementation-defined. We treat the documented `max` as the highest effort tier available.
6. Per-task time was 24-300s in smoke testing. Total wall-clock for 10 tasks × 2 attempts depends on task complexity — expect 20-60 minutes.

## References

- ARC-AGI-1: Chollet, F. *On the Measure of Intelligence.* arXiv:1911.01547. [github.com/fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI).
- ARC-AGI-2: ARC Prize 2025. [arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025](https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025).
- ARC-AGI-3: ARC Prize 2026. [arcprize.org/blog/arc-agi-3-launch](https://arcprize.org/blog/arc-agi-3-launch).
- ARC Prize Verified Testing Policy: [arcprize.org/policy](https://arcprize.org/policy).
- Submission rules summary in this report informed by ARC Prize docs (May 2026).
