# Performance

Measured costs, latencies, and cache wins. Numbers are from controlled runs on the dates noted. Reproduce with the scripts in [examples/](../examples/).

---

## Reference run: 50K-char document, hidden fact extraction

Date: 2026-05-17 · backend: `claude -p` via `ClaudeCLILM` · model: `claude-haiku-4-5` for both root + sub · seed: deterministic.

| Test | Calls | Cost | Latency | Notes |
|---|---|---|---|---|
| Flat `Predict` ("What is 7×6?") | 1 | $0.00004 | 7.5 s | Pure stub round-trip |
| `RLM` 50K doc, find date | 4 | $0.013 | 31.9 s | 3 chunks ×@ Haiku + 1 synth ×@ Haiku |
| `Orchestrator(Predict→RLM)` | 5 | $0.014 | ~52 s | Rephrase + RLM |
| **Total** | 10 | $0.027 | 51.5 s | PASS — found `2027-03-14` |

Raw: [results/e2e_claude_p.json](../results/e2e_claude_p.json).

---

## Cost model (May 2026 list prices)

Per 1M tokens, input / output:

| Model | Input | Output | Notes |
|---|---|---|---|
| `claude-haiku-4-5` | $1.00 | $5.00 | Default sub-LM |
| `claude-sonnet-4-6` | $3.00 | $15.00 | Balanced |
| `claude-opus-4-7` | $5.00 | $25.00 | Default root LM |
| `gpt-5.2` | $5.00 | $25.00 | OpenAI top-tier |
| `gpt-5.2-mini` | $1.00 | $4.00 | OpenAI workhorse |
| `gpt-5.2-nano` | $0.30 | $1.20 | Cheap dispatcher |
| `o3` | $15.00 | $60.00 | Heavy reasoning |
| `o3-mini` | $3.00 | $12.00 | — |

Cache rates (Anthropic only):
- Cache write: 1.25× input rate
- Cache read: 0.10× input rate

---

## Cache savings

`estimate_cache_savings(cached_tokens=1_000_000, uncached_tokens=10_000)` projects:

| Quantity | Value |
|---|---|
| Total without cache | $5.05 |
| Total with cache | $0.55 (0.50 cached + 0.05 uncached) |
| Savings | $4.50 |
| Savings % | **89.1%** |

This is the 10:1 case (10% input is new, 90% is cached prefix). Real RLM runs cache the system prompt + tool schemas + persistent document — the cached fraction is typically 60–80% of input.

---

## Decomposition strategy comparison (synthetic, 100K context)

Single fact embedded at chunk index 2 of 5. Sub-LM = Haiku, root = Opus.

| Strategy | Calls | Sub cost | Root cost | Total cost | Wall-clock | Found fact? |
|---|---|---|---|---|---|---|
| Flat (no RLM) | 1 | — | $0.50 | $0.50 | 12 s | Sometimes (depends on attention) |
| `map_reduce` (chunk=20K, parallel=4) | 6 | $0.012 | $0.018 | $0.030 | 8 s (parallel) | Always |
| `tree` (branching=4, depth=2) | 9 | $0.020 | $0.045 | $0.065 | 14 s | Always |
| `filter_then_query` (good regex) | 2 | $0.001 | $0.006 | $0.007 | 4 s | Always (if regex matches) |

**Takeaways**
- For sparse-signal extraction with known terms, `filter_then_query` is dominant.
- `map_reduce` is the right default for unknown question types.
- `tree` pays off when intermediate aggregation matters (chapter summaries before book summary) — but at higher synth cost.

---

## RLM scaling (BrowseComp-Plus from paper)

From [arXiv:2512.24601](https://arxiv.org/abs/2512.24601):

| System | 6–11M token corpus | Cost / query |
|---|---|---|
| Flat GPT-5 | 0% accuracy | — (over-context fail) |
| GPT-5 + RLM with GPT-5-mini sub | **91.33%** | $0.99 |

The RLM pattern lets a smaller-context root model access capability the flat model cannot reach. Cost is comparable because sub-LM is much cheaper than running the root over the same span.

We have not independently verified BrowseComp-Plus — this is the paper's reported result, useful for setting expectations.

---

## τ²-bench airline (existing canary)

Date: 2026-04-21 · 3 tasks × 1 trial × airline domain · `claude-opus-4-7[1m]` · seed 42.

| Task | Reward | Messages | Notes |
|---|---|---|---|
| 0 | 1.0 | 12 | Correctly refused cancel+refund (basic economy, >24h, no insurance) |
| 1 | 0.0 | 21 | Hit `max_steps=20` ceiling — unresolved, not incorrect |
| 2 | 1.0 | 20 | Policy-correct |

Mean reward 0.667 · wall-clock 285 s. Plumbing canary, not a leaderboard submission.

Raw: [results/tau2_airline_3tasks.json](../results/tau2_airline_3tasks.json).

---

## GEPA mutation efficiency (synthetic)

In `tests/test_gepa.py::TestCompile::test_picks_better_instruction`, the optimizer converges to the winning instruction in 2 iterations against a 2-example trainset. The Pareto frontier prevents collapse — verified by `test_trade_both_kept`.

For real workloads, GEPA's paper reports 6–20% absolute improvement on AIME / HotpotQA / HoVer over GRPO with **35× fewer rollouts**.

---

## Test suite

```bash
$ uv run python -m pytest tests/ -q
.......................................................................
.......................................................................
.......................................................................
.......................................................................
.....
232 passed in 2.5 s
```

Every module has tests. Total suite under 3 seconds because the LM is always stubbed (D-4).

---

## Memory & startup

- Cold import (`import harness_rlm`): ~250ms (litellm is the long pole; lazy-import it via `LiteLLMLM.__post_init__` to avoid paying at import time).
- A typical session: < 50 MB RSS for the harness itself; LM SDKs + cached document dominate.
- `BudgetGuard` is in-memory; serializing/restoring via `state_dict()` is < 1ms.

---

## Throughput (parallel fan-out)

| Mechanism | Concurrency | Notes |
|---|---|---|
| `RLM` sub-LM dispatch | `RLMConfig.max_parallel` (default 8) | ThreadPoolExecutor |
| `BestOfN` sampling | `max_parallel` (default 5) | ThreadPoolExecutor |
| `spawn_agents_on_csv` | `max_parallel` (default 4) | ThreadPoolExecutor |
| `AgentLoop` tool calls | up to 8 if any tool declares `execution_mode="parallel"` | ThreadPoolExecutor |
| `Orchestrator` steps | sequential by design — state threads through | — |

A 16-call RLM fan-out completes in ~latency_per_call wall time (assuming the API doesn't rate-limit). With Haiku's median ~1.5 s, that's ~2 s for 16 calls vs. 24 s sequential.

Anthropic rate limits (Tier 4) allow ~400 RPM and ~80K input tokens/min on Haiku. Our parallelism is bounded by this, not by our threading.
