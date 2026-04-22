# What is a Recursive Language Model?

## One paragraph

A Recursive Language Model (RLM) is an inference pattern from [Zhang, Kraska, Khattab — arXiv:2512.24601](https://arxiv.org/abs/2512.24601) (MIT CSAIL, December 2025). The root LLM gets a sandboxed Python REPL where its input context lives as a named variable (`context`). Instead of feeding the whole context into the prompt, the LLM writes code to slice, filter, and index the variable — and when it needs semantic reasoning over a slice, it calls `llm_query(prompt, chunk)` which dispatches to a cheap sub-LLM (Haiku-4-5, GPT-5-mini). The root aggregates sub-answers and emits `FINAL(answer)`. "Recursive" means the root can nest sub-LLM calls across multiple turns; in the paper's experiments the depth is capped at 1.

## The problem RLMs solve

Two problems, one paradigm:

### Problem 1 — Hard context limits

Frontier models have 128K–2M token windows. A single SEC 10-K filing runs ~500K tokens. A forensic investigation bundling 40 articles runs ~2.2M. BrowseComp-Plus's 1K-docs setting runs 6–11M. Flat LLMs cannot *attempt* these tasks — the input doesn't fit.

### Problem 2 — Context rot

Long before the hard limit, attention degrades. Aggregation and multi-hop tasks drop sharply past ~50K tokens even on 1M-context models. The signal dilutes.

RLMs address both: the REPL abstraction is indifferent to input size. You feed in 10M tokens; they live as a string in a Python process, not as tokens in the LLM's attention.

## The numbers that matter

From the paper's Table 1 (cross-referenced in `/Users/rshah/rlm-research/R1_benchmarks_market.md`):

| Benchmark | Size | RLM Score | Best flat baseline | Delta |
|---|---|---|---|---|
| **BrowseComp-Plus (1K docs)** | 6–11M tokens | **91.33%** (GPT-5+mini) | 0% flat GPT-5 (over-context); 70.47% Summary Agent; 51% CodeAct+BM25 | +20.9 pts over next-best scaffold |
| **OOLONG** | 131K tokens | **56.50%** (GPT-5) | 44% flat GPT-5; 46% Summary Agent | +10.5 pts |
| **OOLONG-Pairs (F1)** | 32K tokens | **58.00** (GPT-5) | F1 0.04 flat GPT-5; 24.67 CodeAct | +33.3 pts |
| **LongBench-v2 CodeQA** | 23K–4.2M tokens | **62%** (GPT-5) | 58% Summary Agent GPT-5; 24% base | +4 pts |

**BrowseComp-Plus is the headline.** Flat GPT-5 scores 0% — the input doesn't fit its window. RLM+GPT-5+GPT-5-mini scores 91.33% at $0.99/query. The paradigm is not "cheaper within an existing frontier"; it's "accesses a regime flat LLMs cannot enter."

## The ablation — what actually earns the score

The paper separates the REPL from the recursion:

| Benchmark | RLM full | RLM, REPL only (no recursion) | Delta |
|---|---|---|---|
| OOLONG GPT-5 | 56.50 | 36.00 | **+20.5 pts** |
| OOLONG-Pairs GPT-5 | 58.00 | 43.93 | **+14.07 pts** |
| BrowseComp+ Qwen3-Coder | 44.66 | 46.00 | -1.34 |
| LongBench-v2 CodeQA | 62.00 | 66.00 | -4.00 |

**Recursion helps aggregation-dense tasks, hurts retrieval-dense tasks.** The REPL alone does most of the long-context heavy-lifting; adding sub-LLM recursion buys you cross-chunk synthesis at the cost of more tokens. On retrieval, recursion can actively hurt — the sub-LLM's interpretation of a chunk trumps the root's eventual read.

## When NOT to use an RLM

Three cases where RLMs waste money:

1. **Short contexts (<50K tokens).** The flat LLM already fits. RLM overhead (root plan + REPL cells + sub-LLM calls) is pure loss. τ²-bench airline policies are ~5K chars — an RLM here never triggers decomposition and runs as a trivially-wrapped flat agent.
2. **Symbolic/deterministic work.** Prime Intellect's RLMEnv includes a `math-python` benchmark where RLM *underperforms* direct LLM. If the task is `numpy`/`scipy`/`sympy` computation, give the LLM direct tool access; don't wrap it in an RLM.
3. **Single-pass retrieval from a pre-indexed corpus.** If you have embeddings + a vector DB, RAG beats RLM on cost and latency. RLMs win when no good index exists and the task requires *reading* the corpus.

## The reproducibility caveat

Independent replication by [anothercodingblog.com](https://www.anothercodingblog.com/p/recursive-language-models-work-but) showed **0/6 to 6/6** accuracy across identical-input runs on a 2.2M-token aggregation task. This is a variance failure the paper does not address. For production use, RLMs need:

- **pass^k reporting** (multiple trials, unbiased estimator — not pass@1 point estimates)
- **Cost-capped budgets** (tail cost can spike — the paper's median is $0.99; p95 is not published)
- **Trajectory logging** (so you can audit which sub-LM call drove the answer)

This repo's adapters ship trajectory + budget machinery for exactly this reason.

## Why a "harness adapter" approach vs a standalone library

Two incumbent RLM implementations exist as standalone libraries:

- [`dspy.RLM`](https://dspy.ai/api/modules/RLM/) — Python library using Pyodide-in-Deno as REPL, integrated with DSPy's optimizer
- [`alexzhang13/rlm`](https://github.com/alexzhang13/rlm) — reference implementation with 6 sandbox backends (Local/Docker/Modal/E2B/Daytona/Prime Intellect)

Both work fine for standalone use. Neither integrates with the coding-agent harnesses most developers already run. Running the RLM pattern *inside* Claude Code / Goose / Codex / OpenCode means:

1. The adapter inherits the harness's tool ecosystem (Read, Write, MCP servers, existing skills)
2. Users don't learn a new CLI — they invoke `/rlm` from the harness they already use
3. Trajectory logs + session state integrate with the harness's own observability
4. Budget enforcement uses the harness's hooks / permissions, not a new external system

Cost consequence: if you're already in Claude Code doing non-RLM work, switching to `dspy.RLM` means leaving the harness and losing its tool context. The adapter keeps everything in one place.

## Further reading

- Paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- Author's blog: [alexzhang13.github.io/blog/2025/rlm/](https://alexzhang13.github.io/blog/2025/rlm/)
- Prime Intellect's RLMEnv training environment: [primeintellect.ai/blog/rlm](https://www.primeintellect.ai/blog/rlm)
- DSPy-first perspective: [Isaac Miller's blog](https://blog.isaacbmiller.com/posts/rlm)
- Independent replication + variance data: [anothercodingblog.com](https://www.anothercodingblog.com/p/recursive-language-models-work-but)
- Commercial wedge analysis (local): `/Users/rshah/rlm-research/R1_benchmarks_market.md`
