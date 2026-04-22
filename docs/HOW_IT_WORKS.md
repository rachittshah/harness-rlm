# How harness-rlm works

## The four-primitive mental model

Strip the paper's notation away and every RLM is built from four primitives:

| Primitive | What it does | Why it matters |
|---|---|---|
| **context-as-variable** | Input corpus is a named Python variable (`context`), not prompt tokens | Enables slicing, filtering, indexing without sending the whole thing to the LLM |
| **llm_query(prompt, chunk)** | Callable that dispatches to a sub-LLM (cheap) and returns a string | Semantic operations over slices at 1/10th the root's cost |
| **FINAL(answer)** | Sentinel that halts the loop and commits the answer | Lets the root decide *when* to stop; budgets are hard caps, FINAL is soft commit |
| **budget** | Hard limits on iterations, sub-LLM calls, output tokens | Bounds worst-case cost — independent replications show tail costs can explode without caps |

Everything else in an RLM implementation is machinery around these four.

## The loop

```
user_query + context (may be 10M tokens)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Root LLM  (Claude Opus 4.7 [1M], GPT-5, ...)     │
│                                                   │
│  sees: metadata about context                     │
│        (size, preview, SHA, line count)           │
│        — NOT the full context                     │
│                                                   │
│  emits: Python code cells                         │
│       + llm_query(prompt, chunk_slice)            │
│       + FINAL(answer)  when done                  │
└──────┬─────────────────────────────────┬──────────┘
       │                                 │
       ▼                                 ▼
┌──────────────┐                ┌──────────────────────┐
│ Python REPL  │                │ Sub-LLM dispatcher   │
│              │                │                      │
│ Executes     │                │ MCP:                 │
│ code cells.  │                │   llm_query tool   → │
│              │                │   Anthropic API    → │
│ Returns      │                │   Haiku-4-5          │
│ stdout       │                │                      │
│ (trim 10K ch)│                │ (bypasses harness    │
│              │                │  Task-tool 50K tax)  │
└──────┬───────┘                └──────────┬───────────┘
       │                                   │
       └───────────────┬───────────────────┘
                       │
                       ▼
              Returns to root LLM
                       │
                       ▼
              Loop until FINAL or budget cap
```

## Why the MCP server is load-bearing

The naive Claude Code implementation: use the `Task` tool to spawn a sub-agent per `llm_query`. **Don't do this.**

Every subagent spawn re-injects CLAUDE.md + skills + MCP tool schemas into the sub-agent's context — measured at ~50K tokens per spawn by [Jung Jae-hoon's benchmark](https://dev.to/jungjaehoon/why-claude-code-subagents-waste-50k-tokens-per-turn-and-how-to-fix-it-41ma), corroborated by independent measurement in the user's MEMORY.md. For an RLM with 40 sub-calls, this 50K tax per call is the difference between:

- **Naive path** (Task tool for every sub-LLM): **$1.73/run** on a 2.2M-token aggregation task
- **MCP path** (direct Anthropic API via `rlm-mcp-server`): **$0.79/run** — same task, same sub-LLM quality

The MCP server short-circuits the entire harness pipeline: it's a tiny stdio process exposing one tool (`llm_query(prompt, model)`) that forwards to `anthropic.Anthropic.messages.create`. No system prompt bloat, no tool schema injection, no skill re-loading. Just prompt in, response out.

**This pattern is non-negotiable** in every adapter in this repo.

## Harness primitive coverage

From the harness landscape analysis (`/Users/rshah/rlm-research/R6_harness_landscape.md` §3), each harness covers the 8 RLM primitives with a mix of NATIVE (N=3 pts), SHIM-OK (S=2), SHIM-EXPENSIVE (E=1), IMPOSSIBLE (X=0):

| Harness | Score /24 | Exec | llm_query | Parallel | Context-var | FINAL | Trajectory | Routing | Budget |
|---|---|---|---|---|---|---|---|---|---|
| **Goose** | **22** | N | N | N | S | S | N | N | N |
| **OpenCode** | **21** | N | N | N | S | S | N | N | S |
| **Claude Code** | **19** | S | N | S | S | S | N | N | S |
| Codex CLI | 17 | N | E | E | S | N | N | N | S |
| Cline | 17 | N | S | N | E | S | S | N | S |
| Cursor CLI | 16 | N | E | E | S | S | N | N | S |
| Aider | 12 | N | X | X | S | E | S | N | S |

**Why Claude Code is #1 MVP despite lower raw score:** implementation maturity, ubiquitous hook ecosystem, per-subagent model routing at definition time, and user familiarity. Shim effort ≤60 LOC per primitive.

**Why Goose is #2 despite higher raw score:** newer community; docs in flux (block.github.io/goose/ URLs 404; goose-docs.ai is the current canonical); Rust subagent boot adds ~7s per dispatch.

**Why Aider is dropped:** architect/editor is a two-stage pipeline (reasoner → editor), not recursion. The Python API is officially "not supported or documented." No clean RLM shape.

## The shared core vs adapter-specific split

```
harness-rlm/
├── src/harness_rlm/core.py         ← BudgetGuard, chunk_context, parse_ingest_directives (pure Python)
├── src/harness_rlm/mcp_server.py   ← universal — any MCP-enabled harness can call it
├── skill/SKILL.md                  ← harness-agnostic RLM loop spec (Open Agent Skills Standard)
└── adapters/
    ├── claude_code/                ← adds Claude-specific frontmatter, hooks, subagent
    ├── goose/                      ← adds Goose recipe YAML, subrecipe, extensions block
    ├── codex/                      ← adds Codex skill scope, scripts/, references/, openai.yaml
    └── opencode/                   ← adds OpenCode TypeScript plugin, agent markdown
```

Each adapter's job:

1. **Express the RLM loop in the harness's native format.** Claude Code has `SKILL.md` with Anthropic-specific fields, Goose has `recipe.yaml`, Codex has `SKILL.md` with Open Agent Skills Standard fields, OpenCode has `plugin.ts`. Different formats, same instructions.
2. **Wire the MCP server into the harness's MCP config.** Every adapter ships `mcp-config.md` with the exact config block.
3. **Enforce the budget using the harness's hook/middleware mechanism.** Claude Code PreToolUse hooks; Codex orchestrator script called from the skill body; OpenCode `tool.execute.before` hook; Goose recipe-level `retry.checks`.
4. **Fall back to the harness's native sub-agent if MCP is unavailable.** Cost regression (the 50K tax), but at least the loop runs.

## Cost and the amortization threshold

From R6 §5 — per-harness overhead as % of total cost for a 2.2M-token aggregation task, and the input-size threshold at which the adapter's overhead falls below 20%:

| Harness | Overhead per sub-call | Amortization threshold |
|---|---|---|
| Native `dspy.RLM` | ~1K tokens | any scale |
| Claude Code (optimized, direct-API MCP) | ~1K tokens | ~30K input |
| OpenCode | ~3K tokens | ~75K input |
| Goose | ~5K tokens | ~125K input |
| Codex / Cursor (shell-out-to-self) | ~8K tokens | ~200K input |
| **Claude Code (naive, Task tool)** | **~50K tokens** | **~800K input** |

**Takeaway:** the adapter is worth it when your task is ≥500K tokens. Below that, direct API orchestration (or a standalone `dspy.RLM` call) beats any harness. The RLM paradigm itself (vs flat LLM) pays off much earlier — around 50K tokens where context rot kicks in on a frontier model.

## Trajectory and observability

Every run writes three files to `/tmp/rlm/`:

| File | What's in it | Writer |
|---|---|---|
| `state.json` | Session counters: `llm_calls`, `iter`. Budget hooks read/write this. | Budget hook |
| `sub_calls.jsonl` | Per-sub-LLM-call log: timestamp, model, input tokens, output tokens, cost | MCP server |
| `trajectory.jsonl` | Per-tool-call log: timestamp, tool, input preview, output chars | Trajectory hook |

Together they reconstruct exactly what the root LLM did, which chunks were queried, and where the money went. For post-hoc audit:

```python
from harness_rlm.trajectory import read_trajectory
from pathlib import Path

entries = read_trajectory(Path("/tmp/rlm"))
for e in entries:
    print(e["timestamp"], e["tool"], e["output_chars"])
```

## Budget enforcement — the hard contract

```python
from harness_rlm.core import BudgetGuard, DEFAULT_BUDGETS, BudgetExceededError

guard = BudgetGuard(DEFAULT_BUDGETS)   # {max_iterations: 20, max_llm_calls: 50, max_output_chars: 10000}
# In the hot path (e.g., a PreToolUse hook):
try:
    guard.check_call()
    guard.increment_call()
except BudgetExceededError as e:
    # emit exit code 2 in hook → Claude Code blocks the tool call
    # or raise in a plugin → OpenCode rejects the invocation
    # or fail the shell test in a recipe → Goose halts
    sys.exit(2)
```

`BudgetGuard.state_dict()` serializes to JSON so the same budget survives across subprocess boundaries — critical for Claude Code where each Bash call is a fresh shell, and for Codex where `codex exec` is stateless per call.

## FAQ

**Q: Why not just use longer context windows?**
A: (a) 1M-context models are expensive per token, (b) context rot hits attention long before the hard limit, (c) BrowseComp-Plus's 1K-docs setting runs 6–11M tokens — past every frontier model's window. RLMs are independent of model context size.

**Q: Does the root LLM need to be expensive?**
A: No. The paper's post-trained `mit-oasys/rlm-qwen3-8b-v0.1` (an 8B model trained on RLM trajectories) scores within striking distance of vanilla GPT-5 on three long-context tasks. Root model quality matters for plan generation; with good training data, smaller models suffice.

**Q: Can the sub-LLM use tools?**
A: In principle yes; the paper doesn't test it. In this repo, by default sub-LLMs are text-only (MCP direct API). If a sub-LLM needs tools, fall back to the harness's native sub-agent primitive — pay the 50K tax; it's worth it for tool access.

**Q: What if the root LLM gets stuck in a loop?**
A: `max_iterations=20` is a hard cap. Hit that and the session aborts with the last draft tagged "BUDGET EXHAUSTED."

**Q: Can I run this offline / against a local model?**
A: The MCP server defaults to Anthropic API. To point it at a local Ollama / vLLM OpenAI-compatible endpoint, swap the `anthropic.Anthropic` client in `src/harness_rlm/mcp_server.py` for an `openai.OpenAI` with `base_url=`. The adapter pattern is model-agnostic by design.

**Q: What about security? The sub-LLM reads untrusted chunks.**
A: Open problem. We call this threat class **SLH-RCD (Sub-LM Hijack via Recursive Context Dispatch)**: adversarial content in a chunk can hijack the sub-LLM, which the root then trusts as tool output. R4 of the research suite names it and surveys mitigations. This repo does not yet ship defenses beyond (a) capability-restricted sub-LM (no tool access by default), and (b) trajectory logging for post-hoc audit. Production deployments on untrusted input should add input sanitization + cross-verification between two independent sub-LMs on any high-confidence claim. See `/Users/rshah/rlm-research/R4_sandbox_safety.md`.
