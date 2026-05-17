# harness-rlm

**Every modern coding-agent harness already ships the eight primitives a Recursive Language Model needs.** Sandboxed exec, sub-agent spawn, model routing, trajectory logging, budget enforcement — Claude Code has them, Goose has them, Codex has them, OpenCode has them. The gap isn't capability. It's naming.

**What that's worth.** Recursive Language Models (Zhang, Kraska, Khattab, [arXiv:2512.24601](https://arxiv.org/abs/2512.24601), Dec 2025) let an LLM programmatically slice its context in a Python REPL and dispatch cheap sub-LLM calls on chunks. On BrowseComp-Plus with 1,000 documents (6–11M tokens), flat GPT-5 scores **0%** — over-context. The same GPT-5 wrapped in the RLM pattern with GPT-5-mini as sub-LLM scores **91.33%** at **$0.99/query**. The paradigm doesn't cost more — it accesses capability flat LLMs cannot reach.

**What this repo ships.** A composable Python harness with SOTA building blocks:

| Layer | Module | Inspired by |
|---|---|---|
| Typed I/O contracts | `signatures.py`, `modules.py`, `structured.py` (Pydantic) | DSPy 2.6 + PydanticAI |
| Long-context decomposition | `rlm.py` — `map_reduce`, `tree`, `filter_then_query` | RLM paper (arXiv:2512.24601) |
| Reflective prompt optimization | `gepa.py` — Pareto frontier + textual feedback | GEPA (arXiv:2507.19457) |
| Tool-using agent loop | `tools.py`, `agent_loop.py` — 4-tool core + 5 hook seams | Pi-mono |
| Declarative subagents + batch | `subagents.py`, `batch.py` — TOML specs + CSV dispatch | OpenAI Codex |
| Multi-step composition | `orchestrator.py` — `SessionStore` + `compress` | Hermes |
| Multi-provider LMs | `llm.py`, `providers.py` — Anthropic + OpenAI + 100+ via LiteLLM | LiteLLM |
| Caching + thinking + streaming | `caching.py`, `streaming.py` | Anthropic SDK 0.40+ |
| Ensemble | `ensemble.py` — `BestOfN`, `SelfConsistency` | Wei et al. (arXiv:2203.11171) |
| MCP-as-client | `mcp_client.py` — wrap any MCP server as `AgentTool`s | MCP spec |
| Trace viz | `trace_viz.py` — `format_trace` + `trace_to_mermaid` | — |

Plus four harness adapters (Claude Code / Goose / Codex / OpenCode), one shared Open Agent Skills Standard skill, and one MCP server for sub-LLM dispatch.

Deep dives:
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — layered architecture with sequence diagrams
- **[docs/COOKBOOK.md](docs/COOKBOOK.md)** — 12 end-to-end recipes
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** — full API surface
- **[docs/DECISIONS.md](docs/DECISIONS.md)** — 15 design-decisions log
- **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** — measured costs + latencies + scaling
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — setup + how to extend
- **[CHANGELOG.md](CHANGELOG.md)** — version history
- **[docs/HARNESS_DESIGN.md](docs/HARNESS_DESIGN.md)** · **[docs/WHAT_IS_RLM.md](docs/WHAT_IS_RLM.md)** · **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)**

---

## Adapters

| Adapter | Status | Primitive score | Install |
|---|---|---|---|
| [Claude Code](adapters/claude_code/) | τ²-bench verified | 19/24 | `./install.sh --harness claude-code` |
| [Goose](adapters/goose/) | Recipe + subrecipe | **22/24** (highest) | `./install.sh --harness goose` |
| [Codex (OpenAI)](adapters/codex/) | Open Agent Skills Standard | 17/24 | `./install.sh --harness codex` |
| [OpenCode](adapters/opencode/) | TypeScript plugin | 21/24 | `./install.sh --harness opencode` |
| Cursor CLI | Roadmap — waiting on native sub-agent | 16/24 | — |
| Cline | Roadmap — waiting on MCP-accessible subagents | 17/24 | — |
| Aider | Dropped — architect/editor is a pipeline, not recursion | 12/24 | — |

Primitive scores from the harness landscape analysis (R6 §3): 7 harnesses × 8 primitives, each cell coded NATIVE / SHIM-OK / SHIM-EXPENSIVE / IMPOSSIBLE.

## Quickstart

```bash
git clone https://github.com/rachittshah/harness-rlm
cd harness-rlm
uv sync --extra dev
./install.sh --harness all          # or --harness claude-code
export ANTHROPIC_API_KEY=sk-ant-...

# Smoke-test the MCP server (single Haiku round-trip, ~$0.00008)
uv run python src/harness_rlm/mcp_server.py --selftest
# → [selftest] PASS

# Exercise the skill via Claude Code headless
claude -p "/rlm summarize /file /path/to/100k_doc.md"
```

Register the MCP server in your harness's config — each adapter's `README.md` + `mcp-config.md` has the exact config block.

## Architecture

```
harness-rlm/
├── src/harness_rlm/
│   ├── core.py          ← BudgetGuard, chunk_context, ingest parsing
│   ├── trajectory.py    ← session state + append-only trajectory log
│   ├── models.py        ← Pydantic request/response contracts
│   ├── mcp_server.py    ← MCP transport for harness-native sub-LLM dispatch
│   │   ─ NEW LAYER ─────────────────────────────────────────────────────
│   ├── signatures.py    ← DSPy-style typed I/O contracts
│   ├── modules.py       ← Module base + Predict, ChainOfThought, Retry
│   ├── llm.py           ← Anthropic LM client + global default
│   ├── claude_cli_lm.py ← Shell-out LM provider (`claude -p` backend)
│   ├── rlm.py           ← RLM Module: strategy-driven recursive decomposition
│   ├── gepa.py          ← Pareto-frontier reflective prompt optimizer
│   ├── orchestrator.py  ← Multi-step Orchestrator + SessionStore + compress()
│   ├── tools.py         ← Pi-style AgentTool + 4-tool core (read/write/edit/bash)
│   ├── agent_loop.py    ← Pi-style hooked agent loop (5 hook seams)
│   ├── subagents.py     ← Codex-style declarative TOML subagents + sandbox tiers
│   ├── batch.py         ← Codex-style spawn_agents_on_csv batch dispatch
│   ├── harness.py       ← Top-level run() + CLI
│   └── __main__.py      ← `python -m harness_rlm`
├── skill/
│   └── SKILL.md         ← harness-agnostic RLM loop spec (Open Agent Skills Standard)
├── adapters/
│   ├── claude_code/     ← skill + subagent + 2 hooks
│   ├── goose/           ← recipe + subrecipe + MCP config
│   ├── codex/           ← skill + budget orchestrator + typed-FINAL JSON schema
│   └── opencode/        ← TypeScript plugin + custom subagent
├── .harness-rlm/agents/ ← Codex-style subagent TOML files (project-local)
├── tau2_integration/    ← tau2-bench custom agent (Claude Code headless)
├── tests/               ← 193 pytest tests (all passing)
├── examples/
│   └── e2e_claude_p.py  ← End-to-end demo via `claude -p` (no API key needed)
└── install.sh           ← --harness {claude-code|goose|codex|opencode|all}
```

The split is deliberate: each layer is usable standalone. `Predict` works without `Orchestrator`; `AgentLoop` works without `RLM`; `GEPA` optimizes any `Module`. Per-feature deep-dive: **[docs/HARNESS_DESIGN.md](docs/HARNESS_DESIGN.md)**.

## τ²-Bench plumbing check (2026-04-21)

End-to-end verification that Claude Code headless → `ClaudeHeadlessAgent` → tau2 orchestrator runs cleanly. **Not a leaderboard submission** — the leaderboard requires 50 tasks × 4 trials × 3 domains; this is 3 tasks × 1 trial × 1 domain as plumbing canary.

| Task | Reward | Messages | Notes |
|---|---|---|---|
| 0 | 1.0 | 12 | Agent correctly refused cancel+refund (basic economy, >24h, no insurance) |
| 1 | 0.0 | 21 | Hit `max_steps=20` ceiling — conversation not resolved, not incorrect action |
| 2 | 1.0 | 20 | Policy-correct |

Mean reward 0.667 · wall-clock 285s · `claude-opus-4-7[1m]` · seed 42.

τ²-bench airline policies are ~5K chars — too short to trigger the RLM loop (threshold >100K tokens). The right showcase for harness-rlm is BrowseComp-Plus / OOLONG-Pairs / LongBench-v2, where RLM decomposition beats flat LLMs by orders of magnitude. See [docs/WHAT_IS_RLM.md](docs/WHAT_IS_RLM.md#when-not-to-use-an-rlm).

Reproduce:
```bash
export ANTHROPIC_API_KEY=...; export OPENAI_API_KEY=...
uv pip install -e /path/to/tau2-bench
uv run python examples/run_tau2_py.py --num-tasks 3 --agent-llm 'claude-opus-4-7[1m]' \
  --out results/tau2_airline_3tasks.json
```

## Tests

```bash
uv sync --extra dev
uv run python -m pytest tests/ -q
# 193 passed in ~3s
```

## End-to-end via `claude -p` (no API key needed)

```bash
uv run python examples/e2e_claude_p.py
```

This exercises `ClaudeCLILM` → `Predict` → `RLM` → `Orchestrator` end-to-end using `claude -p` (Claude Code headless) as the LM backend. Verified on 2026-05-17: RLM correctly extracted a hidden date from a 50K-char document in 4 calls / $0.013 / 32s. Results: [results/e2e_claude_p.json](results/e2e_claude_p.json).

## Roadmap

- **Cross-harness tau2 runs.** `tau2_integration/claude_headless_agent.py` is Claude-Code-only; parameterizing on `--agent-bin` adds codex/goose backends via small command-builder dispatch.
- **BrowseComp-Plus demo.** The right benchmark to showcase harness-rlm — 6–11M token inputs, where RLMs beat flat LLMs 91% vs 0%. `examples/browsecomp_demo.py` pending.
- **Cursor CLI adapter.** Blocked on native sub-agent primitive from Anysphere.
- **Cline adapter.** Blocked on MCP-accessible subagents (current no-MCP restriction breaks context-as-variable).
- **Post-trained root model.** The paper ships `mit-oasys/rlm-qwen3-8b-v0.1` (+28.3% over vanilla Qwen3-8B). A post-train against this repo's trajectory format is worth measuring.

## References

- RLM paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) · [author blog](https://alexzhang13.github.io/blog/2025/rlm/)
- Reference library: [alexzhang13/rlm](https://github.com/alexzhang13/rlm)
- DSPy RLM module: [dspy.ai/api/modules/RLM](https://dspy.ai/api/modules/RLM/)
- CLI precedent: [viplismism/rlm-cli](https://github.com/viplismism/rlm-cli)
- Open Agent Skills Standard: [agentskills.io](https://agentskills.io/specification)
- τ²-Bench: [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
