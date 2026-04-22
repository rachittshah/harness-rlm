# harness-rlm

**Use any existing coding-agent harness as a Recursive Language Model (RLM) substrate.**

Based on [Zhang, Kraska, Khattab — "Recursive Language Models" (arXiv 2512.24601)](https://arxiv.org/abs/2512.24601). Instead of forcing context into one call, RLMs give the root model programmatic access to context via a REPL and recursive sub-LLM calls. This repo ships thin **adapters** that make four production coding harnesses behave as RLM substrates, without forking any of them.

## Four adapters, one core

| Adapter | Status | Primitive score (R6 §3) | MVP effort | Install |
|---|---|---|---|---|
| [Claude Code](adapters/claude_code/) | ✅ Shipped, τ²-bench verified | 19/24 | 1–2 days | `./install.sh --harness claude-code` |
| [Goose](adapters/goose/) | ✅ Recipe + subrecipe | 22/24 (highest) | 2–3 days | `./install.sh --harness goose` |
| [Codex (OpenAI)](adapters/codex/) | ✅ Open Agent Skills standard-compliant | 17/24 | 2–3 days | `./install.sh --harness codex` |
| [OpenCode](adapters/opencode/) | ✅ TypeScript plugin | 21/24 | 2–3 days | `./install.sh --harness opencode` |
| Cursor CLI | Roadmap (R6 §7: wait for native sub-agent) | 16/24 | — | — |
| Cline | Roadmap (no-MCP restriction blocks context-as-variable) | 17/24 | — | — |
| Aider | ✗ Dropped (architect/editor is a pipeline, not recursion) | 12/24 | — | — |

Install all four: `./install.sh --harness all`.

## Repo architecture

```
harness-rlm/
├── src/harness_rlm/
│   ├── core.py            ← pure-Python helpers (BudgetGuard, chunk_context, parse_ingest_directives)
│   ├── mcp_server.py      ← universal sub-LLM dispatch via direct Anthropic API (any MCP-enabled harness)
│   ├── trajectory.py      ← trajectory logging
│   └── models.py          ← Pydantic contracts
├── skill/
│   └── SKILL.md           ← harness-agnostic RLM loop spec, Open Agent Skills Standard format
├── adapters/
│   ├── claude_code/       ← Claude Code adapter (skill + subagent + 2 hooks)
│   ├── goose/             ← Goose recipe + subrecipe + MCP config
│   ├── codex/             ← Codex skill (Open Agent Skills Standard) + budget orchestrator
│   └── opencode/          ← OpenCode TypeScript plugin + custom subagent
├── tau2_integration/      ← tau2-bench custom agent (invokes Claude Code headless today; cross-harness WIP)
├── tests/                 ← 87 pytest tests, all passing
└── install.sh             ← --harness {claude-code|goose|codex|opencode|all}
```

## Why "any harness" works

1. **MCP server is universal.** `src/harness_rlm/mcp_server.py` exposes `llm_query(prompt, model)` over stdio MCP — every modern harness (Claude Code, Goose, OpenCode, Codex, Cursor, Cline) supports MCP. Sub-LLM dispatch is decoupled from the harness.
2. **Open Agent Skills Standard.** The shared `skill/SKILL.md` uses the open standard (name + description only, no harness-specific frontmatter). Each adapter wraps it with harness-specific metadata and hooks.
3. **Budget enforcement via hooks.** Every harness has *some* hook / middleware / wrapper-script mechanism. The `BudgetGuard` Python class in `core.py` is harness-agnostic — each adapter wires it into the local hook system.

## The non-negotiable design constraint

**Text-only sub-LLM calls go through the MCP server (direct Anthropic API), NOT the harness's native sub-agent primitive.** Native sub-agents (Claude Code's `Task`, OpenCode's `Task`, Goose subrecipes, Codex's shell-out) all carry harness overhead — re-injection of system prompt + CLAUDE.md + MCP tool schemas. For Claude Code this has been measured at ~50K tokens per spawn. For an RLM with 40 sub-calls, this is the difference between $1.73/run and $0.79/run. Every adapter enforces this rule.

## Install

```bash
git clone https://github.com/rachittshah/harness-rlm
cd harness-rlm
uv sync --extra dev
./install.sh --harness all   # or --harness claude-code|goose|codex|opencode
```

Then register the MCP server in your harness's config — each adapter's `README.md` + `mcp-config.md` has the exact config block.

## Usage — Claude Code example

```bash
# Interactive
claude
> /rlm summarize this policy and find contradictions /file /path/to/100k_doc.md

# Headless
claude -p "/rlm summarize this policy /file /path/to/doc.md"
```

Other harnesses: see per-adapter README.

## τ²-Bench submission (initial end-to-end, 2026-04-21)

Canary run to verify the plumbing: `claude -p` (Claude Code headless) → `ClaudeHeadlessAgent` → tau2 orchestrator → airline domain, user simulator on GPT-4.1.

| Task | Reward | Messages | Notes |
|---|---|---|---|
| 0 | 1.0 ✓ | 12 | Agent correctly refused cancel+refund (basic economy, >24h, no insurance) |
| 1 | 0.0 ✗ | 21 | Hit `max_steps=20` ceiling — conversation not resolved, not incorrect action |
| 2 | 1.0 ✓ | 20 | Policy-correct |

**Mean reward**: 0.667 (2/3) · **Wall-clock**: 285s · **Model**: `claude-opus-4-7[1m]` · **Seed**: 42

Results file: `results/tau2_airline_3tasks.json`. Full trajectories in `/tmp/rlm/tau2_invocations.jsonl`.

**This is not a leaderboard submission.** τ²-bench leaderboard requires 50 tasks × 4 trials per domain across airline+retail+telecom — this 3-task run is plumbing verification only. See [this repo's rationale](#) for why τ²-bench-airline is the wrong showcase for harness-rlm (short policies don't trigger RLM decomposition); the right showcase is BrowseComp-Plus / OOLONG-Pairs / LongBench-v2.

**Reproduce**: 
```bash
export ANTHROPIC_API_KEY=sk-ant-... ; export OPENAI_API_KEY=sk-...
uv pip install -e /path/to/tau2-bench
uv run python examples/run_tau2_py.py --num-tasks 3 --agent-llm 'claude-opus-4-7[1m]' \
  --out results/tau2_airline_3tasks.json
```

## Tests

```bash
uv sync --extra dev
uv run python -m pytest tests/ -q
# 87 passed in ~2s
```

## Roadmap

- **Cross-harness tau2-bench runs.** Parameterize `tau2_integration/claude_headless_agent.py` on `--agent-bin` so `codex exec` and `goose run` can be the tau2 agent's backend CLI with minor command-builder swaps.
- **BrowseComp-Plus demo.** Where the RLM decomposition actually matters — 6–11M token inputs, paper reports 91.33% vs 0% for flat GPT-5.
- **Cursor CLI adapter.** Waiting for native sub-agent primitive from Anysphere; shell-out-to-self shim works but is expensive.
- **Cline adapter.** Waiting for MCP-accessible subagents (current no-MCP restriction blocks context-as-variable).

## References

- RLM paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- Reference library: [alexzhang13/rlm](https://github.com/alexzhang13/rlm)
- DSPy module: [dspy.ai/api/modules/RLM](https://dspy.ai/api/modules/RLM/)
- CLI precedent: [viplismism/rlm-cli](https://github.com/viplismism/rlm-cli)
- Open Agent Skills Standard: [agentskills.io](https://agentskills.io/specification)
- τ²-Bench: [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
- Primitive-scoring research: `/Users/rshah/rlm-research/R6_harness_landscape.md` (local)
