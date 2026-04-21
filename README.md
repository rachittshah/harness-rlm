# harness-rlm

**Use any existing coding-agent harness as a Recursive Language Model (RLM) substrate.**

Based on [Zhang, Kraska, Khattab — "Recursive Language Models" (arXiv 2512.24601)](https://arxiv.org/abs/2512.24601). Instead of forcing context into one call, RLMs give the root model programmatic access to context via a REPL and recursive sub-LLM calls. This repo ships the **adapter** that makes Claude Code (and other harnesses) behave as RLM substrates, without forking the harness.

## What ships

| Component | Purpose |
|---|---|
| `claude_skill/rlm/` | Claude Code `/rlm` skill — orchestrates decompose → parallel-dispatch → synthesize loop |
| `claude_skill/agents/` | `rlm-subquery-haiku` sub-agent — cheap sub-LLM with Read-only tool access |
| `claude_skill/hooks/` | `budget_guard`, `trajectory_log`, `finalize` — enforce call caps + structured output |
| `src/harness_rlm/mcp_server.py` | `rlm-mcp-server` — bypasses Claude Code's 50K-token Task-tool tax by calling Anthropic API directly for text-only sub-LLM queries |
| `tau2_integration/` | tau2-bench custom agent that invokes Claude Code headless as the RLM |

## Why this works

Every modern coding harness already ships the 8 RLM primitives distilled from the paper — sandboxed exec, sub-agent spawn, model routing, trajectory logging, budget enforcement. The gap is **naming**: treating `Bash` as REPL, `Task` as `llm_query`, `maxTurns` as budget. The skill provides the semantic layer.

## The non-negotiable design constraint

**Text-only sub-LLM calls go through the MCP server (direct Anthropic API), NOT the Task tool.** The Task tool re-injects CLAUDE.md + skills + MCP descriptions (~50K tokens per spawn). For an RLM with 40 sub-calls, this 50K tax becomes $1.73/run vs $0.79/run on the MCP-direct path.

## Install

```bash
git clone https://github.com/rachittshah/harness-rlm
cd harness-rlm
uv sync
./install.sh   # installs skill + subagent + hooks to ~/.claude/
```

## Usage

```bash
# Interactive
claude
> /rlm summarize this policy and find contradictions /file /path/to/100k_doc.md

# Headless
claude -p "/rlm summarize this policy /file /path/to/doc.md"
```

## τ-bench evaluation

```bash
# Install tau2-bench alongside
uv sync --extra tau2

# Run the Claude-headless agent on airline domain, 3 tasks
bash examples/run_tau2.sh
```

## Initial τ²-Bench submission (2026-04-21)

End-to-end verification: `claude -p` (Claude Code headless) → `ClaudeHeadlessAgent` → tau2 orchestrator → airline domain, user simulator on GPT-4.1.

| Task | Reward | Messages | Notes |
|---|---|---|---|
| 0 | 1.0 ✓ | 12 | Agent correctly refused cancel+refund (basic economy, >24h, no insurance) |
| 1 | 0.0 ✗ | 21 | Hit `max_steps=20` ceiling — conversation not resolved, not incorrect action |
| 2 | 1.0 ✓ | 20 | Policy-correct |

**Mean reward**: 0.667 (2/3) · **Wall-clock**: 285s · **Model**: `claude-opus-4-7[1m]` (Opus 4.7 with 1M context window) · **Seed**: 42 · **Max steps per task**: 20

Results file: `results/tau2_airline_3tasks.json`. Full trajectories in `/tmp/rlm/tau2_invocations.jsonl`.

**Reproduce**: 
```bash
export ANTHROPIC_API_KEY=sk-ant-... ; export OPENAI_API_KEY=sk-...
uv pip install -e /path/to/tau2-bench
uv run python examples/run_tau2_py.py --num-tasks 3 --agent-llm 'claude-opus-4-7[1m]' \
  --out results/tau2_airline_3tasks.json
```

## References

- RLM paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- Reference library: [alexzhang13/rlm](https://github.com/alexzhang13/rlm)
- DSPy module: [dspy.ai/api/modules/RLM](https://dspy.ai/api/modules/RLM/)
- CLI precedent: [viplismism/rlm-cli](https://github.com/viplismism/rlm-cli)
- τ²-Bench: [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
