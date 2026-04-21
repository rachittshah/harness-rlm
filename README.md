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

## References

- RLM paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- Reference library: [alexzhang13/rlm](https://github.com/alexzhang13/rlm)
- DSPy module: [dspy.ai/api/modules/RLM](https://dspy.ai/api/modules/RLM/)
- CLI precedent: [viplismism/rlm-cli](https://github.com/viplismism/rlm-cli)
- τ²-Bench: [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
