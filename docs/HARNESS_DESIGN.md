# harness-rlm — design

A composable Python harness built from four sources, each picked for what it solves:

| Source | Feature stolen | Lives in |
|---|---|---|
| **DSPy** (Khattab et al.) | Typed `Signature` contracts, `Module` composition | `signatures.py`, `modules.py` |
| **RLM paper** (Zhang/Kraska/Khattab, arXiv:2512.24601) | Recursive decomposition over long context | `rlm.py` |
| **GEPA** (Agrawal/Khattab, arXiv:2507.19457) | Pareto-frontier reflective prompt evolution | `gepa.py` |
| **Hermes agent** (NousResearch) | Multi-step orchestration + session memory + compression | `orchestrator.py` |
| **Pi-mono** (Mario Zechner) | 4-tool minimal core + hooked agent loop + content/details split | `tools.py`, `agent_loop.py` |
| **Codex** (OpenAI) | Declarative TOML subagents + CSV batch + sandbox tiers + AGENTS.md | `subagents.py`, `batch.py` |

## Layers

```
┌──────────────────────────────────────────────────────────────┐
│ Top-level API:  run(question, context)        harness.py     │
│                  └─ Pi simplicity                            │
├──────────────────────────────────────────────────────────────┤
│ Multi-step:     Orchestrator, SessionStore     orchestrator. │
│                  └─ Hermes session memory                    │
│ Multi-agent:    SubagentSpec (TOML), dispatch  subagents.py  │
│ Tool-using:     AgentLoop + AgentTool          agent_loop.py │
│                  └─ Pi hooks + Codex sandbox                 │
│ Optimizer:      GEPA(metric, reflection_lm)    gepa.py       │
│                  └─ Pareto + reflective mutation             │
│ Batch:          spawn_agents_on_csv            batch.py      │
│                  └─ Codex pattern                            │
├──────────────────────────────────────────────────────────────┤
│ Modules:        Predict, ChainOfThought,       modules.py    │
│                 Retry, RLM                     rlm.py        │
│                  └─ DSPy signature → prompt → parse          │
│ Signatures:     "question -> answer"           signatures.py │
│ LM clients:     LM (Anthropic API),            llm.py        │
│                 ClaudeCLILM (shell out)        claude_cli_lm │
├──────────────────────────────────────────────────────────────┤
│ Primitives:     BudgetGuard, chunk_context     core.py       │
│                 trajectory + audit log         trajectory.py │
│ MCP server:     llm_query over stdio           mcp_server.py │
└──────────────────────────────────────────────────────────────┘
```

The split is deliberate: each layer is usable alone. You can call `Predict` without ever touching `Orchestrator`; you can run `AgentLoop` without ever touching `RLM`; you can run `GEPA` against any `Module`.

## Quick examples

### Flat Q&A — DSPy-style signature, one LM call
```python
from harness_rlm import Predict, LM, configure
configure(lm=LM(model="claude-haiku-4-5-20251001"))

qa = Predict("question -> answer")
pred = qa(question="What is the capital of France?")
print(pred.answer)
```

### Long-context Q&A — RLM decomposition
```python
from harness_rlm import RLM, RLMConfig, LM

doc = open("100k_log.txt").read()
rlm = RLM(
    "question, document -> answer",
    long_context_field="document",
    config=RLMConfig(chunk_size=20_000, max_parallel=4),
    root_lm=LM(model="claude-opus-4-7"),
    sub_lm=LM(model="claude-haiku-4-5"),   # cheap sub
)
pred = rlm(question="Find the OOM root cause.", document=doc)
```

### Pi-style top-level — one function does the right thing
```python
from harness_rlm import run
answer = run("Find the OOM root cause.", context=open("100k_log.txt").read())
print(answer)
```

### Prompt optimization — GEPA evolves the instruction
```python
from harness_rlm import GEPA, ScoreWithFeedback, Predict, LM

qa = Predict("question -> answer")

def metric(example, pred):
    ok = pred.answer.lower() == example["gold"].lower()
    return ScoreWithFeedback(
        score=1.0 if ok else 0.0,
        feedback="exact match" if ok else f"got {pred.answer!r}, want {example['gold']!r}",
    )

trainset = [
    {"question": "capital of France?", "gold": "Paris"},
    {"question": "2 + 2?", "gold": "4"},
]
GEPA(metric=metric, max_iterations=4, reflection_lm=LM(model="claude-opus-4-7")).compile(
    qa, trainset
)
# qa now carries the best-scoring instruction.
```

### Multi-step orchestration — chain Modules with state
```python
from harness_rlm import Orchestrator, Step, Predict, RLM, SessionStore

steps = [
    Step("rephrase",  Predict("question -> rephrased"),
         input_builder=lambda s: {"question": s["raw_question"]}),
    Step("answer", RLM("question, document -> answer", long_context_field="document"),
         input_builder=lambda s: {"question": s["rephrase"]["rephrased"], "document": s["doc"]}),
]
result = Orchestrator(steps).run(
    initial_state={"raw_question": "What happened?", "doc": long_text},
    session_store=SessionStore(name="my-run"),
)
```

### Tool-using agent — Pi-style loop with hooks
```python
from harness_rlm import AgentLoop, AgentLoopConfig, PI_CORE_TOOLS

loop = AgentLoop(
    tools=PI_CORE_TOOLS,    # read/write/edit/bash/finish_task
    config=AgentLoopConfig(
        model="claude-opus-4-7",
        max_turns=20,
        # Hooks: context compaction, policy gates, etc.
        transform_context=lambda msgs: msgs[-10:],            # only last 10
        before_tool_call=lambda name, args: name == "bash" and "rm -rf" in args.get("command", ""),
    ),
)
result = loop.run("Find and fix the bug in main.py.")
print(result.final_text)
```

### Declarative subagents — Codex-style TOML
```toml
# .harness-rlm/agents/explorer.toml
name = "explorer"
description = "Read-only codebase explorer."
model = "claude-haiku-4-5-20251001"
sandbox_mode = "read-only"
tools = ["read", "bash", "finish_task"]
instructions = "Trace execution paths. Cite files. Don't propose fixes."
```

```python
from harness_rlm import discover, dispatch
specs = discover()
result = dispatch(specs["explorer"], "Where is BudgetGuard defined?")
print(result.final_text)
```

### CSV batch dispatch — Codex pattern for evals
```python
from harness_rlm import spawn_agents_on_csv, Predict
qa = Predict("question -> answer")
result = spawn_agents_on_csv(
    "eval_questions.csv",
    module=qa,
    input_template={"question": "{question}"},
    output_csv_path="results.csv",
    max_parallel=8,
    max_retries=2,
)
print(f"ok: {result.ok}/{result.total}  err: {result.err}")
```

## Cost & budgets

`BudgetGuard` caps `max_iterations` (20), `max_llm_calls` (50), `max_output_chars` (10K/cell). Every `LM` instance keeps cumulative token/cost counters. The MCP server and direct `LM` both append to `/tmp/rlm/sub_calls.jsonl` so you have a single audit trail.

## What we deliberately did NOT build

- **Recursive subagents.** Codex caps `max_depth=1` by default for a reason — deeper nesting loses observability without measurable benefit. `Orchestrator` chains sequentially instead.
- **Plugin registry.** Pi's "skills as CLI READMEs" pattern is better — drop scripts in `~/agent-tools/`, let `bash` discover them. We support this without owning it.
- **Async-everywhere.** Sync API with parallel ThreadPoolExecutor for fan-out. Easier to reason about; the parallelism that matters (sub-LM dispatch) is in `RLM` and `batch.py`.

## Verification

- 193 pytest tests across all modules. `uv run python -m pytest tests/ -q`.
- End-to-end demo: `uv run python examples/e2e_claude_p.py` — RLM on 50K-char doc with hidden fact, runs via `claude -p` (no API key needed), found the fact in 4 calls / $0.013 / 32s.
