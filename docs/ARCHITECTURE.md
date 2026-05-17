# Architecture

A composable harness for building LLM applications. Designed around three observations:

1. **Long context is a programming problem, not a model problem.** Wrapping a flat LLM in a recursive decomposition loop with a cheap sub-LLM beats raw long-context models on 6–11M token inputs ([Zhang et al., 2025](https://arxiv.org/abs/2512.24601)).
2. **Prompts are programs.** Typed I/O contracts beat string templates; the optimizer's job is to evolve the prompt as a first-class artifact ([GEPA, Agrawal et al., 2025](https://arxiv.org/abs/2507.19457)).
3. **Tool-using agents need 4 tools, not 40.** Pi's minimalism beats a registry — drop scripts on `$PATH`, let `bash` discover them ([Zechner, 2025](https://mariozechner.at/posts/2025-11-30-pi-coding-agent/)).

## Module map

```mermaid
flowchart TD
    classDef sota fill:#e8f5ff,stroke:#3070d0,color:#1c3358
    classDef core fill:#fff7e0,stroke:#b08a30,color:#5a4d20

    user[User code / CLI / claude -p] --> run[harness.run]:::sota
    user --> orch[Orchestrator]:::sota
    user --> loop[AgentLoop]:::sota
    user --> dispatch[subagents.dispatch]:::sota
    user --> batch[spawn_agents_on_csv]:::sota

    run --> rlm[RLM Module]:::sota
    run --> pred[Predict / ChainOfThought]
    orch --> pred
    orch --> rlm
    loop --> tools[AgentTool registry<br/>read/write/edit/bash/finish]:::sota
    loop --> mcpc[MCPToolset]:::sota
    dispatch --> loop

    rlm --> sub[Sub-LM dispatch<br/>map_reduce / tree / filter]
    pred --> sig[Signature → render → parse]
    sig --> lm[LM]

    sub --> lm
    sub --> async_lm[AsyncLM]:::sota
    sub --> claude_cli[ClaudeCLILM]:::sota
    sub --> openai[OpenAILM]:::sota
    sub --> litellm[LiteLLMLM]:::sota

    lm --> caching[Prompt caching]:::sota
    lm --> thinking[Extended thinking]:::sota
    lm --> guard[BudgetGuard]:::core
    lm --> log[Audit log<br/>sub_calls.jsonl]:::core

    gepa[GEPA optimizer]:::sota --> pred
    gepa --> rlm
    gepa --> typed[TypedPredict<br/>Pydantic schema]:::sota

    ensemble[BestOfN / SelfConsistency]:::sota --> pred
    ensemble --> typed
```

Three groupings:

- **Core (yellow)**: `core.py`, `models.py`, `trajectory.py` — primitives every other layer uses. Zero deps beyond stdlib + pydantic.
- **DSPy-inspired**: `signatures.py`, `modules.py`, `rlm.py`, `gepa.py` — composable typed callables.
- **SOTA (blue)**: everything else — multi-provider LMs, caching, thinking, ensembles, streaming, MCP-as-client, tool loop, subagents, batch.

## End-to-end execution: RLM

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant RLM
    participant BudgetGuard
    participant Sub as Sub-LM (Haiku)
    participant Root as Root LM (Opus)
    participant Log as Audit log

    User->>RLM: rlm(question, document=100KB)
    RLM->>RLM: chunk_context(doc, 20K, overlap=200)
    Note over RLM: 5 chunks
    RLM->>BudgetGuard: check_call() for each
    BudgetGuard-->>RLM: ok
    par parallel sub-LM dispatch
        RLM->>Sub: chunk[0] + question
        Sub-->>RLM: partial[0]
        Sub-->>Log: append SubCallLog
    and
        RLM->>Sub: chunk[1] + question
        Sub-->>RLM: partial[1]
    and
        RLM->>Sub: chunk[2] + question
        Sub-->>RLM: partial[2]
    end
    RLM->>Root: synthesize(partials, question)
    Root-->>RLM: final answer
    Root-->>Log: append SubCallLog
    RLM-->>User: Prediction(answer, trace)
```

## End-to-end execution: GEPA optimization

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant GEPA
    participant Student as Student Module
    participant Reflection as Reflection LM
    participant Metric

    User->>GEPA: compile(student, trainset, valset)
    GEPA->>Student: forward(ex_1), forward(ex_2), ...
    Student-->>GEPA: predictions
    GEPA->>Metric: metric(ex, pred) for each
    Metric-->>GEPA: ScoreWithFeedback(score, feedback)
    GEPA->>GEPA: build seed Candidate (Pareto frontier = [seed])

    loop max_iterations
        GEPA->>GEPA: sample parent from Pareto frontier (softmax-weighted)
        GEPA->>Reflection: prompt(parent_instruction + feedback)
        Reflection-->>GEPA: revised instruction
        GEPA->>Student: forward with revised instruction
        Student-->>GEPA: child predictions
        GEPA->>Metric: score child
        Metric-->>GEPA: ScoreWithFeedback
        GEPA->>GEPA: child dominates? drop dominated members
        GEPA->>GEPA: child non-dominated? add to frontier
    end

    GEPA->>Student: forward against valset (best candidates)
    Student-->>GEPA: val scores
    GEPA->>GEPA: pick winner by val_mean
    GEPA->>Student: set_instruction(winner.instruction)
    GEPA-->>User: GEPAResult(best, pareto, history, rollouts)
```

## End-to-end execution: AgentLoop with tools

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Loop as AgentLoop
    participant Hooks
    participant LM as Anthropic LM
    participant Tool

    User->>Loop: run("Find the bug in main.py")
    Loop->>Hooks: get_steering_messages()?
    Hooks-->>Loop: []
    Loop->>Hooks: transform_context(msgs)?
    Hooks-->>Loop: msgs (or compressed)
    Loop->>LM: messages.create(system, messages, tools)
    LM-->>Loop: Message(text + tool_use blocks)
    Loop->>Hooks: before_tool_call(name, args)?
    Hooks-->>Loop: false (allowed)
    Loop->>Tool: execute(args)
    Tool-->>Loop: ToolResult(content, details, terminate)
    Loop->>Hooks: after_tool_call(name, args, result)?
    Hooks-->>Loop: result
    Note over Loop: if terminate → break outer
    Loop->>Hooks: should_stop_after_turn(ctx)?
    Hooks-->>Loop: false
    Loop->>Hooks: prepare_next_turn(ctx)?
    Hooks-->>Loop: {"model": "claude-sonnet-4-6"}
    Note over Loop: model swapped mid-run
    Loop->>LM: messages.create with new model
    LM-->>Loop: final text (no tool_use)
    Loop->>Hooks: get_follow_up_messages()?
    Hooks-->>Loop: []
    Loop-->>User: AgentLoopResult(final_text, turns, cost_usd, events)
```

## Decomposition strategies

`RLM` ships three first-class strategies:

```mermaid
flowchart LR
    subgraph map_reduce[map_reduce — flat fan-out]
        m_in[100KB ctx] --> m1[chunk 0]
        m_in --> m2[chunk 1]
        m_in --> m3[chunk 2]
        m1 --> m_sub[sub-LM]
        m2 --> m_sub
        m3 --> m_sub
        m_sub --> m_synth[root synth]
    end
```

```mermaid
flowchart TB
    subgraph tree[tree — hierarchical]
        t_in[100KB ctx] --> t_lvl1_a[level 1 chunk A]
        t_in --> t_lvl1_b[level 1 chunk B]
        t_lvl1_a --> t_lvl2_a1[level 2 leaf A1]
        t_lvl1_a --> t_lvl2_a2[level 2 leaf A2]
        t_lvl1_b --> t_lvl2_b1[level 2 leaf B1]
        t_lvl1_b --> t_lvl2_b2[level 2 leaf B2]
        t_lvl2_a1 --> t_synth_a[synth A]
        t_lvl2_a2 --> t_synth_a
        t_lvl2_b1 --> t_synth_b[synth B]
        t_lvl2_b2 --> t_synth_b
        t_synth_a --> t_root[root synth]
        t_synth_b --> t_root
    end
```

```mermaid
flowchart LR
    subgraph filter[filter_then_query — grep-first]
        f_in[100KB ctx] --> f_grep[regex filter]
        f_grep --> f_chunk[chunk hits]
        f_chunk --> f_sub[sub-LM on hits only]
        f_sub --> f_synth[root synth]
    end
```

| Strategy | When to use | Cost vs flat | Accuracy lift |
|---|---|---|---|
| `map_reduce` | Uniform doc, no structure | ~Nx (N chunks) | Up to 90%+ on >100K ctx |
| `tree` | Hierarchical doc (books, code) | ~N log N | Better aggregation, more synth tokens |
| `filter_then_query` | Sparse signal (logs, transcripts) | Much less than flat | High when grep pattern is good |

## Cost model

Per-call cost is computed from the model's pricing table:

```
cost_usd = (input_tokens × $/M_input + output_tokens × $/M_output) / 1_000_000
```

Pricing tables live in `llm.py` (Anthropic) and `providers.py` (OpenAI). Unknown models fall back to the cheapest rate (Haiku / gpt-5-mini).

For multi-stage runs:

```
total_cost = Σ(stage_cost)
           = Σ(input_tokens × rate_in + output_tokens × rate_out)
```

A typical RLM run with 50K-char context, 3 chunks, Opus root + Haiku sub:
- Sub-LM × 3 calls: 50K chars ÷ 3 ≈ 4K tokens input each, ~200 tokens output each = 3 × (4K × $1 + 200 × $5) / 1M ≈ $0.015
- Root synth × 1 call: ~1K tokens input, ~500 tokens output = (1K × $5 + 500 × $25) / 1M ≈ $0.018
- **Total**: ~$0.033

With prompt caching enabled on the root system block (assuming it's reused 10× in a session):
- First call: 1.25× input rate on the cached portion (write)
- Calls 2–10: 0.1× input rate on the cached portion (hit)
- Net savings: ~80% on the cached fraction (typically 60–80% of root input)

## Trajectory & audit

Every LM call appends one JSON line to `/tmp/rlm/sub_calls.jsonl`:

```json
{
  "timestamp": "2026-05-17T12:51:14Z",
  "prompt_preview": "Answer using ONLY the chunk below...",
  "response_chars": 142,
  "model": "claude-haiku-4-5-20251001",
  "cost_usd": 0.0012
}
```

`Trace` (in-memory, per-call) carries the same data plus latency. `Orchestrator` rolls Trace across steps; `SessionStore` persists the rolled trace to `/tmp/rlm/{session}/events.jsonl`.

## Module size guide

| Module | LOC | Purpose | Read me if you want to |
|---|---|---|---|
| `core.py` | 286 | BudgetGuard, chunker, ingest, skill loader | Understand budget enforcement |
| `signatures.py` | 234 | Typed I/O + prompt rendering + parsing | Extend the prompt format |
| `modules.py` | 230 | Module base, Predict, ChainOfThought, Retry | Build new modules |
| `llm.py` | 280 | Anthropic LM + caching + thinking | Wire a new caching strategy |
| `rlm.py` | 380 | RLM with 3 strategies | Add a 4th decomposition strategy |
| `gepa.py` | 240 | Pareto-frontier reflective optimizer | Tune the mutation prompt |
| `harness.py` | 220 | Pi-style run() + CLI | Add CLI flags |
| `orchestrator.py` | 270 | Multi-step composition + compaction | Build pipelines |
| `tools.py` | 290 | AgentTool + 4-tool core | Add a custom tool |
| `agent_loop.py` | 320 | Tool-using loop with 5 hooks | Plug in policy gates |
| `subagents.py` | 240 | Declarative TOML subagents | Define a custom agent role |
| `batch.py` | 220 | CSV-driven batch dispatch | Run evals |
| `mcp_client.py` | 200 | Wrap external MCP servers | Use community MCP tools |
| `caching.py` | 92 | Anthropic prompt caching helpers | Add cache breakpoints |
| `structured.py` | 175 | Pydantic-typed outputs | Get type-safe responses |
| `providers.py` | 240 | OpenAI + LiteLLM providers | Add a new provider |
| `ensemble.py` | 195 | BestOfN + SelfConsistency | Aggregate multiple samples |
| `streaming.py` | 235 | Sync stream() + AsyncLM | Build a streaming UI |
| `trace_viz.py` | 130 | Pretty-print + Mermaid | Visualize an execution |

Total: ~4.5K LOC of production code + ~3K LOC of tests (232 tests).
