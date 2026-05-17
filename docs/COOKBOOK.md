# Cookbook

Twelve end-to-end recipes. Each is a complete, runnable Python program. Pair with [docs/ARCHITECTURE.md](ARCHITECTURE.md) for the underlying model and [docs/API_REFERENCE.md](API_REFERENCE.md) for the per-class surface.

---

## 1. One-shot Q&A

```python
from harness_rlm import Predict, LM, configure
configure(lm=LM(model="claude-haiku-4-5-20251001"))

qa = Predict("question -> answer")
print(qa(question="What is 2+2?").answer)
```

---

## 2. Chain-of-thought with explicit reasoning

```python
from harness_rlm import ChainOfThought

cot = ChainOfThought("question -> answer")
pred = cot(question="If a train leaves Pune at 2pm at 60km/h...")
print("Reasoning:", pred.reasoning)
print("Answer:", pred.answer)
```

---

## 3. Long-context RLM (recursive decomposition)

```python
from harness_rlm import RLM, RLMConfig, LM

doc = open("100k_log.txt").read()
rlm = RLM(
    "question, document -> answer",
    long_context_field="document",
    config=RLMConfig(
        strategy="map_reduce",
        chunk_size=20_000,
        max_parallel=4,
        max_llm_calls=20,
    ),
    root_lm=LM(model="claude-opus-4-7"),
    sub_lm=LM(model="claude-haiku-4-5-20251001"),
)
pred = rlm(question="What error caused the OOM?", document=doc)
print(pred.answer)
print(f"Cost: ${pred.trace.cost_usd:.4f}  Calls: {pred.trace.calls}")
```

---

## 4. Tree-recursive RLM for hierarchical docs

```python
from harness_rlm import RLM, RLMConfig

book = open("textbook.md").read()  # 500K chars, 12 chapters
rlm = RLM(
    "question, document -> answer",
    long_context_field="document",
    config=RLMConfig(
        strategy="tree",
        tree_branching=4,    # 4-ary tree
        tree_max_depth=3,    # 3 levels
        flat_char_threshold=30_000,
        max_llm_calls=50,
    ),
)
print(rlm(question="Summarize the book in 5 bullets.", document=book).answer)
```

---

## 5. Pi-style one-liner

```python
from harness_rlm import run

answer = run(
    "Find the deprecation warning.",
    context=open("huge_log.txt").read(),
)
print(answer)
```

---

## 6. Typed (Pydantic) structured outputs

```python
from pydantic import BaseModel
from harness_rlm import TypedPredict

class Review(BaseModel):
    sentiment: str       # "positive" | "negative" | "neutral"
    score: int           # 1–10
    main_complaint: str | None

review = TypedPredict("text -> review", output_model=Review)
pred = review(text="This product is amazing but ships slow.")
print(pred.value.sentiment, pred.value.score, pred.value.main_complaint)
# Type-safe access — pred.value is a Review instance.
```

---

## 7. Self-consistency for reasoning tasks

```python
from harness_rlm import SelfConsistency, ChainOfThought

cot = ChainOfThought("question -> answer")
sc = SelfConsistency(cot, n=5)   # 5 samples, vote on `answer`
pred = sc(question="What is 23 × 47?")
print(pred.answer)
# 5 reasoning paths may differ, but majority-vote on the answer is more reliable.
```

---

## 8. GEPA prompt optimization

```python
from harness_rlm import GEPA, ScoreWithFeedback, Predict, LM

qa = Predict("question -> answer")

def metric(example, pred):
    ok = pred.answer.lower().strip() == example["gold"].lower().strip()
    return ScoreWithFeedback(
        score=1.0 if ok else 0.0,
        feedback="exact match" if ok
                 else f"got {pred.answer!r}; expected {example['gold']!r}",
    )

trainset = [
    {"question": "capital of France?",      "gold": "Paris"},
    {"question": "2 + 2?",                  "gold": "4"},
    {"question": "first president of USA?", "gold": "George Washington"},
]
valset = [
    {"question": "capital of Germany?",     "gold": "Berlin"},
]

result = GEPA(
    metric=metric,
    max_iterations=4,
    reflection_lm=LM(model="claude-opus-4-7"),
).compile(qa, trainset, valset)

print("Best:", result.best.instruction)
print("Pareto frontier size:", len(result.pareto))
# qa is mutated in place to carry the best instruction.
```

---

## 9. Multi-step orchestration with state

```python
from harness_rlm import Orchestrator, Step, Predict, RLM, SessionStore

steps = [
    Step(
        name="extract",
        module=Predict("text -> entities"),
        input_builder=lambda s: {"text": s["raw_doc"]},
    ),
    Step(
        name="qa",
        module=RLM("question, document -> answer", long_context_field="document"),
        input_builder=lambda s: {
            "question": f"Summarise everything about {s['extract']['entities']}.",
            "document": s["raw_doc"],
        },
    ),
]
session = SessionStore(name="my_run")
result = Orchestrator(steps).run(
    initial_state={"raw_doc": open("doc.txt").read()},
    session_store=session,
)
print(result.state["qa"]["answer"])
print(f"Saved {len(session.read())} events to {session.path}")
```

---

## 10. Tool-using agent (Pi style)

```python
from harness_rlm import AgentLoop, AgentLoopConfig, PI_CORE_TOOLS

loop = AgentLoop(
    tools=PI_CORE_TOOLS,    # read/write/edit/bash/finish_task
    config=AgentLoopConfig(
        model="claude-opus-4-7",
        max_turns=15,
        # Policy gate: deny destructive bash commands.
        before_tool_call=lambda name, args: (
            name == "bash" and any(
                x in args.get("command", "") for x in ["rm -rf", "drop database"]
            )
        ),
    ),
)
result = loop.run(
    "Read src/main.py, find the function that handles login, and explain it."
)
print(result.final_text)
print(f"Turns: {result.turns}  Tool calls: {result.tool_call_count}")
```

---

## 11. Declarative subagent (Codex TOML)

`.harness-rlm/agents/code_explorer.toml`:
```toml
name = "code_explorer"
description = "Read-only code explorer; cites file:line."
model = "claude-haiku-4-5-20251001"
sandbox_mode = "read-only"
tools = ["read", "bash", "finish_task"]
instructions = """
Find the requested symbol or behaviour. Cite file:line. Don't propose fixes.
"""
```

```python
from harness_rlm import discover, dispatch
specs = discover()
res = dispatch(specs["code_explorer"], "Where is BudgetGuard defined?")
print(res.final_text)
```

---

## 12. CSV batch evaluation

`questions.csv`:
```csv
id,question,gold
q1,Capital of France?,Paris
q2,What is 7 × 8?,56
q3,Largest planet?,Jupiter
```

```python
from harness_rlm import spawn_agents_on_csv, Predict

qa = Predict("question -> answer")
result = spawn_agents_on_csv(
    "questions.csv",
    module=qa,
    input_template={"question": "{question}"},
    output_csv_path="predictions.csv",
    max_parallel=8,
    max_retries=2,
)
print(f"ok: {result.ok}/{result.total}  err: {result.err}")
# Score predictions.csv against the `gold` column in your favourite tool.
```

---

## Bonus: streaming output to a terminal

```python
from harness_rlm import LM, stream
import sys

lm = LM(model="claude-haiku-4-5-20251001")
for event in stream(lm, "Write a haiku about recursion."):
    if event.kind == "text_delta":
        sys.stdout.write(event.text)
        sys.stdout.flush()
    elif event.kind == "done":
        print(f"\n\nCost: ${event.result.cost_usd:.6f}")
```

---

## Bonus: prompt caching for long-running sessions

```python
from harness_rlm import LM

lm = LM(
    model="claude-opus-4-7",
    enable_caching=True,
    thinking_budget=4096,      # Claude 4 extended thinking
)
# First call: writes the system block to cache (1.25× input cost).
# Calls 2-N: hits the cache (0.1× input cost) for ~5 minutes.
for q in questions:
    pred = lm(q, system=LONG_SYSTEM_PROMPT)
    print(pred.text)

print("Cache stats:", lm.stats())
# {"calls": N, "cache_read_tokens": 8500, "cache_write_tokens": 950, ...}
```

---

## Bonus: connecting to community MCP servers

```python
from harness_rlm import MCPToolset, AgentLoop, FINISH_TOOL

with MCPToolset(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/repo"],
    name_prefix="fs__",
) as fs:
    loop = AgentLoop([*fs.tools(), FINISH_TOOL])
    result = loop.run("Find all TODO comments in the repo.")
    print(result.final_text)
```
