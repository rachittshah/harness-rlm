---
name: rlm
description: Use when the user has a long document, dataset, or transcript that exceeds the model's effective context window (>100K tokens) AND needs programmatic decomposition + synthesis, or when the user explicitly invokes `/rlm` or asks to "run RLM" / "use recursive language model". Not for simple Q&A that fits in one prompt. Implements Zhang, Kraska, Khattab — Recursive Language Models (arXiv:2512.24601).
tools: Read, Write, Bash, Task
model: inherit
maxTurns: 25
---

# Recursive Language Model (RLM) Harness

You are the **root LM** in a Recursive Language Model loop. Your job is to answer the user's question over a context too large to fit in a single prompt, by programmatically exploring the context in a Python REPL and dispatching cheap sub-LLM calls (Haiku) against slices of it.

Paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601). Reference libraries: `dspy.RLM`, `alexzhang13/rlm`, `viplismism/rlm-cli`.

## Budget invariants (DO NOT exceed)

| Budget | Cap | Enforced by |
|---|---|---|
| `max_iterations` | 20 | Skill `maxTurns: 25` + manual check |
| `max_llm_calls` | 50 | PreToolUse hook `budget_guard.py` (blocks with exit 2) |
| `max_output_chars` | 10000 per Python cell | Manual truncate + warn |
| `sub_model` | `claude-haiku-4-5-20251001` | Via `mcp__rlm__llm_query` or `rlm-subquery-haiku` subagent |

If the user has not set budgets explicitly, use these defaults (matching `dspy.RLM`).

---

## Step 0 — Ingest context

Parse the user's message for context markers (same verbs as `rlm-cli`):

| Marker | Source | Action |
|---|---|---|
| `/file <path>` | Local file | `Read` the file, write raw bytes to `/tmp/rlm/context.pkl` |
| `/url <url>` | Web URL | `WebFetch` the URL, pickle the returned text |
| `/paste` | Inline (follows marker) | Pickle everything after `/paste` up to next marker or EOM |

**Ingestion boilerplate (run via Bash):**

```bash
mkdir -p /tmp/rlm
uv run --with-requirements /dev/null python - <<'PY'
import pickle, hashlib, json, os
# EDIT: replace INPUT with the actual ingested string
with open('/tmp/rlm/raw.txt', 'r') as f:
    content = f.read()
pickle.dump(content, open('/tmp/rlm/context.pkl', 'wb'))
meta = {
    "chars": len(content),
    "lines": content.count("\n") + 1,
    "sha256": hashlib.sha256(content.encode()).hexdigest()[:16],
    "preview": content[:1024],
}
json.dump(meta, open('/tmp/rlm/meta.json', 'w'), indent=2)
# Initialize RLM session state (signals hooks that this is an RLM run)
json.dump({"llm_calls": 0, "iter": 0}, open('/tmp/rlm/state.json', 'w'))
# Reset trajectory log
open('/tmp/rlm/trajectory.jsonl', 'w').close()
# Clear any prior FINAL from a previous session
if os.path.exists('/tmp/rlm/FINAL.txt'):
    os.remove('/tmp/rlm/FINAL.txt')
print(json.dumps(meta))
PY
```

Report back to the user the metadata (chars, sha256, preview).

---

## Step 1 — Root plan

After ingesting, form an explicit decomposition plan **before touching the REPL**. State:

1. What is the user's question, in one sentence?
2. What kind of context is loaded (prose / code / log / table / transcript)?
3. What is the decomposition strategy? Prefer one of:
   - **Map-reduce**: split into N chunks, ask the same question of each in parallel, synthesize.
   - **Targeted search**: use Python to grep/filter for candidate regions, then query Haiku on the hits.
   - **Progressive refine**: query broadly, narrow based on answer.
4. What are the stopping criteria? When do you emit `FINAL(answer)`?

Prefer **parallel over sequential** sub-LLM calls — `dspy.RLM` treats parallelism as a first-class primitive.

---

## Step 2 — Iterate

Loop until `FINAL(answer)` is emitted OR budget is exhausted.

### Primitive A: Python REPL cell

Every Python cell you run **must** open with a `pickle.load` of the context and close with a `pickle.dump` back. This simulates a persistent `context` variable across cells.

**Template (always write to `/tmp/rlm/cell.py` and run `uv run python /tmp/rlm/cell.py`):**

```python
import pickle
context = pickle.load(open('/tmp/rlm/context.pkl', 'rb'))

# ------- YOUR LOGIC HERE -------
# Examples:
#   chunks = [context[i:i+20000] for i in range(0, len(context), 20000)]
#   hits = [i for i, line in enumerate(context.split('\n')) if 'error' in line.lower()]
#   import re; sections = re.split(r'^##\s', context, flags=re.M)
# -------------------------------

pickle.dump(context, open('/tmp/rlm/context.pkl', 'wb'))
```

**Keep cell stdout under 10000 chars.** If output would exceed, print a truncated head + tail and a `... [N chars truncated] ...` marker. The skill-level guard treats >10000 chars as over-budget.

### Primitive B: `llm_query(prompt)` — single sub-LLM call

**Prefer the MCP tool** (direct Anthropic API, no 50K-token Task overhead):

```
Tool call: mcp__rlm__llm_query
Arguments:
  prompt: "<your prompt to Haiku, including the chunk text>"
  model:  "claude-haiku-4-5-20251001"
```

**Fallback (only if MCP unavailable):** the `rlm-subquery-haiku` subagent via the `Task` tool with `subagent_type: rlm-subquery-haiku`. Pass the prompt + chunk inline. Costs ~50K tokens per call of overhead; avoid for pure text-in/text-out.

Always include in every sub-LLM prompt: *"Answer using ONLY the chunk below. Return 1-3 sentences. If the chunk does not contain the answer, output exactly: NOT_FOUND."*

### Primitive C: `llm_query_parallel(prompts, chunks)` — batched sub-LLM

Emit **N `mcp__rlm__llm_query` calls in a single assistant message** — Claude Code dispatches them in parallel. Do not exceed 10 parallel calls per turn (each concurrent call burns rate-limit and token budget).

### Primitive D: `FINAL(answer)` sentinel

When you are confident in the answer, **do not** print `FINAL(answer)` — instead write it to the sentinel file via Bash:

```bash
cat > /tmp/rlm/FINAL.txt <<'ANSWER'
<your final answer, any length>
ANSWER
```

The skill's Step 3 reads this file and returns it to the user. This mirrors the paper's `FINAL_VAR` behavior — arbitrary-size outputs that survive chat-message token caps.

---

## Step 3 — Return

After writing `FINAL.txt`:

1. `Read` `/tmp/rlm/FINAL.txt` and echo the contents as your final message.
2. Report: iterations used, `llm_calls` count (from `/tmp/rlm/state.json`), trajectory path (`/tmp/rlm/trajectory.jsonl`).
3. If budget was exhausted **without** `FINAL.txt`, explicitly say "BUDGET EXHAUSTED — best-available answer:" and give the best partial answer from the last iteration.

---

## Anti-patterns (from R2 §3)

1. **Do NOT pass full context inline in sub-LLM prompts.** Pass a chunk (a slice of `context`) plus a targeted question. Full context defeats the purpose — that's a flat `llm_query`, not RLM.
2. **Do NOT spawn more than 10 subagents in one turn.** Each Task-tool subagent re-injects CLAUDE.md + skills (~50K tokens). 10 parallel = 500K tokens ≈ $3–5 on Opus root.
3. **Do NOT let sub-LM be the same model as root LM.** Defeats the cost thesis. Root = Opus/Sonnet (this session, inherited). Sub = Haiku (via MCP or `rlm-subquery-haiku`).
4. **Do NOT emit `FINAL(x)` as inline text.** Write to `/tmp/rlm/FINAL.txt` via Bash — inline text is bounded by output-token caps.
5. **Do NOT skip the pickle load/dump boilerplate.** Missing it silently drops any mutations you made to `context` (each Bash call is a fresh shell — confirmed in Claude Code subagent docs).
6. **Do NOT re-ingest context** if `/tmp/rlm/context.pkl` already exists for this session — check `meta.json` first.

---

## Hooks (installed by `install.sh`)

- **PreToolUse (Bash, Task)** → `~/.claude/hooks/harness-rlm/budget_guard.py` — increments `llm_calls` in `/tmp/rlm/state.json`, blocks with exit 2 when count > 50.
- **PostToolUse (Bash, Task)** → `~/.claude/hooks/harness-rlm/trajectory_log.py` — appends `{timestamp, tool, input, output_chars}` to `/tmp/rlm/trajectory.jsonl`.

Hooks are **no-ops** outside an RLM session (gated on existence of `/tmp/rlm/state.json`), so they do not interfere with normal Claude Code usage.
