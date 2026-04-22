---
name: rlm
description: Use when the user has a long document, dataset, or transcript that exceeds the model's effective context window (>100K tokens) AND needs programmatic decomposition + synthesis, or when the user explicitly invokes `/rlm` or asks to "run RLM" / "use recursive language model". Not for simple Q&A that fits in one prompt. Implements Zhang, Kraska, Khattab — Recursive Language Models (arXiv:2512.24601).
---

# Recursive Language Model (RLM) Harness

You are the **root LM** in a Recursive Language Model loop. Your job is to answer the user's question over a context too large to fit in a single prompt, by programmatically exploring the context in a code-execution environment and dispatching cheap sub-LLM calls (Haiku or equivalent) against slices of it.

Paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601). Reference libraries: `dspy.RLM`, `alexzhang13/rlm`, `viplismism/rlm-cli`.

This skill is **harness-agnostic**. Adapter-specific plumbing (tool names, hook paths, session storage) is documented in the adapter's own SKILL.md under `adapters/<harness>/`. The instructions below apply verbatim in any harness that supports the Open Agent Skills standard.

## Budget invariants (DO NOT exceed)

| Budget | Cap | Enforced by |
|---|---|---|
| `max_iterations` | 20 | Harness max-turn setting + manual check |
| `max_llm_calls` | 50 | Pre-tool hook (blocks when exceeded) |
| `max_output_chars` | 10000 per code cell | Manual truncate + warn |
| `sub_model` | Cheap model (Haiku 4.5 or equivalent) | Via `rlm` MCP server's `llm_query` tool, or your harness's native sub-agent mechanism |

If the user has not set budgets explicitly, use these defaults (matching `dspy.RLM`).

---

## Step 0 — Ingest context

Parse the user's message for context markers (same verbs as `rlm-cli`):

| Marker | Source | Action |
|---|---|---|
| `/file <path>` | Local file | Read the file, persist raw bytes to the session store (default: `/tmp/rlm/context.pkl`) |
| `/url <url>` | Web URL | Fetch the URL, persist the returned text |
| `/paste` | Inline (follows marker) | Persist everything after `/paste` up to next marker or end-of-message |

**Ingestion boilerplate (run via your harness's shell-execution tool):**

```bash
mkdir -p /tmp/rlm
python - <<'PY'
import pickle, hashlib, json, os
# EDIT: replace with the actual ingested string
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
   - **Targeted search**: grep/filter the context for candidate regions, then query the sub-LLM on the hits.
   - **Progressive refine**: query broadly, narrow based on answer.
4. What are the stopping criteria? When do you emit `FINAL(answer)`?

Prefer **parallel over sequential** sub-LLM calls — `dspy.RLM` treats parallelism as a first-class primitive.

---

## Step 2 — Iterate

Loop until `FINAL(answer)` is emitted OR budget is exhausted.

### Primitive A: Code-execution cell

Every Python cell you run **must** open with a `pickle.load` of the context and close with a `pickle.dump` back. This simulates a persistent `context` variable across cells, because most harnesses spawn a fresh shell per execution.

**Template (write to `/tmp/rlm/cell.py` and run with your Python runner):**

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

**Prefer the `rlm` MCP server's `llm_query` tool** (direct Anthropic API, no harness re-injection overhead):

```
Tool call: llm_query   (from the `rlm` MCP server)
Arguments:
  prompt: "<your prompt to Haiku, including the chunk text>"
  model:  "claude-haiku-4-5-20251001"
```

**Fallback (only if the MCP server is unavailable):** your harness's native sub-agent mechanism (e.g. `Task` in Claude Code, subrecipes in Goose, sub-sessions in OpenCode). Native sub-agents typically cost 5K–50K tokens of harness-level overhead per dispatch. Prefer MCP for pure text-in / text-out; reserve native sub-agents for sub-LLM calls that themselves need tool access.

Always include in every sub-LLM prompt: *"Answer using ONLY the chunk below. Return 1-3 sentences. If the chunk does not contain the answer, output exactly: NOT_FOUND."*

### Primitive C: `llm_query_parallel(prompts, chunks)` — batched sub-LLM

Emit **N `llm_query` calls in a single assistant message** — most harnesses dispatch them in parallel. Do not exceed 10 parallel calls per turn (each concurrent call burns rate-limit and token budget).

### Primitive D: `FINAL(answer)` sentinel

When you are confident in the answer, **do not** print `FINAL(answer)` as chat text — instead write it to the sentinel file via your shell tool:

```bash
cat > /tmp/rlm/FINAL.txt <<'ANSWER'
<your final answer, any length>
ANSWER
```

The skill's Step 3 reads this file and returns it to the user. This mirrors the paper's `FINAL_VAR` behavior — arbitrary-size outputs that survive chat-message token caps.

---

## Step 3 — Return

After writing `FINAL.txt`:

1. Read `/tmp/rlm/FINAL.txt` and echo the contents as your final message.
2. Report: iterations used, `llm_calls` count (from `/tmp/rlm/state.json`), trajectory path (`/tmp/rlm/trajectory.jsonl`).
3. If budget was exhausted **without** `FINAL.txt`, explicitly say "BUDGET EXHAUSTED — best-available answer:" and give the best partial answer from the last iteration.

---

## Anti-patterns (from R2 §3)

1. **Do NOT pass full context inline in sub-LLM prompts.** Pass a chunk (a slice of `context`) plus a targeted question. Full context defeats the purpose — that's a flat `llm_query`, not RLM.
2. **Do NOT spawn more than 10 sub-agents in one turn.** Each native sub-agent re-injects the harness's system prompt + skills + MCP descriptions. On Claude Code that's ~50K tokens/spawn; 10 parallel = 500K tokens ≈ $3–5 on Opus root.
3. **Do NOT let the sub-LM be the same model as the root LM.** Defeats the cost thesis. Root = Opus/Sonnet (or equivalent reasoning model). Sub = Haiku (or equivalent cheap model).
4. **Do NOT emit `FINAL(x)` as inline text.** Write to `/tmp/rlm/FINAL.txt` via your shell tool — inline text is bounded by output-token caps.
5. **Do NOT skip the pickle load/dump boilerplate.** Missing it silently drops any mutations you made to `context` (each shell call is typically a fresh shell — state does not persist in-memory across cells in any mainstream coding harness).
6. **Do NOT re-ingest context** if `/tmp/rlm/context.pkl` already exists for this session — check `meta.json` first.

---

## Session storage contract

All session artifacts live under `/tmp/rlm/` by default. Adapters may override the base directory, but the filenames are part of the contract:

| File | Purpose | Written by |
|---|---|---|
| `context.pkl` | Pickled context (the big thing) | Step 0 ingest + every cell's dump |
| `meta.json` | `{chars, lines, sha256, preview}` | Step 0 ingest |
| `state.json` | `{llm_calls, iter, ...}` — also gates adapter hooks (no-op outside an RLM session) | Step 0 + budget hook |
| `trajectory.jsonl` | Append-only log: `{timestamp, tool, input, output_chars}` | Adapter post-tool hook |
| `sub_calls.jsonl` | Audit trail of every `llm_query` call (prompt preview, cost, tokens) | `rlm` MCP server |
| `FINAL.txt` | The root LM's final answer | Step 2 primitive D |
| `cell.py` | Ephemeral Python cell source | Rewritten each cell |

Adapters are responsible for installing pre/post tool hooks that read `state.json` for budget enforcement and append to `trajectory.jsonl` for audit. Hooks must be **no-ops outside an RLM session** — gate on the existence of `state.json`.
