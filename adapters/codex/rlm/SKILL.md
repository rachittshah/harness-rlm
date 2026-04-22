---
name: rlm
description: Use when the user has a long document, dataset, or transcript that exceeds the model's effective context window (>100K tokens) AND needs programmatic decomposition + synthesis, or when the user explicitly invokes this skill (`$rlm` or asks to "run RLM" / "use recursive language model"). Not for simple Q&A that fits in one prompt. Implements Zhang, Kraska, Khattab — Recursive Language Models (arXiv:2512.24601).
license: Apache-2.0
compatibility: Requires OpenAI Codex CLI, Python 3.12+, uv, and an `rlm` MCP server registered in ~/.codex/config.toml exposing `llm_query`.
metadata:
  author: harness-rlm
  version: "0.1"
  upstream: https://github.com/rachittshah/harness-rlm
  paper: https://arxiv.org/abs/2512.24601
---

# Recursive Language Model (RLM) Harness — Codex CLI

You are the **root LM** inside an OpenAI Codex CLI session acting as a Recursive Language Model loop. Your job is to answer the user's question over a context too large to fit in a single prompt, by programmatically exploring the context via Codex's sandboxed shell and dispatching cheap sub-LLM calls against slices of it.

Paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601). Reference libraries: `dspy.RLM`, `alexzhang13/rlm`, `viplismism/rlm-cli`. For deeper mechanics see `references/loop.md`.

## Codex primitive mapping

| RLM primitive | Codex mapping | Notes |
|---|---|---|
| Code execution | Codex's native sandboxed shell (`bash`, `python3`) | Each shell invocation is a fresh process — state must be pickled to disk. |
| `llm_query(prompt)` sub-LLM | `rlm` MCP server's `llm_query` tool | Registered in `~/.codex/config.toml`. See `mcp-config.md` in the adapter root. |
| Context-as-variable | Pickled to `/tmp/rlm/context.pkl` | Load+dump every cell. Same pattern as the Claude Code adapter (Codex shells are stateless, like Claude Code's Bash). |
| FINAL / SUBMIT | `--output-schema` enforced JSON `{"final": "..."}` + sentinel file `/tmp/rlm/FINAL.txt` | Codex's `--output-schema` is the strictest FINAL in the cohort — use it when invoked via `codex exec`. |
| Trajectory log | Codex emits `--json` JSONL stream per turn | Captured by wrapper when run headless. In interactive mode use `/tmp/rlm/trajectory.jsonl` via `scripts/rlm_orchestrator.py`. |
| Budget enforcement | `scripts/rlm_orchestrator.py` counter in `/tmp/rlm/state.json` | Codex has no native PreToolUse hook — the orchestrator script must be invoked before every `llm_query`. |

## Budget invariants (DO NOT exceed)

| Budget | Cap | Enforced by |
|---|---|---|
| `max_iterations` | 20 | Manual check against `/tmp/rlm/state.json::iter` |
| `max_llm_calls` | 50 | `scripts/rlm_orchestrator.py check` (fails hard with exit 2 when count > 50) |
| `max_output_chars` | 10000 per shell cell | Manual truncate + warn; anything larger, print head+tail with `... [N chars truncated] ...` |
| `sub_model` | `claude-haiku-4-5-20251001` (default) | Passed as `model` arg to `llm_query` tool |

If the user has not set budgets explicitly, use these defaults (matching `dspy.RLM`).

---

## Step 0 — Ingest context

Parse the user's message for context markers (same verbs as `rlm-cli`):

| Marker | Source | Action |
|---|---|---|
| `/file <path>` | Local file | Read the file, write raw bytes to `/tmp/rlm/context.pkl` |
| `/url <url>` | Web URL | Fetch the URL, pickle the returned text |
| `/paste` | Inline (follows marker) | Pickle everything after `/paste` up to next marker or EOM |

**Ingestion boilerplate** (run via Codex shell):

```bash
mkdir -p /tmp/rlm
uv run --with-requirements /dev/null python - <<'PY'
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
# Initialize RLM session state
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

After ingesting, form an explicit decomposition plan **before touching the shell**. State:

1. What is the user's question, in one sentence?
2. What kind of context is loaded (prose / code / log / table / transcript)?
3. What is the decomposition strategy? Prefer one of:
   - **Map-reduce**: split into N chunks, ask the same question of each, synthesize.
   - **Targeted search**: use Python/grep to filter for candidate regions, then query sub-LLM on the hits.
   - **Progressive refine**: query broadly, narrow based on answer.
4. What are the stopping criteria? When do you emit `FINAL(answer)`?

Prefer **parallel over sequential** sub-LLM calls.

---

## Step 2 — Iterate

Loop until `FINAL(answer)` is emitted OR budget is exhausted.

### Primitive A: Python/shell cell

Every cell that mutates context **must** open with a `pickle.load` and close with a `pickle.dump`. Codex shells are stateless per call (same constraint as Claude Code's Bash tool — confirmed in R6 §2.5).

**Template** (write to `/tmp/rlm/cell.py` then `uv run python /tmp/rlm/cell.py`):

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

**Keep cell stdout under 10000 chars.** If output would exceed, print a truncated head + tail and a `... [N chars truncated] ...` marker.

### Primitive B: `llm_query(prompt)` — single sub-LLM call via MCP

Before each sub-LLM call, increment the budget counter by running:

```bash
uv run python /absolute/path/to/rlm/scripts/rlm_orchestrator.py check
```

If that exits non-zero, stop and emit the best-available answer.

Then call the MCP tool (Codex auto-discovers MCP tools from `~/.codex/config.toml`):

```
Use the `llm_query` tool from the `rlm` MCP server.
Arguments:
  prompt: "<your prompt to Haiku, including the chunk text>"
  model:  "claude-haiku-4-5-20251001"
```

**Fallback (shell-out-to-self, expensive)**: if no MCP is registered, the root LM can invoke `codex exec -p "<sub-prompt>" --json` via shell. This pays full process-init cost on every call (R6 §2.5: SHIM-EXPENSIVE). Avoid when possible.

Always include in every sub-LLM prompt: *"Answer using ONLY the chunk below. Return 1-3 sentences. If the chunk does not contain the answer, output exactly: NOT_FOUND."*

### Primitive C: Parallel sub-LLM

Codex supports `supports_parallel_tool_calls = true` per MCP server. Emit **N `llm_query` calls in one turn** — Codex dispatches them concurrently. Cap at 10 parallel calls per turn.

### Primitive D: `FINAL(answer)` sentinel

When confident in the answer, **write it to the sentinel file** via shell:

```bash
cat > /tmp/rlm/FINAL.txt <<'ANSWER'
<your final answer, any length>
ANSWER
```

If the session was started with `codex exec --output-schema /absolute/path/to/rlm/references/final_schema.json`, emit the answer as the final assistant message in the form `{"final": "..."}` — Codex will enforce schema validity and terminate cleanly. See `references/loop.md` for the schema.

---

## Step 3 — Return

After writing `FINAL.txt`:

1. Read `/tmp/rlm/FINAL.txt` and echo the contents as your final message.
2. Report: iterations used, `llm_calls` count (from `/tmp/rlm/state.json`), trajectory path (`/tmp/rlm/trajectory.jsonl`).
3. If budget was exhausted **without** `FINAL.txt`, explicitly say "BUDGET EXHAUSTED — best-available answer:" and give the best partial answer from the last iteration.

---

## Anti-patterns

1. **Do NOT pass full context inline in sub-LLM prompts.** Pass a chunk plus a targeted question. Full context defeats the purpose — that's a flat `llm_query`, not RLM.
2. **Do NOT fall back to shell-out-to-self unless MCP is unavailable.** Every `codex exec` subprocess pays full init cost (R6 §2.5). MCP is far cheaper for text-only sub-LLM dispatch.
3. **Do NOT let sub-LM be the same model as root LM.** Defeats the cost thesis. Root = gpt-5 / opus-class. Sub = Haiku (via the `rlm` MCP server).
4. **Do NOT emit `FINAL(x)` as inline text when using `codex exec --output-schema`.** The schema enforces a typed JSON response — inline prose will fail validation.
5. **Do NOT skip the pickle load/dump boilerplate.** Codex shells are stateless per call — any in-memory mutation is lost.
6. **Do NOT re-ingest context** if `/tmp/rlm/context.pkl` already exists for this session — check `meta.json` first.
7. **Do NOT skip the orchestrator's `check` call.** Without it, nothing enforces the budget (Codex has no PreToolUse hook).

---

## Known Codex-specific limitations (honest accounting)

- **No native sub-agent primitive.** All sub-LLM dispatch must go through MCP (preferred) or shell-out-to-self (expensive). R6 §2.5 scores Codex as SHIM-EXPENSIVE on `llm_query` without an MCP server.
- **No native PreToolUse hook.** Budget enforcement relies on the root LM calling `rlm_orchestrator.py check` before each sub-LLM call. If the root LM forgets, the budget is not enforced. The MCP server's own `/tmp/rlm/sub_calls.jsonl` log is the authoritative cost ledger.
- **No `--skill` flag for `codex exec`.** Skills are only invokable in interactive mode via `/skills` or `$rlm`. For headless runs, prompt Codex with the skill body inline or paste the instructions in the prompt.
- **`AGENTS.md` is project scaffolding, NOT sub-agent spawn.** Do not confuse the two.

For deeper loop mechanics, see `references/loop.md`.
