# harness-rlm — Goose adapter

The Goose adapter packages the Recursive Language Model pattern (arXiv:2512.24601)
as a pair of Goose recipes plus an MCP extension registration. The root Goose
agent (the user's default model — typically Sonnet or Opus) plans the
decomposition of a long context and dispatches cheap per-chunk sub-LLM calls
through `rlm-mcp-server`, which proxies direct to Anthropic's API. The loop
halts when the root writes `FINAL(answer)` to `/tmp/rlm/FINAL.txt` or exhausts
the turn / sub-call budget.

## Why Goose

Goose scores **22/24** on the RLM primitive coverage rubric in
[R6 §2.2](../../../../rlm-research/R6_harness_landscape.md) — the highest of
the seven harnesses surveyed. Four rubric cells are covered by **native**
Goose primitives with no shims:

| RLM primitive   | Goose native feature                                              |
| --------------- | ----------------------------------------------------------------- |
| `exec`          | `goose run --recipe ... -t "..."` headless mode                   |
| `llm_query`     | MCP stdio extension → `rlm-mcp-server` tool                       |
| parallel fan-out | `sub_recipes` run in parallel by default (up to 10 concurrent)   |
| routing         | per-recipe `settings.goose_provider` / `goose_model` override    |
| budget          | `settings.max_turns`, `retry.timeout_seconds`, `--max-turns` flag |
| JSON output     | `--output-format json` / `stream-json`                            |
| trajectory log  | session file (native) + `/tmp/rlm/sub_calls.jsonl` (MCP server)  |

The two non-native cells (`FINAL` sentinel and dynamic decomposition) are
bridged by (a) writing to a sentinel file via the builtin `developer`
extension, and (b) the recipe's `instructions:` block that tells the root LM
how to plan on the fly. Both are cheap shims.

## Prerequisites

- **Goose** — install from <https://goose-docs.ai/docs/getting-started/installation>. Confirmed governed by the Linux Foundation Agentic AI Foundation as of April 2026.
- **`uv`** — the MCP server launches via `uv run`. Install: <https://docs.astral.sh/uv/getting-started/installation/>.
- **`ANTHROPIC_API_KEY`** — export in the shell that launches `goose`.
- **harness_rlm package importable** — run `uv sync` in the repo root at least once, or `uv pip install -e /path/to/harness-rlm`.

## Installation

```bash
cd adapters/goose
./install-goose.sh
```

The installer:

1. Copies `recipes/rlm.yaml` and `recipes/rlm-subquery.yaml` into
   `~/.config/goose/recipes/`.
2. Prints the exact YAML block to paste into `~/.config/goose/config.yaml`
   under `extensions:` (we don't auto-edit the config — YAML edits in bash are
   fragile).
3. Prints a three-step smoke-test (MCP selftest → `--render-recipe` →
   end-to-end headless run).

Full manual registration instructions: [`mcp-config.md`](./mcp-config.md).

## Usage

### Headless RLM run

```bash
goose run \
  --recipe ~/.config/goose/recipes/rlm.yaml \
  --params user_query="What are the three main risks discussed?" \
  --params context_path="/path/to/long_doc.md" \
  --no-session \
  --output-format json
```

- `--params user_query=...` — the question to answer.
- `--params context_path=...` — absolute path to a long document loaded by the
  root LM via the bundled `developer` extension. Optional; if empty, pass the
  context inline via `-t "..."`.
- `--no-session` — don't persist a Goose session file (clean CI runs).
- `--output-format json` — JSON stdout; easier to parse. Use `stream-json` if
  you want token-by-token streaming.

The final answer lands in `/tmp/rlm/FINAL.txt` (configurable via
`--params final_path=...`). Every sub-LLM dispatch is audit-logged to
`/tmp/rlm/sub_calls.jsonl` by the MCP server itself.

### Optional knobs

| Param           | Default                              | Effect                                     |
| --------------- | ------------------------------------ | ------------------------------------------ |
| `sub_model`     | `claude-haiku-4-5-20251001`          | Model for every `llm_query` sub-call.      |
| `max_turns`     | 40                                   | Hard cap on root LM turns.                 |
| `max_sub_calls` | 50                                   | Soft cap on `llm_query` dispatches.        |
| `final_path`    | `/tmp/rlm/FINAL.txt`                 | Where the root writes the halt sentinel.   |

### Override the root model

Goose respects the user's default (`GOOSE_MODEL`) unless the recipe overrides
it. To force the root onto Sonnet for a specific run:

```bash
goose run --recipe ~/.config/goose/recipes/rlm.yaml \
  --params user_query="..." \
  --model claude-sonnet-4-6-20251001
```

## Primitive mapping (how the recipes satisfy the RLM rubric)

See R6 §2.2 for the full 7-primitive × 7-harness grid. Adapter-specific
mapping:

| RLM primitive          | This adapter's implementation                                                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Root LM orchestration  | `rlm.yaml` `instructions:` block — explicit 5-step loop contract (load → plan → dispatch → synthesise → halt).             |
| Context loading        | Bundled `developer` extension reads `context_path` inline; user can also embed context via `-t`.                           |
| Sub-LLM dispatch       | `mcp__rlm-mcp-server__llm_query(prompt, model, max_tokens?, system?)` — one tool, no context re-injection tax.             |
| Per-chunk sub-agent    | `sub_recipes: - name: rlm_subquery` → invoked by the root via Goose's auto-generated `run_subrecipe_rlm_subquery` tool.     |
| Parallel fan-out       | Sub-recipes run in parallel by default per Goose docs; `sequential_when_repeated: true` forces serial when needed.         |
| Budget / halt          | `settings.max_turns: 40` + `retry.timeout_seconds: 1200` + sentinel check at `{{ final_path }}`.                            |
| JSON final             | `response.json_schema` on the root recipe + `--output-format json` on the CLI.                                             |
| Audit trail            | Goose session file (native) + `/tmp/rlm/sub_calls.jsonl` (MCP server, one line per dispatch with cost + token counts).     |

## Limitations

1. **Recipes are YAML-first.** Goose recipes are declarative workflow
   specs — the root LM can plan mid-run, but the *shape* of the
   sub-recipe (parameters, extensions) is fixed at file-write time. If you
   need fully dynamic decomposition (e.g. spawning N variable sub-recipes
   with different prompts), the root should call `llm_query` directly rather
   than `run_subrecipe_rlm_subquery`. The sub-recipe is a structured option,
   not the default.

2. **Sub-recipe boot cost.** Per R6 §8, Goose subagents incur ≈7 s of
   process-boot overhead per dispatch (Rust boot + auth + extension load).
   For >30 sub-calls this is amortised by parallelism (10-way cap), but for
   short runs `llm_query` via MCP is strictly cheaper. The recipe's
   `instructions:` tell the root to prefer `llm_query`.

3. **MCP server path.** The recipes reference the MCP server via
   `python -m harness_rlm.mcp_server`, which assumes the package is on
   `sys.path`. Run `uv sync` in the repo root first, or use the alternative
   absolute-path recipe variant documented in [`mcp-config.md`](./mcp-config.md).

4. **Halt sentinel is file-based.** Goose has no first-class "stop tool" that
   a recipe can emit. We use a file at `/tmp/rlm/FINAL.txt` (configurable)
   plus a `retry.checks` shell test. This works but means a stale FINAL.txt
   from a previous run can short-circuit a new one — the installer does not
   clean this on launch (add `rm -f /tmp/rlm/FINAL.txt` to your invocation if
   you're running in a loop).

5. **No hooks equivalent.** Claude Code's PreToolUse hook (used for the
   Haiku-mix budget guard in `../claude_code/`) has no direct parallel in
   Goose. Budget discipline lives inside the recipe's `instructions:` block
   and the MCP server's cost logger — not an external interceptor.

## Files

- [`recipes/rlm.yaml`](./recipes/rlm.yaml) — root RLM orchestrator recipe.
- [`recipes/rlm-subquery.yaml`](./recipes/rlm-subquery.yaml) — per-chunk sub-LLM sub-recipe.
- [`mcp-config.md`](./mcp-config.md) — manual MCP-extension registration guide.
- [`install-goose.sh`](./install-goose.sh) — installer that copies recipes and prints the config block.

The MCP server itself lives upstream at `../../src/harness_rlm/mcp_server.py`
and is shared across all harness adapters in this repo.

## References

- Recursive Language Models — Zhang, Kraska, Khattab (2026). [arXiv:2512.24601](https://arxiv.org/abs/2512.24601).
- R6 harness landscape — `/Users/rshah/rlm-research/R6_harness_landscape.md` §2.2, scoring Goose 22/24.
- Goose recipe reference — <https://goose-docs.ai/docs/guides/recipes/recipe-reference>.
- Goose config files — <https://goose-docs.ai/docs/guides/config-files/>.
- Goose CLI `run` command — <https://goose-docs.ai/docs/guides/goose-cli-commands>.
