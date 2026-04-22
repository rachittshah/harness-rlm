# harness-rlm — OpenAI Codex CLI adapter

Makes OpenAI Codex CLI behave as a Recursive Language Model (RLM) substrate,
per [Zhang, Kraska, Khattab — *Recursive Language Models* (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601).

## What this adapter does

Packages the RLM loop (decompose long context → dispatch cheap sub-LLM calls →
synthesize) as a Codex skill conforming to the
[Open Agent Skills Standard](https://agentskills.io/specification). The skill
body instructs Codex's root model to:

1. Ingest a long context into `/tmp/rlm/context.pkl`
2. Decompose programmatically via Codex's sandboxed shell
3. Dispatch sub-LLM calls to Haiku via the `rlm` MCP server's `llm_query` tool
4. Synthesize and emit `FINAL(answer)` — either via `--output-schema` (strict
   JSON) or via `/tmp/rlm/FINAL.txt` (interactive)

## Primitive mapping (R6 §2.5 rubric)

| # | Primitive | Codex cell | How this adapter closes the gap |
|---|-----------|------------|----------------------------------|
| 1 | Code execution | NATIVE | Codex's sandboxed shell; Python via `uv run`. |
| 2 | `llm_query(prompt)` | SHIM-EXPENSIVE → NATIVE | **`rlm` MCP server** (load-bearing). Without MCP, falls back to shell-out-to-self which is 10-100× more expensive. |
| 3 | Parallel sub-LLM | SHIM-EXPENSIVE → NATIVE | MCP server declares `supports_parallel_tool_calls = true`. |
| 4 | Context-as-variable | SHIM-OK | Pickle to `/tmp/rlm/context.pkl`, same pattern as Claude Code adapter. |
| 5 | FINAL / SUBMIT | NATIVE | Codex's `--output-schema` flag enforces a typed JSON FINAL — strictest in the cohort. |
| 6 | Trajectory log | NATIVE | `codex exec --json` emits JSONL events. Interactive mode: `scripts/rlm_orchestrator.py log`. |
| 7 | Sub-LM model routing | NATIVE | Root LM is Codex's model; sub-LM is an arg to `llm_query`. |
| 8 | Budget enforcement | SHIM-OK | `scripts/rlm_orchestrator.py check` — Codex has no native PreToolUse hook. |

Reference: `/Users/rshah/rlm-research/R6_harness_landscape.md` §2.5.

## Layout

```
adapters/codex/
├── README.md                    # This file
├── install-codex.sh             # Installer: copies skill tree, prints MCP config
├── mcp-config.md                # Detailed MCP server registration guide
└── rlm/                         # Skill tree (copied to ~/.codex/skills/rlm/)
    ├── SKILL.md                 # Open-standard-compliant skill body
    ├── scripts/
    │   └── rlm_orchestrator.py  # Budget/trajectory enforcement (no hook system)
    ├── references/
    │   ├── loop.md              # Deeper RLM mechanics (progressive disclosure)
    │   └── final_schema.json    # JSON Schema for --output-schema
    └── agents/
        └── openai.yaml          # Optional Codex UI metadata
```

## Installation

```bash
# 1. Install the skill tree into ~/.codex/skills/rlm/
bash /path/to/harness-rlm/adapters/codex/install-codex.sh

# 2. Paste the printed [mcp_servers.rlm] block into ~/.codex/config.toml
#    (see mcp-config.md for the exact block and field reference)

# 3. Export your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Verify
codex mcp list          # expect 'rlm' in list
uv run python /path/to/harness-rlm/src/harness_rlm/mcp_server.py --selftest
```

## Usage

### Interactive

```bash
codex
> $rlm summarize this file /file /path/to/100k_doc.md
```

Codex's `$skill-name` mention syntax loads the skill into the active session.

### Headless with structured FINAL (recommended for pipelines)

```bash
codex exec \
    --output-schema ~/.codex/skills/rlm/references/final_schema.json \
    --json \
    --sandbox workspace-write \
    "Act as the rlm skill from ~/.codex/skills/rlm/SKILL.md. User query: summarize /path/to/doc.md"
```

The `--output-schema` flag enforces that Codex's final assistant message is
valid JSON matching `{"final": "...", "iterations": N, "llm_calls": N}`.
This is the tightest RLM termination signal in the cohort.

### Pipeline-friendly stdin

```bash
cat /path/to/large_doc.md | codex exec - \
    --output-schema ~/.codex/skills/rlm/references/final_schema.json \
    --json
```

## Limitations (honest accounting)

1. **No native sub-agent primitive.** The `rlm` MCP server is load-bearing.
   Without it, sub-LLM calls fall back to `codex exec -p "<sub-prompt>" --json`,
   which pays full process init on every call. R6 §3 measured this at 10-100×
   the MCP path's cost for a 10-sub-LLM-call run.

2. **No `--skill` CLI flag.** Verified against
   [developers.openai.com/codex/cli/reference](https://developers.openai.com/codex/cli/reference)
   on 2026-04-21. Headless mode requires pasting the skill body inline or
   instructing Codex to activate the skill by path. Interactive mode supports
   `$rlm` mention syntax.

3. **No native PreToolUse hook.** Budget enforcement is cooperative:
   `scripts/rlm_orchestrator.py check` must be invoked by the root LM before
   each sub-LLM call. The MCP server's `/tmp/rlm/sub_calls.jsonl` log is the
   authoritative cost ledger if the root LM forgets to call the orchestrator.

4. **Sub-LLM recursion depth is 1.** The MCP server is a stateless text-in /
   text-out proxy (same as all harness-rlm adapters). Matches the paper's
   experimental setup.

## Smoke-test commands

```bash
# (1) MCP server selftest (no Codex needed)
uv run python /path/to/harness-rlm/src/harness_rlm/mcp_server.py --selftest
# → '[selftest] PASS' with sub-$0.001 cost

# (2) Orchestrator selftest
echo '{"llm_calls": 0, "iter": 0}' > /tmp/rlm/state.json
uv run python ~/.codex/skills/rlm/scripts/rlm_orchestrator.py check
uv run python ~/.codex/skills/rlm/scripts/rlm_orchestrator.py status
# → llm_calls should read 1

# (3) Codex MCP discovery
codex mcp list
# → 'rlm' in list

# (4) End-to-end headless (smallest possible payload)
mkdir -p /tmp/rlm
echo 'hello world' > /tmp/rlm/raw.txt
codex exec \
    --output-schema ~/.codex/skills/rlm/references/final_schema.json \
    --sandbox workspace-write \
    --json \
    "Act as the rlm skill. User query: what does /tmp/rlm/raw.txt say?"
```

## Sources (verified 2026-04-21)

- [Codex Skills documentation](https://developers.openai.com/codex/skills)
- [Codex MCP configuration](https://developers.openai.com/codex/mcp)
- [Codex non-interactive / `codex exec`](https://developers.openai.com/codex/noninteractive)
- [Codex CLI flag reference](https://developers.openai.com/codex/cli/reference)
- [Open Agent Skills Standard specification](https://agentskills.io/specification)
- [openai/codex GitHub](https://github.com/openai/codex)
- Internal: `/Users/rshah/rlm-research/R6_harness_landscape.md` §2.5 (Codex CLI assessment)
