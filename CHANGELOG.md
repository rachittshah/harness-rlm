# Changelog

All notable changes to harness-rlm. Versions follow [SemVer](https://semver.org/).

## [0.3.0] — 2026-05-17

SOTA upgrade. Five phases, no breaking changes.

### Added
- **Prompt caching** (`caching.py`). `cached_text_block`, `cached_system_block`, `cached_tools`, `estimate_cache_savings`. `LM(enable_caching=True)` toggles automatic system-block caching with `cache_control={"type": "ephemeral"}`.
- **Extended thinking** (`llm.py`). `LM(thinking_budget=N)` enables Claude 4 extended thinking. Thinking content lands in `LMResult.thinking` separate from `LMResult.text`. Per-call override on `__call__`.
- **`TypedPredict[T]`** (`structured.py`). Pass a Pydantic `BaseModel` as `output_model`. Returns `TypedPrediction.value` as a typed instance. Validation errors feed back to the LM for self-correction across retries.
- **Multi-provider LMs** (`providers.py`). `OpenAILM` (direct openai SDK) + `LiteLLMLM` (100+ backends). Same call surface as `LM` — modules swap providers transparently.
- **`BestOfN` and `SelfConsistency`** (`ensemble.py`). Parallel sampling with vote aggregation. Wei et al., arXiv:2203.11171.
- **Streaming + `AsyncLM`** (`streaming.py`). Sync `stream(lm, prompt)` generator + asyncio `AsyncLM`. Auto-updates cumulative counters.
- **Trace pretty-printer + Mermaid** (`trace_viz.py`). `format_trace(trace, color=True)` for terminals; `trace_to_mermaid(trace)` for Markdown / docs.
- **Tree-recursive RLM** (`rlm.py`). New `strategy="tree"` with `tree_branching` + `tree_max_depth`. Hierarchical decomposition for books, multi-document corpora, code repos. Parallel within each level.
- **MCP-as-client** (`mcp_client.py`). `MCPToolset(command, args)` connects to an external MCP server and exposes its tools as `AgentTool` instances. Internal background asyncio loop so sync `AgentLoop` works.
- **Cumulative cache stats**. `LM.stats()` now returns `cache_read_tokens` + `cache_write_tokens` for cost auditing.

### Changed
- `LMResult` carries new fields: `thinking`, `cache_read_tokens`, `cache_write_tokens`, `stop_reason`. Existing fields unchanged.
- `RLMConfig` adds `tree_branching` and `tree_max_depth` (no-ops unless `strategy="tree"`).

### Docs
- `docs/ARCHITECTURE.md` — full layered architecture with three Mermaid diagrams (RLM sequence, GEPA sequence, AgentLoop sequence) and strategy-comparison flowcharts.
- `docs/COOKBOOK.md` — 12 end-to-end recipes covering every module.
- `docs/API_REFERENCE.md` — module-by-module API with signatures and contracts.
- `docs/DECISIONS.md` — 15 architecture decisions log entries.
- `docs/PERFORMANCE.md` — measured costs, latencies, scaling notes, cache wins.
- `CONTRIBUTING.md` — setup, code style, commit format, review checklist.
- `CHANGELOG.md` (this file).

### Stats
- 232 tests passing (+45 from 0.2.0).
- ~4.5K LOC production + ~3K LOC tests.

## [0.2.0] — 2026-05-17

Major refactor: the original harness-rlm gains a composable module layer plus three multi-source integrations.

### Added
- **DSPy-style core**: `Signature`, `Module`, `Predict`, `ChainOfThought`, `Retry`.
- **`LM` client** with audit log + cumulative counters.
- **`RLM` Module**: programmatic recursive decomposition (`map_reduce`, `filter_then_query`).
- **`GEPA` optimizer**: Pareto-frontier reflective prompt evolution.
- **Pi-style top-level**: `harness.run(question, context)` one-liner + CLI.
- **`Orchestrator` + `SessionStore` + `compress`**: multi-step composition (Hermes pattern).
- **Pi-style `AgentTool` + `AgentLoop`** with 5 hook seams.
- **Codex-style declarative subagents** (TOML) + sandbox tiers + AGENTS.md.
- **Codex-style `spawn_agents_on_csv`** batch dispatch.
- **`ClaudeCLILM`**: shell-out provider for `claude -p`. Works without `ANTHROPIC_API_KEY`.
- **End-to-end demo** via `claude -p`: extracted hidden date from 50K-char doc in 4 calls / $0.013.

### Stats
- 193 tests passing.

## [0.1.0] — 2026-04-22

Initial release.

- Four harness adapters: Claude Code, Goose, Codex, OpenCode.
- One shared `SKILL.md` (Open Agent Skills Standard).
- One MCP server (`mcp_server.py`) for sub-LLM dispatch.
- `BudgetGuard`, `chunk_context`, `parse_ingest_directives`, trajectory log.
- τ²-bench plumbing canary (3 tasks, mean reward 0.667).
- 87 tests passing.
