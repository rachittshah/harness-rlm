# Design decisions

Architecture decisions log. Each entry: the decision, alternatives considered, why we picked what we picked, and (when relevant) when to revisit.

---

## D-1 Pareto frontier over greedy best-so-far (GEPA)

**Decision.** Track non-dominated candidates over the per-example score vector, not just the single-best mean.

**Alternatives.**
- Greedy hill-climbing on mean trainset score — simpler.
- Beam search over top-K by mean — easier to tune.

**Why.** Mean collapses trade-offs ("good on math, bad on commonsense" vs. "okay at both"). Pareto preserves them. GEPA's original paper shows this prevents local-optima collapse with 35× fewer rollouts than RL. We confirmed: in `tests/test_gepa.py::TestParetoUpdate::test_trade_both_kept`, two trade candidates both survive.

**Revisit when.** Trainsets exceed 200 examples — Pareto frontier grows superlinearly in the worst case; we may need a cap.

---

## D-2 Strategy-driven RLM, not code-driven

**Decision.** The `RLM` Module exposes named strategies (`map_reduce`, `tree`, `filter_then_query`) rather than letting the root LM write arbitrary Python.

**Alternatives.**
- Code-driven (paper-faithful): root LM generates Python in a REPL sandbox.
- Free-form prompt: root LM decides decomposition turn-by-turn.

**Why.**
- Safer — no `exec` sandbox attack surface.
- Easier to test deterministically (no LM in the decomposition path).
- Easier for GEPA to optimize — the strategy lives in the signature instruction.
- Easier to bound — `RLMConfig` is one struct.

The skill-driven version (in `skill/SKILL.md`) handles the code-driven flavour when running inside Claude Code. The Python harness uses strategies.

**Revisit when.** A benchmark demands strategies our menu doesn't cover. Add another `Literal` member, not a code escape hatch.

---

## D-3 Sync API with parallel ThreadPoolExecutor, not async-everywhere

**Decision.** Default API is synchronous. `concurrent.futures.ThreadPoolExecutor` for fan-out (RLM sub-LM dispatch, BestOfN, batch). `AsyncLM` is opt-in for users who genuinely need an event loop.

**Alternatives.**
- Native async (`async def forward`) throughout.
- `asyncio.run_in_executor` shims.

**Why.** The places parallelism matters (sub-LM fan-out, parallel candidates) are I/O-bound but coarse-grained — threads are fine and avoid colour-splitting the codebase. Most users compose modules sync; the asyncio version is fewer than 10% of users. We pay no language tax to keep both paths.

**Revisit when.** A single user needs to fan out >100 concurrent LM calls. Threads do work, but the asyncio path is more memory-efficient at that scale.

---

## D-4 No mocking the LM in tests; stub objects instead

**Decision.** Tests construct lightweight `_StubLM` dataclasses that mimic `LM.__call__` and `LMResult`. We never patch `anthropic.Anthropic`.

**Alternatives.**
- `pytest-mock` to patch the SDK.
- `responses` / `httpx_mock` for HTTP-level mocking.
- A live anthropic call against an LLM proxy.

**Why.** The harness *uses* the LM as a callable. A stub callable is the smallest test fixture — no SDK version coupling, no network. CI runs in 2 seconds because every test is local.

**Revisit when.** We ship a feature that depends on SDK-specific behaviour (retries, streaming back-pressure). At that point, add a few opt-in integration tests with `pytest -m integration`.

---

## D-5 SOTA features are additive, not breaking

**Decision.** Caching, thinking, streaming, multi-provider all flip on via constructor flags / per-call args. Default behaviour matches pre-SOTA.

**Alternatives.**
- New top-level `SOTALM` class.
- Implicit "best practice" defaults that change between versions.

**Why.** Backwards compat is cheap. Anyone who pinned `harness-rlm==0.1.0` and upgrades to 0.2+ sees no behaviour change unless they ask for it.

---

## D-6 Pi's `terminate` flag, not a separate `finish()` path

**Decision.** Tools signal end-of-loop via `ToolResult(terminate=True)`. The `FINISH_TOOL` is just a tool that always sets the flag.

**Alternatives.**
- Special-case `finish_task` in `AgentLoop`.
- A separate `FinishingAgentLoop` class.

**Why.** Pi's design unifies "I'm done" into the same mechanism as any other tool result. Useful for tools other than `finish_task`: a `submit_answer(answer, confidence)` tool can also short-circuit. Tested in `test_agent_loop.py::TestHooks::test_after_tool_call_can_rewrite` — an `after_tool_call` hook can promote any tool's result to `terminate=True`.

---

## D-7 Codex's "child can downgrade but never escalate" sandbox

**Decision.** `subagents.dispatch` takes a `parent_sandbox` arg and uses the more restrictive of `(parent, spec.sandbox_mode)`. Read-only mode strips write tools at dispatch time.

**Alternatives.**
- OS-level sandboxing (chroot, seccomp).
- A capability registry.

**Why.** Codex's pattern is good enough for 95% of subagent use. Real isolation belongs at the OS level — we don't pretend otherwise. The sandbox enum is advisory + tool-stripping. Tested in `test_subagents.py::TestSandboxEnforcement`.

**Revisit when.** Users start running untrusted code via subagents. At that point, integrate with `bubblewrap` or `firecracker`.

---

## D-8 LiteLLM for multi-provider, not handwritten adapters

**Decision.** Build `OpenAILM` directly (only one provider; small surface) but use LiteLLM for everything else.

**Alternatives.**
- Write per-provider adapters for OpenRouter, Bedrock, Vertex, etc.
- A single LiteLLM-only provider.

**Why.**
- LiteLLM covers 100+ providers and is well-maintained.
- Direct `OpenAILM` is one place where the strictly-typed SDK gives us reasoning_effort + base_url cleanly.
- We sidestep LiteLLM's known footguns by setting `num_retries=0` (rule from `rules/llm-gotchas.md`).

**Revisit when.** LiteLLM becomes a >100MB dep or its retry policy starts hurting us. At that point, write provider-direct adapters as needed.

---

## D-9 No mocking of the network at the agent loop level

**Decision.** `AgentLoop` tests construct a fake `_FakeClient` and patch the client attribute. The Anthropic SDK is never actually called.

**Alternatives.**
- Hit a localhost LLM proxy in tests.
- Use the SDK's `respx`-style mock.

**Why.** Same as D-4 but for the loop. The control-flow surface is small; the LM-message round-trip is a small interface; fakes are simpler than mocks.

---

## D-10 Mermaid diagrams in trace_viz, not third-party libs

**Decision.** `trace_to_mermaid()` returns a string that any Markdown renderer (GitHub, Notion, mermaid.live) can render. No graphviz, no matplotlib.

**Alternatives.**
- `pydot` / `graphviz-python` — produces images.
- `rich.tree` — already a dep.

**Why.** Zero deps. Mermaid is the de-facto Markdown diagram format in 2026. Users can paste output into a PR description or docs page and get a labelled graph.

**Revisit when.** We need interactive visualisation (drilldown, filtering). At that point, ship a separate `harness-rlm-ui` package.

---

## D-11 Anthropic prompt caching is opt-in per LM instance

**Decision.** `enable_caching=False` default. When True, the system block becomes a cached content list automatically.

**Alternatives.**
- Always cache when system is set.
- Cache only via explicit `cached_system_block()` calls.

**Why.** Caching has costs (1.25× input on writes). For one-shot or short-prompt workloads, it's a net loss. Making it explicit forces users to know what they're optimizing.

The `estimate_cache_savings()` helper makes the math visible upfront.

---

## D-12 We don't build a plugin registry

**Decision.** No `harness-rlm install <plugin>` mechanism. Users compose modules in Python.

**Alternatives.**
- A pluggy-based registry.
- npm-style plugin install.

**Why.** Pi's argument: registries front-load token cost. The actual extension surface that matters (custom modules, custom tools, custom subagents via TOML) is already covered by Python imports + filesystem conventions. We support `~/agent-tools/` on `$PATH` for CLI-as-tool but don't own it.

**Revisit when.** A non-Python audience materialises (TS-only users via OpenCode). At that point, freeze a JSON-RPC interface and wrap it.

---

## D-13 Trace pretty-printer is sync + dependency-free

**Decision.** `format_trace` and `trace_to_mermaid` are plain functions that emit strings. No tty detection, no Rich integration.

**Alternatives.**
- Auto-detect tty and emit ANSI colours / Rich panels.
- Multiple output formats (HTML, JSON, etc.).

**Why.** Composability. A function that returns a string can be piped, logged, tested. ANSI codes are gated by `color=True` so callers control terminal output. Adding HTML output later is one new function, not a refactor.

---

## D-14 RLM tree-recursion respects budget at every level

**Decision.** The `_tree_decompose` recursive function returns the best partial when `BudgetGuard.check_call()` raises, instead of propagating the error.

**Alternatives.**
- Bubble the `BudgetExceededError` all the way up.
- Hard-truncate the tree before dispatch.

**Why.** Long tree runs are dynamic — a parent doesn't know at split time whether its children will exhaust budget. Catching the budget exception inside `_tree_decompose` gives a graceful "best-effort answer" with a clear marker (`BUDGET_EXHAUSTED`). The trace records this so users see exactly where the cutoff was. Tested in `test_rlm_tree_and_mcp.py::TestTreeRLM::test_tree_with_budget_exhaustion`.

---

## D-15 ClaudeCLILM pipes prompt via stdin

**Decision.** `subprocess.run(input=prompt, ...)` rather than passing the prompt as a positional CLI arg.

**Alternatives.**
- `argv`-passed prompt.
- A file written to a tmpfile then `--prompt-file`.

**Why.** The argparse layer in `claude` interprets prompts starting with `-` or `---` as flags. stdin sidesteps the whole class of escape bugs. Tested empirically with prompts containing `---` separators (the standard Signature render).

Caveat: bare mode (`--bare`) requires `ANTHROPIC_API_KEY`. Default `bare=False` so OAuth-only users get a working harness.
