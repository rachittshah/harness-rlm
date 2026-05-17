# Contributing

This doc covers: project setup, adding a module / tool / provider / RLM strategy / subagent / test, and the commit format we follow.

---

## Setup

```bash
git clone https://github.com/rachittshah/harness-rlm
cd harness-rlm
uv sync --extra dev
uv run python -m pytest tests/ -q   # 232 tests in ~2.5s
uv run ruff check src/ tests/
```

If you don't have `uv`: `pip install uv` first. We avoid `pip install` because it doesn't lock cleanly.

---

## Adding a new Module

A `Module` is any callable that:
1. Has a `Signature` (typed I/O contract).
2. Returns a `Prediction` (or a subclass) with a `Trace`.
3. Optionally calls an LM via `self._lm()`.

Minimal template:

```python
# src/harness_rlm/my_module.py
from harness_rlm.modules import Module, Prediction, Trace
from harness_rlm.signatures import Signature

class MyModule(Module):
    def forward(self, **inputs) -> Prediction:
        trace = Trace(module=self.name)
        # Do your work, calling self._lm()(prompt) as needed.
        result = self._lm()(self._render(inputs))
        trace.record(result, label="my_module")
        return Prediction(fields={"answer": result.text.strip()}, trace=trace)

    def _render(self, inputs):
        return self.signature.render_prompt(inputs)
```

Then:
- Add to `src/harness_rlm/__init__.py` imports + `__all__`.
- Add `tests/test_my_module.py` with `_StubLM` (see `tests/test_modules.py` for the pattern).
- Cite in [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

---

## Adding a new tool

```python
from harness_rlm.tools import AgentTool, ToolResult

WEATHER_TOOL = AgentTool(
    name="get_weather",
    description="Get current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    execute=lambda args: ToolResult.text(
        f"Sunny in {args['city']}, 72°F",
        details={"temp_f": 72, "condition": "sunny"},
    ),
)
```

For Python functions, `from_function(fn)` does the right thing — it inspects the signature and builds a basic schema.

Then add tests under `tests/test_tools.py`.

---

## Adding a new LM provider

A provider is any object with:
- `__call__(self, prompt, *, system=None, model=None, max_tokens=None) -> LMResult`
- `total_calls`, `total_input_tokens`, `total_output_tokens`, `total_cost_usd` counters
- A `stats()` method returning the cumulative dict

See `src/harness_rlm/providers.py::OpenAILM` as a reference template. Pricing tables belong inline; fall back to a sensible default for unknown models.

---

## Adding an RLM strategy

1. Add a `Literal` member to the `Strategy` type in `src/harness_rlm/rlm.py`.
2. Add an `_<your_strategy>_decompose` method on `RLM` (see `_tree_decompose` for the recursive pattern).
3. Branch on the strategy in `forward()`.
4. Add an `RLMConfig` field for strategy-specific knobs.
5. Add tests under `tests/test_rlm*.py` — use the `_StubLM` pattern, verify call counts and trace labels.
6. Mention in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) under "Decomposition strategies".

---

## Adding a declarative subagent

Drop a TOML file at `.harness-rlm/agents/<name>.toml`:

```toml
name = "my_role"
description = "Short description of what this subagent does."
model = "claude-haiku-4-5-20251001"
sandbox_mode = "read-only"   # or workspace-write, danger
tools = ["read", "bash", "finish_task"]
instructions = """
What the subagent should do, step by step.
"""
```

Then `dispatch(spec, task)` runs it through `AgentLoop` with the spec's tools.

---

## Code style

- **Format**: ruff handles it. We use `ruff check` and (in this repo) the post-tool hook auto-formats.
- **Types**: parameter annotations everywhere. Return types where non-trivial.
- **Comments**: rare. Code should be self-explanatory. The exception: invariants ("This pickle must be loaded by the same Python version") and gotchas ("Anthropic SDK 0.40+ required for cache_control").
- **Tests**: every module gets `tests/test_<module>.py`. Stub the LM. Don't hit the network in CI.

---

## Commit format

Phase commits use this template:

```
Phase <N>: <short title>

<bulleted what + why summary>

<test count line>
<ruff line>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

We aim for one logical change per commit. Tests + docs go in the same commit as the code they describe.

---

## Reviewing

If you're reviewing a PR:

1. Run `uv run python -m pytest tests/ -q` locally.
2. Check that the new feature is exported from `__init__.py` and listed in `docs/API_REFERENCE.md`.
3. For RLM / GEPA / Orchestrator changes, verify the trace events are still helpful.
4. For new LM providers, run a `--selftest` against the real provider once before approving.

---

## What this repo deliberately doesn't support

(See [docs/DECISIONS.md](docs/DECISIONS.md) for full reasoning.)

- A plugin registry (D-12).
- Code-driven RLM via arbitrary `exec` (D-2).
- Async-everywhere (D-3).
- Auto-detecting the LM in tests (D-4).

If you need any of these, open an issue with a use case — we'll consider opt-in extensions.

---

## Release process

1. Bump `__version__` in `src/harness_rlm/__init__.py`.
2. Update [CHANGELOG.md](CHANGELOG.md) with the diff summary.
3. Run the test suite end-to-end: `uv run python -m pytest tests/ -q`.
4. Tag: `git tag v0.X.0 && git push origin v0.X.0`.
5. (Future) publish to PyPI: `uv build && uv publish`.
