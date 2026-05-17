# API Reference

Generated from source. Each entry: signature, args, return, example.

For architectural context see [ARCHITECTURE.md](ARCHITECTURE.md); for usage patterns see [COOKBOOK.md](COOKBOOK.md).

---

## Signatures (`harness_rlm.signatures`)

### `class Signature`

Declarative input/output contract.

**Constructor forms** — pass one of:
```python
Signature("question, context -> answer")             # shorthand
Signature(inputs=["question"], outputs=["answer"])   # list form
Signature(
    inputs={"q": "the question"},                    # dict form (with desc)
    outputs={"a": "the answer"},
    instruction="Be brief.",
)
```

**Methods**
| Method | Returns | Purpose |
|---|---|---|
| `input_names()` | `list[str]` | Names of input fields. |
| `output_names()` | `list[str]` | Names of output fields. |
| `with_instruction(s)` | `Signature` | Copy with a new instruction. |
| `render_prompt(values)` | `str` | Render the LM prompt from input values. |
| `parse_response(text)` | `dict[str, str]` | Parse LM output back to fields. |

**Raises**
- `ValueError` on construction if outputs empty or both shorthand+explicit passed.
- `SignatureParseError` on `parse_response` if any output field is missing.

---

## LM clients

### `class LM` — Anthropic (`harness_rlm.llm`)

```python
LM(
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1024,
    api_key: str | None = None,
    log_path: Path | None = Path("/tmp/rlm/sub_calls.jsonl"),
    enable_caching: bool = False,
    thinking_budget: int | None = None,
)
```

**Call** — `lm(prompt, *, system=None, model=None, max_tokens=None, enable_caching=None, thinking_budget=None) -> LMResult`

**Cumulative counters** (thread-safe)
- `total_calls`, `total_input_tokens`, `total_output_tokens`, `total_cost_usd`
- `total_cache_read_tokens`, `total_cache_write_tokens`

**`stats()`** — snapshot dict of the above.

### `class OpenAILM` — OpenAI (`harness_rlm.providers`)

```python
OpenAILM(
    model: str = "gpt-5-mini",
    max_tokens: int = 1024,
    api_key: str | None = None,
    reasoning_effort: str | None = None,    # "low" | "medium" | "high"
    base_url: str | None = None,            # Azure / custom proxies
)
```

Drop-in for `LM`. Pricing table covers gpt-5{,.2,-mini,-nano} and o3{,-mini}.

### `class LiteLLMLM` — universal (`harness_rlm.providers`)

```python
LiteLLMLM(
    model: str = "openai/gpt-5-mini",
    max_tokens: int = 1024,
    num_retries: int = 0,
    api_base: str | None = None,
    extra_kwargs: dict = {},
)
```

100+ providers via LiteLLM. Model strings: `openai/...`, `anthropic/...`, `openrouter/...`, `bedrock/...`, `ollama/...`, etc.

### `class ClaudeCLILM` — `claude -p` (`harness_rlm.claude_cli_lm`)

```python
ClaudeCLILM(
    model: str | None = None,
    bin: str = "claude",
    timeout_s: int = 120,
    bare: bool = False,    # requires ANTHROPIC_API_KEY when True
)
```

Shell-out provider for environments where direct API isn't available but `claude` CLI is authenticated via OAuth.

### `class AsyncLM` — asyncio (`harness_rlm.streaming`)

Same surface as `LM`, but `await lm(prompt)` and `async for evt in lm.stream(prompt)`.

---

## Modules

### `class Module` (abstract base)

```python
class Module:
    signature: Signature
    lm: LM | None
    def forward(self, **inputs) -> Prediction: ...
    def __call__(self, **inputs) -> Prediction: ...
    def set_instruction(self, s: str) -> None: ...
    def get_instruction(self) -> str: ...
```

### `Predict(signature, *, lm=None, name=None)`

One LM call: signature → prompt → LM → parsed Prediction.

### `ChainOfThought(signature, *, lm=None, name=None)`

`Predict` + injected `reasoning` output field that the LM fills first.

### `Retry(child, *, max_attempts=3, name=None)`

Wrap a Module to retry on `SignatureParseError`.

### `RLM(signature, long_context_field, config, root_lm, sub_lm, name=None)`

Recursive long-context module.

```python
RLM(
    "question, document -> answer",
    long_context_field="document",
    config=RLMConfig(strategy="map_reduce"|"tree"|"filter_then_query"),
    root_lm=LM(model="claude-opus-4-7"),
    sub_lm=LM(model="claude-haiku-4-5-20251001"),
)
```

### `TypedPredict[T](signature, *, output_model, lm=None, max_retries=2)`

Pydantic-validated outputs. `output_model` must be a `BaseModel` subclass. Returns `TypedPrediction` whose `.value` is a model instance.

### `BestOfN(child, *, n=5, vote_fn=None, max_parallel=5, keep_all=False)`

N parallel samples → majority vote on the first non-reasoning output field. Pass `vote_fn` to customise aggregation.

### `SelfConsistency(child, *, n=5, answer_field="answer")`

Variant of `BestOfN` tuned for `ChainOfThought` — votes on the answer field regardless of differing reasoning paths.

---

## RLM configuration

```python
@dataclass
class RLMConfig:
    strategy: "map_reduce"|"filter_then_query"|"tree"|"auto" = "map_reduce"
    chunk_size: int = 20_000
    overlap: int = 200
    max_parallel: int = 8
    max_iterations: int = 20
    max_llm_calls: int = 50
    flat_char_threshold: int = 80_000
    filter_pattern: str | None = None
    sub_instruction: str = DEFAULT_SUB_INSTRUCTION
    synth_instruction: str = DEFAULT_SYNTH_INSTRUCTION
    sub_max_tokens: int = 512
    synth_max_tokens: int = 1024
    tree_branching: int = 4
    tree_max_depth: int = 4
```

---

## GEPA optimizer

```python
GEPA(
    metric: Callable[[dict, Prediction], ScoreWithFeedback],
    *,
    max_iterations: int = 8,
    reflection_lm: LM | None = None,
    minibatch_size: int = 3,
    rng_seed: int | None = None,
)

result = gepa.compile(student: Module, trainset, valset=None) -> GEPAResult
# student is mutated in place to carry the best instruction.

GEPAResult:
    best: Candidate
    pareto: list[Candidate]
    history: list[Candidate]
    rollouts: int

Candidate:
    instruction: str
    scores: list[float]      # per trainset example
    feedback: list[str]
    val_mean: float | None
```

The metric must return `ScoreWithFeedback(score: float in [0,1], feedback: str)`. Feedback drives evolution — be specific.

---

## Top-level API

### `run(question, context=None, *, root_model="claude-opus-4-7", sub_model="claude-haiku-4-5-20251001", ...) -> RunResult`

The Pi-style one-liner. Routes to flat `Predict` if `context` is short (or None), to `RLM` otherwise. Optional `optimize_with=(trainset, metric)` runs GEPA before answering.

```python
RunResult:
    answer: str
    fields: dict
    trace: dict | None
    cost_usd: float
    calls: int
```

### CLI: `harness-rlm "question" [--context-file FOO] [--json] [--max-calls N] [--instruction ...]`

---

## Orchestrator

```python
Step(name: str, module: Module, input_builder: Callable[[dict], dict], store_as: str | None = None)

Orchestrator(steps: list[Step], *, continue_on_error: bool = False)
    .run(initial_state=None, *, session_store: SessionStore | None = None) -> OrchestratorResult

OrchestratorResult:
    state: dict          # accumulated per-step outputs
    trace: Trace         # combined cost/tokens
    steps: list[dict]    # per-step records
```

### `SessionStore(name: str, base_dir: Path = Path("/tmp/rlm"))`

Append-only JSONL log under `{base_dir}/{name}/events.jsonl`. Methods: `append(event)`, `read()`, `clear()`.

### `compress(text, *, target_chars=4_000, lm=None, max_tokens=2_000) -> str`

LM-summarised compaction. No-op if `text` already fits.

---

## Tools (Pi style)

### `AgentTool`

```python
AgentTool(
    name: str,                  # snake_case identifier
    description: str,
    parameters: dict,           # JSON Schema
    execute: Callable[[dict], ToolResult],
    execution_mode: "sequential"|"parallel" = "sequential",
    label: str | None = None,
)
```

### `ToolResult`

```python
ToolResult(
    content: list[dict],    # {"type": "text", "text": "..."}
    details: Any = None,    # structured payload for UI/trace
    terminate: bool = False,
    error: str | None = None,
)

# Helpers:
ToolResult.text("hi", details={...}, terminate=False)
ToolResult.fail("description of failure")
```

### Builtin 4-tool core

- `READ_TOOL` — read file
- `WRITE_TOOL` — write file (creates parents)
- `EDIT_TOOL` — exact-match replace
- `BASH_TOOL` — shell command (60s default timeout, 20K output cap)
- `FINISH_TOOL` — emit final answer and end loop (terminate=True)

All bundled as `PI_CORE_TOOLS`.

### `from_function(fn, *, name=None, description=None, parameters=None) -> AgentTool`

Wrap any Python callable. Derives JSON Schema from `inspect.signature`. Docstring becomes description.

---

## AgentLoop (tool-using)

```python
AgentLoop(tools: list[AgentTool], config: AgentLoopConfig | None = None, *, api_key=None)
    .run(user_message: str) -> AgentLoopResult

AgentLoopConfig:
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 4096
    max_turns: int = 20
    system: str | None = None
    # Hooks (Pi-inspired):
    transform_context: Callable | None     # compaction
    should_stop_after_turn: Callable | None
    prepare_next_turn: Callable | None     # model swap mid-run
    before_tool_call: Callable | None      # return True to block
    after_tool_call: Callable | None       # rewrite result
    get_steering_messages: Callable | None # async user input
    get_follow_up_messages: Callable | None

AgentLoopResult:
    final_text: str
    messages: list[dict]
    turns: int
    terminated_by_tool: bool
    tool_call_count: int
    cost_usd: float
    events: list[dict]
```

---

## Subagents (Codex style)

```python
discover(search_paths=None) -> dict[str, SubagentSpec]
load_spec(path) -> SubagentSpec
dispatch(spec, task, *, parent_sandbox="danger", tool_registry=None,
         max_turns=12, api_key=None) -> AgentLoopResult
load_agents_md(start_dir=None, max_bytes=32*1024) -> str

SubagentSpec:
    name: str                                # safe identifier
    description: str
    model: str = "claude-haiku-4-5-20251001"
    instructions: str = ""
    reasoning_effort: "low"|"medium"|"high" = "medium"
    sandbox_mode: "read-only"|"workspace-write"|"danger" = "read-only"
    tools: list[str] = ["read", "bash", "finish_task"]
```

TOML files live at `.harness-rlm/agents/<name>.toml` (project) or `~/.harness-rlm/agents/<name>.toml` (personal). Project-local wins.

---

## CSV batch

```python
spawn_agents_on_csv(
    csv_path: Path,
    *,
    module: Module,
    input_template: dict[str, str],      # {"question": "{column_name}"}
    output_csv_path: Path,
    job_id: str | None = None,
    item_id_column: str = "id",
    max_parallel: int = 4,
    max_retries: int = 1,
    output_schema: dict | None = None,
    on_progress: Callable | None = None,
) -> BatchResult

BatchResult:
    job_id: str
    total: int
    ok: int
    err: int
    rows: list[BatchJobResult]
    output_csv: Path
    elapsed_s: float
```

---

## Caching

```python
cached_text_block(text) -> dict
cached_system_block(text) -> list[dict]
cached_tools(tools) -> list[dict]
estimate_cache_savings(cached_tokens, uncached_tokens, *, base_rate_per_million=5.0) -> dict
```

---

## Streaming

```python
stream(lm: LM, prompt: str, *, system=None, model=None, max_tokens=None) -> Generator[StreamEvent]

StreamEvent:
    kind: "text_delta"|"thinking_delta"|"done"
    text: str = ""
    result: LMResult | None = None
```

---

## Trace visualization

```python
format_trace(trace: Trace, *, color=False) -> str
trace_to_mermaid(trace: Trace, *, title=None) -> str
```

---

## MCP-as-client

```python
MCPToolset(
    command: str,           # e.g. "npx"
    args: list[str] = [],   # e.g. ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
    env: dict[str, str] | None = None,
    name_prefix: str = "",
)
    .start()                # spawn server + initialize session
    .tools() -> list[AgentTool]
    .close()
    # Or use as context manager.
```

---

## Core primitives

```python
BudgetGuard(budgets=DEFAULT_BUDGETS)
    .increment_call() / .check_call()
    .increment_iteration() / .check_iteration()
    .record_output(n) / .check_output(n)
    .state_dict() / BudgetGuard.from_state_dict(state)

DEFAULT_BUDGETS = {
    "max_iterations": 20,
    "max_llm_calls": 50,
    "max_output_chars": 10_000,
}

chunk_context(text, chunk_size=5_000, overlap=200) -> list[str]
parse_ingest_directives(msg) -> list[dict]  # /file /url /paste markers
load_shared_skill() -> str                  # frontmatter-stripped SKILL.md
```
