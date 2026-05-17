"""harness-rlm: Use any existing coding-agent harness as a Recursive Language Model substrate.

Top-level imports surface the most commonly used pieces. Sub-modules are
deliberately small enough to read end-to-end:

    core       — BudgetGuard, chunker, ingest parser, skill loader
    trajectory — session log
    signatures — typed I/O contracts (DSPy-inspired)
    modules    — Module / Predict / ChainOfThought / Retry
    llm        — direct Anthropic LM client + global default
    rlm        — RLM module: recursive decomposition over long context
    gepa       — Pareto-frontier reflective prompt optimizer
    harness    — Pi-style one-liner top-level API
    orchestrator — Hermes-style multi-agent + trajectory
    mcp_server — MCP transport so other harnesses can call us as a sub-LLM
"""

from harness_rlm.core import (
    DEFAULT_BUDGETS,
    BudgetExceededError,
    BudgetGuard,
    chunk_context,
    load_shared_skill,
    parse_ingest_directives,
)
from harness_rlm.llm import LM, configure, get_lm
from harness_rlm.models import LLMQueryRequest, LLMQueryResponse, SubCallLog
from harness_rlm.modules import (
    ChainOfThought,
    Module,
    Predict,
    Prediction,
    Retry,
    Trace,
)
from harness_rlm.agent_loop import AgentLoop, AgentLoopConfig, AgentLoopResult
from harness_rlm.batch import BatchJobResult, BatchResult, spawn_agents_on_csv
from harness_rlm.claude_cli_lm import ClaudeCLILM
from harness_rlm.gepa import GEPA, Candidate, GEPAResult, ScoreWithFeedback
from harness_rlm.harness import RunResult, run
from harness_rlm.orchestrator import (
    Orchestrator,
    OrchestratorResult,
    SessionStore,
    Step,
    compress,
)
from harness_rlm.rlm import RLM, RLMConfig
from harness_rlm.signatures import Field_, Signature, SignatureParseError
from harness_rlm.subagents import (
    SubagentSpec,
    discover,
    dispatch,
    load_agents_md,
    load_spec,
)
from harness_rlm.tools import (
    BASH_TOOL,
    EDIT_TOOL,
    FINISH_TOOL,
    PI_CORE_TOOLS,
    READ_TOOL,
    WRITE_TOOL,
    AgentTool,
    ToolResult,
    from_function,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # models
    "LLMQueryRequest",
    "LLMQueryResponse",
    "SubCallLog",
    # core
    "BudgetGuard",
    "BudgetExceededError",
    "DEFAULT_BUDGETS",
    "chunk_context",
    "parse_ingest_directives",
    "load_shared_skill",
    # signatures
    "Signature",
    "Field_",
    "SignatureParseError",
    # modules
    "Module",
    "Predict",
    "ChainOfThought",
    "Retry",
    "Prediction",
    "Trace",
    # llm
    "LM",
    "ClaudeCLILM",
    "configure",
    "get_lm",
    # rlm
    "RLM",
    "RLMConfig",
    # gepa
    "GEPA",
    "GEPAResult",
    "Candidate",
    "ScoreWithFeedback",
    # top-level harness
    "run",
    "RunResult",
    # orchestrator
    "Orchestrator",
    "OrchestratorResult",
    "Step",
    "SessionStore",
    "compress",
    # tools (Pi-style AgentTool)
    "AgentTool",
    "ToolResult",
    "READ_TOOL",
    "WRITE_TOOL",
    "EDIT_TOOL",
    "BASH_TOOL",
    "FINISH_TOOL",
    "PI_CORE_TOOLS",
    "from_function",
    # agent loop (Pi-style with hooks)
    "AgentLoop",
    "AgentLoopConfig",
    "AgentLoopResult",
    # subagents (Codex-style TOML)
    "SubagentSpec",
    "discover",
    "dispatch",
    "load_spec",
    "load_agents_md",
    # csv batch (Codex-style)
    "spawn_agents_on_csv",
    "BatchResult",
    "BatchJobResult",
]
