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
from harness_rlm.signatures import Field_, Signature, SignatureParseError

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
    "configure",
    "get_lm",
]
