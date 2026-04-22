"""harness-rlm: Use any existing coding-agent harness as a Recursive Language Model substrate."""

from harness_rlm.core import (
    DEFAULT_BUDGETS,
    BudgetExceededError,
    BudgetGuard,
    chunk_context,
    load_shared_skill,
    parse_ingest_directives,
)
from harness_rlm.models import LLMQueryRequest, LLMQueryResponse, SubCallLog

__version__ = "0.1.0"

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
]
