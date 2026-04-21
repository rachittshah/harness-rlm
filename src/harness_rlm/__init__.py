"""harness-rlm: Use any existing coding-agent harness as a Recursive Language Model substrate."""

from harness_rlm.models import LLMQueryRequest, LLMQueryResponse, SubCallLog

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LLMQueryRequest",
    "LLMQueryResponse",
    "SubCallLog",
]
