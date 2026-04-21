"""Pydantic data classes for the RLM MCP server."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Default sub-LLM model — Haiku 4.5 per R3 §4 token economics (Apr 2026).
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class LLMQueryRequest(BaseModel):
    """Request payload for the `llm_query` MCP tool."""

    prompt: str = Field(..., description="User prompt sent to the sub-LLM.")
    model: str = Field(
        default=DEFAULT_MODEL,
        description="Anthropic model ID (e.g. claude-haiku-4-5-20251001).",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=64_000,
        description="Maximum output tokens for the completion.",
    )
    system: str | None = Field(
        default=None,
        description="Optional system prompt prepended to the request.",
    )


class LLMQueryResponse(BaseModel):
    """Structured response returned by the `llm_query` MCP tool."""

    content: str = Field(..., description="Text content returned by the model.")
    input_tokens: int = Field(..., ge=0, description="Input tokens consumed.")
    output_tokens: int = Field(..., ge=0, description="Output tokens generated.")
    model: str = Field(..., description="Model ID that produced the response.")
    cost_usd: float = Field(..., ge=0.0, description="Estimated cost in USD.")


class SubCallLog(BaseModel):
    """One line in /tmp/rlm/sub_calls.jsonl — audit trail of every sub-LLM call."""

    timestamp: str = Field(..., description="ISO-8601 timestamp (UTC).")
    prompt_preview: str = Field(
        ...,
        description="First 200 chars of the prompt (for debugging without leaking full context).",
    )
    response_chars: int = Field(..., ge=0, description="Length of response content in chars.")
    model: str = Field(..., description="Model ID used for the call.")
    cost_usd: float = Field(..., ge=0.0, description="Cost in USD for this sub-call.")
