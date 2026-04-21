"""Tests for harness_rlm.models Pydantic classes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_llm_query_request_defaults():
    """Only `prompt` is required; other fields get defaults."""
    from harness_rlm.models import DEFAULT_MODEL, LLMQueryRequest

    req = LLMQueryRequest(prompt="hello")
    assert req.prompt == "hello"
    assert req.model == DEFAULT_MODEL
    assert req.max_tokens == 1024
    assert req.system is None


def test_llm_query_request_requires_prompt():
    """`prompt` is a required field; missing it raises ValidationError."""
    from harness_rlm.models import LLMQueryRequest

    with pytest.raises(ValidationError) as exc_info:
        LLMQueryRequest()
    errs = exc_info.value.errors()
    assert any(e["loc"] == ("prompt",) for e in errs)


def test_llm_query_request_accepts_known_model():
    """Explicit model IDs override the default."""
    from harness_rlm.models import LLMQueryRequest

    req = LLMQueryRequest(prompt="hi", model="claude-sonnet-4-6-20250101")
    assert req.model == "claude-sonnet-4-6-20250101"


def test_llm_query_request_unknown_model_accepted_as_string():
    """Model field is a string — unknown IDs don't raise at construction time.
    (Pricing fallback is handled downstream in compute_cost, not the Pydantic model.)
    """
    from harness_rlm.models import LLMQueryRequest

    req = LLMQueryRequest(prompt="hi", model="totally-made-up-model")
    assert req.model == "totally-made-up-model"


def test_llm_query_request_max_tokens_bounds():
    """max_tokens has ge=1, le=64_000 constraints."""
    from harness_rlm.models import LLMQueryRequest

    # valid boundary
    req = LLMQueryRequest(prompt="x", max_tokens=1)
    assert req.max_tokens == 1
    req = LLMQueryRequest(prompt="x", max_tokens=64_000)
    assert req.max_tokens == 64_000

    with pytest.raises(ValidationError):
        LLMQueryRequest(prompt="x", max_tokens=0)
    with pytest.raises(ValidationError):
        LLMQueryRequest(prompt="x", max_tokens=64_001)


def test_llm_query_request_system_optional():
    """system may be str or None; explicitly set takes precedence over default."""
    from harness_rlm.models import LLMQueryRequest

    req = LLMQueryRequest(prompt="x", system="be terse")
    assert req.system == "be terse"
    req = LLMQueryRequest(prompt="x", system=None)
    assert req.system is None


def test_llm_query_response_roundtrip():
    """All fields required; construction with valid values works."""
    from harness_rlm.models import LLMQueryResponse

    resp = LLMQueryResponse(
        content="hi there",
        input_tokens=10,
        output_tokens=4,
        model="claude-haiku-4-5-20251001",
        cost_usd=0.00003,
    )
    assert resp.content == "hi there"
    assert resp.input_tokens == 10
    assert resp.output_tokens == 4
    assert resp.cost_usd == pytest.approx(0.00003)


def test_llm_query_response_requires_all_fields():
    """Every field on LLMQueryResponse is required (no defaults)."""
    from harness_rlm.models import LLMQueryResponse

    with pytest.raises(ValidationError):
        LLMQueryResponse()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        LLMQueryResponse(content="x")  # type: ignore[call-arg]


def test_llm_query_response_rejects_negative_numbers():
    """input_tokens/output_tokens/cost_usd are ge=0."""
    from harness_rlm.models import LLMQueryResponse

    with pytest.raises(ValidationError):
        LLMQueryResponse(
            content="x",
            input_tokens=-1,
            output_tokens=0,
            model="claude-haiku-4-5",
            cost_usd=0.0,
        )
    with pytest.raises(ValidationError):
        LLMQueryResponse(
            content="x",
            input_tokens=0,
            output_tokens=0,
            model="claude-haiku-4-5",
            cost_usd=-0.01,
        )


def test_sub_call_log_roundtrip():
    """All fields required; serialises as JSON cleanly for jsonl output."""
    from harness_rlm.models import SubCallLog

    entry = SubCallLog(
        timestamp="2026-04-21T22:34:56Z",
        prompt_preview="hello world",
        response_chars=42,
        model="claude-haiku-4-5-20251001",
        cost_usd=0.000123,
    )
    assert entry.response_chars == 42
    # Confirm JSON serialisation works — this is how it lands in sub_calls.jsonl.
    payload = entry.model_dump_json()
    assert "claude-haiku-4-5-20251001" in payload
    assert "hello world" in payload


def test_sub_call_log_requires_all_fields():
    """SubCallLog has no defaults — omission is a validation error."""
    from harness_rlm.models import SubCallLog

    with pytest.raises(ValidationError):
        SubCallLog()  # type: ignore[call-arg]
