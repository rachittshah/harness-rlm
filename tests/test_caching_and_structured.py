"""Tests for caching helpers + Pydantic-typed structured outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from harness_rlm.caching import (
    cached_system_block,
    cached_text_block,
    cached_tools,
    estimate_cache_savings,
)
from harness_rlm.llm import LMResult
from harness_rlm.signatures import SignatureParseError
from harness_rlm.structured import TypedPredict


# ---------------------------------------------------------------------------
# Stub LM
# ---------------------------------------------------------------------------
@dataclass
class _StubLM:
    canned: list[str] = field(default_factory=list)
    model: str = "stub"
    max_tokens: int = 1024

    def __post_init__(self):
        self.calls = 0
        self.received: list[str] = []

    def __call__(self, prompt, *, system=None, model=None, max_tokens=None):
        self.received.append(prompt)
        idx = min(self.calls, len(self.canned) - 1)
        text = self.canned[idx] if self.canned else ""
        self.calls += 1
        return LMResult(
            text=text,
            model=self.model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
            cost_usd=0.0001,
            latency_s=0.01,
        )


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------
class TestCaching:
    def test_cached_text_block(self):
        b = cached_text_block("hello")
        assert b["type"] == "text"
        assert b["text"] == "hello"
        assert b["cache_control"] == {"type": "ephemeral"}

    def test_cached_system_returns_list(self):
        out = cached_system_block("you are helpful")
        assert isinstance(out, list)
        assert out[0]["cache_control"] == {"type": "ephemeral"}

    def test_cached_tools_marks_only_last(self):
        tools = [
            {"name": "a", "description": "a", "input_schema": {}},
            {"name": "b", "description": "b", "input_schema": {}},
            {"name": "c", "description": "c", "input_schema": {}},
        ]
        out = cached_tools(tools)
        assert "cache_control" not in out[0]
        assert "cache_control" not in out[1]
        assert out[2]["cache_control"] == {"type": "ephemeral"}

    def test_cached_tools_idempotent(self):
        tools = [
            {"name": "a", "description": "a", "input_schema": {}, "cache_control": {"type": "ephemeral"}},
            {"name": "b", "description": "b", "input_schema": {}, "cache_control": {"type": "ephemeral"}},
        ]
        out = cached_tools(tools)
        assert "cache_control" not in out[0]
        assert out[1]["cache_control"] == {"type": "ephemeral"}

    def test_cached_tools_empty(self):
        assert cached_tools([]) == []

    def test_savings_estimate(self):
        s = estimate_cache_savings(cached_tokens=1_000_000, uncached_tokens=10_000)
        # Caching must save money overall.
        assert s["total_with_cache"] < s["total_without_cache"]
        assert s["savings_usd"] > 0
        assert s["savings_pct"] > 0


# ---------------------------------------------------------------------------
# Pydantic structured outputs
# ---------------------------------------------------------------------------
class AnswerSchema(BaseModel):
    result: int
    explanation: str


class TestTypedPredict:
    def test_happy_path(self):
        lm = _StubLM(
            canned=['{"result": 4, "explanation": "two plus two"}']
        )
        m = TypedPredict("question -> answer", output_model=AnswerSchema, lm=lm)  # type: ignore[arg-type]
        pred = m(question="2+2?")
        assert pred.value.result == 4
        assert pred.value.explanation == "two plus two"
        # Fields mirror model_dump.
        assert pred.fields["result"] == 4

    def test_json_inside_code_fence(self):
        lm = _StubLM(
            canned=['Here you go:\n```json\n{"result": 5, "explanation": "five"}\n```']
        )
        m = TypedPredict("q -> a", output_model=AnswerSchema, lm=lm)  # type: ignore[arg-type]
        pred = m(q="x")
        assert pred.value.result == 5

    def test_validation_failure_retries(self):
        # First: missing required field. Second: valid.
        lm = _StubLM(
            canned=[
                '{"result": 4}',  # missing explanation
                '{"result": 4, "explanation": "ok"}',
            ]
        )
        m = TypedPredict("q -> a", output_model=AnswerSchema, lm=lm, max_retries=2)  # type: ignore[arg-type]
        pred = m(q="x")
        assert pred.value.result == 4
        # Second prompt should include the validation error feedback.
        assert "validation" in lm.received[1].lower() or "previous" in lm.received[1].lower()

    def test_max_retries_exhausted_raises(self):
        lm = _StubLM(canned=["{garbage"])  # always fails
        m = TypedPredict("q -> a", output_model=AnswerSchema, lm=lm, max_retries=1)  # type: ignore[arg-type]
        with pytest.raises(SignatureParseError, match="failed after"):
            m(q="x")

    def test_rejects_non_basemodel(self):
        with pytest.raises(TypeError, match="BaseModel"):
            TypedPredict("q -> a", output_model=dict)  # type: ignore[arg-type]

    def test_schema_appears_in_prompt(self):
        lm = _StubLM(canned=['{"result": 1, "explanation": "x"}'])
        m = TypedPredict("q -> a", output_model=AnswerSchema, lm=lm)  # type: ignore[arg-type]
        m(q="?")
        first = lm.received[0]
        assert '"result"' in first
        assert '"explanation"' in first
