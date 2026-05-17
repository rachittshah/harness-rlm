"""Tests for harness_rlm.modules (LM is mocked — no network)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from harness_rlm.llm import LMResult, configure
from harness_rlm.modules import ChainOfThought, Predict, Retry, Trace
from harness_rlm.signatures import Signature, SignatureParseError


# ---------------------------------------------------------------------------
# Stub LM — never hits the network.
# ---------------------------------------------------------------------------
@dataclass
class _StubLM:
    """Test double for harness_rlm.llm.LM. Records prompts; returns canned text."""

    canned: list[str]
    model: str = "stub-model"
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        self.received: list[str] = []
        self.calls = 0

    def __call__(self, prompt, *, system=None, model=None, max_tokens=None):
        self.received.append(prompt)
        idx = min(self.calls, len(self.canned) - 1)
        self.calls += 1
        return LMResult(
            text=self.canned[idx],
            model=self.model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(self.canned[idx]) // 4,
            cost_usd=0.0001,
            latency_s=0.01,
        )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
class TestPredict:
    def test_happy_path(self):
        lm = _StubLM(canned=["answer: forty-two"])
        m = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        pred = m(q="what?")
        assert pred.answer == "forty-two"
        assert pred["answer"] == "forty-two"
        assert pred.trace is not None
        assert pred.trace.calls == 1
        assert "what?" in lm.received[0]

    def test_unknown_field_raises_attribute_error(self):
        lm = _StubLM(canned=["answer: hi"])
        m = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        pred = m(q="x")
        with pytest.raises(AttributeError, match="no field"):
            pred.does_not_exist

    def test_parse_failure_raises(self):
        lm = _StubLM(canned=["this has no label"])
        m = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        with pytest.raises(SignatureParseError):
            m(q="x")

    def test_set_instruction_changes_prompt(self):
        lm = _StubLM(canned=["answer: ok"])
        m = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        m.set_instruction("Be terse.")
        m(q="x")
        assert "Be terse." in lm.received[0]


# ---------------------------------------------------------------------------
# ChainOfThought
# ---------------------------------------------------------------------------
class TestChainOfThought:
    def test_adds_reasoning_field(self):
        lm = _StubLM(canned=["reasoning: I thought about it\nanswer: 42"])
        m = ChainOfThought("q -> answer", lm=lm)  # type: ignore[arg-type]
        assert "reasoning" in m.signature.output_names()
        pred = m(q="x")
        assert pred.reasoning == "I thought about it"
        assert pred.answer == "42"

    def test_idempotent_when_reasoning_already_present(self):
        sig = Signature(inputs=["q"], outputs=["reasoning", "answer"])
        lm = _StubLM(canned=["reasoning: r\nanswer: a"])
        m = ChainOfThought(sig, lm=lm)  # type: ignore[arg-type]
        # Should not duplicate the field.
        assert m.signature.output_names().count("reasoning") == 1


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------
class TestRetry:
    def test_succeeds_on_second_attempt(self):
        lm = _StubLM(canned=["garbage", "answer: ok"])
        child = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        wrapped = Retry(child, max_attempts=2)
        pred = wrapped(q="x")
        assert pred.answer == "ok"
        # Combined trace should reflect both LM calls.
        assert pred.trace.calls == 1  # only the successful child call counted
        # Parse error appears in events.
        labels = [e.get("label") for e in pred.trace.events]
        assert "parse_error" in labels

    def test_gives_up_after_max(self):
        lm = _StubLM(canned=["bad1", "bad2", "bad3"])
        child = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        wrapped = Retry(child, max_attempts=2)
        with pytest.raises(SignatureParseError):
            wrapped(q="x")


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------
class TestTrace:
    def test_absorb_sums(self):
        a = Trace(module="a", calls=1, cost_usd=0.01, latency_s=0.5)
        b = Trace(module="b", calls=2, cost_usd=0.02, latency_s=0.3)
        a.absorb(b)
        assert a.calls == 3
        assert a.cost_usd == pytest.approx(0.03)
        assert a.latency_s == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Global configure() works
# ---------------------------------------------------------------------------
class TestGlobalConfigure:
    def test_uses_global_when_no_lm(self, monkeypatch):
        lm = _StubLM(canned=["answer: from-global"])
        configure(lm=lm)  # type: ignore[arg-type]
        try:
            m = Predict("q -> answer")  # no lm= kwarg
            pred = m(q="x")
            assert pred.answer == "from-global"
        finally:
            # Reset for other tests — set a fresh default.
            from harness_rlm import llm as _llm_mod

            _llm_mod._settings.lm = None
