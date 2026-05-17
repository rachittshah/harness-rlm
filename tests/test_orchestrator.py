"""Tests for harness_rlm.orchestrator — Orchestrator, SessionStore, compress."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from harness_rlm.llm import LMResult
from harness_rlm.modules import Predict
from harness_rlm.orchestrator import (
    Orchestrator,
    SessionStore,
    Step,
    compress,
)


@dataclass
class _StubLM:
    canned: list[str] = field(default_factory=list)
    model: str = "stub-model"
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt, *, system=None, model=None, max_tokens=None):
        idx = min(self.calls, len(self.canned) - 1)
        text = self.canned[idx] if self.canned else "answer: stub"
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
# Orchestrator
# ---------------------------------------------------------------------------
class TestOrchestrator:
    def test_single_step(self):
        lm = _StubLM(canned=["answer: 42"])
        m = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        steps = [
            Step(
                name="qa",
                module=m,
                input_builder=lambda state: {"q": "what?"},
            )
        ]
        result = Orchestrator(steps).run()
        assert result.state["qa"]["answer"] == "42"
        assert result.trace.calls == 1

    def test_state_threads_forward(self):
        lm1 = _StubLM(canned=["answer: hello"])
        lm2 = _StubLM(canned=["answer: HELLO"])
        first = Predict("q -> answer", lm=lm1)  # type: ignore[arg-type]
        second = Predict("q -> answer", lm=lm2)  # type: ignore[arg-type]
        steps = [
            Step("first", first, lambda s: {"q": "say hello"}),
            # Second step reads first step's answer.
            Step(
                "second",
                second,
                lambda s: {"q": f"shout: {s['first']['answer']}"},
            ),
        ]
        result = Orchestrator(steps).run()
        assert result.state["first"]["answer"] == "hello"
        assert result.state["second"]["answer"] == "HELLO"
        # Cost rolls up across both modules.
        assert result.trace.calls == 2

    def test_failure_aborts(self):
        lm = _StubLM(canned=["bad output"])  # No "answer:" label → parse error
        m = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        steps = [
            Step("a", m, lambda s: {"q": "x"}),
            Step("b", m, lambda s: {"q": "y"}),
        ]
        result = Orchestrator(steps).run()
        # First step failed, second step did not run.
        assert result.steps[0]["ok"] is False
        assert len(result.steps) == 1

    def test_continue_on_error(self):
        lm = _StubLM(canned=["bad", "answer: ok"])
        m1 = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        m2 = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        steps = [
            Step("a", m1, lambda s: {"q": "x"}),
            Step("b", m2, lambda s: {"q": "y"}),
        ]
        result = Orchestrator(steps, continue_on_error=True).run()
        # Both steps recorded — first failed, second succeeded.
        assert result.steps[0]["ok"] is False
        assert result.steps[1]["ok"] is True
        assert result.state["b"]["answer"] == "ok"

    def test_duplicate_step_names_raise(self):
        lm = _StubLM(canned=["answer: x"])
        m = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        steps = [
            Step("x", m, lambda s: {"q": "."}),
            Step("x", m, lambda s: {"q": "."}),
        ]
        with pytest.raises(ValueError, match="unique"):
            Orchestrator(steps)

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Orchestrator([])


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------
class TestSessionStore:
    def test_append_and_read(self, tmp_path):
        store = SessionStore(name="test", base_dir=tmp_path)
        store.append({"kind": "event", "value": 1})
        store.append({"kind": "event", "value": 2})
        events = store.read()
        assert len(events) == 2
        assert events[0]["value"] == 1
        assert events[1]["value"] == 2
        # Auto-stamped timestamp.
        assert "ts" in events[0]

    def test_clear(self, tmp_path):
        store = SessionStore(name="test", base_dir=tmp_path)
        store.append({"x": 1})
        assert store.read() != []
        store.clear()
        assert store.read() == []

    def test_orchestrator_writes_to_session_store(self, tmp_path):
        lm = _StubLM(canned=["answer: ok"])
        m = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        store = SessionStore(name="orchtest", base_dir=tmp_path)
        steps = [Step("a", m, lambda s: {"q": "x"})]
        Orchestrator(steps).run(session_store=store)
        events = store.read()
        assert len(events) == 1
        assert events[0]["kind"] == "step"
        assert events[0]["fields"]["answer"] == "ok"


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------
class TestCompress:
    def test_passthrough_when_short(self):
        out = compress("short", target_chars=100)
        assert out == "short"

    def test_calls_lm_when_long(self):
        lm = _StubLM(canned=["summarized version"])
        long = "x" * 1000
        out = compress(long, target_chars=100, lm=lm)  # type: ignore[arg-type]
        assert out == "summarized version"
        assert lm.calls == 1
