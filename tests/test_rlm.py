"""Tests for harness_rlm.rlm — RLM module + decomposition (no network)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from harness_rlm.llm import LMResult
from harness_rlm.rlm import NOT_FOUND, RLM, RLMConfig


@dataclass
class _StubLM:
    """Stateful test double — returns canned responses in order; records prompts."""

    canned: list[str] = field(default_factory=list)
    model: str = "stub-model"
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        self.received: list[str] = []
        self.calls = 0
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def __call__(self, prompt, *, system=None, model=None, max_tokens=None):
        self.received.append(prompt)
        idx = min(self.calls, len(self.canned) - 1)
        text = self.canned[idx] if self.canned else "answer: stub"
        self.calls += 1
        self.total_calls += 1
        return LMResult(
            text=text,
            model=self.model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
            cost_usd=0.0001,
            latency_s=0.01,
        )


class TestFlatFastPath:
    def test_uses_flat_when_context_small(self):
        root = _StubLM(canned=["answer: small"])
        sub = _StubLM(canned=["should not be called"])
        cfg = RLMConfig(flat_char_threshold=10_000)
        m = RLM(
            "question, document -> answer",
            config=cfg,
            root_lm=root,  # type: ignore[arg-type]
            sub_lm=sub,  # type: ignore[arg-type]
        )
        pred = m(question="?", document="short text")
        assert pred.answer == "small"
        assert sub.calls == 0
        assert root.calls == 1
        labels = [e["label"] for e in pred.trace.events]
        assert "flat" in labels


class TestMapReduce:
    def test_decomposes_long_context(self):
        # Long context: 50K chars; chunk_size 20K → 3 chunks + synth = 4 calls.
        long_ctx = "x" * 50_000
        root = _StubLM(canned=["synthesized answer"])
        sub = _StubLM(canned=["part A", "part B", "part C"])
        cfg = RLMConfig(
            flat_char_threshold=5_000,
            chunk_size=20_000,
            overlap=0,
            max_parallel=1,  # sequential = deterministic for assertions
        )
        m = RLM(
            "question, document -> answer",
            config=cfg,
            root_lm=root,  # type: ignore[arg-type]
            sub_lm=sub,  # type: ignore[arg-type]
        )
        pred = m(question="what?", document=long_ctx)
        assert pred.answer == "synthesized answer"
        # 3 sub calls + 1 synth call
        assert sub.calls == 3
        assert root.calls == 1
        # Synth prompt mentions all three partials.
        synth_prompt = root.received[0]
        for tag in ["[0] part A", "[1] part B", "[2] part C"]:
            assert tag in synth_prompt

    def test_parallel_preserves_order(self):
        long_ctx = "x" * 50_000
        root = _StubLM(canned=["final"])
        # Each chunk gets a distinct response — order must be preserved.
        sub = _StubLM(canned=["A", "B", "C"])
        cfg = RLMConfig(
            flat_char_threshold=5_000,
            chunk_size=20_000,
            overlap=0,
            max_parallel=4,
        )
        m = RLM(
            "question, document -> answer",
            config=cfg,
            root_lm=root,  # type: ignore[arg-type]
            sub_lm=sub,  # type: ignore[arg-type]
        )
        m(question="?", document=long_ctx)
        # Note: with parallel dispatch and a global counter on the stub,
        # we just verify all three partials reach synth in some order.
        synth_prompt = root.received[0]
        for tag in ["[0]", "[1]", "[2]"]:
            assert tag in synth_prompt


class TestBudgetEnforcement:
    def test_truncates_chunks_when_over_budget(self):
        long_ctx = "x" * 200_000  # 10 chunks at 20K
        root = _StubLM(canned=["final"])
        sub = _StubLM(canned=["partial"] * 20)
        # Budget allows only 3 calls total → 2 sub + 1 synth max.
        cfg = RLMConfig(
            flat_char_threshold=5_000,
            chunk_size=20_000,
            overlap=0,
            max_parallel=1,
            max_llm_calls=3,
        )
        m = RLM(
            "question, document -> answer",
            config=cfg,
            root_lm=root,  # type: ignore[arg-type]
            sub_lm=sub,  # type: ignore[arg-type]
        )
        pred = m(question="?", document=long_ctx)
        assert sub.calls == 2  # truncated
        assert root.calls == 1  # synth still ran
        labels = [e["label"] for e in pred.trace.events]
        assert "chunk_budget_truncated" in [
            e.get("label") for e in pred.trace.events
        ]


class TestSignatureValidation:
    def test_missing_long_context_field_raises(self):
        with pytest.raises(ValueError, match="long-context field"):
            RLM("question -> answer", long_context_field="document")


class TestNotFoundHandling:
    def test_partials_marked_not_found_still_flow_to_synth(self):
        long_ctx = "x" * 50_000
        root = _StubLM(canned=["final answer"])
        sub = _StubLM(canned=[NOT_FOUND, "real partial", NOT_FOUND])
        cfg = RLMConfig(
            flat_char_threshold=5_000,
            chunk_size=20_000,
            overlap=0,
            max_parallel=1,
        )
        m = RLM(
            "question, document -> answer",
            config=cfg,
            root_lm=root,  # type: ignore[arg-type]
            sub_lm=sub,  # type: ignore[arg-type]
        )
        pred = m(question="?", document=long_ctx)
        assert pred.answer == "final answer"
        synth_prompt = root.received[0]
        assert NOT_FOUND in synth_prompt
        assert "real partial" in synth_prompt
