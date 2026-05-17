"""Tests for harness_rlm.ensemble — BestOfN, SelfConsistency, majority_vote_on."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from harness_rlm.ensemble import BestOfN, SelfConsistency, majority_vote_on
from harness_rlm.llm import LMResult
from harness_rlm.modules import ChainOfThought, Predict


@dataclass
class _StubLM:
    canned: list[str] = field(default_factory=list)
    model: str = "stub"
    max_tokens: int = 1024

    def __post_init__(self):
        self.calls = 0
        # Thread-safe-ish since BestOfN uses threads — but tests run serially.
        import threading

        self._lock = threading.Lock()

    def __call__(self, prompt, *, system=None, model=None, max_tokens=None):
        with self._lock:
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


class TestMajorityVote:
    def test_picks_most_common(self):
        from harness_rlm.modules import Prediction

        preds = [
            Prediction(fields={"answer": "A"}),
            Prediction(fields={"answer": "B"}),
            Prediction(fields={"answer": "A"}),
        ]
        vote = majority_vote_on("answer")
        winner = vote(preds)
        assert winner.fields["answer"] == "A"

    def test_normalizes_whitespace_and_case(self):
        from harness_rlm.modules import Prediction

        preds = [
            Prediction(fields={"answer": "Paris"}),
            Prediction(fields={"answer": " paris "}),
            Prediction(fields={"answer": "Lyon"}),
        ]
        vote = majority_vote_on("answer")
        winner = vote(preds)
        assert winner.fields["answer"].lower().strip() == "paris"


class TestBestOfN:
    def test_picks_majority_answer(self):
        # 3 samples: A, B, A — A wins.
        lm = _StubLM(canned=["answer: A", "answer: B", "answer: A"])
        child = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        m = BestOfN(child, n=3, max_parallel=1)
        pred = m(q="?")
        assert pred.answer == "A"
        # Trace shows N samples and the vote distribution.
        labels = [e.get("label") for e in pred.trace.events]
        assert "best_of_n" in labels

    def test_n_one_is_passthrough(self):
        lm = _StubLM(canned=["answer: X"])
        child = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        m = BestOfN(child, n=1)
        pred = m(q="?")
        assert pred.answer == "X"
        assert lm.calls == 1

    def test_invalid_n_raises(self):
        lm = _StubLM(canned=["answer: X"])
        child = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="n must be"):
            BestOfN(child, n=0)

    def test_custom_vote_fn(self):
        from harness_rlm.modules import Prediction

        lm = _StubLM(canned=["answer: A", "answer: B", "answer: C"])
        child = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]

        # Custom vote: alphabetically last.
        def vote(preds: list[Prediction]) -> Prediction:
            return max(preds, key=lambda p: p.fields["answer"])

        m = BestOfN(child, n=3, vote_fn=vote, max_parallel=1)
        pred = m(q="?")
        assert pred.answer == "C"


class TestSelfConsistency:
    def test_majority_on_answer_ignores_reasoning(self):
        # Reasoning differs each time but answer converges on "42".
        lm = _StubLM(
            canned=[
                "reasoning: path one\nanswer: 42",
                "reasoning: path two\nanswer: 99",
                "reasoning: path three\nanswer: 42",
                "reasoning: path four\nanswer: 42",
                "reasoning: path five\nanswer: 99",
            ]
        )
        child = ChainOfThought("q -> answer", lm=lm)  # type: ignore[arg-type]
        m = SelfConsistency(child, n=5, max_parallel=1)
        pred = m(q="?")
        assert pred.answer == "42"

    def test_invalid_answer_field_raises(self):
        lm = _StubLM(canned=["answer: x"])
        child = Predict("q -> answer", lm=lm)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="not in child outputs"):
            SelfConsistency(child, answer_field="nope")
