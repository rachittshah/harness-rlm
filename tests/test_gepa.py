"""Tests for harness_rlm.gepa — reflective Pareto optimizer (no network)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from harness_rlm.gepa import GEPA, Candidate, ScoreWithFeedback
from harness_rlm.llm import LMResult
from harness_rlm.modules import Predict


# ---------------------------------------------------------------------------
# Stub LMs
# ---------------------------------------------------------------------------
@dataclass
class _StubLM:
    canned: list[str] = field(default_factory=list)
    model: str = "stub-model"
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        self.received: list[str] = []
        self.calls = 0

    def __call__(self, prompt, *, system=None, model=None, max_tokens=None):
        self.received.append(prompt)
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
# Candidate dominance
# ---------------------------------------------------------------------------
class TestCandidate:
    def test_dominance(self):
        a = Candidate(instruction="A", scores=[1.0, 0.8, 0.6])
        b = Candidate(instruction="B", scores=[0.9, 0.8, 0.5])
        assert a.dominates(b)
        assert not b.dominates(a)

    def test_no_dominance_when_equal(self):
        a = Candidate(instruction="A", scores=[0.5, 0.5])
        b = Candidate(instruction="B", scores=[0.5, 0.5])
        assert not a.dominates(b)
        assert not b.dominates(a)

    def test_no_dominance_when_trade(self):
        # Each better on a different example → neither dominates.
        a = Candidate(instruction="A", scores=[1.0, 0.0])
        b = Candidate(instruction="B", scores=[0.0, 1.0])
        assert not a.dominates(b)
        assert not b.dominates(a)


# ---------------------------------------------------------------------------
# Full compile loop
# ---------------------------------------------------------------------------
class TestCompile:
    def test_picks_better_instruction(self):
        """Simulated optimizer where:
        - reflection_lm always proposes the literal "GOOD" instruction.
        - student replies "answer: ok" iff instruction is "GOOD", else "bad".
        - metric scores 1.0 for "ok", 0.0 for "bad".
        """
        # Reflection LM proposes "GOOD" every time.
        reflection_lm = _StubLM(canned=["GOOD"])

        # The student LM is a more elaborate stub — its output depends on the
        # *content* of the prompt (specifically, whether "GOOD" is in there).
        class _ConditionalLM:
            calls = 0

            def __call__(self, prompt, *, system=None, model=None, max_tokens=None):
                self.calls += 1
                txt = "answer: ok" if "GOOD" in prompt else "answer: bad"
                return LMResult(
                    text=txt,
                    model="x",
                    input_tokens=10,
                    output_tokens=2,
                    cost_usd=0.0001,
                    latency_s=0.01,
                )

        student_lm = _ConditionalLM()
        student = Predict(
            "question -> answer",
            lm=student_lm,  # type: ignore[arg-type]
        )

        # Metric: 1.0 if answer == "ok", else 0.0.
        def metric(example, pred):
            ok = pred.answer == "ok"
            return ScoreWithFeedback(
                score=1.0 if ok else 0.0,
                feedback="correct" if ok else "answer was 'bad', want 'ok'",
            )

        trainset = [{"question": q} for q in ["q1", "q2"]]
        valset = [{"question": "v1"}]

        gepa = GEPA(
            metric=metric,
            max_iterations=2,
            reflection_lm=reflection_lm,  # type: ignore[arg-type]
            minibatch_size=2,
            rng_seed=0,
        )
        result = gepa.compile(student, trainset, valset)

        # The best candidate must be the "GOOD" one (score 1.0).
        assert result.best.mean == 1.0
        assert "GOOD" in result.best.instruction
        # Student was actually mutated.
        assert "GOOD" in student.get_instruction()
        # Frontier has at least one candidate.
        assert len(result.pareto) >= 1

    def test_empty_trainset_raises(self):
        gepa = GEPA(metric=lambda ex, p: ScoreWithFeedback(score=1.0))
        with pytest.raises(ValueError, match="at least one"):
            gepa.compile(Predict("q -> a"), [])

    def test_score_clamp_validation(self):
        with pytest.raises(ValueError, match="score must be"):
            ScoreWithFeedback(score=1.5)


# ---------------------------------------------------------------------------
# Pareto update
# ---------------------------------------------------------------------------
class TestParetoUpdate:
    def test_dominated_new_candidate_rejected(self):
        gepa = GEPA(metric=lambda ex, p: ScoreWithFeedback(score=1.0))
        frontier = [Candidate(instruction="A", scores=[1.0, 1.0])]
        new = Candidate(instruction="B", scores=[0.5, 0.5])
        out = gepa._update_pareto(frontier, new)
        assert len(out) == 1
        assert out[0].instruction == "A"

    def test_new_candidate_drops_dominated(self):
        gepa = GEPA(metric=lambda ex, p: ScoreWithFeedback(score=1.0))
        frontier = [Candidate(instruction="A", scores=[0.5, 0.5])]
        new = Candidate(instruction="B", scores=[1.0, 1.0])
        out = gepa._update_pareto(frontier, new)
        assert len(out) == 1
        assert out[0].instruction == "B"

    def test_trade_both_kept(self):
        gepa = GEPA(metric=lambda ex, p: ScoreWithFeedback(score=1.0))
        frontier = [Candidate(instruction="A", scores=[1.0, 0.0])]
        new = Candidate(instruction="B", scores=[0.0, 1.0])
        out = gepa._update_pareto(frontier, new)
        assert len(out) == 2
