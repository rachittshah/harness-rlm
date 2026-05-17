"""GEPA — Genetic-Pareto reflective prompt optimizer.

Implements Agrawal/Khattab et al. (arXiv:2507.19457) at the module level:
mutate a `Module`'s signature instruction by reflecting on its failure traces,
keep a Pareto frontier of candidates over the trainset, and return the best
candidate against an optional validation set.

The optimizer is intentionally minimal:
  - One predictor at a time (the `student` Module).
  - The metric returns text feedback alongside a scalar score — that text is
    the only thing the reflection_lm sees besides the prior instruction.
  - Selection is Pareto-frontier sampling so we don't collapse to a single
    local optimum.

API mirrors `dspy.GEPA.compile(student, trainset, valset)`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from harness_rlm.llm import LM, get_lm
from harness_rlm.modules import Module, Prediction


@dataclass
class ScoreWithFeedback:
    """Result of the metric function.

    score:    numeric score in [0, 1]. Higher is better.
    feedback: text feedback the reflection_lm uses to propose improvements.
              Be specific — "answer was off-topic", "answer hallucinated X",
              "answer was correct but verbose" — beats generic praise.
    """

    score: float
    feedback: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0,1] (got {self.score})")


Metric = Callable[[dict, Prediction], ScoreWithFeedback]


@dataclass
class Candidate:
    """One candidate instruction + its per-example scores."""

    instruction: str
    scores: list[float] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)
    val_mean: float | None = None

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def dominates(self, other: Candidate) -> bool:
        """Pareto dominance over the trainset scores.

        Self dominates other iff self >= other on every example AND > on at
        least one. Lengths must match for this to make sense.
        """
        if len(self.scores) != len(other.scores):
            return False
        better_anywhere = False
        for s, o in zip(self.scores, other.scores):
            if s < o:
                return False
            if s > o:
                better_anywhere = True
        return better_anywhere


@dataclass
class GEPAResult:
    """Output of `GEPA.compile`. The best candidate + the full search history."""

    best: Candidate
    pareto: list[Candidate]
    history: list[Candidate]
    rollouts: int


class GEPA:
    """Reflective Pareto-frontier prompt optimizer.

    Args:
        metric:          fn(example, pred) -> ScoreWithFeedback.
        max_iterations:  Hard cap on mutation rounds (each round = 1 minibatch
                         eval + 1 reflection call + 1 full-train eval of the
                         proposed candidate).
        reflection_lm:   LM that proposes new instructions from feedback.
                         Defaults to the global LM. Use a strong model here.
        minibatch_size:  Number of trainset examples evaluated per mutation.
        rng_seed:        Set for reproducible Pareto sampling.

    The student's signature is mutated in place at the end — call
    `.compile(student, ...)` and the student now has the best instruction.
    """

    def __init__(
        self,
        metric: Metric,
        *,
        max_iterations: int = 8,
        reflection_lm: LM | None = None,
        minibatch_size: int = 3,
        rng_seed: int | None = None,
    ) -> None:
        self.metric = metric
        self.max_iterations = max_iterations
        self.reflection_lm = reflection_lm
        self.minibatch_size = minibatch_size
        self._rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
        self.rollouts = 0

    # ---- public ------------------------------------------------------------
    def compile(
        self,
        student: Module,
        trainset: Sequence[dict],
        valset: Sequence[dict] | None = None,
    ) -> GEPAResult:
        """Optimize `student.signature.instruction` against the trainset.

        The student is mutated in place to carry the best-scoring instruction.
        """
        if not trainset:
            raise ValueError("trainset must contain at least one example.")

        original_instruction = student.get_instruction()
        seed = self._evaluate_candidate(student, original_instruction, trainset)
        pareto: list[Candidate] = [seed]
        history: list[Candidate] = [seed]

        for _ in range(self.max_iterations):
            parent = self._sample_parent(pareto)
            child_instr = self._propose_mutation(
                parent_instruction=parent.instruction,
                trainset=trainset,
                parent_feedback=parent.feedback,
                student=student,
            )
            if not child_instr or child_instr == parent.instruction:
                # Reflection returned nothing usable — skip.
                continue
            child = self._evaluate_candidate(student, child_instr, trainset)
            history.append(child)
            pareto = self._update_pareto(pareto, child)

        # Pick the best by val_mean if a valset exists, else by trainset mean.
        if valset:
            for c in pareto:
                c.val_mean = self._evaluate_mean(student, c.instruction, valset)
            best = max(pareto, key=lambda c: (c.val_mean or 0.0, c.mean))
        else:
            best = max(pareto, key=lambda c: c.mean)

        # Mutate the student to carry the winning instruction.
        student.set_instruction(best.instruction)

        return GEPAResult(
            best=best,
            pareto=pareto,
            history=history,
            rollouts=self.rollouts,
        )

    # ---- evaluation --------------------------------------------------------
    def _evaluate_candidate(
        self,
        student: Module,
        instruction: str,
        examples: Sequence[dict],
    ) -> Candidate:
        cand = Candidate(instruction=instruction)
        student.set_instruction(instruction)
        for ex in examples:
            self.rollouts += 1
            inputs = self._split_inputs(student, ex)
            try:
                pred = student(**inputs)
                result = self.metric(ex, pred)
            except Exception as e:  # noqa: BLE001 — surface any failure as 0 + feedback
                result = ScoreWithFeedback(score=0.0, feedback=f"error: {e}")
            cand.scores.append(result.score)
            cand.feedback.append(result.feedback)
        return cand

    def _evaluate_mean(
        self,
        student: Module,
        instruction: str,
        examples: Sequence[dict],
    ) -> float:
        cand = self._evaluate_candidate(student, instruction, examples)
        return cand.mean

    def _split_inputs(self, student: Module, example: dict) -> dict:
        """Pull only the declared inputs from the example."""
        wanted = set(student.signature.input_names())
        return {k: v for k, v in example.items() if k in wanted}

    # ---- mutation ----------------------------------------------------------
    def _propose_mutation(
        self,
        parent_instruction: str,
        trainset: Sequence[dict],
        parent_feedback: list[str],
        student: Module,
    ) -> str:
        """Ask the reflection_lm for an improved instruction.

        The prompt includes the prior instruction, a minibatch of examples with
        their feedback, and asks for a single revised instruction.
        """
        lm = self.reflection_lm or get_lm()

        # Subsample feedback so the prompt doesn't balloon.
        n = min(self.minibatch_size, len(parent_feedback))
        idxs = self._rng.sample(range(len(parent_feedback)), n)
        feedback_block = "\n".join(
            f"[ex{i}] score={parent_feedback[i] or '(empty)'}" for i in idxs
        )

        prompt = (
            "You are tuning a single instruction string used by an LLM module.\n"
            "Below is the CURRENT instruction and feedback from running it on\n"
            "several examples. Propose a SINGLE revised instruction that\n"
            "addresses the failure modes. Keep it concise (<= 4 sentences).\n"
            "Do NOT include any preamble — only the new instruction text.\n"
            "\n---\nCURRENT INSTRUCTION:\n"
            f"{parent_instruction or '(empty)'}\n"
            "\n---\nFEEDBACK FROM RECENT RUNS:\n"
            f"{feedback_block or '(no feedback)'}\n"
            "\n---\nREVISED INSTRUCTION:"
        )
        result = lm(prompt, max_tokens=512)
        return result.text.strip()

    # ---- Pareto bookkeeping -----------------------------------------------
    def _update_pareto(
        self,
        frontier: list[Candidate],
        new_candidate: Candidate,
    ) -> list[Candidate]:
        """Insert `new_candidate` into the frontier, dropping anything it dominates."""
        # First check if frontier dominates the new candidate — if so, ignore.
        for c in frontier:
            if c.dominates(new_candidate):
                return frontier
        # Otherwise add it and drop dominated members.
        survivors = [c for c in frontier if not new_candidate.dominates(c)]
        survivors.append(new_candidate)
        return survivors

    def _sample_parent(self, frontier: list[Candidate]) -> Candidate:
        """Sample weighted by mean-score (softmax with temperature 0.5)."""
        if len(frontier) == 1:
            return frontier[0]
        scores = [c.mean for c in frontier]
        # Softmax with T=0.5 (sharper than uniform but not greedy).
        max_s = max(scores)
        weights = [math.exp((s - max_s) / 0.5) for s in scores]
        total = sum(weights) or 1.0
        weights = [w / total for w in weights]
        # weighted choice
        r = self._rng.random()
        acc = 0.0
        for cand, w in zip(frontier, weights):
            acc += w
            if r <= acc:
                return cand
        return frontier[-1]


__all__ = ["GEPA", "GEPAResult", "Candidate", "ScoreWithFeedback", "Metric"]
