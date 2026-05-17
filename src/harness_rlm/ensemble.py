"""Ensemble modules — sample multiple completions, aggregate the answer.

Two modules:

  BestOfN(child, n)
      Runs `child` N times in parallel. Aggregator picks a winner.
      Default aggregator: majority vote on the primary output field.
      Configurable: pass a `vote_fn(predictions) -> Prediction`.

  SelfConsistency(child, n, answer_field='answer')
      Wraps a reasoning module (e.g. ChainOfThought). Runs N samples,
      groups by `answer_field` value, returns the prediction whose
      answer has the most support. Wei et al., "Self-Consistency"
      (arXiv:2203.11171) — proven to lift GSM8K/MATH accuracy by 5-15%.

Both modules use ThreadPoolExecutor for parallelism; each child call gets
a fresh signature instance to avoid concurrent mutation. The aggregated
Prediction carries the combined Trace (calls/tokens/cost).

API mirrors DSPy's `dspy.BestOfN` and `dspy.SelfConsistency`.
"""

from __future__ import annotations

import concurrent.futures
from collections import Counter
from typing import Any, Callable

from harness_rlm.modules import Module, Prediction, Trace
from harness_rlm.signatures import Signature

VoteFn = Callable[[list[Prediction]], Prediction]


def majority_vote_on(field_name: str) -> VoteFn:
    """Return a vote_fn that picks the prediction whose `field_name` is most common."""

    def vote(predictions: list[Prediction]) -> Prediction:
        if not predictions:
            raise ValueError("majority_vote_on: empty predictions list")
        counts = Counter(_normalize(p.fields.get(field_name, "")) for p in predictions)
        winner, _ = counts.most_common(1)[0]
        # Return the FIRST prediction matching the winning answer (preserves trace).
        for p in predictions:
            if _normalize(p.fields.get(field_name, "")) == winner:
                return p
        return predictions[0]  # unreachable

    return vote


def _normalize(s: Any) -> str:
    """Loose normalisation for vote keys: strip, lower, collapse whitespace."""
    if not isinstance(s, str):
        return str(s).strip().lower()
    return " ".join(s.strip().lower().split())


# ---------------------------------------------------------------------------
# BestOfN
# ---------------------------------------------------------------------------
class BestOfN(Module):
    """Sample child N times in parallel, pick a winner via vote_fn.

    Args:
        child:         Module to sample.
        n:             Number of samples (default 5).
        vote_fn:       Aggregator. Default: majority_vote_on the first
                       non-reasoning output of the child's signature.
        max_parallel:  ThreadPool size.
        keep_all:      If True, the returned Prediction.trace includes every
                       sample (large traces). Default False — only winner.
    """

    def __init__(
        self,
        child: Module,
        *,
        n: int = 5,
        vote_fn: VoteFn | None = None,
        max_parallel: int = 5,
        keep_all: bool = False,
        name: str | None = None,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1 (got {n})")
        super().__init__(child.signature, lm=child.lm, name=name or f"BestOfN(n={n}, {child.name})")
        self.child = child
        self.n = n
        # Pick a sensible default vote field.
        outs = child.signature.output_names()
        default_field = next((o for o in outs if o.lower() != "reasoning"), outs[0])
        self.vote_fn = vote_fn or majority_vote_on(default_field)
        self.max_parallel = max_parallel
        self.keep_all = keep_all

    def forward(self, **inputs: Any) -> Prediction:
        # Run N samples in parallel.
        workers = max(1, min(self.max_parallel, self.n))
        if workers == 1 or self.n == 1:
            samples = [self.child(**inputs) for _ in range(self.n)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(self.child, **inputs) for _ in range(self.n)]
                samples = [f.result() for f in futures]

        winner = self.vote_fn(samples)

        # Build a combined trace.
        combined = Trace(module=self.name)
        for s in samples:
            if s.trace is not None:
                combined.absorb(s.trace)
        combined.events.append(
            {
                "label": "best_of_n",
                "n": self.n,
                "winner_fields": winner.fields,
                "vote_distribution": Counter(
                    _normalize(s.fields.get(_first_nonreasoning_field(self.signature), ""))
                    for s in samples
                ).most_common(),
            }
        )

        # Preserve per-sample traces if requested.
        if self.keep_all:
            combined.events.append(
                {
                    "label": "samples",
                    "traces": [s.trace.to_dict() if s.trace else None for s in samples],
                }
            )

        return Prediction(fields=dict(winner.fields), trace=combined)


def _first_nonreasoning_field(sig: Signature) -> str:
    outs = sig.output_names()
    return next((o for o in outs if o.lower() != "reasoning"), outs[0])


# ---------------------------------------------------------------------------
# SelfConsistency — alias of BestOfN tuned for ChainOfThought.
# ---------------------------------------------------------------------------
class SelfConsistency(BestOfN):
    """Self-consistency for chain-of-thought reasoning.

    Wei et al. (arXiv:2203.11171). Identical to `BestOfN` with the vote field
    explicitly tied to the answer (not the reasoning), so divergent reasoning
    paths that converge on the same answer get aggregated correctly.
    """

    def __init__(
        self,
        child: Module,
        *,
        n: int = 5,
        answer_field: str = "answer",
        max_parallel: int = 5,
        name: str | None = None,
    ) -> None:
        if answer_field not in child.signature.output_names():
            raise ValueError(
                f"answer_field {answer_field!r} not in child outputs "
                f"({child.signature.output_names()})"
            )
        super().__init__(
            child,
            n=n,
            vote_fn=majority_vote_on(answer_field),
            max_parallel=max_parallel,
            name=name or f"SelfConsistency(n={n}, answer={answer_field})",
        )


__all__ = ["BestOfN", "SelfConsistency", "majority_vote_on", "VoteFn"]
