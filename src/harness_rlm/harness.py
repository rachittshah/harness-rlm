"""Pi-style top-level API.

One function — `run` — that does the right thing for the common case. Hides
LM construction, RLM vs flat dispatch, and trace assembly behind sensible
defaults. The pi-mono mantra: small surface, do the obvious thing.

Examples:
    # Short question, no long context — flat Predict under the hood.
    answer = run("What is 2+2?")

    # Long-context question — RLM under the hood.
    answer = run("Find the bug.", context=open('huge.log').read())

    # Optimize against a trainset, then call.
    answer = run("...", context="...", optimize_with=(trainset, metric))

    # Use specific models (overrides defaults).
    answer = run("...", root_model="claude-opus-4-7", sub_model="claude-haiku-4-5")
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from harness_rlm.gepa import GEPA, Metric
from harness_rlm.llm import LM
from harness_rlm.models import DEFAULT_MODEL
from harness_rlm.modules import Predict, Prediction
from harness_rlm.rlm import RLM, RLMConfig
from harness_rlm.signatures import Signature

DEFAULT_ROOT_MODEL = "claude-opus-4-7"
DEFAULT_SUB_MODEL = DEFAULT_MODEL  # Haiku 4.5 — cheap default


@dataclass
class RunResult:
    """Return value from `run` — answer plus full trace and cost telemetry."""

    answer: str
    fields: dict[str, Any]
    trace: dict[str, Any] | None
    cost_usd: float
    calls: int

    def __str__(self) -> str:
        return self.answer


def run(
    question: str,
    context: str | None = None,
    *,
    root_model: str = DEFAULT_ROOT_MODEL,
    sub_model: str = DEFAULT_SUB_MODEL,
    flat_threshold: int = 80_000,
    optimize_with: tuple[Sequence[dict], Metric] | None = None,
    max_iterations: int = 20,
    max_llm_calls: int = 50,
    instruction: str | None = None,
) -> RunResult:
    """Run a one-shot question, optionally over a long context.

    Args:
        question:       The user's question.
        context:        Optional long context (document/log/transcript).
        root_model:     Model for the root LM (synthesis / flat answer).
        sub_model:      Model for sub-LM chunk queries (only used if context
                        is long enough to trigger RLM).
        flat_threshold: chars under which we skip RLM and call flat.
        optimize_with:  (trainset, metric) — if passed, run GEPA on the
                        module before answering. Trainset items must include
                        the same keys as the runtime call.
        max_iterations: BudgetGuard cap for RLM iterations.
        max_llm_calls:  BudgetGuard cap for LLM calls.
        instruction:    Override the default instruction for the module.

    Returns:
        RunResult with the answer string + full trace.
    """
    root = LM(model=root_model)
    sub = LM(model=sub_model)

    if context is None or context.strip() == "":
        # Pure question — single Predict call.
        sig = Signature("question -> answer")
        if instruction:
            sig = sig.with_instruction(instruction)
        module = Predict(sig, lm=root)
        if optimize_with:
            module = _optimize(module, optimize_with, reflection_lm=root)
        pred = module(question=question)
        return _to_result(pred, root, sub)

    # Long-context path — RLM.
    sig = Signature("question, document -> answer")
    if instruction:
        sig = sig.with_instruction(instruction)
    cfg = RLMConfig(
        flat_char_threshold=flat_threshold,
        max_iterations=max_iterations,
        max_llm_calls=max_llm_calls,
    )
    module = RLM(
        sig,
        long_context_field="document",
        config=cfg,
        root_lm=root,
        sub_lm=sub,
    )
    if optimize_with:
        module = _optimize(module, optimize_with, reflection_lm=root)
    pred = module(question=question, document=context)
    return _to_result(pred, root, sub)


def _optimize(
    module,
    optimize_with: tuple[Sequence[dict], Metric],
    *,
    reflection_lm: LM,
):
    trainset, metric = optimize_with
    gepa = GEPA(metric=metric, reflection_lm=reflection_lm)
    gepa.compile(module, trainset)
    return module


def _to_result(pred: Prediction, root: LM, sub: LM) -> RunResult:
    cost = root.total_cost_usd + sub.total_cost_usd
    calls = root.total_calls + sub.total_calls
    answer_field = next(
        (k for k in pred.fields if k.lower() != "reasoning"),
        next(iter(pred.fields), ""),
    )
    return RunResult(
        answer=pred.fields.get(answer_field, ""),
        fields=pred.fields,
        trace=pred.trace.to_dict() if pred.trace else None,
        cost_usd=round(cost, 6),
        calls=calls,
    )


# ---------------------------------------------------------------------------
# CLI — `python -m harness_rlm "What is X?" --context-file FOO.md`
# ---------------------------------------------------------------------------
def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="harness-rlm",
        description="Recursive Language Model harness — answer questions over long contexts.",
    )
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument(
        "--context-file",
        type=Path,
        help="Path to a file containing long context.",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Inline context string.",
    )
    parser.add_argument(
        "--root-model",
        default=DEFAULT_ROOT_MODEL,
        help=f"Root LM model (default: {DEFAULT_ROOT_MODEL}).",
    )
    parser.add_argument(
        "--sub-model",
        default=DEFAULT_SUB_MODEL,
        help=f"Sub-LM model (default: {DEFAULT_SUB_MODEL}).",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=50,
        help="Hard cap on LLM calls.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full result JSON instead of just the answer.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Override the default instruction the module uses.",
    )
    args = parser.parse_args()

    ctx: str | None = None
    if args.context_file:
        ctx = args.context_file.read_text(encoding="utf-8")
    elif args.context:
        ctx = args.context

    result = run(
        args.question,
        context=ctx,
        root_model=args.root_model,
        sub_model=args.sub_model,
        max_llm_calls=args.max_calls,
        instruction=args.instruction,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "answer": result.answer,
                    "fields": result.fields,
                    "trace": result.trace,
                    "cost_usd": result.cost_usd,
                    "calls": result.calls,
                },
                indent=2,
                default=str,
            )
        )
    else:
        print(result.answer)
    return 0


def main() -> None:
    """Entrypoint for `python -m harness_rlm`."""
    sys.exit(_cli())


__all__ = ["run", "RunResult", "main"]
