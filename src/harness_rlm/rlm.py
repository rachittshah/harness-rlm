"""RLM — Recursive Language Model module.

Implements Zhang/Kraska/Khattab (arXiv:2512.24601) as a `Module`. The long-
context field is decomposed into chunks; a cheap sub-LM is dispatched on each
chunk in parallel; the root LM synthesizes the partial answers into the final
answer. The whole flow honours `BudgetGuard` and emits a single `Trace`.

This is the programmatic flavour. The skill-driven flavour (root agent inside
Claude Code writes Python in a REPL) lives in `skill/SKILL.md`. Both share the
same budget envelope and audit-log path.

Three decomposition strategies are first-class:
    - "map_reduce"        — split context → ask each chunk → merge (default)
    - "filter_then_query" — grep candidate regions → ask only hits → merge
    - "auto"              — root LM picks the strategy from the signature

Selection is GEPA-friendly: the strategy lives in the `instruction` field of
the signature, so prompt evolution can rewrite it without touching the loop.
"""

from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass
from typing import Any, Literal

from harness_rlm.core import BudgetExceededError, BudgetGuard, chunk_context
from harness_rlm.llm import LM, get_lm
from harness_rlm.modules import Module, Prediction, Trace
from harness_rlm.signatures import Signature

Strategy = Literal["map_reduce", "filter_then_query", "auto"]

# Threshold below which we skip recursion entirely — flat call is faster
# *and* equally accurate when the context fits in one prompt.
DEFAULT_FLAT_CHAR_THRESHOLD = 80_000

# Used by the sub-LM to indicate a chunk does not contain the answer.
NOT_FOUND = "NOT_FOUND"

DEFAULT_SUB_INSTRUCTION = (
    "Answer using ONLY the chunk below. Return 1-3 sentences. "
    f"If the chunk does not contain the answer, output exactly: {NOT_FOUND}"
)

DEFAULT_SYNTH_INSTRUCTION = (
    "Synthesize a single coherent answer from the partial answers below. "
    "Ignore any that say NOT_FOUND. If every partial says NOT_FOUND, say so."
)


@dataclass
class RLMConfig:
    """All RLM knobs in one place. Optimizers may mutate fields here."""

    strategy: Strategy = "map_reduce"
    chunk_size: int = 20_000
    overlap: int = 200
    max_parallel: int = 8
    # Budget envelope — also enforced by BudgetGuard wrapped around the loop.
    max_iterations: int = 20
    max_llm_calls: int = 50
    # Threshold under which we fall back to a flat call.
    flat_char_threshold: int = DEFAULT_FLAT_CHAR_THRESHOLD
    # Per-chunk filter (only used by "filter_then_query") — a regex applied
    # line-wise. Lines that don't match are dropped before chunking.
    filter_pattern: str | None = None
    # Custom instructions (otherwise the defaults above are used).
    sub_instruction: str = DEFAULT_SUB_INSTRUCTION
    synth_instruction: str = DEFAULT_SYNTH_INSTRUCTION
    # Max output tokens for sub-LM and synth-LM calls respectively.
    sub_max_tokens: int = 512
    synth_max_tokens: int = 1024


class RLM(Module):
    """Recursive Language Model module.

    Args:
        signature:  Must declare a *long-context* input field (passed via
                    `long_context_field`) and at least one output (typically
                    "answer"). E.g. `Signature("question, document -> answer")`.
        long_context_field: Name of the input field that holds the long context.
        config:     `RLMConfig` instance — chunking, parallelism, budgets.
        root_lm:    LM used for the final synthesis (defaults to global).
        sub_lm:     LM used for chunk queries — pass Haiku here for cheap
                    parallel calls. Defaults to global (root_lm) which is
                    inefficient; pass an explicit cheap LM for the cost win.

    Returns:
        Prediction with the declared output fields plus a Trace.
    """

    def __init__(
        self,
        signature: Signature | str,
        *,
        long_context_field: str = "document",
        config: RLMConfig | None = None,
        root_lm: LM | None = None,
        sub_lm: LM | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(signature, lm=root_lm, name=name)
        if long_context_field not in self.signature.input_names():
            raise ValueError(
                f"Signature must declare the long-context field "
                f"{long_context_field!r} as an input. "
                f"Got inputs={self.signature.input_names()}."
            )
        self.long_context_field = long_context_field
        self.config = config or RLMConfig()
        self.sub_lm = sub_lm
        # The single declared output field — used for short-circuit + synth.
        outs = self.signature.output_names()
        # Pick the first non-reasoning field as "the" answer slot.
        self._answer_field = next(
            (n for n in outs if n.lower() != "reasoning"), outs[0]
        )

    # ---- runtime ----------------------------------------------------------
    def _root(self) -> LM:
        return self.lm or get_lm()

    def _sub(self) -> LM:
        return self.sub_lm or self._root()

    def forward(self, **inputs: Any) -> Prediction:
        trace = Trace(module=self.name)
        guard = BudgetGuard(
            budgets={
                "max_iterations": self.config.max_iterations,
                "max_llm_calls": self.config.max_llm_calls,
                "max_output_chars": 10_000,
            }
        )

        ctx: str = inputs[self.long_context_field]
        other_inputs = {k: v for k, v in inputs.items() if k != self.long_context_field}

        # Fast path: context small enough to fit flat — skip decomposition.
        if len(ctx) <= self.config.flat_char_threshold:
            return self._flat_call(inputs, trace, guard)

        # Slow path: decompose, dispatch, synth.
        chunks = self._make_chunks(ctx)
        partials = self._fan_out(chunks, other_inputs, trace, guard)
        answer = self._synthesize(partials, other_inputs, trace, guard)

        return Prediction(fields={self._answer_field: answer}, trace=trace)

    # ---- chunk + filter ---------------------------------------------------
    def _make_chunks(self, ctx: str) -> list[str]:
        if self.config.strategy == "filter_then_query" and self.config.filter_pattern:
            pat = re.compile(self.config.filter_pattern, re.IGNORECASE)
            lines = ctx.splitlines()
            kept = [line for line in lines if pat.search(line)]
            # If filter wipes the context, fall back to the original.
            if not kept:
                ctx_filtered = ctx
            else:
                ctx_filtered = "\n".join(kept)
            return chunk_context(
                ctx_filtered,
                chunk_size=self.config.chunk_size,
                overlap=self.config.overlap,
            )
        return chunk_context(
            ctx,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
        )

    # ---- dispatch ---------------------------------------------------------
    def _fan_out(
        self,
        chunks: list[str],
        other_inputs: dict[str, Any],
        trace: Trace,
        guard: BudgetGuard,
    ) -> list[str]:
        """Dispatch sub-LM calls over chunks in parallel; collect partials."""
        # Pre-check budget: refuse if we don't have headroom for all chunks +
        # one synth call.
        needed = len(chunks) + 1
        if guard.llm_calls + needed > guard.budgets["max_llm_calls"]:
            # Truncate chunks to fit budget — keep the head, drop the tail.
            available = guard.budgets["max_llm_calls"] - guard.llm_calls - 1
            chunks = chunks[: max(0, available)]
            trace.events.append(
                {
                    "label": "chunk_budget_truncated",
                    "kept": len(chunks),
                    "budget_remaining": available,
                }
            )

        partials: list[str] = []
        sub_lm = self._sub()

        # Build per-chunk prompt by reusing the signature with sub_instruction.
        def call_one(chunk: str) -> str:
            prompt = self._render_sub_prompt(chunk, other_inputs)
            result = sub_lm(prompt, max_tokens=self.config.sub_max_tokens)
            return result.text, result  # noqa: E501

        max_workers = max(1, min(self.config.max_parallel, len(chunks)))
        if max_workers == 1 or len(chunks) <= 1:
            # Sequential — simpler, easier to debug.
            for chunk in chunks:
                text, result = call_one(chunk)
                trace.record(result, label="sub_call")
                guard.increment_call()
                partials.append(text)
        else:
            # Parallel — preserves chunk order in `partials`.
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(call_one, c) for c in chunks]
                for fut in futures:
                    text, result = fut.result()
                    trace.record(result, label="sub_call_parallel")
                    guard.increment_call()
                    partials.append(text)
        return partials

    def _render_sub_prompt(self, chunk: str, other_inputs: dict[str, Any]) -> str:
        """Render the prompt sent to the sub-LM for one chunk."""
        # We do NOT use signature.render_prompt here — sub calls have a
        # different shape (one chunk + the non-context inputs).
        lines = [self.config.sub_instruction, "---"]
        for k, v in other_inputs.items():
            lines.append(f"{k}: {v}")
        lines.append("---")
        lines.append("CHUNK:")
        lines.append(chunk)
        lines.append("---")
        lines.append("Answer (1-3 sentences, or NOT_FOUND):")
        return "\n".join(lines)

    # ---- synth ------------------------------------------------------------
    def _synthesize(
        self,
        partials: list[str],
        other_inputs: dict[str, Any],
        trace: Trace,
        guard: BudgetGuard,
    ) -> str:
        try:
            guard.check_call()
        except BudgetExceededError:
            # No headroom — return the best partial we have, or a failure note.
            usable = [p for p in partials if NOT_FOUND not in p]
            if usable:
                return usable[0]
            return "BUDGET_EXHAUSTED: no synthesis possible"

        lines = [self.config.synth_instruction, "---"]
        for k, v in other_inputs.items():
            lines.append(f"{k}: {v}")
        lines.append("---")
        lines.append("Partial answers (one per source chunk):")
        for i, p in enumerate(partials):
            lines.append(f"[{i}] {p}")
        lines.append("---")
        lines.append("Final answer:")
        prompt = "\n".join(lines)
        result = self._root()(prompt, max_tokens=self.config.synth_max_tokens)
        trace.record(result, label="synth")
        guard.increment_call()
        return result.text.strip()

    # ---- flat fallback ----------------------------------------------------
    def _flat_call(
        self,
        inputs: dict[str, Any],
        trace: Trace,
        guard: BudgetGuard,
    ) -> Prediction:
        """Single root-LM call when context fits in one prompt."""
        # Reuse the declared signature directly — render + parse.
        prompt = self.signature.render_prompt(inputs)
        result = self._root()(prompt, max_tokens=self.config.synth_max_tokens)
        trace.record(result, label="flat")
        guard.increment_call()
        try:
            parsed = self.signature.parse_response(result.text)
        except Exception:
            # If parse fails on the flat path, fall back to "whole text is the answer".
            parsed = {self._answer_field: result.text.strip()}
        return Prediction(fields=parsed, trace=trace)


__all__ = ["RLM", "RLMConfig", "Strategy", "NOT_FOUND"]
