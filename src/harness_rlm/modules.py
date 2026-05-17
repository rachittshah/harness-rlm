"""Composable modules — the unit of work the harness runs.

Modules are callable, signature-typed, and stateless wrt their inputs. They are
also serialisable: their `signature` (with `instruction`) is the only knob an
optimizer mutates, so saving a tuned module is just saving its signature.

Three modules ship here:
    Predict          — one LM call, signature in/out
    ChainOfThought   — adds a `reasoning` output field that the LM fills first
    Retry            — re-invokes a child module N times if parse fails

`RLM` lives in `rlm.py` to keep the recursive logic in its own file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from harness_rlm.llm import LM, LMResult, get_lm
from harness_rlm.signatures import Signature, SignatureParseError


@dataclass
class Prediction:
    """Output of a Module call. Behaves like a dict but exposes attributes.

    Examples:
        pred = predict(question="...")
        pred.answer            # field access
        pred["answer"]         # dict access
        pred.trace.calls       # how many LM calls this prediction cost
    """

    fields: dict[str, Any] = field(default_factory=dict)
    trace: Trace | None = None

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails.
        if name in self.fields:
            return self.fields[name]
        raise AttributeError(
            f"Prediction has no field {name!r}. Available: {sorted(self.fields)}"
        )

    def __getitem__(self, key: str) -> Any:
        return self.fields[key]

    def __contains__(self, key: object) -> bool:
        return key in self.fields

    def __repr__(self) -> str:
        return f"Prediction(fields={self.fields!r}, trace={self.trace!r})"

    def to_json(self) -> str:
        payload = {"fields": self.fields}
        if self.trace is not None:
            payload["trace"] = self.trace.to_dict()
        return json.dumps(payload, default=str)


@dataclass
class Trace:
    """Per-call telemetry. Attached to the `Prediction` a module returns."""

    module: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    # Free-form events — sub-modules may append (model, prompt_preview, latency).
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, lm_result: LMResult, *, label: str = "") -> None:
        self.calls += 1
        self.input_tokens += lm_result.input_tokens
        self.output_tokens += lm_result.output_tokens
        self.cost_usd += lm_result.cost_usd
        self.latency_s += lm_result.latency_s
        self.events.append(
            {
                "label": label,
                "model": lm_result.model,
                "input_tokens": lm_result.input_tokens,
                "output_tokens": lm_result.output_tokens,
                "cost_usd": round(lm_result.cost_usd, 6),
                "latency_s": round(lm_result.latency_s, 3),
            }
        )

    def absorb(self, other: Trace) -> None:
        """Roll a sub-module's trace into this one."""
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        self.latency_s += other.latency_s
        self.events.extend(other.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_s": round(self.latency_s, 3),
            "events": self.events,
        }


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Module:
    """Callable unit. Subclasses override `forward(**inputs) -> Prediction`.

    A module owns:
      - signature: the typed contract (mutated by optimizers)
      - lm:        the LM client to use (None means "use global default")
    """

    def __init__(
        self,
        signature: Signature | str,
        *,
        lm: LM | None = None,
        name: str | None = None,
    ) -> None:
        self.signature = (
            signature if isinstance(signature, Signature) else Signature(signature)
        )
        self.lm = lm
        self.name = name or type(self).__name__

    # Resolve the LM at call-time so configure() takes effect retroactively.
    def _lm(self) -> LM:
        return self.lm or get_lm()

    def forward(self, **inputs: Any) -> Prediction:
        raise NotImplementedError

    def __call__(self, **inputs: Any) -> Prediction:
        return self.forward(**inputs)

    # ---- optimizer hooks --------------------------------------------------
    def set_instruction(self, instruction: str) -> None:
        """Mutate the signature's instruction in place (used by GEPA)."""
        self.signature = self.signature.with_instruction(instruction)

    def get_instruction(self) -> str:
        return self.signature.instruction


# ---------------------------------------------------------------------------
# Predict — the simplest useful module.
# ---------------------------------------------------------------------------
class Predict(Module):
    """One LM call. Renders signature → prompt, calls LM, parses response.

    On parse failure, raises SignatureParseError so callers can wrap with
    `Retry` (or fall back to ad-hoc handling).
    """

    def forward(self, **inputs: Any) -> Prediction:
        prompt = self.signature.render_prompt(inputs)
        result = self._lm()(prompt)
        trace = Trace(module=self.name)
        trace.record(result, label="predict")
        try:
            parsed = self.signature.parse_response(result.text)
        except SignatureParseError as e:
            # Tag the error with the trace so the caller can read tokens spent.
            e.add_note(f"trace: {trace.to_dict()}")
            raise
        return Prediction(fields=parsed, trace=trace)


# ---------------------------------------------------------------------------
# ChainOfThought — Predict with a reasoning slot.
# ---------------------------------------------------------------------------
class ChainOfThought(Module):
    """Predict with an extra `reasoning` output field rendered before the answer.

    The LM is steered to fill `reasoning` first, then the declared output(s).
    This is the dspy.ChainOfThought pattern in 30 lines.
    """

    REASONING_FIELD = "reasoning"

    def __init__(
        self,
        signature: Signature | str,
        *,
        lm: LM | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(signature, lm=lm, name=name)
        # Inject the reasoning field at the head of outputs (idempotent).
        if self.REASONING_FIELD not in self.signature.output_names():
            from harness_rlm.signatures import Field_

            self.signature.outputs.insert(
                0,
                Field_(
                    name=self.REASONING_FIELD,
                    desc="step-by-step reasoning before the answer",
                ),
            )

    def forward(self, **inputs: Any) -> Prediction:
        prompt = self.signature.render_prompt(inputs)
        result = self._lm()(prompt)
        trace = Trace(module=self.name)
        trace.record(result, label="chain_of_thought")
        parsed = self.signature.parse_response(result.text)
        return Prediction(fields=parsed, trace=trace)


# ---------------------------------------------------------------------------
# Retry — re-run a child module up to N times on parse error.
# ---------------------------------------------------------------------------
class Retry(Module):
    """Wrap a child module to retry on SignatureParseError.

    The retry prompt is the original prompt plus a corrective system note
    ("your previous response could not be parsed; follow the format exactly").
    """

    def __init__(self, child: Module, *, max_attempts: int = 3, name: str | None = None):
        super().__init__(child.signature, lm=child.lm, name=name or f"Retry({child.name})")
        self.child = child
        self.max_attempts = max_attempts

    def forward(self, **inputs: Any) -> Prediction:
        last_err: SignatureParseError | None = None
        combined = Trace(module=self.name)
        for attempt in range(self.max_attempts):
            try:
                pred = self.child(**inputs)
                if pred.trace is not None:
                    combined.absorb(pred.trace)
                pred.trace = combined
                return pred
            except SignatureParseError as e:
                last_err = e
                combined.events.append(
                    {"label": "parse_error", "attempt": attempt + 1, "msg": str(e)[:200]}
                )
                continue
        # All attempts failed — re-raise the last parse error with trace.
        assert last_err is not None
        last_err.add_note(f"retry_trace: {combined.to_dict()}")
        raise last_err


__all__ = [
    "Module",
    "Predict",
    "ChainOfThought",
    "Retry",
    "Prediction",
    "Trace",
]
