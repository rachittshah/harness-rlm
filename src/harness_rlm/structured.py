"""Pydantic-typed structured outputs for harness modules.

`TypedPredict` accepts a Pydantic `BaseModel` class as the output schema,
renders the schema into the prompt, parses the LM's JSON response, and
validates against the model. The result is a type-safe Pydantic instance,
not a `dict`.

Why a separate module: the base `Signature` is intentionally untyped (one
parser path, easy to extend). Pydantic-validated outputs need:
  - schema rendering (instead of `name: <your name>` lines)
  - JSON extraction from possibly-decorated LM responses
  - validation feedback in case of failure

This is the path most production agent frameworks (PydanticAI, OpenAI
Structured Outputs, BAML) optimize for.

Usage:
    from pydantic import BaseModel
    from harness_rlm import TypedPredict

    class Answer(BaseModel):
        result: int
        explanation: str

    qa = TypedPredict("question -> answer", output_model=Answer)
    pred = qa(question="What is 2+2?")
    # pred.value is now an `Answer` instance.
    print(pred.value.result, pred.value.explanation)
"""

from __future__ import annotations

import json
import re
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from harness_rlm.llm import LM
from harness_rlm.modules import Module, Prediction, Trace
from harness_rlm.signatures import Signature, SignatureParseError

T = TypeVar("T", bound=BaseModel)


class TypedPrediction(Prediction, Generic[T]):
    """Prediction with a validated Pydantic instance in `value`."""

    def __init__(
        self,
        value: T,
        fields: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> None:
        super().__init__(fields=fields or {}, trace=trace)
        self.value = value

    def __repr__(self) -> str:
        return f"TypedPrediction(value={self.value!r}, trace={self.trace!r})"


JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```|(\{.*\})", re.DOTALL)


def _extract_json(text: str) -> str:
    """Pull the first JSON object from text. Handles ```json``` fences."""
    m = JSON_RE.search(text)
    if not m:
        # Fallback: last-ditch attempt with bracket counting.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise SignatureParseError(
                f"No JSON object found in response. First 500 chars: {text[:500]!r}"
            )
        return text[start : end + 1]
    return (m.group(1) or m.group(2) or "").strip()


class TypedPredict(Module, Generic[T]):
    """Predict whose output is validated against a Pydantic model.

    Args:
        signature:      Same as `Predict` — input names matter; output names
                        are derived from the model fields.
        output_model:   Pydantic BaseModel class. Required.
        lm:             LM client (or global default).
        max_retries:    Retries on JSON parse / validation errors. The retry
                        prompt includes the validation error so the LM can
                        self-correct.
    """

    def __init__(
        self,
        signature: Signature | str,
        *,
        output_model: type[T],
        lm: LM | None = None,
        max_retries: int = 2,
        name: str | None = None,
    ) -> None:
        if not issubclass(output_model, BaseModel):
            raise TypeError(
                f"output_model must be a Pydantic BaseModel subclass (got {output_model!r})"
            )
        super().__init__(signature, lm=lm, name=name)
        self.output_model = output_model
        self.max_retries = max_retries

    def forward(self, **inputs: Any) -> TypedPrediction[T]:
        trace = Trace(module=self.name)
        prompt = self._render_prompt(inputs)
        lm = self._lm()
        last_error: str | None = None

        for attempt in range(self.max_retries + 1):
            full_prompt = prompt
            if attempt > 0 and last_error:
                full_prompt = (
                    prompt
                    + "\n\nThe previous response failed validation:\n"
                    + last_error
                    + "\n\nReturn ONLY valid JSON matching the schema above."
                )
            result = lm(full_prompt)
            trace.record(result, label=f"typed_predict.attempt{attempt + 1}")

            try:
                raw_json = _extract_json(result.text)
                payload = json.loads(raw_json)
            except (SignatureParseError, json.JSONDecodeError) as e:
                last_error = f"json parse: {e}"
                continue

            try:
                value = self.output_model.model_validate(payload)
            except ValidationError as e:
                last_error = f"validation: {e}"
                continue

            return TypedPrediction(
                value=value,
                fields=value.model_dump(),
                trace=trace,
            )

        # All retries exhausted.
        err = SignatureParseError(
            f"TypedPredict failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )
        err.add_note(f"trace: {trace.to_dict()}")
        raise err

    def _render_prompt(self, inputs: dict[str, Any]) -> str:
        """Render a JSON-schema-aware prompt."""
        # Schema string.
        schema = self.output_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        lines: list[str] = []
        if self.signature.instruction:
            lines.append(self.signature.instruction)
        if self.signature.inputs:
            lines.append("---")
            for f in self.signature.inputs:
                desc = f" ({f.desc})" if f.desc else ""
                lines.append(f"{f.name}{desc}: {inputs[f.name]}")
        lines.append("---")
        lines.append("Respond with a SINGLE JSON object matching this schema:")
        lines.append(f"```json\n{schema_str}\n```")
        lines.append("Return only the JSON object, no commentary.")
        return "\n".join(lines)


__all__ = ["TypedPredict", "TypedPrediction"]
