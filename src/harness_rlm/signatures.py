"""Typed input/output contracts for harness modules.

DSPy-inspired. A `Signature` declares what fields a module consumes and produces,
plus an optional natural-language instruction the LM should follow.

Two construction styles, both first-class:

    # Shorthand string (terse, like dspy):
    sig = Signature("question, context -> answer")

    # Explicit (clearer when you want field-level descriptions):
    sig = Signature(
        inputs={"question": "the user's query", "context": "background text"},
        outputs={"answer": "1-3 sentence response"},
        instruction="Answer using only the context.",
    )

The string form is parsed once; both forms produce the same internal shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SHORTHAND_RE = re.compile(r"^\s*(.+?)\s*->\s*(.+?)\s*$")


@dataclass
class Field_:
    """One named field with an optional description."""

    name: str
    desc: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"Field name must be alphanumeric/underscore (got {self.name!r})"
            )


@dataclass
class Signature:
    """Declarative input/output contract.

    The canonical internal representation is two lists of `Field_`. The
    constructor accepts several conveniences (dict, list of names, single
    string) and normalises them.

    Attributes:
        inputs:      ordered list of input fields
        outputs:     ordered list of output fields
        instruction: optional natural-language guidance for the LM

    Examples:
        >>> Signature("q -> a").inputs[0].name
        'q'
        >>> Signature(inputs=["q"], outputs=["a"]).outputs[0].name
        'a'
        >>> Signature(inputs={"q": "the question"}, outputs={"a": "answer"}).inputs[0].desc
        'the question'
    """

    inputs: list[Field_] = field(default_factory=list)
    outputs: list[Field_] = field(default_factory=list)
    instruction: str = ""

    def __init__(
        self,
        shorthand: str | None = None,
        *,
        inputs: list[str] | dict[str, str] | list[Field_] | None = None,
        outputs: list[str] | dict[str, str] | list[Field_] | None = None,
        instruction: str = "",
    ) -> None:
        # Accept either a shorthand string OR explicit inputs/outputs, not both.
        if shorthand is not None:
            if inputs is not None or outputs is not None:
                raise ValueError(
                    "Pass either a shorthand string OR inputs/outputs, not both."
                )
            parsed_in, parsed_out = _parse_shorthand(shorthand)
            self.inputs = parsed_in
            self.outputs = parsed_out
        else:
            self.inputs = _coerce_fields(inputs or [])
            self.outputs = _coerce_fields(outputs or [])

        if not self.outputs:
            raise ValueError("Signature must declare at least one output field.")

        self.instruction = instruction.strip()

    # ---- introspection ----------------------------------------------------
    def input_names(self) -> list[str]:
        return [f.name for f in self.inputs]

    def output_names(self) -> list[str]:
        return [f.name for f in self.outputs]

    def with_instruction(self, instruction: str) -> Signature:
        """Return a copy with a new instruction (used by optimizers)."""
        return Signature(
            inputs=[Field_(f.name, f.desc) for f in self.inputs],
            outputs=[Field_(f.name, f.desc) for f in self.outputs],
            instruction=instruction,
        )

    # ---- rendering --------------------------------------------------------
    def render_prompt(self, values: dict[str, Any]) -> str:
        """Render a prompt the LM can answer.

        Layout (each section omitted when empty):
            <instruction>
            ---
            <input field>: <value>
            ...
            ---
            Respond in this exact format:
            <output field>: ...

        The output schema is explicit so the parser is unambiguous.
        """
        missing = [f.name for f in self.inputs if f.name not in values]
        if missing:
            raise KeyError(
                f"Signature missing required inputs: {missing}. "
                f"Got {sorted(values.keys())}"
            )

        parts: list[str] = []
        if self.instruction:
            parts.append(self.instruction.strip())

        if self.inputs:
            parts.append("---")
            for f in self.inputs:
                desc = f" ({f.desc})" if f.desc else ""
                parts.append(f"{f.name}{desc}: {values[f.name]}")

        parts.append("---")
        parts.append("Respond in this exact format (one per line):")
        for f in self.outputs:
            desc = f" — {f.desc}" if f.desc else ""
            parts.append(f"{f.name}: <your {f.name}>{desc}")

        return "\n".join(parts)

    def parse_response(self, text: str) -> dict[str, str]:
        """Parse the LM's raw response into a {output_name: value} dict.

        Handles two common layouts:
          1. `name: value` per line (the format we asked for).
          2. Markdown-bolded names: `**name**: value`.

        Unknown lines are ignored. Missing fields raise — the caller can
        choose to retry with a stricter prompt.
        """
        result: dict[str, str] = {}
        wanted = {f.name for f in self.outputs}

        # Greedy: each output field captures everything between its label
        # and the next output label (or end of text). Handles multiline values.
        # Build a regex that matches any of the field labels.
        if not wanted:
            return result

        labels = "|".join(re.escape(n) for n in wanted)
        pattern = re.compile(
            rf"(?:^|\n)(?:\*\*)?(?P<name>{labels})(?:\*\*)?\s*:\s*"
            rf"(?P<value>.*?)(?=\n(?:\*\*)?(?:{labels})(?:\*\*)?\s*:|\Z)",
            re.DOTALL | re.IGNORECASE,
        )

        for m in pattern.finditer(text):
            name = m.group("name").lower()
            # Case-fold lookup back to declared casing.
            for canon in wanted:
                if canon.lower() == name and canon not in result:
                    result[canon] = m.group("value").strip()
                    break

        missing = wanted - result.keys()
        if missing:
            raise SignatureParseError(
                f"Response missing fields: {sorted(missing)}. "
                f"Raw output (first 500 chars): {text[:500]!r}"
            )
        return result


class SignatureParseError(ValueError):
    """Raised when the LM's response cannot be parsed against the signature."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_shorthand(s: str) -> tuple[list[Field_], list[Field_]]:
    m = _SHORTHAND_RE.match(s)
    if not m:
        raise ValueError(
            f"Shorthand must look like 'a, b -> c' (got {s!r})"
        )
    lhs = [x.strip() for x in m.group(1).split(",") if x.strip()]
    rhs = [x.strip() for x in m.group(2).split(",") if x.strip()]
    if not rhs:
        raise ValueError(f"Shorthand has no outputs after '->': {s!r}")
    return [Field_(n) for n in lhs], [Field_(n) for n in rhs]


def _coerce_fields(
    spec: list[str] | dict[str, str] | list[Field_],
) -> list[Field_]:
    if isinstance(spec, dict):
        return [Field_(name=k, desc=v) for k, v in spec.items()]
    out: list[Field_] = []
    for item in spec:
        if isinstance(item, Field_):
            out.append(item)
        elif isinstance(item, str):
            out.append(Field_(name=item))
        else:
            raise TypeError(
                f"Field spec must be str | dict | Field_, got {type(item).__name__}"
            )
    return out


__all__ = ["Signature", "Field_", "SignatureParseError"]
