"""Hermes-style orchestrator — compose modules, log trajectories, compress.

Three building blocks for multi-step workflows:

  Orchestrator       — runs an ordered list of (name, module, input_builder)
                       steps. Each step can read prior outputs via the
                       `state` dict the input_builder receives. Trace is
                       merged across steps so cost/calls roll up.

  SessionStore       — append-only JSONL log under a session directory.
                       Same path conventions as trajectory.py for cross-tool
                       compatibility (sub_calls.jsonl, trajectory.jsonl).

  compress(...)      — LM-summarised compaction of a long text. Used to keep
                       a rolling history under a target char budget across
                       multi-turn sessions (Hermes' `/compress` equivalent).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from harness_rlm.llm import LM, get_lm
from harness_rlm.modules import Module, Prediction, Trace

# Type for the function each step uses to build its inputs from prior state.
InputBuilder = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Step:
    """One unit of work in an Orchestrator pipeline."""

    name: str
    module: Module
    input_builder: InputBuilder
    # Optional: where to stash this step's prediction in `state`.
    # Defaults to `state[name] = pred.fields`.
    store_as: str | None = None


@dataclass
class OrchestratorResult:
    """Return value of `Orchestrator.run`."""

    state: dict[str, Any]
    trace: Trace
    steps: list[dict[str, Any]] = field(default_factory=list)

    def cost_usd(self) -> float:
        return round(self.trace.cost_usd, 6)


class Orchestrator:
    """Runs a list of `Step`s sequentially, threading state forward.

    Each step's `input_builder` receives the current `state` dict and returns
    a kwargs dict to pass into the module. After the module runs, its
    `Prediction.fields` are stored under `state[step.store_as or step.name]`.

    Failures abort the run by default — pass `continue_on_error=True` to
    record the error in `state[step.name]["error"]` and proceed.
    """

    def __init__(self, steps: list[Step], *, continue_on_error: bool = False) -> None:
        if not steps:
            raise ValueError("Orchestrator needs at least one step.")
        names = [s.name for s in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"Step names must be unique. Got {names}.")
        self.steps = steps
        self.continue_on_error = continue_on_error

    def run(
        self,
        initial_state: dict[str, Any] | None = None,
        *,
        session_store: SessionStore | None = None,
    ) -> OrchestratorResult:
        state: dict[str, Any] = dict(initial_state or {})
        combined = Trace(module="Orchestrator")
        step_logs: list[dict[str, Any]] = []

        for step in self.steps:
            t0 = _now_iso()
            inputs = step.input_builder(state)
            try:
                pred: Prediction = step.module(**inputs)
                if pred.trace is not None:
                    combined.absorb(pred.trace)
                key = step.store_as or step.name
                state[key] = pred.fields
                step_logs.append(
                    {
                        "step": step.name,
                        "started": t0,
                        "ended": _now_iso(),
                        "ok": True,
                        "input_keys": sorted(inputs.keys()),
                        "output_keys": sorted(pred.fields.keys()),
                        "calls": pred.trace.calls if pred.trace else 0,
                        "cost_usd": (
                            round(pred.trace.cost_usd, 6) if pred.trace else 0.0
                        ),
                    }
                )
                if session_store is not None:
                    session_store.append(
                        {
                            "kind": "step",
                            "name": step.name,
                            "started": t0,
                            "ended": _now_iso(),
                            "ok": True,
                            "fields": pred.fields,
                            "trace": pred.trace.to_dict() if pred.trace else None,
                        }
                    )
            except Exception as e:  # noqa: BLE001 — orchestrator owns the error policy
                step_logs.append(
                    {
                        "step": step.name,
                        "started": t0,
                        "ended": _now_iso(),
                        "ok": False,
                        "error": str(e),
                    }
                )
                if session_store is not None:
                    session_store.append(
                        {
                            "kind": "step_error",
                            "name": step.name,
                            "started": t0,
                            "ended": _now_iso(),
                            "error": str(e),
                        }
                    )
                if not self.continue_on_error:
                    return OrchestratorResult(
                        state=state, trace=combined, steps=step_logs
                    )
                state[step.name] = {"error": str(e)}

        return OrchestratorResult(state=state, trace=combined, steps=step_logs)


# ---------------------------------------------------------------------------
# SessionStore — cross-session, append-only JSONL log.
# ---------------------------------------------------------------------------
DEFAULT_SESSION_BASE = Path("/tmp/rlm")


@dataclass
class SessionStore:
    """Append-only JSONL log shared across orchestrator runs.

    File path: {base}/{name}/events.jsonl. Safe to share across processes —
    each event is one JSON line, atomically appended.
    """

    name: str = "default"
    base_dir: Path = field(default_factory=lambda: DEFAULT_SESSION_BASE)

    @property
    def path(self) -> Path:
        return self.base_dir / self.name / "events.jsonl"

    def append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Stamp the event with a timestamp if the caller didn't provide one.
        event.setdefault("ts", _now_iso())
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entries.append(json.loads(raw))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Corrupt event line {line_no} in {self.path}: {e}"
                    ) from e
        return entries

    def clear(self) -> None:
        """Wipe events for this session — useful in tests."""
        if self.path.exists():
            self.path.unlink()


# ---------------------------------------------------------------------------
# compress — LM-based summarisation for rolling history.
# ---------------------------------------------------------------------------
COMPRESS_INSTRUCTION = (
    "Compress the text below into a faithful summary. Preserve concrete facts, "
    "numbers, names, decisions, and unresolved questions. Drop pleasantries and "
    "repetition. Target {target_chars} characters."
)


def compress(
    text: str,
    *,
    target_chars: int = 4_000,
    lm: LM | None = None,
    max_tokens: int = 2_000,
) -> str:
    """Return an LM-summarised version of `text` targeting `target_chars`.

    No-op if `text` already fits. Single-call — for very long inputs, run RLM
    first and then compress the synthesis. The summary is faithful by prompt,
    but verification is the caller's responsibility.
    """
    if len(text) <= target_chars:
        return text
    used_lm = lm or get_lm()
    prompt = (
        COMPRESS_INSTRUCTION.format(target_chars=target_chars)
        + "\n---\n"
        + text
        + "\n---\nSummary:"
    )
    result = used_lm(prompt, max_tokens=max_tokens)
    return result.text.strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "Orchestrator",
    "OrchestratorResult",
    "Step",
    "SessionStore",
    "compress",
]
