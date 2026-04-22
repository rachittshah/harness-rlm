"""Harness-agnostic core helpers for the RLM runtime.

Any adapter (Claude Code, Goose, Codex, OpenCode, ...) can import from here.
Zero harness dependencies: stdlib + pydantic only. Keep it that way — adapter-
specific behaviour belongs in adapters/<harness>/.

Exports:
    BudgetGuard           — tracks iterations + llm_calls + output_chars with hard caps
    BudgetExceededError   — raised by BudgetGuard when a cap is crossed
    chunk_context         — overlap-based text chunker
    parse_ingest_directives — parses /file, /url, /paste markers from a user message
    load_shared_skill     — returns the harness-agnostic SKILL.md body (no frontmatter)
    DEFAULT_BUDGETS       — canonical default caps (matches dspy.RLM + R2 skill)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default budget envelope — mirrors the /rlm skill in R2 and dspy.RLM defaults.
# ---------------------------------------------------------------------------
DEFAULT_BUDGETS: dict[str, int] = {
    "max_iterations": 20,
    "max_llm_calls": 50,
    "max_output_chars": 10_000,
}


class BudgetExceededError(RuntimeError):
    """Raised when a BudgetGuard cap is crossed.

    Attributes:
        budget: the name of the cap that was hit (e.g. "max_llm_calls").
        limit:  the configured cap value.
        actual: the observed value that triggered the breach.
    """

    def __init__(self, budget: str, limit: int, actual: int) -> None:
        self.budget = budget
        self.limit = limit
        self.actual = actual
        super().__init__(
            f"RLM budget exceeded: {budget} hit {actual} (cap={limit}). "
            f"Emit FINAL(answer) to halt cleanly."
        )


# ---------------------------------------------------------------------------
# BudgetGuard — in-memory counter + cap enforcement.
# ---------------------------------------------------------------------------
@dataclass
class BudgetGuard:
    """Tracks RLM budget counters in memory. Persistence is the caller's job.

    Usage:
        g = BudgetGuard()                # defaults: 20 iters / 50 calls / 10K chars
        g.check_call()                   # raises if over cap
        g.increment_call()
        g.check_output(len(output))
        state = g.state_dict()           # serialise to JSON
        g2 = BudgetGuard.from_state_dict(state)   # restore
    """

    budgets: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_BUDGETS))
    iterations: int = 0
    llm_calls: int = 0
    total_output_chars: int = 0

    def __post_init__(self) -> None:
        # Fill any missing keys from defaults so callers can pass partial dicts.
        for k, v in DEFAULT_BUDGETS.items():
            self.budgets.setdefault(k, v)

    # -------- counters ------------------------------------------------------
    def increment_iteration(self) -> int:
        """Bump the iteration counter; return new value. Does not raise."""
        self.iterations += 1
        return self.iterations

    def increment_call(self) -> int:
        """Bump the llm_calls counter; return new value. Does not raise.

        Pair with `check_call()` BEFORE incrementing to enforce the cap.
        """
        self.llm_calls += 1
        return self.llm_calls

    def record_output(self, n_chars: int) -> int:
        """Add to the running total of output chars. Does not raise."""
        self.total_output_chars += max(0, int(n_chars))
        return self.total_output_chars

    # -------- checks --------------------------------------------------------
    def check_call(self) -> None:
        """Raise BudgetExceededError if the NEXT llm_call would exceed the cap."""
        cap = self.budgets["max_llm_calls"]
        if self.llm_calls + 1 > cap:
            raise BudgetExceededError("max_llm_calls", cap, self.llm_calls + 1)

    def check_iteration(self) -> None:
        """Raise BudgetExceededError if the NEXT iteration would exceed the cap."""
        cap = self.budgets["max_iterations"]
        if self.iterations + 1 > cap:
            raise BudgetExceededError("max_iterations", cap, self.iterations + 1)

    def check_output(self, n_chars: int) -> None:
        """Raise BudgetExceededError if a single cell emits > max_output_chars.

        Per-cell cap (not cumulative) — this mirrors the skill-level guard
        that treats >10K chars from one Python cell as over-budget.
        """
        cap = self.budgets["max_output_chars"]
        if n_chars > cap:
            raise BudgetExceededError("max_output_chars", cap, n_chars)

    # -------- persistence ---------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of counters + budgets."""
        return {
            "budgets": dict(self.budgets),
            "iterations": self.iterations,
            "llm_calls": self.llm_calls,
            "total_output_chars": self.total_output_chars,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> BudgetGuard:
        """Reconstruct a BudgetGuard from a prior state_dict() output.

        Unknown keys are ignored; missing counters default to 0; missing budget
        keys default to DEFAULT_BUDGETS. This makes the snapshot forward-compat
        with future counter additions.
        """
        budgets = dict(DEFAULT_BUDGETS)
        budgets.update(state.get("budgets") or {})
        return cls(
            budgets=budgets,
            iterations=int(state.get("iterations", 0)),
            llm_calls=int(state.get("llm_calls", 0)),
            total_output_chars=int(state.get("total_output_chars", 0)),
        )


# ---------------------------------------------------------------------------
# chunk_context — overlap-based chunker.
# ---------------------------------------------------------------------------
def chunk_context(text: str, chunk_size: int = 5_000, overlap: int = 200) -> list[str]:
    """Split `text` into overlapping chunks.

    Args:
        text: the input string.
        chunk_size: target length of each chunk in chars. Must be > 0.
        overlap: number of chars each chunk shares with the previous one.
            Must satisfy 0 <= overlap < chunk_size.

    Returns:
        List of chunk strings. Empty input returns []. A text shorter than
        chunk_size returns a single-element list.

    Raises:
        ValueError: if chunk_size <= 0 or overlap is out of range.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0 (got {chunk_size})")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < chunk_size "
            f"(got overlap={overlap}, chunk_size={chunk_size})"
        )

    if not text:
        return []

    # If text fits in one chunk, return it as-is (avoids a trailing empty chunk
    # when len(text) is exactly chunk_size).
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# parse_ingest_directives — /file /url /paste parsing.
# ---------------------------------------------------------------------------
# Markers live on their own token-boundary; we match at start-of-string or after
# whitespace. `/paste` consumes everything that follows up to the next marker or
# end-of-message (this mirrors rlm-cli's behaviour).
_MARKER_RE = re.compile(
    r"(?:^|\s)(?P<kind>/file|/url|/paste)(?:\s+(?P<arg>[^\s].*?))?(?=\s+/file\s|\s+/url\s|\s+/paste\s|\Z)",
    re.DOTALL,
)


def parse_ingest_directives(msg: str) -> list[dict[str, Any]]:
    """Parse `/file <path>`, `/url <url>`, `/paste <text>` markers.

    Returns a list of directive dicts in the order they appear in `msg`.
    Each dict has shape:
        {"kind": "file",  "path": "<path>"}
        {"kind": "url",   "url":  "<url>"}
        {"kind": "paste", "text": "<text>"}   # text may be "" if /paste has no body

    Input without any markers returns [].
    """
    if not msg:
        return []

    out: list[dict[str, Any]] = []
    for m in _MARKER_RE.finditer(msg):
        kind = m.group("kind").lstrip("/")
        arg = (m.group("arg") or "").strip()
        if kind == "file":
            out.append({"kind": "file", "path": arg})
        elif kind == "url":
            out.append({"kind": "url", "url": arg})
        elif kind == "paste":
            out.append({"kind": "paste", "text": arg})
    return out


# ---------------------------------------------------------------------------
# load_shared_skill — read the package-embedded SKILL.md body.
# ---------------------------------------------------------------------------
# Resolution order, in preference:
#   1. A sibling `skill/SKILL.md` inside the installed package (hatch wheel include).
#   2. A `skill/SKILL.md` walking up from this file to the repo root (editable/dev).
# Frontmatter (between two leading `---` lines) is stripped before returning.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text, count=1).lstrip()


def _candidate_skill_paths() -> list[Path]:
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    # Packaged layout: <site-packages>/harness_rlm/skill/SKILL.md (if bundled).
    candidates.append(here.parent / "skill" / "SKILL.md")
    # Editable/dev layout: repo_root/skill/SKILL.md. Walk upward looking for it.
    for parent in here.parents:
        candidates.append(parent / "skill" / "SKILL.md")
        # Don't walk past the filesystem root.
        if parent == parent.parent:
            break
    return candidates


def load_shared_skill() -> str:
    """Return the body of the harness-agnostic SKILL.md (frontmatter stripped).

    Raises:
        FileNotFoundError: if no skill/SKILL.md is discoverable on disk.
    """
    for p in _candidate_skill_paths():
        if p.is_file():
            return _strip_frontmatter(p.read_text(encoding="utf-8"))
    checked = "\n  ".join(str(p) for p in _candidate_skill_paths())
    raise FileNotFoundError(
        "Could not locate skill/SKILL.md. Searched:\n  " + checked
    )


__all__ = [
    "BudgetGuard",
    "BudgetExceededError",
    "DEFAULT_BUDGETS",
    "chunk_context",
    "parse_ingest_directives",
    "load_shared_skill",
]
