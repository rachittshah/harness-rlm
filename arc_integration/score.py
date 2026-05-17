"""ARC-AGI scoring helpers.

Strict grid comparison — every cell must match. No partial credit.

Pass@k: a task passes iff ANY of the k attempts produces an exact-match output.
The ARC-AGI Kaggle leaderboard uses pass@2.

Submission format (per Kaggle / ARC Prize):
    {
      "task_id_1": {
        "attempt_1": [[r0c0, r0c1, ...], ...],
        "attempt_2": [[r0c0, r0c1, ...], ...]
      },
      ...
    }

Note: for tasks with multiple test inputs, each test input has its own list
of attempts. We follow the single-test-input convention here (matches the
vast majority of ARC-AGI-1 and ARC-AGI-2 tasks).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


Grid = list[list[int]]


def grids_equal(a: Grid | None, b: Grid | None) -> bool:
    """Strict bit-exact comparison. None never matches."""
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for ca, cb in zip(ra, rb):
            if ca != cb:
                return False
    return True


def task_passes_at_k(
    attempts: list[Grid | None],
    gold: Grid,
    k: int = 2,
) -> bool:
    """Does any of the first k attempts match the gold?"""
    for attempt in attempts[:k]:
        if grids_equal(attempt, gold):
            return True
    return False


def score_run(
    predictions: dict[str, dict[str, Grid | None]],
    gold: dict[str, Grid],
    k: int = 2,
) -> dict[str, Any]:
    """Score a full ARC-AGI run.

    Args:
        predictions: {task_id: {"attempt_1": grid, "attempt_2": grid, ...}}
        gold:        {task_id: expected_output_grid}
        k:           pass@k cutoff (default 2)

    Returns:
        {
          "total": N,
          "passed_at_k": M,
          "pass_rate_pct": (M/N)*100,
          "per_task": [{"task_id": "...", "passed": bool, "best_attempt": int|null}, ...]
        }
    """
    per_task: list[dict[str, Any]] = []
    passed = 0
    for task_id, gold_grid in gold.items():
        attempts_dict = predictions.get(task_id, {})
        # Ordered list of attempts.
        attempt_keys = sorted(
            [k for k in attempts_dict if k.startswith("attempt_")],
            key=lambda s: int(s.split("_")[1]) if s.split("_")[1].isdigit() else 99,
        )
        attempts = [attempts_dict.get(ak) for ak in attempt_keys]
        ok = task_passes_at_k(attempts, gold_grid, k=k)
        best_idx: int | None = None
        for i, att in enumerate(attempts[:k]):
            if grids_equal(att, gold_grid):
                best_idx = i + 1
                break
        if ok:
            passed += 1
        per_task.append(
            {
                "task_id": task_id,
                "passed": ok,
                "best_attempt": best_idx,
                "num_attempts": len(attempts),
            }
        )

    total = len(gold)
    return {
        "k": k,
        "total": total,
        "passed_at_k": passed,
        "pass_rate_pct": round((passed / total * 100) if total else 0.0, 2),
        "per_task": per_task,
    }


def load_task(path: Path) -> dict[str, Any]:
    """Load a single ARC-AGI task JSON. Returns the parsed dict."""
    return json.loads(path.read_text(encoding="utf-8"))


def expected_outputs(task: dict[str, Any]) -> list[Grid]:
    """Return the list of test-input expected outputs from a task dict."""
    return [t["output"] for t in task.get("test", [])]


__all__ = [
    "Grid",
    "grids_equal",
    "task_passes_at_k",
    "score_run",
    "load_task",
    "expected_outputs",
]
