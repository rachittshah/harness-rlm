"""Trajectory management helpers for the RLM runtime.

A "session" is one `/rlm` invocation. It has:
  - state.json: counters + budget (call_count, started, max_calls)
  - trajectory.jsonl: append-only log of every step (sub-call, decision, etc.)
  - sub_calls.jsonl: audit trail written by the MCP server
  - FINAL.txt: synthesized answer written by `finalize(...)`

The MCP server and the Claude Code /rlm skill share these files via
/tmp/rlm/ (or an override) so the skill-side budget guard can enforce caps.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SESSION_DIR = Path("/tmp/rlm")
DEFAULT_MAX_CALLS = 50


def _now_iso() -> str:
    """ISO-8601 UTC timestamp with trailing Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_session(
    session_dir: Path = DEFAULT_SESSION_DIR,
    max_calls: int = DEFAULT_MAX_CALLS,
) -> Path:
    """Create the session dir and seed state.json.

    Returns the session_dir so callers can chain. Safe to call more than once —
    reinitialising wipes the counter (intended; each `/rlm` run is a fresh session).
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    state_path = session_dir / "state.json"
    state = {
        "call_count": 0,
        "started": _now_iso(),
        "max_calls": max_calls,
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return session_dir


def append_log(entry: dict[str, Any], session_dir: Path = DEFAULT_SESSION_DIR) -> None:
    """Append one JSON line to trajectory.jsonl.

    Caller owns the entry shape; we just ensure it serialises and lands on disk.
    The dir is created if missing so callers don't have to worry about ordering.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, default=str)
    traj_path = session_dir / "trajectory.jsonl"
    with traj_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_trajectory(session_dir: Path = DEFAULT_SESSION_DIR) -> list[dict[str, Any]]:
    """Read all trajectory entries. Returns [] if the file is missing."""
    traj_path = session_dir / "trajectory.jsonl"
    if not traj_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with traj_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entries.append(json.loads(raw))
            except json.JSONDecodeError as e:
                # Surface bad lines loudly — silent swallow would hide corruption.
                raise ValueError(f"Corrupt trajectory line {line_no} in {traj_path}: {e}") from e
    return entries


def finalize(answer: str, session_dir: Path = DEFAULT_SESSION_DIR) -> Path:
    """Write the synthesized answer to FINAL.txt. Returns the written path."""
    session_dir.mkdir(parents=True, exist_ok=True)
    final_path = session_dir / "FINAL.txt"
    final_path.write_text(answer, encoding="utf-8")
    return final_path
