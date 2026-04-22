#!/usr/bin/env python3
"""RLM budget + trajectory enforcement for the Codex CLI adapter.

Codex has no native PreToolUse hook (unlike Claude Code). This script is the
adapter's replacement: the /rlm skill body instructs the root LM to invoke
`rlm_orchestrator.py check` via the Codex shell before every sub-LLM call.
If the call count exceeds max_llm_calls, this script exits 2, which the root
LM treats as a hard stop signal.

Same shape as `claude_skill/hooks/budget_guard.py` but callable directly as a
CLI rather than as a hook.

Usage:
    rlm_orchestrator.py check              # increment + enforce cap
    rlm_orchestrator.py log <tool> <preview>  # append trajectory entry
    rlm_orchestrator.py status             # print current state JSON
    rlm_orchestrator.py reset              # clear state (start new session)

Stdlib only. No external deps.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

STATE_PATH = Path("/tmp/rlm/state.json")
TRAJECTORY_PATH = Path("/tmp/rlm/trajectory.jsonl")
MAX_LLM_CALLS = int(os.environ.get("RLM_MAX_LLM_CALLS", "50"))
MAX_ITERATIONS = int(os.environ.get("RLM_MAX_ITERATIONS", "20"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {"llm_calls": 0, "iter": 0}
    try:
        return json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {"llm_calls": 0, "iter": 0}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state))


def cmd_check() -> int:
    """Increment llm_calls counter and enforce MAX_LLM_CALLS cap.

    Exits 0 if under cap, 2 if over. If /tmp/rlm/state.json doesn't exist,
    this is NOT an RLM session — exit 0 as a no-op (mirrors the Claude Code
    hook's gating).
    """
    if not STATE_PATH.exists():
        # Not an RLM session — treat as no-op.
        return 0

    state = _load_state()
    state["llm_calls"] = int(state.get("llm_calls", 0)) + 1
    calls = state["llm_calls"]

    try:
        _save_state(state)
    except OSError:
        pass  # Still enforce cap even if persist failed.

    if calls > MAX_LLM_CALLS:
        sys.stderr.write(
            f"RLM budget exceeded ({MAX_LLM_CALLS} llm calls). "
            f"Current count: {calls}. "
            f"Emit FINAL(answer) via `cat > /tmp/rlm/FINAL.txt` to halt cleanly.\n"
        )
        return 2

    return 0


def cmd_log(tool: str, preview: str) -> int:
    """Append a trajectory entry. Best-effort, never blocks."""
    if not STATE_PATH.exists():
        return 0

    entry = {
        "timestamp": _now_iso(),
        "tool": tool,
        "preview": preview[:500],
    }
    try:
        TRAJECTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRAJECTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        sys.stderr.write(f"warning: could not append trajectory log: {e}\n")
    return 0


def cmd_status() -> int:
    """Print state.json contents as a one-line JSON summary."""
    state = _load_state()
    state["_max_llm_calls"] = MAX_LLM_CALLS
    state["_max_iterations"] = MAX_ITERATIONS
    state["_state_path"] = str(STATE_PATH)
    state["_trajectory_path"] = str(TRAJECTORY_PATH)
    state["_session_active"] = STATE_PATH.exists()
    print(json.dumps(state))
    return 0


def cmd_reset() -> int:
    """Clear state + trajectory. Used to start a fresh RLM session."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _save_state({"llm_calls": 0, "iter": 0})
    # Truncate trajectory log
    TRAJECTORY_PATH.write_text("")
    # Remove any stale FINAL.txt
    final = Path("/tmp/rlm/FINAL.txt")
    if final.exists():
        final.unlink()
    print(json.dumps({"status": "reset", "state_path": str(STATE_PATH)}))
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: rlm_orchestrator.py {check|log|status|reset} [args...]\n"
        )
        return 1

    cmd = sys.argv[1]
    if cmd == "check":
        return cmd_check()
    if cmd == "log":
        if len(sys.argv) < 4:
            sys.stderr.write("usage: rlm_orchestrator.py log <tool> <preview>\n")
            return 1
        return cmd_log(sys.argv[2], sys.argv[3])
    if cmd == "status":
        return cmd_status()
    if cmd == "reset":
        return cmd_reset()

    sys.stderr.write(f"unknown subcommand: {cmd}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
