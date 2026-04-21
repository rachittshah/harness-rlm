#!/usr/bin/env python3
"""PreToolUse hook: enforce RLM max_llm_calls budget.

Reads a Claude Code hook payload from stdin (JSON with `tool_name`, `tool_input`,
`session_id`). Increments /tmp/rlm/state.json::llm_calls on each Bash or Task call
during an RLM session, and blocks with exit code 2 when the count exceeds
max_llm_calls (default 50).

A session is considered an RLM session iff /tmp/rlm/state.json already exists
(the /rlm skill's Step 0 creates it during ingest). Outside an RLM session, this
hook is a no-op.

Stdlib only. No external deps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STATE_PATH = Path("/tmp/rlm/state.json")
MAX_LLM_CALLS = 50
GUARDED_TOOLS = {"Bash", "Task"}


def main() -> int:
    # 1. Read hook input from stdin. Hook payloads are JSON on a single stream.
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        # Malformed input — don't block (exit 0 = allow).
        return 0

    tool_name = payload.get("tool_name", "")

    # 2. Only guard Bash/Task. Read/Write/Grep/Glob are free.
    if tool_name not in GUARDED_TOOLS:
        return 0

    # 3. Gate on RLM session: state.json must exist.
    if not STATE_PATH.exists():
        return 0

    # 4. Increment counter atomically (single-process; no lock needed for CC hooks).
    try:
        state = json.loads(STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        state = {"llm_calls": 0, "iter": 0}

    state["llm_calls"] = int(state.get("llm_calls", 0)) + 1
    calls = state["llm_calls"]

    try:
        STATE_PATH.write_text(json.dumps(state))
    except OSError:
        pass  # If we can't write, still check the cap in-memory below.

    # 5. Enforce cap. Exit 2 = blocking error; stderr message is surfaced to the
    #    root LM by Claude Code.
    if calls > MAX_LLM_CALLS:
        sys.stderr.write(
            f"RLM budget exceeded ({MAX_LLM_CALLS} llm calls). "
            f"Current count: {calls}. "
            f"Emit FINAL(answer) via `cat > /tmp/rlm/FINAL.txt` to halt cleanly.\n"
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
