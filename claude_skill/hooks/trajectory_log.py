#!/usr/bin/env python3
"""PostToolUse hook: append a trajectory record to /tmp/rlm/trajectory.jsonl.

Reads a Claude Code hook payload from stdin (JSON with `tool_name`, `tool_input`,
`tool_response`, `session_id`). For Bash/Task calls during an RLM session
(gated on /tmp/rlm/state.json existence), appends one JSON line:

    {"timestamp": "2026-04-21T22:34:56.789Z",
     "tool": "Bash",
     "input": <summary of tool_input, <=256 chars>,
     "output_chars": <len of tool_response string>}

Never blocks (always exits 0). Outside an RLM session, this hook is a no-op.
Stdlib only.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

STATE_PATH = Path("/tmp/rlm/state.json")
TRAJECTORY_PATH = Path("/tmp/rlm/trajectory.jsonl")
LOGGED_TOOLS = {"Bash", "Task"}
MAX_INPUT_SUMMARY_CHARS = 256


def _summarize_input(tool_input: object) -> str:
    """Flatten tool_input to a short string for logging."""
    if isinstance(tool_input, dict):
        # Prefer 'command' (Bash) or 'prompt' (Task) or 'description' fields.
        for key in ("command", "prompt", "description"):
            if key in tool_input and isinstance(tool_input[key], str):
                s = tool_input[key]
                break
        else:
            s = json.dumps(tool_input, default=str)
    else:
        s = str(tool_input)

    if len(s) > MAX_INPUT_SUMMARY_CHARS:
        s = s[: MAX_INPUT_SUMMARY_CHARS - 3] + "..."
    return s


def _output_length(tool_response: object) -> int:
    """Compute a char-count for the tool response."""
    if tool_response is None:
        return 0
    if isinstance(tool_response, str):
        return len(tool_response)
    try:
        return len(json.dumps(tool_response, default=str))
    except (TypeError, ValueError):
        return len(str(tool_response))


def main() -> int:
    # Never block — exit 0 on any error.
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_name = payload.get("tool_name", "")
    if tool_name not in LOGGED_TOOLS:
        return 0

    if not STATE_PATH.exists():
        return 0

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "input": _summarize_input(payload.get("tool_input", {})),
        "output_chars": _output_length(payload.get("tool_response", "")),
    }

    try:
        TRAJECTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRAJECTORY_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # Silent failure — never block.

    return 0


if __name__ == "__main__":
    sys.exit(main())
