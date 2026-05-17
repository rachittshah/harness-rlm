"""ARC-AGI-3 access stub — NOT a working run yet.

ARC-AGI-3 is the interactive games benchmark (launched Mar 2026). Unlike
ARC-AGI-1 and ARC-AGI-2 which ship as JSON files on GitHub, ARC-AGI-3 requires
access to a hosted API at three.arcprize.org / docs.arcprize.org.

## Blocker

The API requires `ARC_API_KEY`. Anonymous endpoints (per docs) exist but
return `unauthorized` when called without the header set:

    $ curl -sS "https://three.arcprize.org/api/games"
    unauthorized

To unblock:
1. Register at https://arcprize.org/api-keys (free, gets an anonymous-tier key)
2. `export ARC_API_KEY=<your-key>`
3. Install the toolkit: `pip install arc-agi` (verify package name on docs)
4. Use `arc_agi.Arcade()` from the docs quickstart

## What this file gives you

A scaffold for when the key is available. Wire `claude -p` as the policy:
each game step, build a prompt describing the current grid + action history,
call claude -p, parse one of ACTION1..ACTION7 from the response, submit.

The submission/scoring follows the ARC-AGI-3 protocol:
- score normalized to 100% = human baseline
- skill-acquisition efficiency, not raw completion
- frontier models currently score <0.5% (per ARC Prize blog, Mar 2026)

## Reference SOTA (May 2026)

- Humans: 100%
- Gemini 3.1 Pro: 0.37%
- Opus 4.6: 0.25%
- GPT-5.4: 0.26%
- Best purpose-built agent: 12.58%

Floor is wide open — any non-trivial score is publishable. But scaffolding
dominates raw model capability here.

## Status

Not run in this canary. Documented as a known gap. Re-enable by setting
ARC_API_KEY and removing this stub.
"""

from __future__ import annotations

import os
import sys
from typing import Any


def precheck() -> dict[str, Any]:
    """Return a status dict reporting whether ARC-AGI-3 can be run."""
    api_key = os.environ.get("ARC_API_KEY")
    status: dict[str, Any] = {
        "blocker": None,
        "ready": False,
        "api_key_set": bool(api_key),
        "next_steps": [],
    }
    if not api_key:
        status["blocker"] = "ARC_API_KEY env var not set"
        status["next_steps"] = [
            "Register at https://arcprize.org/api-keys",
            "export ARC_API_KEY=<the-key>",
            "Install the ARC-AGI toolkit (`pip install arc-agi` — verify on docs.arcprize.org)",
            "Re-run this script",
        ]
        return status
    # If the key is set we still need the toolkit installed.
    try:
        import arc_agi  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        status["blocker"] = "arc-agi Python toolkit not installed"
        status["next_steps"] = [
            "pip install arc-agi (verify exact name on docs.arcprize.org)",
            "Then `arc_agi.Arcade()` will list public games",
        ]
        return status
    status["ready"] = True
    status["next_steps"] = [
        "List games: `python -c 'import arc_agi; print(arc_agi.Arcade().list_games())'`",
        "Wire a claude -p policy: for each step, prompt claude with current grid + action history, parse ACTION1..ACTION7 from response",
        "Submit actions via env.step(GameAction.ACTIONx)",
    ]
    return status


def main() -> int:
    import json

    status = precheck()
    print(json.dumps(status, indent=2))
    return 0 if status["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
