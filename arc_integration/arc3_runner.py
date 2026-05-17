"""ARC-AGI-3 runner — claude -p as the policy.

Each turn:
  1. Read the current frame (grid) + available actions from the env.
  2. Ask claude -p which action to take next, given history.
  3. Parse the response into a GameAction enum.
  4. env.step(action). Repeat until WIN, LOSE, or max-turns.

Per-game wall-clock varies wildly (interactive games). We cap max_turns to
limit cost. We also cap parallelism since each "game" is a sequence of
claude -p calls and reset has tighter rate limits than static eval.

Submission: scorecards land on the user's ARC API account when ARC_API_KEY
is set. Without a key the SDK uses an anonymous key (results not attached to
an account).

Usage:
    export ARC_API_KEY=<your-key>
    uv run python -m arc_integration.arc3_runner \\
        --num-games 5 --max-turns 80 --parallel 3 \\
        --model "claude-opus-4-7[1m]" --effort max \\
        --out results/arc/arc3_5games.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# System prompt: tells claude -p how to play
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are playing an ARC-AGI-3 interactive puzzle game. Each turn, you will see:\n"
    "  - the current game frame as a grid of integers 0-9 (each integer is a colour),\n"
    "  - the action history (your recent actions + frame transitions),\n"
    "  - the list of available actions (subset of 1-7).\n\n"
    "Your job: pick the SINGLE next action that makes progress toward winning.\n\n"
    "Action semantics (default — game-specific details vary):\n"
    "  ACTION1..5  — directional / interaction inputs (mapped per game)\n"
    "  ACTION6     — point action: requires (x, y) coordinates (we don't support these here)\n"
    "  ACTION7     — undo / step back\n\n"
    "STRICT OUTPUT FORMAT: respond with ONLY a JSON object of the form:\n"
    '  {"action": N, "reason": "<one short sentence>"}\n'
    "where N is an integer 1-5 or 7 from the available_actions list. "
    "If only ACTION6 is available, return action: 6 (we will skip it).\n"
    "NO commentary, NO markdown fences, NO explanation outside the JSON object."
)


# ---------------------------------------------------------------------------
# claude -p invocation
# ---------------------------------------------------------------------------
def invoke_claude(
    prompt: str,
    *,
    model: str | None,
    effort: str | None,
    mcp_config: str | None,
    timeout_s: int,
    claude_bin: str = "claude",
) -> tuple[str, int, str | None]:
    """Call `claude -p` and return (stdout, elapsed_ms, error_or_None)."""
    if shutil.which(claude_bin) is None:
        return "", 0, f"claude binary not found: {claude_bin}"
    cmd: list[str] = [
        claude_bin,
        "-p",
        "--output-format",
        "text",
        "--permission-mode",
        "bypassPermissions",
    ]
    if model:
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["--effort", effort])
    if mcp_config:
        cmd.extend(["--mcp-config", mcp_config])
    cmd.extend(["--append-system-prompt", SYSTEM_PROMPT])
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", int((time.monotonic() - t0) * 1000), f"timeout after {timeout_s}s"
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if proc.returncode != 0:
        return "", elapsed_ms, f"claude exit {proc.returncode}: {proc.stderr.strip()[:300]}"
    return proc.stdout or "", elapsed_ms, None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_action(response: str, available_actions: list[int]) -> tuple[int | None, str]:
    """Return (action_int_or_None, reason). action must be in available_actions."""
    if not response or not response.strip():
        return None, "empty response"
    text = response.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    for m in JSON_OBJ_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        a = obj.get("action")
        reason = obj.get("reason", "")
        if isinstance(a, int) and a in available_actions:
            return a, str(reason)
        if isinstance(a, str) and a.isdigit() and int(a) in available_actions:
            return int(a), str(reason)
    return None, f"no valid action in available_actions={available_actions}"


# ---------------------------------------------------------------------------
# Frame rendering — turn ndarray into a compact string for the prompt
# ---------------------------------------------------------------------------
def render_frame(frame: Any) -> str:
    """Render the first sub-frame as space-separated rows. Frames are lists of ndarrays."""
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None  # type: ignore[assignment]
    if not frame:
        return "(empty frame)"
    sub = frame[0] if isinstance(frame, list) else frame
    if np is not None and isinstance(sub, np.ndarray):
        sub_list = sub.tolist()
    else:
        sub_list = list(sub) if hasattr(sub, "__iter__") else [[sub]]
    return "\n".join(" ".join(str(int(c)) for c in row) for row in sub_list)


# ---------------------------------------------------------------------------
# Play one game
# ---------------------------------------------------------------------------
def play_one_game(
    game_id: str,
    *,
    max_turns: int,
    model: str | None,
    effort: str | None,
    mcp_config: str | None,
    timeout_s: int,
) -> dict[str, Any]:
    """Play one game start-to-finish with claude -p as the policy."""
    import arc_agi
    from arcengine import GameAction

    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    fd = env.reset()
    history: list[dict[str, Any]] = []

    # Defensive: some SDK paths return frames where .state is None on the
    # initial reset. Default to "NOT_FINISHED" and continue — we'll re-read
    # state every turn from env.step() which always returns a fresh FrameDataRaw.
    state_obj = getattr(fd, "state", None) if fd is not None else None
    state_name = (
        getattr(state_obj, "name", str(state_obj)) if state_obj is not None else "NOT_FINISHED"
    )
    levels_done = int(getattr(fd, "levels_completed", 0)) if fd is not None else 0
    won = False

    for turn in range(max_turns):
        if fd is None:
            break
        avail = (
            [int(a) for a in (fd.available_actions or [])]
            if hasattr(fd, "available_actions")
            else []
        )
        if not avail:
            break
        frame_str = render_frame(getattr(fd, "frame", None) or [])
        # Build prompt
        prompt_parts = [
            f"## Game: {game_id}",
            f"Turn: {turn + 1}/{max_turns}",
            f"State: {state_name}",
            f"Levels completed: {levels_done}",
            f"Available actions: {avail}",
            "",
            "Recent history (last 6 turns):",
        ]
        for h in history[-6:]:
            prompt_parts.append(
                f"  turn {h['turn']}: action={h['action']} → state={h['state']} levels={h['levels']}"
            )
        prompt_parts += [
            "",
            "Current frame (grid of integers 0-9):",
            frame_str,
            "",
            'Reply with JSON: {"action": N, "reason": "..."}',
        ]
        prompt = "\n".join(prompt_parts)

        raw, elapsed_ms, err = invoke_claude(
            prompt,
            model=model,
            effort=effort,
            mcp_config=mcp_config,
            timeout_s=timeout_s,
        )
        action_int, reason = parse_action(raw, avail)
        if action_int is None:
            # Fallback: pick the first available action — never stall.
            action_int = avail[0]
            reason = f"FALLBACK ({reason})"
        try:
            action_enum = getattr(GameAction, f"ACTION{action_int}")
        except AttributeError:
            action_enum = getattr(GameAction, "ACTION1")
        try:
            fd = env.step(action_enum)
        except Exception as e:  # noqa: BLE001
            history.append({"turn": turn + 1, "action": action_int, "error": str(e)})
            break

        if fd is None:
            history.append(
                {"turn": turn + 1, "action": action_int, "error": "env.step returned None"}
            )
            break
        state_obj = getattr(fd, "state", None)
        state_name = (
            getattr(state_obj, "name", str(state_obj)) if state_obj is not None else state_name
        )
        levels_done = int(getattr(fd, "levels_completed", levels_done) or levels_done)
        history.append(
            {
                "turn": turn + 1,
                "action": action_int,
                "reason": reason[:160],
                "state": state_name,
                "levels": levels_done,
                "elapsed_ms": elapsed_ms,
                "claude_err": err,
            }
        )

        # Terminal states.
        if state_name in {"WIN", "GAME_OVER"}:
            won = state_name == "WIN"
            break

    # Score from scorecard.
    final_state = state_name
    try:
        sc = arc.get_scorecard() if hasattr(arc, "get_scorecard") else None
    except Exception:  # noqa: BLE001
        sc = None
    score_info: dict[str, Any] = {}
    if sc is not None:
        try:
            score_info = sc.model_dump() if hasattr(sc, "model_dump") else dict(sc)
        except Exception:  # noqa: BLE001
            score_info = {"repr": repr(sc)[:300]}

    return {
        "game_id": game_id,
        "turns_used": len(history),
        "max_turns": max_turns,
        "final_state": final_state,
        "levels_completed": levels_done,
        "won": won,
        "scorecard_id": str(env.scorecard_id) if env.scorecard_id else None,
        "history": history,
        "scorecard": score_info,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="ARC-AGI-3 runner via claude -p")
    p.add_argument(
        "--num-games",
        type=int,
        default=5,
        help="How many environments to play (0 = all 25 public).",
    )
    p.add_argument("--max-turns", type=int, default=60, help="Per-game hard cap on turns.")
    p.add_argument("--parallel", type=int, default=3, help="Concurrent games.")
    p.add_argument("--model", default="claude-opus-4-7[1m]")
    p.add_argument("--effort", default="max")
    p.add_argument(
        "--mcp-config",
        default="results/mcp_config_harness_rlm.json",
        help="Path to MCP server config JSON (harness-rlm).",
    )
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--out", type=Path, default=Path("results/arc/arc3_run.json"))
    p.add_argument(
        "--game-ids", default="", help="Comma-separated explicit game IDs (overrides --num-games)."
    )
    p.add_argument("--checkpoint-every", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Need ARC_API_KEY for the SDK (anonymous works too but results aren't attached).
    if not os.environ.get("ARC_API_KEY"):
        print(
            "[arc3] WARN: ARC_API_KEY not set — anonymous key will be used "
            "(scorecards won't be attached to an account).",
            file=sys.stderr,
        )

    import arc_agi

    arc = arc_agi.Arcade()
    envs = arc.available_environments
    print(f"[arc3] Found {len(envs)} environments")

    if args.game_ids:
        wanted = {g.strip() for g in args.game_ids.split(",") if g.strip()}
        # Match by title (case-insensitive prefix) or game_id (prefix).
        chosen = [
            e
            for e in envs
            if e.title.lower() in {w.lower() for w in wanted}
            or any(e.game_id.lower().startswith(w.lower()) for w in wanted)
        ]
    else:
        n = args.num_games or len(envs)
        chosen = envs[:n]
    game_ids = [e.title.lower() for e in chosen]
    if not game_ids:
        print("[arc3] ERROR: no games selected.", file=sys.stderr)
        return 2

    mcp_config_path: str | None = None
    if args.mcp_config:
        c = Path(args.mcp_config)
        if c.is_file():
            mcp_config_path = str(c.resolve())

    cfg = {
        "num_games": len(game_ids),
        "game_ids": game_ids,
        "max_turns": args.max_turns,
        "parallel": args.parallel,
        "model": args.model,
        "effort": args.effort,
        "mcp_config": mcp_config_path,
        "timeout_s": args.timeout,
        "arc_api_key_source": "env (ARC_API_KEY)" if os.environ.get("ARC_API_KEY") else "anonymous",
        "started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return 0

    print(
        f"[arc3] playing {len(game_ids)} games, parallel={args.parallel}, "
        f"max_turns={args.max_turns}, mcp={mcp_config_path}",
        flush=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    t_start = time.monotonic()

    def _flush() -> None:
        elapsed = round(time.monotonic() - t_start, 2)
        wins = sum(1 for r in results if r.get("won"))
        levels = sum(int(r.get("levels_completed", 0)) for r in results)
        payload = {
            "config": cfg,
            "elapsed_s": elapsed,
            "summary": {
                "games_played": len(results),
                "wins": wins,
                "win_rate_pct": round(wins / max(1, len(results)) * 100, 2),
                "total_levels": levels,
            },
            "results": results,
            "partial": len(results) < len(game_ids),
        }
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    def _run_one(gid: str) -> dict[str, Any]:
        try:
            return play_one_game(
                gid,
                max_turns=args.max_turns,
                model=args.model,
                effort=args.effort,
                mcp_config=mcp_config_path,
                timeout_s=args.timeout,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[arc3] {gid} CRASHED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return {"game_id": gid, "error": f"{type(e).__name__}: {e}", "won": False}

    try:
        if args.parallel <= 1:
            for i, gid in enumerate(game_ids, start=1):
                print(f"[arc3] [{i}/{len(game_ids)}] starting {gid}", flush=True)
                r = _run_one(gid)
                results.append(r)
                print(
                    f"[arc3] [{i}/{len(game_ids)}] {gid} "
                    f"won={r.get('won')} levels={r.get('levels_completed')} "
                    f"turns={r.get('turns_used')} final={r.get('final_state')}",
                    flush=True,
                )
                if i % args.checkpoint_every == 0:
                    _flush()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
                futures = {ex.submit(_run_one, gid): gid for gid in game_ids}
                done_count = 0
                for fut in concurrent.futures.as_completed(futures):
                    gid = futures[fut]
                    r = fut.result()
                    results.append(r)
                    done_count += 1
                    print(
                        f"[arc3] [{done_count}/{len(game_ids)}] {gid} "
                        f"won={r.get('won')} levels={r.get('levels_completed')} "
                        f"turns={r.get('turns_used')} final={r.get('final_state')}",
                        flush=True,
                    )
                    if done_count % args.checkpoint_every == 0:
                        _flush()
    finally:
        _flush()
        wins = sum(1 for r in results if r.get("won"))
        print(
            f"\n[arc3] PLAYED: {len(results)}  WINS: {wins}  "
            f"WIN%: {round(wins / max(1, len(results)) * 100, 2)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
