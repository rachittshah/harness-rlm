#!/usr/bin/env python3
"""Run tau2-bench using harness-rlm's custom agents, from pure Python.

This path is more reliable than the ``tau2 run`` CLI because agent
registration has to happen *before* any orchestrator is constructed — and
doing that via the CLI requires patching the registry import. Running
programmatically sidesteps that entirely.

Example:
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...          # for the user simulator
    cd /Users/rshah/harness-rlm
    uv run python examples/run_tau2_py.py --num-tasks 1 --dry-run
    uv run python examples/run_tau2_py.py --num-tasks 3 --domain airline \\
        --agent harness-rlm/claude-headless --user-llm openai/gpt-4.1 \\
        --out /Users/rshah/harness-rlm/results/tau2_airline.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# Put the repo root on sys.path so `tau2_integration` is importable even when
# this script is executed directly (not via `python -m`).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run tau2 with harness-rlm agents.")
    p.add_argument("--domain", default="airline", help="tau2 domain name.")
    p.add_argument(
        "--agent",
        default="harness-rlm/claude-headless",
        choices=["harness-rlm/claude-headless", "harness-rlm/rlm"],
        help="Registered tau2 agent name.",
    )
    p.add_argument(
        "--user-llm",
        default="openai/gpt-4.1",
        help="LLM identifier for the user simulator (passed to tau2).",
    )
    p.add_argument(
        "--agent-llm",
        default=None,
        help=(
            "Optional agent LLM override. For harness-rlm/rlm this sets the "
            "root model (default claude-sonnet-4-6). Ignored by "
            "harness-rlm/claude-headless."
        ),
    )
    p.add_argument("--num-tasks", type=int, default=1, help="How many tau2 tasks to run.")
    p.add_argument("--num-trials", type=int, default=1, help="Trials per task.")
    p.add_argument("--max-steps", type=int, default=30, help="Max steps per rollout.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "results" / "tau2_run.json"),
        help="Where to write the JSON result summary.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned config and exit without running tau2 / calling any LLM.",
    )
    return p.parse_args()


def _marker(label: str) -> None:
    print(f"========== {label} ==========", flush=True)


def _result_summary(result: Any) -> dict[str, Any]:
    """Extract a JSON-safe summary from a tau2 SimulationRun."""
    reward_info = getattr(result, "reward_info", None)
    messages = getattr(result, "messages", []) or []
    task_id = getattr(result, "task_id", None) or getattr(
        getattr(result, "task", None), "id", None
    )

    summary: dict[str, Any] = {
        "task_id": task_id,
        "reward": getattr(reward_info, "reward", None),
        "num_messages": len(messages),
    }
    if reward_info is not None:
        try:
            summary["reward_info"] = reward_info.model_dump()
        except AttributeError:
            summary["reward_info"] = {
                "reward": getattr(reward_info, "reward", None),
            }
    message_previews: list[dict[str, str]] = []
    for msg in messages[:40]:
        role = getattr(msg, "role", "")
        role_val = role.value if hasattr(role, "value") else str(role)
        content = getattr(msg, "content", None) or ""
        if not isinstance(content, str):
            content = str(content)
        message_previews.append({"role": role_val, "content": content[:300]})
    summary["messages_preview"] = message_previews
    return summary


def main() -> int:
    args = _parse_args()

    if args.dry_run:
        _marker("TAU2 DRY RUN")
        print(
            json.dumps(
                {
                    "domain": args.domain,
                    "agent": args.agent,
                    "user_llm": args.user_llm,
                    "agent_llm": args.agent_llm,
                    "num_tasks": args.num_tasks,
                    "num_trials": args.num_trials,
                    "max_steps": args.max_steps,
                    "seed": args.seed,
                    "out": args.out,
                    "repo_root": str(REPO_ROOT),
                    "ANTHROPIC_API_KEY_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
                    "OPENAI_API_KEY_set": bool(os.environ.get("OPENAI_API_KEY")),
                },
                indent=2,
            ),
            flush=True,
        )
        _marker("END DRY RUN")
        return 0

    # --- Register our agents with tau2 ------------------------------------
    from tau2_integration.register import register as register_agents

    newly_registered = register_agents(overwrite=True)
    print(f"[tau2-integration] registered agents: {newly_registered}", flush=True)

    # --- Build orchestrator + run tasks -----------------------------------
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import (
        build_text_orchestrator,
        get_tasks,
        run_simulation,
    )

    tasks = get_tasks(args.domain)
    tasks = tasks[: args.num_tasks]
    if not tasks:
        print(f"[tau2-integration] ERROR: no tasks found for domain {args.domain!r}")
        return 2

    config_kwargs: dict[str, Any] = {
        "domain": args.domain,
        "agent": args.agent,
        "llm_agent": args.agent_llm or "",
        "llm_user": args.user_llm,
        "num_trials": args.num_trials,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }
    config = TextRunConfig(**config_kwargs)

    _marker("TAU2 RUN START")
    print(
        f"[tau2-integration] domain={args.domain} agent={args.agent} "
        f"tasks={len(tasks)} trials/task={args.num_trials}",
        flush=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    t_start = time.monotonic()
    try:
        for task in tasks:
            for trial in range(args.num_trials):
                print(
                    f"[tau2-integration] -> task={task.id} trial={trial + 1}/{args.num_trials}",
                    flush=True,
                )
                orchestrator = build_text_orchestrator(
                    config, task, seed=args.seed + trial
                )
                try:
                    result = run_simulation(orchestrator)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[tau2-integration] trial crashed: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    traceback.print_exc()
                    summaries.append(
                        {
                            "task_id": task.id,
                            "trial": trial,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                    continue

                summary = _result_summary(result)
                summary["trial"] = trial
                summary.setdefault("task_id", task.id)
                summaries.append(summary)
                print(
                    f"[tau2-integration]    reward={summary.get('reward')!r} "
                    f"messages={summary.get('num_messages')}",
                    flush=True,
                )
    finally:
        elapsed = time.monotonic() - t_start
        payload = {
            "config": config_kwargs,
            "num_tasks": len(tasks),
            "elapsed_sec": round(elapsed, 2),
            "results": summaries,
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        _marker(f"WROTE {out_path}")

    # Summary printout.
    rewards = [s.get("reward") for s in summaries if isinstance(s.get("reward"), (int, float))]
    if rewards:
        avg = sum(rewards) / len(rewards)
        print(
            f"[tau2-integration] mean reward over {len(rewards)} rollouts: {avg:.3f}",
            flush=True,
        )
    _marker("TAU2 RUN END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
