"""ARC-AGI runner: calls `claude -p` per task, scores pass@k.

Each task is a JSON file with:
    {
      "train": [{"input": grid, "output": grid}, ...],
      "test":  [{"input": grid, "output": grid}, ...]
    }

We prompt claude -p with the training examples + the test input grid, asking
for JSON output. We parse, repeat up to `k` times for pass@k, score against
the gold.

Why claude -p (not direct Anthropic API):
  - User's Claude Code Enterprise has unlimited subscription (no per-token billing).
  - Same harness an end-user would invoke. Authentically tests the system.
  - Auto-loads any MCP servers in user/project config — harness-rlm tools
    become available to the model.

CLI:
    uv run python -m arc_integration.runner \\
        --dataset /tmp/ARC-AGI/data/evaluation \\
        --out results/arc/arc1_canary.json \\
        --num-tasks 5 \\
        --model "claude-opus-4-7[1m]" \\
        --effort max \\
        --k 2
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arc_integration.score import Grid, load_task, score_run


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert at the ARC-AGI puzzle benchmark. You will see a few "
    "training input/output grid pairs that demonstrate a transformation rule. "
    "Then you will see a test input grid. Your job is to apply the same "
    "transformation rule to produce the test output grid.\n\n"
    "OUTPUT FORMAT (strict): respond with ONLY a JSON object of the form:\n"
    '  {"output": [[r0c0, r0c1, ...], [r1c0, r1c1, ...], ...]}\n'
    "where each cell is a single integer 0-9. No commentary, no markdown fences, "
    "no explanation. JUST the JSON object."
)


def _grid_to_string(grid: Grid) -> str:
    """Pretty-print a grid as space-separated rows for the prompt."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def build_prompt(task: dict[str, Any], test_idx: int = 0) -> str:
    """Render the train-examples + test-input prompt for one ARC task."""
    parts: list[str] = []
    train = task.get("train", [])
    for i, ex in enumerate(train):
        parts.append(f"## Example {i + 1}")
        parts.append("INPUT:")
        parts.append(_grid_to_string(ex["input"]))
        parts.append("OUTPUT:")
        parts.append(_grid_to_string(ex["output"]))
        parts.append("")
    parts.append("## Test")
    parts.append("INPUT:")
    parts.append(_grid_to_string(task["test"][test_idx]["input"]))
    parts.append("OUTPUT (respond ONLY with the JSON object as specified):")
    return "\n".join(parts)


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

    cmd: list[str] = [claude_bin, "-p"]
    cmd.extend(["--output-format", "text"])
    cmd.extend(["--permission-mode", "bypassPermissions"])
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
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_grid(response: str) -> Grid | None:
    """Pull a 2D integer grid out of claude's response. Returns None on failure."""
    if not response or not response.strip():
        return None
    text = response.strip()

    # Strip code fences if present.
    if text.startswith("```"):
        # Remove the first fence line + the trailing one.
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.strip().startswith("```"))

    # Try to extract the outermost JSON object.
    match = JSON_OBJ_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    grid = obj.get("output")
    if grid is None and isinstance(obj.get("grid"), list):
        grid = obj["grid"]
    if not isinstance(grid, list) or not grid:
        return None
    # Validate it's a list of lists of ints.
    cleaned: Grid = []
    for row in grid:
        if not isinstance(row, list):
            return None
        out_row: list[int] = []
        for cell in row:
            if isinstance(cell, bool):  # bool is a subclass of int — reject
                return None
            if not isinstance(cell, int):
                return None
            if cell < 0 or cell > 9:
                return None
            out_row.append(cell)
        cleaned.append(out_row)
    return cleaned


# ---------------------------------------------------------------------------
# Per-task driver
# ---------------------------------------------------------------------------
def run_one_task(
    task_path: Path,
    *,
    k: int,
    model: str | None,
    effort: str | None,
    mcp_config: str | None,
    timeout_s: int,
) -> dict[str, Any]:
    """Run one ARC task k times; return result dict."""
    task = load_task(task_path)
    task_id = task_path.stem
    # ARC-AGI tasks may have multiple test inputs; we score only test[0].
    # (>99% of public tasks have a single test input.)
    test_input = task["test"][0]["input"]
    test_gold = task["test"][0]["output"]

    prompt = build_prompt(task, test_idx=0)

    attempts: list[Grid | None] = []
    attempt_info: list[dict[str, Any]] = []
    for attempt_idx in range(k):
        raw, elapsed_ms, err = invoke_claude(
            prompt,
            model=model,
            effort=effort,
            mcp_config=mcp_config,
            timeout_s=timeout_s,
        )
        grid = parse_grid(raw)
        attempts.append(grid)
        attempt_info.append(
            {
                "attempt_idx": attempt_idx + 1,
                "elapsed_ms": elapsed_ms,
                "error": err,
                "raw_response_chars": len(raw),
                "raw_response_preview": raw.strip()[:300] if raw else None,
                "parsed_grid": grid,
                "parse_ok": grid is not None,
            }
        )

    # Build the prediction dict in Kaggle submission format.
    pred_for_task: dict[str, Grid | None] = {}
    for i, a in enumerate(attempts):
        pred_for_task[f"attempt_{i + 1}"] = a

    passed = any(_grids_equal(att, test_gold) for att in attempts[:k])
    return {
        "task_id": task_id,
        "test_input_shape": (len(test_input), len(test_input[0]) if test_input else 0),
        "test_gold_shape": (len(test_gold), len(test_gold[0]) if test_gold else 0),
        "k": k,
        "passed_at_k": passed,
        "attempts": attempt_info,
        "prediction": pred_for_task,
    }


def _grids_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(ra == rb for ra, rb in zip(a, b))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="ARC-AGI runner via claude -p")
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to a directory of ARC task JSONs (e.g. /tmp/ARC-AGI/data/evaluation).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/arc/run.json"),
        help="Where to write the result summary JSON.",
    )
    p.add_argument(
        "--num-tasks",
        type=int,
        default=5,
        help="How many tasks to run (capped to dataset size).",
    )
    p.add_argument("--k", type=int, default=2, help="pass@k (default 2 per ARC convention).")
    p.add_argument("--model", default="claude-opus-4-7[1m]")
    p.add_argument("--effort", default="max", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument(
        "--mcp-config",
        default="results/mcp_config_harness_rlm.json",
        help="Path to MCP server config JSON (registers harness-rlm tools).",
    )
    p.add_argument("--timeout", type=int, default=600, help="Per-call timeout, seconds.")
    p.add_argument("--seed", type=int, default=42, help="Reproducible task selection.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan + exit without invoking claude.",
    )
    p.add_argument("--task-ids", default="", help="Comma-separated explicit task IDs to run.")
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Concurrent tasks (each task still runs its k attempts sequentially). Use 8-20 with claude -p Enterprise.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Flush partial results to --out every N tasks (lets you tail progress + survive crashes).",
    )
    args = p.parse_args()

    if not args.dataset.is_dir():
        print(f"ERROR: dataset dir not found: {args.dataset}", file=sys.stderr)
        return 2

    # Discover tasks. Deterministic ordering.
    all_tasks = sorted(args.dataset.glob("*.json"))
    if args.task_ids:
        wanted = {t.strip() for t in args.task_ids.split(",") if t.strip()}
        tasks = [p for p in all_tasks if p.stem in wanted]
    else:
        # Deterministic sample: first N by filename.
        tasks = all_tasks[: args.num_tasks]
    if not tasks:
        print("ERROR: no tasks selected.", file=sys.stderr)
        return 2

    mcp_config_path: str | None = None
    if args.mcp_config:
        candidate = Path(args.mcp_config)
        if candidate.is_file():
            mcp_config_path = str(candidate.resolve())
        else:
            print(
                f"[arc] WARN: --mcp-config {args.mcp_config} not found, "
                f"running claude -p without our MCP server.",
                file=sys.stderr,
            )

    cfg = {
        "dataset": str(args.dataset),
        "num_tasks": len(tasks),
        "k": args.k,
        "model": args.model,
        "effort": args.effort,
        "mcp_config": mcp_config_path,
        "timeout_s": args.timeout,
        "task_ids": [p.stem for p in tasks],
        "started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if args.dry_run:
        print("DRY RUN — would run:")
        print(json.dumps(cfg, indent=2))
        return 0

    print(f"[arc] running {len(tasks)} tasks × k={args.k} attempts each via claude -p")
    print(f"[arc] model={args.model} effort={args.effort} mcp_config={mcp_config_path}")

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    gold: dict[str, Grid] = {}
    predictions: dict[str, dict[str, Grid | None]] = {}

    t_start = time.monotonic()

    def _flush_partial() -> None:
        """Write partial results so a crash or kill doesn't lose data."""
        partial = {
            "config": cfg,
            "elapsed_s": round(time.monotonic() - t_start, 2),
            "scoring": score_run(predictions, gold, k=args.k),
            "results": results,
            "partial": True,
            "completed": len(results),
            "total_planned": len(tasks),
        }
        out_path.write_text(json.dumps(partial, indent=2, default=str))

    def _process_one(idx: int, task_path: Path) -> dict[str, Any]:
        """Inner driver — runnable in a thread."""
        try:
            return run_one_task(
                task_path,
                k=args.k,
                model=args.model,
                effort=args.effort,
                mcp_config=mcp_config_path,
                timeout_s=args.timeout,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[arc] [{idx}] {task_path.stem} CRASHED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return {
                "task_id": task_path.stem,
                "passed_at_k": False,
                "error": f"{type(e).__name__}: {e}",
            }

    try:
        if args.parallel <= 1:
            for i, task_path in enumerate(tasks, start=1):
                print(f"[arc] [{i}/{len(tasks)}] {task_path.stem}", flush=True)
                task_result = _process_one(i, task_path)
                results.append(task_result)
                if "prediction" in task_result:
                    gold[task_path.stem] = load_task(task_path)["test"][0]["output"]
                    predictions[task_path.stem] = task_result["prediction"]
                print(
                    f"[arc]   passed_at_k={task_result.get('passed_at_k')} "
                    f"parses={[a.get('parse_ok') for a in task_result.get('attempts', [])]}",
                    flush=True,
                )
                if i % args.checkpoint_every == 0:
                    _flush_partial()
        else:
            import concurrent.futures as _cf

            print(f"[arc] parallel={args.parallel}", flush=True)
            with _cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
                futures = {
                    ex.submit(_process_one, i, tp): (i, tp) for i, tp in enumerate(tasks, start=1)
                }
                done_count = 0
                for fut in _cf.as_completed(futures):
                    i, task_path = futures[fut]
                    task_result = fut.result()
                    results.append(task_result)
                    if "prediction" in task_result:
                        gold[task_path.stem] = load_task(task_path)["test"][0]["output"]
                        predictions[task_path.stem] = task_result["prediction"]
                    done_count += 1
                    print(
                        f"[arc] [{done_count}/{len(tasks)}] {task_path.stem} "
                        f"passed_at_k={task_result.get('passed_at_k')} "
                        f"parses={[a.get('parse_ok') for a in task_result.get('attempts', [])]}",
                        flush=True,
                    )
                    if done_count % args.checkpoint_every == 0:
                        _flush_partial()
    finally:
        elapsed_s = round(time.monotonic() - t_start, 2)
        scoring = score_run(predictions, gold, k=args.k)
        payload = {
            "config": cfg,
            "elapsed_s": elapsed_s,
            "scoring": scoring,
            "results": results,
            "ended": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n[arc] wrote {out_path}")
        print(
            f"[arc] PASS RATE: {scoring['passed_at_k']}/{scoring['total']} "
            f"= {scoring['pass_rate_pct']}% (pass@{args.k})"
        )
        print(f"[arc] elapsed: {elapsed_s}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
