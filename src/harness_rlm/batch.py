"""Codex-style CSV batch dispatch.

`spawn_agents_on_csv` — read a CSV, spawn one Module call per row (with
`{column}` template substitution into the inputs), collect results into an
output CSV. Supports retries, parallelism, and an optional JSON schema for
the output column.

Use case: evals. You have a CSV of (id, question, gold_answer); you want to
run a Module across all rows in parallel and write (id, question, prediction,
score) back to disk.

API mirrors Codex's `spawn_agents_on_csv` tool. The instruction template lets
you reference any CSV column by name.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from harness_rlm.modules import Module, Prediction


@dataclass
class BatchJobResult:
    """One row's outcome — emitted to output CSV and returned in the result list."""

    job_id: str
    item_id: str
    status: str  # "ok" | "error"
    last_error: str = ""
    result_json: str = ""
    started: str = ""
    ended: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "job_id": self.job_id,
            "item_id": self.item_id,
            "status": self.status,
            "last_error": self.last_error,
            "result_json": self.result_json,
            "started": self.started,
            "ended": self.ended,
        }


@dataclass
class BatchResult:
    """Aggregate batch outcome."""

    job_id: str
    total: int
    ok: int
    err: int
    rows: list[BatchJobResult] = field(default_factory=list)
    output_csv: Path | None = None
    elapsed_s: float = 0.0


def spawn_agents_on_csv(
    csv_path: Path | str,
    *,
    module: Module,
    input_template: dict[str, str],
    output_csv_path: Path | str,
    job_id: str | None = None,
    item_id_column: str = "id",
    max_parallel: int = 4,
    max_retries: int = 1,
    output_schema: dict[str, Any] | None = None,
    on_progress: Callable[[BatchJobResult], None] | None = None,
) -> BatchResult:
    """Run `module` once per CSV row, write a results CSV.

    Args:
        csv_path:          Input CSV with a header row.
        module:            The Module to call per row.
        input_template:    Map of {module_input_name -> "template with {col}".}
                           e.g. {"question": "{question}", "context": "{doc}"}
                           Each {col} is replaced from the row's columns.
        output_csv_path:   Where to write results.
        job_id:            Stable identifier for this batch run.
        item_id_column:    Column in input CSV that names each row.
        max_parallel:      ThreadPool size for module dispatch.
        max_retries:       Per-row retries on exception.
        output_schema:     Optional JSON Schema — when set, `result_json` is
                           validated against it; failures land in `last_error`.
        on_progress:       Optional callback after each row completes.

    Returns:
        BatchResult with aggregate counters and a list of BatchJobResult.
    """
    csv_path = Path(csv_path)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise FileNotFoundError(f"input CSV not found: {csv_path}")

    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"input CSV is empty: {csv_path}")
    if item_id_column not in rows[0]:
        raise ValueError(
            f"item_id_column {item_id_column!r} not in CSV header. "
            f"Available: {sorted(rows[0])}"
        )

    job = job_id or f"batch-{int(time.time())}"
    t0 = time.perf_counter()

    def _one(row: dict[str, str]) -> BatchJobResult:
        item = row.get(item_id_column, "")
        start = _now()
        # Build module inputs by substituting {col} placeholders in templates.
        inputs: dict[str, Any] = {}
        for input_name, template in input_template.items():
            try:
                inputs[input_name] = template.format(**row)
            except KeyError as e:
                return BatchJobResult(
                    job_id=job,
                    item_id=item,
                    status="error",
                    last_error=f"template KeyError: {e} (row keys: {sorted(row)})",
                    started=start,
                    ended=_now(),
                )

        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                pred: Prediction = module(**inputs)
                payload = dict(pred.fields)
                if output_schema:
                    err = _validate_against_schema(payload, output_schema)
                    if err:
                        last_err = err
                        continue
                return BatchJobResult(
                    job_id=job,
                    item_id=item,
                    status="ok",
                    result_json=json.dumps(payload, default=str),
                    started=start,
                    ended=_now(),
                )
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
        return BatchJobResult(
            job_id=job,
            item_id=item,
            status="error",
            last_error=last_err,
            started=start,
            ended=_now(),
        )

    results: list[BatchJobResult] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(max_parallel, len(rows)))
    ) as ex:
        futures = [ex.submit(_one, row) for row in rows]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            results.append(r)
            if on_progress:
                on_progress(r)

    # Sort by item_id for deterministic output order.
    results.sort(key=lambda r: r.item_id)
    _write_csv(output_csv_path, results)

    ok = sum(1 for r in results if r.status == "ok")
    err = len(results) - ok
    return BatchResult(
        job_id=job,
        total=len(results),
        ok=ok,
        err=err,
        rows=results,
        output_csv=output_csv_path,
        elapsed_s=round(time.perf_counter() - t0, 3),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[BatchJobResult]) -> None:
    fieldnames = [
        "job_id",
        "item_id",
        "status",
        "last_error",
        "result_json",
        "started",
        "ended",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())


def _validate_against_schema(payload: dict, schema: dict) -> str:
    """Minimal schema check — required field presence + type coarsely.

    For full JSON Schema validation, callers can plug in jsonschema themselves
    in `on_progress`. We keep this dep-free.
    """
    required = schema.get("required", [])
    for k in required:
        if k not in payload:
            return f"missing required field: {k!r}"
    props = schema.get("properties", {})
    for k, ptype in props.items():
        want = ptype.get("type")
        if k in payload and want == "string" and not isinstance(payload[k], str):
            return f"field {k!r} expected string (got {type(payload[k]).__name__})"
    return ""


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = ["spawn_agents_on_csv", "BatchResult", "BatchJobResult"]
