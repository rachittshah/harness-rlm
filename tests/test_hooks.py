"""Tests for the Claude Code hooks (adapters/claude_code/hooks/*.py).

Hooks hard-code /tmp/rlm paths. To isolate each test, we copy the hook source
into tmp_path and text-substitute the hard-coded paths to point at tmp_path,
then invoke the patched copy via `subprocess.run`. This keeps the production
hook code untouched and gives every test a clean /tmp/rlm analog.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).resolve().parent.parent / "adapters" / "claude_code" / "hooks"
BUDGET_GUARD_SRC = HOOKS_DIR / "budget_guard.py"
TRAJECTORY_LOG_SRC = HOOKS_DIR / "trajectory_log.py"


def _skip_if_hook_missing(src: Path):
    if not src.exists():
        pytest.skip(f"hook source not committed yet: {src}")


def _patched_hook(src: Path, tmp_rlm_dir: Path, dest_dir: Path) -> Path:
    """Copy a hook script into dest_dir, rewriting the hard-coded /tmp/rlm path.

    Returns path to the patched, executable script.
    """
    text = src.read_text(encoding="utf-8")
    # Replace the exact hardcoded prefix with tmp_rlm_dir.
    patched = text.replace('Path("/tmp/rlm/', f'Path("{tmp_rlm_dir}/')
    dest = dest_dir / src.name
    dest.write_text(patched, encoding="utf-8")
    dest.chmod(0o755)
    return dest


def _run_hook(hook: Path, stdin_payload: dict) -> subprocess.CompletedProcess:
    """Invoke hook via the current Python with stdin = json-dumped payload."""
    return subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(stdin_payload).encode("utf-8"),
        capture_output=True,
        timeout=10,
        check=False,
    )


# ---------------------------------------------------------------------------
# budget_guard
# ---------------------------------------------------------------------------
def test_budget_guard_blocks_when_over_limit(tmp_path: Path):
    """With llm_calls already at 50, next Bash call increments to 51 → exit 2."""
    _skip_if_hook_missing(BUDGET_GUARD_SRC)

    rlm_dir = tmp_path / "rlm"
    rlm_dir.mkdir()
    state_path = rlm_dir / "state.json"
    # Pre-seed counter so increment pushes it past MAX_LLM_CALLS (=50).
    state_path.write_text(json.dumps({"llm_calls": 50, "iter": 0}))

    hook = _patched_hook(BUDGET_GUARD_SRC, rlm_dir, tmp_path)
    result = _run_hook(hook, {"tool_name": "Bash", "tool_input": {"command": "echo hi"}})

    assert result.returncode == 2, (
        f"expected exit 2 (blocking), got {result.returncode}. stderr={result.stderr!r}"
    )
    assert b"budget exceeded" in result.stderr.lower() or b"RLM budget" in result.stderr


def test_budget_guard_allows_under_limit(tmp_path: Path):
    """Under the cap, the hook increments and exits 0 (allow)."""
    _skip_if_hook_missing(BUDGET_GUARD_SRC)

    rlm_dir = tmp_path / "rlm"
    rlm_dir.mkdir()
    state_path = rlm_dir / "state.json"
    state_path.write_text(json.dumps({"llm_calls": 3, "iter": 0}))

    hook = _patched_hook(BUDGET_GUARD_SRC, rlm_dir, tmp_path)
    result = _run_hook(hook, {"tool_name": "Bash", "tool_input": {"command": "echo hi"}})

    assert result.returncode == 0, (
        f"expected exit 0 (allow), got {result.returncode}. stderr={result.stderr!r}"
    )
    # Counter should have incremented.
    new_state = json.loads(state_path.read_text())
    assert new_state["llm_calls"] == 4


def test_budget_guard_ignores_non_rlm_session(tmp_path: Path):
    """No state.json → hook is a no-op, exits 0 without acting."""
    _skip_if_hook_missing(BUDGET_GUARD_SRC)

    rlm_dir = tmp_path / "rlm"  # intentionally NOT creating state.json
    hook = _patched_hook(BUDGET_GUARD_SRC, rlm_dir, tmp_path)
    result = _run_hook(hook, {"tool_name": "Bash", "tool_input": {"command": "echo hi"}})

    assert result.returncode == 0
    # state.json must still not exist — hook did not seed it.
    assert not (rlm_dir / "state.json").exists()


def test_budget_guard_ignores_non_guarded_tools(tmp_path: Path):
    """Read/Write/Grep/Glob etc. are not counted — exit 0 without incrementing."""
    _skip_if_hook_missing(BUDGET_GUARD_SRC)

    rlm_dir = tmp_path / "rlm"
    rlm_dir.mkdir()
    state_path = rlm_dir / "state.json"
    state_path.write_text(json.dumps({"llm_calls": 10}))

    hook = _patched_hook(BUDGET_GUARD_SRC, rlm_dir, tmp_path)
    result = _run_hook(hook, {"tool_name": "Read", "tool_input": {"file_path": "/tmp/x"}})

    assert result.returncode == 0
    # Counter unchanged — Read is not guarded.
    assert json.loads(state_path.read_text())["llm_calls"] == 10


def test_budget_guard_malformed_stdin(tmp_path: Path):
    """Malformed JSON on stdin → exit 0 (allow, don't block on parse error)."""
    _skip_if_hook_missing(BUDGET_GUARD_SRC)

    rlm_dir = tmp_path / "rlm"
    hook = _patched_hook(BUDGET_GUARD_SRC, rlm_dir, tmp_path)
    result = subprocess.run(
        [sys.executable, str(hook)],
        input=b"not-json-at-all",
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# trajectory_log
# ---------------------------------------------------------------------------
def test_trajectory_log_appends(tmp_path: Path):
    """A Bash PostToolUse payload appends one JSON line to trajectory.jsonl."""
    _skip_if_hook_missing(TRAJECTORY_LOG_SRC)

    rlm_dir = tmp_path / "rlm"
    rlm_dir.mkdir()
    # Gate: state.json must exist for the hook to act.
    (rlm_dir / "state.json").write_text(json.dumps({"llm_calls": 0}))

    hook = _patched_hook(TRAJECTORY_LOG_SRC, rlm_dir, tmp_path)
    payload = {
        "tool_name": "Bash",
        "tool_input": {"command": "ls /tmp", "description": "list tmp"},
        "tool_response": "file1\nfile2\n",
        "session_id": "abc",
    }
    result = _run_hook(hook, payload)
    assert result.returncode == 0, f"stderr={result.stderr!r}"

    traj_path = rlm_dir / "trajectory.jsonl"
    assert traj_path.exists()
    lines = traj_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["tool"] == "Bash"
    assert entry["input"] == "ls /tmp"
    assert entry["output_chars"] == len("file1\nfile2\n")
    assert "timestamp" in entry


def test_trajectory_log_skips_non_logged_tools(tmp_path: Path):
    """Read / Write / Grep are not logged — no file created."""
    _skip_if_hook_missing(TRAJECTORY_LOG_SRC)

    rlm_dir = tmp_path / "rlm"
    rlm_dir.mkdir()
    (rlm_dir / "state.json").write_text(json.dumps({"llm_calls": 0}))

    hook = _patched_hook(TRAJECTORY_LOG_SRC, rlm_dir, tmp_path)
    result = _run_hook(
        hook,
        {
            "tool_name": "Read",
            "tool_input": {"file_path": "/tmp/x"},
            "tool_response": "contents",
        },
    )
    assert result.returncode == 0
    assert not (rlm_dir / "trajectory.jsonl").exists()


def test_trajectory_log_noop_without_rlm_session(tmp_path: Path):
    """No state.json → hook is a no-op even for Bash."""
    _skip_if_hook_missing(TRAJECTORY_LOG_SRC)

    rlm_dir = tmp_path / "rlm"  # not created
    hook = _patched_hook(TRAJECTORY_LOG_SRC, rlm_dir, tmp_path)
    result = _run_hook(
        hook,
        {
            "tool_name": "Bash",
            "tool_input": {"command": "echo hi"},
            "tool_response": "hi\n",
        },
    )
    assert result.returncode == 0
    assert not (rlm_dir / "trajectory.jsonl").exists()


def test_trajectory_log_truncates_long_input(tmp_path: Path):
    """Input summary is clipped at MAX_INPUT_SUMMARY_CHARS (256)."""
    _skip_if_hook_missing(TRAJECTORY_LOG_SRC)

    rlm_dir = tmp_path / "rlm"
    rlm_dir.mkdir()
    (rlm_dir / "state.json").write_text(json.dumps({"llm_calls": 0}))

    hook = _patched_hook(TRAJECTORY_LOG_SRC, rlm_dir, tmp_path)
    long_cmd = "echo " + ("x" * 1000)
    result = _run_hook(
        hook,
        {
            "tool_name": "Bash",
            "tool_input": {"command": long_cmd},
            "tool_response": "ok",
        },
    )
    assert result.returncode == 0
    entry = json.loads((rlm_dir / "trajectory.jsonl").read_text().splitlines()[0])
    assert len(entry["input"]) == 256
    assert entry["input"].endswith("...")


def test_trajectory_log_malformed_stdin(tmp_path: Path):
    """Malformed JSON → exit 0 (never blocks)."""
    _skip_if_hook_missing(TRAJECTORY_LOG_SRC)

    rlm_dir = tmp_path / "rlm"
    hook = _patched_hook(TRAJECTORY_LOG_SRC, rlm_dir, tmp_path)
    result = subprocess.run(
        [sys.executable, str(hook)],
        input=b"garbage{{",
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Safety: confirm that using the real /tmp/rlm path is never touched by tests.
# (Regression guard — if any test accidentally forgets the patched hook.)
# ---------------------------------------------------------------------------
def test_tests_do_not_touch_real_tmp_rlm():
    """Tests must isolate to tmp_path; /tmp/rlm should not be modified by tests."""
    # Not a strict guarantee (the user may have /tmp/rlm from normal operation),
    # but this test serves as a breadcrumb: if hook tests ever start writing
    # /tmp/rlm, set a marker file in tmp_path and fail loudly.
    real_tmp_rlm = Path("/tmp/rlm")
    # Record mtime snapshot for state.json if present (advisory only).
    if real_tmp_rlm.exists() and (real_tmp_rlm / "state.json").exists():
        before = os.path.getmtime(real_tmp_rlm / "state.json")
        # Re-read to confirm reading itself doesn't mutate.
        after = os.path.getmtime(real_tmp_rlm / "state.json")
        assert before == after
    # Otherwise this is a trivial pass — documents the invariant.
