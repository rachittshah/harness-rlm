"""Tests for harness_rlm.trajectory session helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_init_session_creates_state(tmp_path: Path):
    """init_session creates the dir and seeds state.json with default counters."""
    from harness_rlm.trajectory import init_session

    session_dir = tmp_path / "rlm-session"
    result = init_session(session_dir=session_dir)

    assert result == session_dir
    assert session_dir.is_dir()
    state_path = session_dir / "state.json"
    assert state_path.exists()

    state = json.loads(state_path.read_text())
    assert state["call_count"] == 0
    assert state["max_calls"] == 50
    assert "started" in state
    assert state["started"].endswith("Z")  # UTC marker


def test_init_session_respects_custom_max_calls(tmp_path: Path):
    """Caller can override the default max_calls budget."""
    from harness_rlm.trajectory import init_session

    session_dir = tmp_path / "rlm"
    init_session(session_dir=session_dir, max_calls=7)
    state = json.loads((session_dir / "state.json").read_text())
    assert state["max_calls"] == 7


def test_init_session_reinit_wipes_counter(tmp_path: Path):
    """Re-initialising resets call_count (each /rlm run is a fresh session)."""
    from harness_rlm.trajectory import init_session

    session_dir = tmp_path / "rlm"
    init_session(session_dir=session_dir)
    # Simulate some progress in the previous run.
    state_path = session_dir / "state.json"
    state_path.write_text(json.dumps({"call_count": 17, "started": "x", "max_calls": 50}))

    init_session(session_dir=session_dir)
    state = json.loads(state_path.read_text())
    assert state["call_count"] == 0


def test_append_log_appends_line(tmp_path: Path):
    """append_log writes one JSON line per call; file grows without overwrite."""
    from harness_rlm.trajectory import append_log

    session_dir = tmp_path / "rlm"
    append_log({"step": 1, "action": "spawn"}, session_dir=session_dir)
    append_log({"step": 2, "action": "synth"}, session_dir=session_dir)

    traj_path = session_dir / "trajectory.jsonl"
    lines = traj_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": 1, "action": "spawn"}
    assert json.loads(lines[1]) == {"step": 2, "action": "synth"}


def test_append_log_creates_dir_if_missing(tmp_path: Path):
    """append_log mkdirs the session dir so callers don't have to."""
    from harness_rlm.trajectory import append_log

    session_dir = tmp_path / "deeply" / "nested" / "rlm"
    assert not session_dir.exists()
    append_log({"ok": True}, session_dir=session_dir)
    assert session_dir.is_dir()
    assert (session_dir / "trajectory.jsonl").exists()


def test_append_log_serialises_non_json_defaults(tmp_path: Path):
    """Non-JSON-serialisable values fall through `default=str` (e.g. Path)."""
    from harness_rlm.trajectory import append_log

    session_dir = tmp_path / "rlm"
    append_log({"path": Path("/etc/hosts")}, session_dir=session_dir)
    lines = (session_dir / "trajectory.jsonl").read_text().splitlines()
    assert json.loads(lines[0]) == {"path": "/etc/hosts"}


def test_read_trajectory_returns_all_entries(tmp_path: Path):
    """read_trajectory returns every valid JSON line in order."""
    from harness_rlm.trajectory import append_log, read_trajectory

    session_dir = tmp_path / "rlm"
    entries = [
        {"step": 1, "tool": "Bash"},
        {"step": 2, "tool": "Task"},
        {"step": 3, "tool": "mcp__rlm__llm_query"},
    ]
    for e in entries:
        append_log(e, session_dir=session_dir)

    got = read_trajectory(session_dir=session_dir)
    assert got == entries


def test_read_trajectory_empty_when_missing(tmp_path: Path):
    """If trajectory.jsonl does not exist, return []."""
    from harness_rlm.trajectory import read_trajectory

    session_dir = tmp_path / "rlm"
    assert read_trajectory(session_dir=session_dir) == []


def test_read_trajectory_skips_blank_lines(tmp_path: Path):
    """Blank lines in trajectory.jsonl are tolerated (skipped, not raised)."""
    from harness_rlm.trajectory import read_trajectory

    session_dir = tmp_path / "rlm"
    session_dir.mkdir(parents=True)
    traj_path = session_dir / "trajectory.jsonl"
    traj_path.write_text('{"a": 1}\n\n{"b": 2}\n')
    got = read_trajectory(session_dir=session_dir)
    assert got == [{"a": 1}, {"b": 2}]


def test_read_trajectory_raises_on_corrupt_line(tmp_path: Path):
    """Corrupt JSON lines surface loudly — silent swallow would hide corruption."""
    from harness_rlm.trajectory import read_trajectory

    session_dir = tmp_path / "rlm"
    session_dir.mkdir(parents=True)
    traj_path = session_dir / "trajectory.jsonl"
    traj_path.write_text('{"ok": 1}\nthis is not json\n')
    with pytest.raises(ValueError, match="Corrupt trajectory line 2"):
        read_trajectory(session_dir=session_dir)


def test_finalize_writes_final_txt(tmp_path: Path):
    """finalize writes the synthesized answer to FINAL.txt and returns the path."""
    from harness_rlm.trajectory import finalize

    session_dir = tmp_path / "rlm"
    answer = "The three contradictions are: A, B, C.\n"
    result = finalize(answer, session_dir=session_dir)

    assert result == session_dir / "FINAL.txt"
    assert result.read_text(encoding="utf-8") == answer


def test_finalize_creates_dir_if_missing(tmp_path: Path):
    """finalize mkdirs the session dir if absent."""
    from harness_rlm.trajectory import finalize

    session_dir = tmp_path / "brand" / "new" / "rlm"
    assert not session_dir.exists()
    path = finalize("done", session_dir=session_dir)
    assert path.exists()
    assert path.read_text() == "done"


def test_finalize_overwrites_previous(tmp_path: Path):
    """finalize overwrites a previous FINAL.txt (write, not append)."""
    from harness_rlm.trajectory import finalize

    session_dir = tmp_path / "rlm"
    finalize("first", session_dir=session_dir)
    finalize("second", session_dir=session_dir)
    assert (session_dir / "FINAL.txt").read_text() == "second"
