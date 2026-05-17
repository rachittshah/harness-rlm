"""Tests for harness_rlm.batch — CSV batch dispatch."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from harness_rlm.batch import BatchResult, spawn_agents_on_csv
from harness_rlm.llm import LMResult
from harness_rlm.modules import Predict


@dataclass
class _StubLM:
    canned: list[str] = field(default_factory=list)
    model: str = "stub-model"
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt, *, system=None, model=None, max_tokens=None):
        idx = min(self.calls, len(self.canned) - 1)
        text = self.canned[idx] if self.canned else "answer: stub"
        self.calls += 1
        return LMResult(
            text=text,
            model=self.model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
            cost_usd=0.0001,
            latency_s=0.01,
        )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestBasicDispatch:
    def test_happy_path(self, tmp_path):
        in_csv = tmp_path / "in.csv"
        out_csv = tmp_path / "out.csv"
        _write_csv(
            in_csv,
            [
                {"id": "a", "question": "q1"},
                {"id": "b", "question": "q2"},
                {"id": "c", "question": "q3"},
            ],
        )
        # Stub returns different answers per call.
        lm = _StubLM(canned=["answer: r1", "answer: r2", "answer: r3"])
        module = Predict("question -> answer", lm=lm)  # type: ignore[arg-type]

        result = spawn_agents_on_csv(
            in_csv,
            module=module,
            input_template={"question": "{question}"},
            output_csv_path=out_csv,
            max_parallel=1,  # deterministic ordering of stub canned
        )
        assert isinstance(result, BatchResult)
        assert result.total == 3
        assert result.ok == 3
        assert result.err == 0
        # Output CSV exists and has the right shape.
        with out_csv.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        statuses = {r["status"] for r in rows}
        assert statuses == {"ok"}
        # result_json contains the predicted fields.
        for r in rows:
            payload = json.loads(r["result_json"])
            assert "answer" in payload

    def test_template_substitution(self, tmp_path):
        in_csv = tmp_path / "in.csv"
        _write_csv(
            in_csv,
            [{"id": "a", "name": "alice", "lang": "py"}],
        )
        lm = _StubLM(canned=["answer: ok"])
        module = Predict("question -> answer", lm=lm)  # type: ignore[arg-type]
        result = spawn_agents_on_csv(
            in_csv,
            module=module,
            input_template={"question": "Translate {name} to {lang}"},
            output_csv_path=tmp_path / "out.csv",
            max_parallel=1,
        )
        assert result.ok == 1


class TestErrorHandling:
    def test_template_keyerror_marks_row_error(self, tmp_path):
        in_csv = tmp_path / "in.csv"
        _write_csv(in_csv, [{"id": "a", "x": "1"}])
        lm = _StubLM(canned=["answer: ok"])
        module = Predict("question -> answer", lm=lm)  # type: ignore[arg-type]
        result = spawn_agents_on_csv(
            in_csv,
            module=module,
            input_template={"question": "{nonexistent_column}"},
            output_csv_path=tmp_path / "out.csv",
        )
        assert result.err == 1
        assert "KeyError" in result.rows[0].last_error

    def test_module_exception_retries_then_fails(self, tmp_path):
        in_csv = tmp_path / "in.csv"
        _write_csv(in_csv, [{"id": "a", "question": "q"}])
        # 3 canned "bad" responses → all retries fail.
        lm = _StubLM(canned=["bad without label"] * 5)
        module = Predict("question -> answer", lm=lm)  # type: ignore[arg-type]
        result = spawn_agents_on_csv(
            in_csv,
            module=module,
            input_template={"question": "{question}"},
            output_csv_path=tmp_path / "out.csv",
            max_retries=2,  # 1 initial + 2 retries = 3 attempts
        )
        assert result.err == 1
        # Stub was called 3 times (1 + 2 retries).
        assert lm.calls == 3

    def test_missing_id_column_raises(self, tmp_path):
        in_csv = tmp_path / "in.csv"
        _write_csv(in_csv, [{"question": "q"}])  # no `id`
        lm = _StubLM(canned=["answer: ok"])
        module = Predict("question -> answer", lm=lm)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="not in CSV header"):
            spawn_agents_on_csv(
                in_csv,
                module=module,
                input_template={"question": "{question}"},
                output_csv_path=tmp_path / "out.csv",
            )

    def test_missing_input_csv_raises(self, tmp_path):
        lm = _StubLM(canned=["answer: ok"])
        module = Predict("question -> answer", lm=lm)  # type: ignore[arg-type]
        with pytest.raises(FileNotFoundError):
            spawn_agents_on_csv(
                tmp_path / "nope.csv",
                module=module,
                input_template={"question": "{question}"},
                output_csv_path=tmp_path / "out.csv",
            )


class TestSchema:
    def test_schema_missing_required_marks_error(self, tmp_path):
        in_csv = tmp_path / "in.csv"
        _write_csv(in_csv, [{"id": "a", "question": "q"}])
        lm = _StubLM(canned=["answer: ok"])
        module = Predict("question -> answer", lm=lm)  # type: ignore[arg-type]
        result = spawn_agents_on_csv(
            in_csv,
            module=module,
            input_template={"question": "{question}"},
            output_csv_path=tmp_path / "out.csv",
            output_schema={"required": ["answer", "fingerprint"]},
        )
        # The Module returns {answer: ok} so `fingerprint` is missing.
        assert result.err == 1
        assert "fingerprint" in result.rows[0].last_error


class TestProgress:
    def test_progress_callback_fires_per_row(self, tmp_path):
        in_csv = tmp_path / "in.csv"
        _write_csv(
            in_csv,
            [{"id": "a", "question": "q"}, {"id": "b", "question": "q"}],
        )
        lm = _StubLM(canned=["answer: ok"])
        module = Predict("question -> answer", lm=lm)  # type: ignore[arg-type]
        seen: list[str] = []
        spawn_agents_on_csv(
            in_csv,
            module=module,
            input_template={"question": "{question}"},
            output_csv_path=tmp_path / "out.csv",
            on_progress=lambda r: seen.append(r.item_id),
        )
        assert sorted(seen) == ["a", "b"]
