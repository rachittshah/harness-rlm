"""Tests for the top-level harness `run()` API (no network — patches LM)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from harness_rlm.harness import RunResult, run
from harness_rlm.llm import LMResult


@dataclass
class _StubLM:
    canned: list[str] = field(default_factory=list)
    model: str = "stub-model"
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        self.received: list[str] = []
        self.calls = 0
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def __call__(self, prompt, *, system=None, model=None, max_tokens=None):
        self.received.append(prompt)
        idx = min(self.calls, len(self.canned) - 1)
        text = self.canned[idx] if self.canned else "answer: stub"
        self.calls += 1
        self.total_calls += 1
        cost = 0.0001
        self.total_cost_usd += cost
        return LMResult(
            text=text,
            model=self.model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
            cost_usd=cost,
            latency_s=0.01,
        )


# ---------------------------------------------------------------------------
# Patch LM class so run() doesn't try to construct a real client.
# ---------------------------------------------------------------------------
@pytest.fixture
def patched_lm(monkeypatch):
    """Replace harness_rlm.harness.LM with a stub-producing factory."""
    instances: list[_StubLM] = []

    def fake_lm_ctor(model="x", **kwargs):
        stub = _StubLM(canned=["answer: dummy"], model=model)
        instances.append(stub)
        return stub

    monkeypatch.setattr("harness_rlm.harness.LM", fake_lm_ctor)
    return instances


class TestNoContext:
    def test_flat_call(self, patched_lm):
        result = run("What is 2+2?")
        assert isinstance(result, RunResult)
        assert result.answer == "dummy"
        # One root LM was constructed; sub LM wasn't called.
        assert any(inst.calls > 0 for inst in patched_lm)


class TestLongContext:
    def test_rlm_path(self, patched_lm):
        ctx = "x" * 200_000
        # When run() routes to RLM, root LM gets called only for synth (1×)
        # and sub LM gets called per chunk. We supply enough canned answers.
        # Each new LM() builds its own stub; we need to seed them.
        # The first call to LM is root, second is sub.
        # We can't easily set canned per instance from here, but the default
        # "answer: dummy" works for synth; for sub it just needs any text.

        # Reset canned text per instance — make sub LM say "partial" and root
        # say "final answer". harness.run constructs root first then sub.
        # We mutate instances after the call.
        result = run("Find the bug.", context=ctx, max_llm_calls=20)
        # Whatever the stubs returned: the result must have non-empty fields.
        assert isinstance(result, RunResult)
        assert isinstance(result.cost_usd, float)
        assert result.calls >= 1


class TestCLI:
    def test_argparse_routes(self, patched_lm, capsys, monkeypatch):
        # Force CLI args
        monkeypatch.setattr(
            "sys.argv",
            ["harness-rlm", "What is X?"],
        )
        from harness_rlm.harness import _cli

        rc = _cli()
        assert rc == 0
        captured = capsys.readouterr()
        # The CLI prints just the answer by default.
        assert captured.out.strip() != ""

    def test_argparse_json_output(self, patched_lm, capsys, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["harness-rlm", "What is X?", "--json"],
        )
        from harness_rlm.harness import _cli

        _cli()
        captured = capsys.readouterr()
        # Should be valid JSON.
        payload = json.loads(captured.out)
        assert "answer" in payload
        assert "cost_usd" in payload
        assert "calls" in payload

    def test_argparse_with_context_file(self, patched_lm, capsys, monkeypatch, tmp_path):
        ctx_file = tmp_path / "ctx.txt"
        ctx_file.write_text("some short context", encoding="utf-8")
        monkeypatch.setattr(
            "sys.argv",
            [
                "harness-rlm",
                "What is X?",
                "--context-file",
                str(ctx_file),
            ],
        )
        from harness_rlm.harness import _cli

        rc = _cli()
        assert rc == 0
