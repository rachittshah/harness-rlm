"""Tests for the expanded MCP server tool surface.

Each handler is called directly (skipping the asyncio dispatcher) so the test
suite stays sync + fast. LM-touching handlers monkeypatch the LM class so we
never hit the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from harness_rlm.llm import LMResult
from harness_rlm.mcp_server import (
    _TOOLS,
    _handle_best_of_n,
    _handle_chain_of_thought,
    _handle_chunk_text,
    _handle_compress_text,
    _handle_dispatch_subagent,
    _handle_estimate_cost,
    _handle_list_subagents,
    _handle_predict,
    _handle_rlm_run,
)


@dataclass
class _StubLM:
    canned: list[str] = field(default_factory=list)
    model: str = "stub"
    max_tokens: int = 1024

    def __post_init__(self):
        import threading

        self.calls = 0
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self._lock = threading.Lock()

    def __call__(self, prompt, *, system=None, model=None, max_tokens=None, **kw):
        with self._lock:
            idx = min(self.calls, len(self.canned) - 1)
            text = self.canned[idx] if self.canned else "answer: stub"
            self.calls += 1
            self.total_calls += 1
            self.total_cost_usd += 0.0001
        return LMResult(
            text=text,
            model=self.model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
            cost_usd=0.0001,
            latency_s=0.01,
        )


def _patch_lm(monkeypatch, stub: _StubLM) -> None:
    """Replace harness_rlm.llm.LM with a factory that returns the stub."""
    monkeypatch.setattr("harness_rlm.llm.LM", lambda *a, **kw: stub)


# ---------------------------------------------------------------------------
# Registry / metadata
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_has_all_ten_tools(self):
        expected = {
            "llm_query",
            "rlm_run",
            "predict",
            "chain_of_thought",
            "best_of_n",
            "compress_text",
            "chunk_text",
            "dispatch_subagent",
            "list_subagents",
            "estimate_cost",
        }
        assert expected == set(_TOOLS.keys())

    def test_every_tool_has_required_fields(self):
        for name, spec in _TOOLS.items():
            assert spec["name"] == name
            assert spec["description"]
            assert isinstance(spec["input_schema"], dict)
            assert spec["input_schema"]["type"] == "object"
            assert callable(spec["handler"])


# ---------------------------------------------------------------------------
# Pure (no LM) handlers
# ---------------------------------------------------------------------------
class TestChunkText:
    def test_basic(self):
        out = _handle_chunk_text({"text": "abcdefghij" * 100, "chunk_size": 100, "overlap": 0})
        assert out["count"] == 10
        assert all(len(c) == 100 for c in out["chunks"])

    def test_short_text_one_chunk(self):
        out = _handle_chunk_text({"text": "short", "chunk_size": 1000, "overlap": 100})
        assert out["count"] == 1
        assert out["chunks"][0] == "short"

    def test_empty(self):
        out = _handle_chunk_text({"text": "", "chunk_size": 1000, "overlap": 100})
        assert out["count"] == 0
        assert out["chunks"] == []


class TestEstimateCost:
    def test_haiku(self):
        out = _handle_estimate_cost(
            {"model": "claude-haiku-4-5", "input_tokens": 1_000_000, "output_tokens": 1_000_000}
        )
        assert out["rate_input_per_million"] == pytest.approx(1.0)
        assert out["rate_output_per_million"] == pytest.approx(5.0)
        assert out["cost_total_usd"] == pytest.approx(6.0)

    def test_opus(self):
        out = _handle_estimate_cost(
            {"model": "claude-opus-4-7", "input_tokens": 1_000_000, "output_tokens": 0}
        )
        assert out["cost_total_usd"] == pytest.approx(5.0)

    def test_cached_savings(self):
        out = _handle_estimate_cost(
            {
                "model": "claude-opus-4-7",
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "cached_input_tokens": 1_000_000,
            }
        )
        # 0.9 of input rate saved on cached tokens.
        assert out["cache_savings_usd"] == pytest.approx(4.5)
        assert out["cost_total_usd"] == pytest.approx(0.5)


class TestListSubagents:
    def test_returns_list(self):
        out = _handle_list_subagents({})
        assert "subagents" in out
        assert "count" in out
        assert isinstance(out["subagents"], list)


# ---------------------------------------------------------------------------
# LM-touching handlers (stubbed)
# ---------------------------------------------------------------------------
class TestPredict:
    def test_happy_path(self, monkeypatch):
        stub = _StubLM(canned=["answer: forty-two"])
        _patch_lm(monkeypatch, stub)
        out = _handle_predict({"signature": "question -> answer", "inputs": {"question": "?"}})
        assert out["fields"]["answer"] == "forty-two"
        assert out["cost_usd"] > 0

    def test_with_instruction(self, monkeypatch):
        stub = _StubLM(canned=["answer: brief"])
        _patch_lm(monkeypatch, stub)
        out = _handle_predict(
            {
                "signature": "question -> answer",
                "inputs": {"question": "?"},
                "instruction": "Be terse.",
            }
        )
        assert out["fields"]["answer"] == "brief"


class TestChainOfThought:
    def test_with_reasoning(self, monkeypatch):
        stub = _StubLM(canned=["reasoning: think\nanswer: 42"])
        _patch_lm(monkeypatch, stub)
        out = _handle_chain_of_thought({"signature": "q -> answer", "inputs": {"q": "?"}})
        assert out["fields"]["reasoning"] == "think"
        assert out["fields"]["answer"] == "42"


class TestRlmRun:
    def test_long_doc_decomposition(self, monkeypatch):
        # 200K-char doc → forces decomposition. Same stub serves root + sub.
        stub = _StubLM(canned=["NOT_FOUND", "the date is 2027", "NOT_FOUND", "synth answer"])
        _patch_lm(monkeypatch, stub)
        out = _handle_rlm_run(
            {
                "question": "what date?",
                "document": "x" * 200_000,
                "chunk_size": 80_000,
                "max_parallel": 1,
                "max_llm_calls": 10,
            }
        )
        # Some answer was produced + accounting fields.
        assert "answer" in out
        assert out["calls"] >= 1
        assert out["cost_usd"] >= 0

    def test_short_doc_flat_path(self, monkeypatch):
        stub = _StubLM(canned=["answer: short"])
        _patch_lm(monkeypatch, stub)
        out = _handle_rlm_run({"question": "?", "document": "short text", "max_llm_calls": 5})
        # Flat path → 1 call.
        assert out["calls"] == 1
        assert out["answer"] == "short"


class TestBestOfN:
    def test_self_consistency_path(self, monkeypatch):
        stub = _StubLM(
            canned=[
                "reasoning: a\nanswer: 42",
                "reasoning: b\nanswer: 42",
                "reasoning: c\nanswer: 99",
            ]
        )
        _patch_lm(monkeypatch, stub)
        out = _handle_best_of_n(
            {
                "signature": "q -> answer",
                "inputs": {"q": "?"},
                "n": 3,
                "chain_of_thought": True,
                "max_parallel": 1,
            }
        )
        # Majority of "42" wins.
        assert out["fields"]["answer"] == "42"
        assert out["calls"] == 3
        # Vote distribution surfaces the breakdown.
        assert isinstance(out["vote_distribution"], list)

    def test_predict_path(self, monkeypatch):
        stub = _StubLM(canned=["answer: A", "answer: A", "answer: B"])
        _patch_lm(monkeypatch, stub)
        out = _handle_best_of_n(
            {
                "signature": "q -> answer",
                "inputs": {"q": "?"},
                "n": 3,
                "chain_of_thought": False,
                "max_parallel": 1,
            }
        )
        assert out["fields"]["answer"] == "A"


class TestCompressText:
    def test_compresses_long_text(self, monkeypatch):
        stub = _StubLM(canned=["summarized text"])
        _patch_lm(monkeypatch, stub)
        out = _handle_compress_text({"text": "x" * 10_000, "target_chars": 500})
        assert out["summary"] == "summarized text"
        assert out["original_chars"] == 10_000
        assert out["summary_chars"] == len("summarized text")
        assert out["compression_ratio"] > 1

    def test_short_passthrough(self, monkeypatch):
        stub = _StubLM(canned=["should not be called"])
        _patch_lm(monkeypatch, stub)
        out = _handle_compress_text({"text": "short", "target_chars": 1000})
        # No LM call — passthrough.
        assert out["summary"] == "short"
        assert stub.calls == 0


# ---------------------------------------------------------------------------
# dispatch_subagent error paths
# ---------------------------------------------------------------------------
class TestDispatchSubagent:
    def test_unknown_spec_returns_error(self):
        out = _handle_dispatch_subagent(
            {"spec_name": "this-name-does-not-exist-12345", "task": "test"}
        )
        assert "error" in out
        assert "Unknown subagent" in out["error"]
