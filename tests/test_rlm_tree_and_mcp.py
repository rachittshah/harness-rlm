"""Tests for tree-recursive RLM and MCPToolset import surface."""

from __future__ import annotations

from dataclasses import dataclass, field


from harness_rlm.llm import LMResult
from harness_rlm.rlm import RLM, RLMConfig


@dataclass
class _StubLM:
    canned: list[str] = field(default_factory=list)
    model: str = "stub"
    max_tokens: int = 1024

    def __post_init__(self):
        import threading

        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, prompt, *, system=None, model=None, max_tokens=None):
        with self._lock:
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


class TestTreeRLM:
    def test_tree_strategy_runs_leaf_sub_calls(self):
        long_ctx = "x" * 100_000  # > flat_char_threshold
        # branching=4, depth=1: 4 leaves + 1 root synth.
        root = _StubLM(canned=["root synth answer"])
        sub = _StubLM(canned=["leaf A", "leaf B", "leaf C", "leaf D"])
        cfg = RLMConfig(
            strategy="tree",
            flat_char_threshold=30_000,  # 25K chunks all fit → leaves
            tree_branching=4,
            tree_max_depth=2,
            max_parallel=1,  # sequential = predictable
        )
        m = RLM(
            "question, document -> answer",
            long_context_field="document",
            config=cfg,
            root_lm=root,
            sub_lm=sub,
        )
        pred = m(question="?", document=long_ctx)
        # Sub-LM ran on each leaf (4×), root synthed once.
        assert sub.calls == 4
        assert root.calls == 1
        # Trace labels reflect tree levels.
        labels = [e.get("label", "") for e in pred.trace.events]
        leaf_count = sum(1 for label in labels if label.startswith("tree_leaf"))
        synth_count = sum(1 for label in labels if label.startswith("tree_synth"))
        assert leaf_count == 4
        assert synth_count == 1

    def test_tree_max_depth_forces_leaf(self):
        # With max_depth=0, the root call itself is a leaf — single sub-call.
        long_ctx = "x" * 100_000
        root = _StubLM(canned=[])
        sub = _StubLM(canned=["the whole answer"])
        cfg = RLMConfig(
            strategy="tree",
            flat_char_threshold=1_000,  # forces decomposition
            tree_max_depth=0,
            max_parallel=1,
        )
        m = RLM(
            "question, document -> answer",
            long_context_field="document",
            config=cfg,
            root_lm=root,
            sub_lm=sub,
        )
        m(question="?", document=long_ctx)
        # Max depth=0 hit before any branching → leaf at root.
        assert sub.calls == 1
        assert root.calls == 0

    def test_tree_with_budget_exhaustion(self):
        long_ctx = "x" * 200_000
        root = _StubLM(canned=["synth"])
        sub = _StubLM(canned=["leaf"] * 20)
        cfg = RLMConfig(
            strategy="tree",
            flat_char_threshold=30_000,
            tree_branching=4,
            tree_max_depth=3,
            max_llm_calls=2,  # very tight budget
            max_parallel=1,
        )
        m = RLM(
            "question, document -> answer",
            long_context_field="document",
            config=cfg,
            root_lm=root,
            sub_lm=sub,
        )
        pred = m(question="?", document=long_ctx)
        # Should return something — may be partial — without raising.
        assert pred.answer is not None


class TestMCPClientImport:
    def test_importable(self):
        from harness_rlm.mcp_client import MCPToolset

        assert callable(MCPToolset)

    def test_unconfigured_close_is_safe(self):
        from harness_rlm.mcp_client import MCPToolset

        ts = MCPToolset(command="false")  # never started
        ts.close()  # should not raise
