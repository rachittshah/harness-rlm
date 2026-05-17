"""Tests for streaming.py + trace_viz.py."""

from __future__ import annotations


from harness_rlm.modules import Trace
from harness_rlm.streaming import StreamEvent
from harness_rlm.trace_viz import format_trace, trace_to_mermaid


class TestStreamEvent:
    def test_constructable(self):
        evt = StreamEvent(kind="text_delta", text="hi")
        assert evt.kind == "text_delta"
        assert evt.text == "hi"

    def test_done_with_result(self):
        from harness_rlm.llm import LMResult

        r = LMResult(
            text="hi",
            model="x",
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            latency_s=0.0,
        )
        evt = StreamEvent(kind="done", result=r)
        assert evt.result is r


class TestFormatTrace:
    def test_empty_trace(self):
        t = Trace(module="X")
        out = format_trace(t)
        assert "X" in out
        assert "calls=0" in out

    def test_with_events(self):
        t = Trace(module="X", calls=2, cost_usd=0.05, latency_s=1.23)
        t.events.append({"label": "predict", "model": "haiku", "cost_usd": 0.05})
        t.events.append({"label": "tool_call", "name": "bash"})
        out = format_trace(t)
        assert "predict" in out
        assert "tool_call" in out
        assert "bash" in out

    def test_color_codes_when_enabled(self):
        t = Trace(module="X", calls=1)
        plain = format_trace(t, color=False)
        coloured = format_trace(t, color=True)
        # ANSI escape only in coloured version.
        assert "\033[" not in plain
        assert "\033[" in coloured


class TestTraceToMermaid:
    def test_basic(self):
        t = Trace(module="A", calls=1, cost_usd=0.01)
        t.events.append({"label": "predict", "cost_usd": 0.01})
        out = trace_to_mermaid(t)
        assert out.startswith("```mermaid")
        assert "flowchart TD" in out
        assert "predict" in out

    def test_includes_totals_in_end_node(self):
        t = Trace(module="A", calls=3, cost_usd=0.0123, latency_s=4.56)
        out = trace_to_mermaid(t)
        assert "calls=3" in out
        assert "$0.0123" in out
        assert "4.56s" in out

    def test_title_is_included(self):
        t = Trace(module="A")
        out = trace_to_mermaid(t, title="My Run")
        assert "%% My Run" in out

    def test_special_chars_sanitized(self):
        t = Trace(module="A/B C")
        t.events.append({"label": "step"})
        out = trace_to_mermaid(t)
        # Module id should be sanitised to alphanumeric in the event node id.
        assert "A_B_C_e1" in out
