"""Tests for harness_rlm.agent_loop — tool-using loop with hook seams.

We mock the Anthropic client end-to-end since the loop's value is the control
flow, not the API. The mock returns Message-shaped objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from harness_rlm.agent_loop import AgentLoop, AgentLoopConfig
from harness_rlm.tools import FINISH_TOOL, AgentTool, ToolResult


# ---------------------------------------------------------------------------
# Minimal Message shape — enough for the loop's parser.
# ---------------------------------------------------------------------------
@dataclass
class _Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = None  # type: ignore[assignment]


@dataclass
class _Usage:
    input_tokens: int = 10
    output_tokens: int = 5


@dataclass
class _Message:
    content: list[_Block]
    stop_reason: str = "end_turn"
    usage: _Usage = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.usage is None:
            self.usage = _Usage()


class _FakeMessages:
    def __init__(self, scripted: list[_Message]) -> None:
        self._scripted = scripted
        self._idx = 0
        self.received: list[dict] = []

    def create(self, **kwargs):
        self.received.append(kwargs)
        idx = min(self._idx, len(self._scripted) - 1)
        self._idx += 1
        return self._scripted[idx]


class _FakeClient:
    def __init__(self, scripted: list[_Message]) -> None:
        self.messages = _FakeMessages(scripted)


# ---------------------------------------------------------------------------
# Test scaffolding helpers
# ---------------------------------------------------------------------------
def _patch_client(monkeypatch, loop: AgentLoop, scripted: list[_Message]) -> _FakeClient:
    fake = _FakeClient(scripted)
    monkeypatch.setattr(loop, "_client", fake)
    return fake


def _text(content: str) -> _Message:
    return _Message(content=[_Block(type="text", text=content)], stop_reason="end_turn")


def _tool_call(name: str, args: dict, *, call_id: str = "c1") -> _Message:
    return _Message(
        content=[_Block(type="tool_use", id=call_id, name=name, input=args)],
        stop_reason="tool_use",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestBasicLoop:
    def test_zero_tool_call_ends_immediately(self, monkeypatch):
        loop = AgentLoop([FINISH_TOOL], AgentLoopConfig(max_turns=3))
        _patch_client(monkeypatch, loop, [_text("Hello! No tools needed.")])
        result = loop.run("Say hi.")
        assert result.turns == 1
        assert result.tool_call_count == 0
        assert result.final_text == "Hello! No tools needed."
        assert not result.terminated_by_tool

    def test_terminate_by_finish_tool(self, monkeypatch):
        loop = AgentLoop([FINISH_TOOL], AgentLoopConfig(max_turns=5))
        # Turn 1: call finish_task; loop should terminate before another LM call.
        scripted = [
            _tool_call("finish_task", {"answer": "the answer"}, call_id="c1"),
        ]
        _patch_client(monkeypatch, loop, scripted)
        result = loop.run("Finish please.")
        assert result.terminated_by_tool
        assert result.tool_call_count == 1
        assert result.final_text == "the answer"

    def test_max_turns_caps_loop(self, monkeypatch):
        # Tool returns non-terminating result; LM keeps calling tool forever.
        infinite_tool = AgentTool(
            name="ping",
            description="ping",
            parameters={"type": "object", "properties": {}},
            execute=lambda args: ToolResult.text("pong"),
        )
        loop = AgentLoop([infinite_tool, FINISH_TOOL], AgentLoopConfig(max_turns=3))
        # Script enough tool_calls to exceed max_turns.
        scripted = [_tool_call("ping", {}, call_id=f"c{i}") for i in range(10)]
        _patch_client(monkeypatch, loop, scripted)
        result = loop.run("loop forever")
        assert result.turns >= 3  # capped
        kinds = [e["kind"] for e in result.events]
        assert "max_turns_reached" in kinds


class TestHooks:
    def test_before_tool_call_can_block(self, monkeypatch):
        blocked_calls: list[str] = []

        def deny(name, args):
            blocked_calls.append(name)
            return True  # block

        loop = AgentLoop(
            [FINISH_TOOL],
            AgentLoopConfig(max_turns=3, before_tool_call=deny),
        )
        scripted = [
            _tool_call("finish_task", {"answer": "denied?"}),
            _text("OK, I'll stop."),  # follow-up LM call after tool blocked
        ]
        _patch_client(monkeypatch, loop, scripted)
        result = loop.run("try to finish")
        assert "finish_task" in blocked_calls
        # The blocked finish_task does NOT set terminate=True (it errored),
        # so the loop made another LM call.
        assert result.turns >= 2

    def test_after_tool_call_can_rewrite(self, monkeypatch):
        def rewrite(name, args, result):
            return ToolResult.text("REWRITTEN", terminate=True)

        loop = AgentLoop(
            [FINISH_TOOL],
            AgentLoopConfig(max_turns=3, after_tool_call=rewrite),
        )
        scripted = [_tool_call("finish_task", {"answer": "x"})]
        _patch_client(monkeypatch, loop, scripted)
        result = loop.run("finish")
        # `after_tool_call` rewrote the result to terminate=True with "REWRITTEN".
        assert result.terminated_by_tool
        assert result.final_text == "REWRITTEN"

    def test_should_stop_after_turn_ends_loop(self, monkeypatch):
        ping = AgentTool(
            name="ping",
            description="ping",
            parameters={"type": "object", "properties": {}},
            execute=lambda args: ToolResult.text("pong"),
        )
        loop = AgentLoop(
            [ping, FINISH_TOOL],
            AgentLoopConfig(
                max_turns=10,
                should_stop_after_turn=lambda ctx: ctx["turns"] >= 1,
            ),
        )
        scripted = [_tool_call("ping", {})]
        _patch_client(monkeypatch, loop, scripted)
        result = loop.run("go")
        assert result.turns == 1


class TestValidation:
    def test_no_tools_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            AgentLoop([])

    def test_duplicate_tool_names_raise(self):
        t1 = AgentTool(
            name="x", description="x", parameters={}, execute=lambda a: ToolResult.text("")
        )
        t2 = AgentTool(
            name="x", description="y", parameters={}, execute=lambda a: ToolResult.text("")
        )
        with pytest.raises(ValueError, match="unique"):
            AgentLoop([t1, t2])
