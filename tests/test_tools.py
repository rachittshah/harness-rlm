"""Tests for harness_rlm.tools — AgentTool, builtin core, from_function."""

from __future__ import annotations

import pytest

from harness_rlm.tools import (
    BASH_TOOL,
    EDIT_TOOL,
    FINISH_TOOL,
    PI_CORE_TOOLS,
    READ_TOOL,
    WRITE_TOOL,
    AgentTool,
    ToolResult,
    from_function,
)


class TestToolResult:
    def test_text_helper(self):
        r = ToolResult.text("hi", details={"x": 1})
        assert r.content == [{"type": "text", "text": "hi"}]
        assert r.details == {"x": 1}
        assert not r.terminate
        assert r.error is None

    def test_fail_helper(self):
        r = ToolResult.fail("boom")
        assert r.error == "boom"
        assert "ERROR" in r.content[0]["text"]


class TestRegisterValidation:
    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="snake_case"):
            AgentTool(
                name="bad name!",
                description="x",
                parameters={},
                execute=lambda a: ToolResult.text(""),
            )

    def test_label_defaults_to_name(self):
        t = AgentTool(
            name="ok",
            description="x",
            parameters={},
            execute=lambda a: ToolResult.text(""),
        )
        assert t.label == "ok"


class TestBuiltinTools:
    def test_read_then_write(self, tmp_path):
        path = tmp_path / "x.txt"
        wresult = WRITE_TOOL.execute({"path": str(path), "content": "hello"})
        assert wresult.error is None
        rresult = READ_TOOL.execute({"path": str(path)})
        assert rresult.content[0]["text"] == "hello"

    def test_edit_unique_match(self, tmp_path):
        path = tmp_path / "x.txt"
        path.write_text("hello world", encoding="utf-8")
        r = EDIT_TOOL.execute({"path": str(path), "old_string": "hello", "new_string": "GREETINGS"})
        assert r.error is None
        assert path.read_text() == "GREETINGS world"

    def test_edit_replace_all(self, tmp_path):
        path = tmp_path / "x.txt"
        path.write_text("a a a", encoding="utf-8")
        r = EDIT_TOOL.execute(
            {"path": str(path), "old_string": "a", "new_string": "B", "replace_all": True}
        )
        assert r.error is None
        assert path.read_text() == "B B B"

    def test_edit_unique_match_required(self, tmp_path):
        path = tmp_path / "x.txt"
        path.write_text("a a a", encoding="utf-8")
        # Without replace_all, multiple matches must fail.
        r = EDIT_TOOL.execute({"path": str(path), "old_string": "a", "new_string": "B"})
        assert r.error and "match" in r.error

    def test_read_missing_file(self):
        r = READ_TOOL.execute({"path": "/nope/does/not/exist"})
        assert r.error and "not found" in r.error

    def test_bash_runs(self):
        r = BASH_TOOL.execute({"command": "echo hello"})
        assert r.error is None
        assert "hello" in r.content[0]["text"]

    def test_bash_nonzero_surfaced(self):
        r = BASH_TOOL.execute({"command": "exit 7"})
        assert "exit 7" in r.content[0]["text"]

    def test_finish_sets_terminate(self):
        r = FINISH_TOOL.execute({"answer": "all done"})
        assert r.terminate
        assert r.content[0]["text"] == "all done"

    def test_pi_core_count(self):
        assert len(PI_CORE_TOOLS) == 5
        names = [t.name for t in PI_CORE_TOOLS]
        for expected in ["read", "write", "edit", "bash", "finish_task"]:
            assert expected in names


class TestFromFunction:
    def test_wraps_simple(self):
        def add(a, b):
            """Add two numbers."""
            return int(a) + int(b)

        t = from_function(add)
        assert t.name == "add"
        assert "Add" in t.description
        r = t.execute({"a": "2", "b": "3"})
        assert r.content[0]["text"] == "5"
        assert r.details == 5

    def test_exceptions_become_errors(self):
        def boom():
            raise ValueError("nope")

        t = from_function(boom)
        r = t.execute({})
        assert r.error and "nope" in r.error
