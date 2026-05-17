"""Pi-style `AgentTool` interface — typed tools with content/details split.

Adapted from pi-mono (Mario Zechner). Two split channels:

  content   — what the LLM sees (text or image)
  details   — what the trace / UI sees (arbitrary structured payload)

A tool can also set `terminate=True` to end the agent loop without another LM
turn (think `finish_task`).

Tools register their JSON Schema for parameters via the `parameters` dict so
the agent loop can validate inputs and surface them to the LM.
"""

from __future__ import annotations

import inspect
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

ToolContent = list[dict[str, Any]]  # [{"type": "text", "text": "..."}]
ExecutionMode = Literal["sequential", "parallel"]


@dataclass
class ToolResult:
    """Return value of a tool. `content` goes to the LLM; `details` to the UI."""

    content: ToolContent
    details: Any = None
    terminate: bool = False
    error: str | None = None

    @classmethod
    def text(cls, text: str, *, details: Any = None, terminate: bool = False) -> ToolResult:
        return cls(
            content=[{"type": "text", "text": text}],
            details=details,
            terminate=terminate,
        )

    @classmethod
    def fail(cls, error: str) -> ToolResult:
        return cls(
            content=[{"type": "text", "text": f"ERROR: {error}"}],
            error=error,
        )


@dataclass
class AgentTool:
    """One callable tool exposed to the agent loop.

    Attributes:
        name:        Stable identifier (snake_case). Used by the LM to invoke.
        description: Short one-liner. The LM reads this to decide when to use.
        parameters:  JSON Schema for the tool's args. Mirrors the OpenAI/Anthropic
                     tool-use contract.
        execute:     fn(args: dict) -> ToolResult. Synchronous; the loop runs
                     parallel calls in threads when execution_mode='parallel'.
        execution_mode: "sequential" (default — wait for each) or "parallel".
        label:       Human-readable label for trace UI. Defaults to `name`.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    execute: Callable[[dict[str, Any]], ToolResult]
    execution_mode: ExecutionMode = "sequential"
    label: str | None = None

    def __post_init__(self) -> None:
        if not self.name.replace("_", "").isalnum():
            raise ValueError(f"Tool name must be snake_case alphanumeric (got {self.name!r})")
        if self.label is None:
            self.label = self.name


# ---------------------------------------------------------------------------
# Pi-style minimal 4-tool core: read / write / edit / bash.
# Plus a 5th: finish_task (terminates the loop).
# ---------------------------------------------------------------------------
def _read_tool(args: dict) -> ToolResult:
    path = Path(args["path"]).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ToolResult.fail(f"file not found: {path}")
    except PermissionError:
        return ToolResult.fail(f"permission denied: {path}")
    # Cap at 100K chars to keep the agent loop's context manageable.
    if len(text) > 100_000:
        text = text[:100_000] + f"\n\n[truncated, full {len(text):,} chars]"
    return ToolResult.text(text, details={"path": str(path), "chars": len(text)})


def _write_tool(args: dict) -> ToolResult:
    path = Path(args["path"]).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    return ToolResult.text(
        f"wrote {len(args['content'])} chars to {path}",
        details={"path": str(path), "chars": len(args["content"])},
    )


def _edit_tool(args: dict) -> ToolResult:
    path = Path(args["path"]).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ToolResult.fail(f"file not found: {path}")
    old, new = args["old_string"], args["new_string"]
    occurrences = text.count(old)
    if occurrences == 0:
        return ToolResult.fail(f"old_string not found in {path}. Tool calls must be exact.")
    if occurrences > 1 and not args.get("replace_all"):
        return ToolResult.fail(
            f"old_string matches {occurrences} times — pass replace_all=true or expand context."
        )
    new_text = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
    path.write_text(new_text, encoding="utf-8")
    return ToolResult.text(
        f"edited {path} ({occurrences} replacement{'s' if occurrences > 1 else ''})",
        details={"path": str(path), "occurrences": occurrences},
    )


def _bash_tool(args: dict) -> ToolResult:
    cmd = args["command"]
    timeout = int(args.get("timeout_s", 60))
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.fail(f"command timed out after {timeout}s")
    stdout = proc.stdout[-20_000:]  # cap output
    stderr = proc.stderr[-2_000:]
    summary = stdout
    if stderr:
        summary += f"\n--- stderr ---\n{stderr}"
    if proc.returncode != 0:
        summary = f"[exit {proc.returncode}]\n{summary}"
    return ToolResult.text(
        summary or f"[exit {proc.returncode}, no output]",
        details={
            "exit_code": proc.returncode,
            "stdout_chars": len(proc.stdout),
            "stderr_chars": len(proc.stderr),
        },
    )


def _finish_tool(args: dict) -> ToolResult:
    """Terminate the agent loop with a final answer. Pi-style `terminate=True`."""
    answer = args.get("answer", "")
    return ToolResult.text(answer, details={"answer": answer}, terminate=True)


# Concrete instances — opinionated defaults. Override / extend at registration time.
READ_TOOL = AgentTool(
    name="read",
    description="Read a file from disk and return its text content.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or ~/-expanded path."},
        },
        "required": ["path"],
    },
    execute=_read_tool,
)

WRITE_TOOL = AgentTool(
    name="write",
    description="Write text content to a file (creates parent dirs; overwrites existing).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
    execute=_write_tool,
)

EDIT_TOOL = AgentTool(
    name="edit",
    description="Replace exact text in a file. old_string must match uniquely (or pass replace_all=true).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
    },
    execute=_edit_tool,
)

BASH_TOOL = AgentTool(
    name="bash",
    description="Run a shell command. Output is truncated to 20K chars; default timeout 60s.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout_s": {"type": "integer", "default": 60, "minimum": 1, "maximum": 600},
        },
        "required": ["command"],
    },
    execute=_bash_tool,
)

FINISH_TOOL = AgentTool(
    name="finish_task",
    description="Emit the final answer and end the agent loop. Use when you are confident.",
    parameters={
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "The final answer to return."},
        },
        "required": ["answer"],
    },
    execute=_finish_tool,
)

# Pi-style 4-tool core + finish.
PI_CORE_TOOLS: list[AgentTool] = [READ_TOOL, WRITE_TOOL, EDIT_TOOL, BASH_TOOL, FINISH_TOOL]


# ---------------------------------------------------------------------------
# Helpers for callers building tools from plain Python functions.
# ---------------------------------------------------------------------------
def from_function(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> AgentTool:
    """Wrap a Python function as an AgentTool.

    The function's docstring becomes the description if `description` is None.
    The function's return value becomes ToolResult.text(str(return), details=return).
    """
    sig = inspect.signature(fn)
    derived_params = parameters or {
        "type": "object",
        "properties": {p: {"type": "string"} for p in sig.parameters.keys()},
        "required": [p for p, v in sig.parameters.items() if v.default is inspect.Parameter.empty],
    }

    def _execute(args: dict[str, Any]) -> ToolResult:
        try:
            result = fn(**args)
            return ToolResult.text(str(result), details=result)
        except Exception as e:  # noqa: BLE001
            return ToolResult.fail(f"{type(e).__name__}: {e}")

    doc_lines = (inspect.getdoc(fn) or "").splitlines()
    derived_desc = description or (doc_lines[0] if doc_lines else fn.__name__)
    return AgentTool(
        name=name or fn.__name__,
        description=derived_desc,
        parameters=derived_params,
        execute=_execute,
    )


__all__ = [
    "AgentTool",
    "ToolResult",
    "ToolContent",
    "ExecutionMode",
    "READ_TOOL",
    "WRITE_TOOL",
    "EDIT_TOOL",
    "BASH_TOOL",
    "FINISH_TOOL",
    "PI_CORE_TOOLS",
    "from_function",
]
