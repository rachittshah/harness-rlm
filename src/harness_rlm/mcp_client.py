"""Wrap an external MCP server as a set of AgentTools.

The reverse of `mcp_server.py`: instead of exposing OUR functions to an MCP
client, we connect to an existing MCP server (over stdio) and expose ITS
tools as `AgentTool` instances that any `AgentLoop` can use.

Usage:
    from harness_rlm import MCPToolset, AgentLoop, FINISH_TOOL

    toolset = MCPToolset(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    )
    try:
        with toolset:
            tools = toolset.tools()                  # list[AgentTool]
            loop = AgentLoop([*tools, FINISH_TOOL])
            result = loop.run("List the files in the directory.")
    finally:
        toolset.close()

Or as a context manager:
    with MCPToolset(command="...", args=[...]) as ts:
        loop = AgentLoop([*ts.tools(), FINISH_TOOL])
        ...

Each MCP tool's input_schema is preserved verbatim. The tool's `execute`
function makes the MCP call and returns a `ToolResult` with the raw text
content for the LM and the structured payload as `details`.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Any

from harness_rlm.tools import AgentTool, ToolResult


@dataclass
class MCPToolset:
    """Connection to one stdio MCP server.

    The toolset owns a background asyncio event loop running in a thread; each
    tool call posts a coroutine to that loop and waits for the result. This
    lets sync `AgentLoop` use async MCP tools without colour-splitting the
    whole codebase.

    Args:
        command:    Binary to spawn (e.g. "npx", "uv", "node").
        args:       Argv tail (e.g. ["-y", "@modelcontextprotocol/server-fs"]).
        env:        Optional env vars for the spawned process.
        name_prefix: If multiple toolsets are loaded, prefix each tool name
                     to disambiguate (e.g. "fs__read_file").
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    name_prefix: str = ""
    _thread: threading.Thread | None = None
    _loop: asyncio.AbstractEventLoop | None = None
    _ready: threading.Event = field(default_factory=threading.Event)
    _session: Any = None
    _exit_stack: Any = None
    _tools_cached: list[dict[str, Any]] = field(default_factory=list)

    def __enter__(self) -> "MCPToolset":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def start(self) -> None:
        """Boot the background event loop + spawn the MCP server process."""
        if self._thread is not None:
            return

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._init_session())
            self._ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run, daemon=True, name="mcp-toolset")
        self._thread.start()
        # Wait up to 30s for the MCP server to come up.
        if not self._ready.wait(timeout=30):
            raise RuntimeError(
                f"MCP server {self.command!r} did not initialise within 30s."
            )

    async def _init_session(self) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.command,
            args=list(self.args),
            env=self.env,
        )
        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        # Cache the tool list once at startup; MCP servers can refresh it but
        # for our wrapper we treat the initial set as canonical.
        listing = await self._session.list_tools()
        self._tools_cached = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema or {"type": "object", "properties": {}},
            }
            for t in listing.tools
        ]

    def close(self) -> None:
        """Shut down the MCP server + event loop."""
        if self._loop is None:
            return

        async def _cleanup() -> None:
            if self._exit_stack is not None:
                await self._exit_stack.aclose()

        try:
            fut = asyncio.run_coroutine_threadsafe(_cleanup(), self._loop)
            fut.result(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None
        self._loop = None

    def tools(self) -> list[AgentTool]:
        """Return the MCP server's tools as AgentTool instances."""
        if not self._tools_cached:
            raise RuntimeError(
                "MCPToolset not started — call .start() or use as context manager."
            )
        return [self._wrap(spec) for spec in self._tools_cached]

    # ---- internal --------------------------------------------------------
    def _wrap(self, spec: dict[str, Any]) -> AgentTool:
        """Build an AgentTool whose execute() posts a coroutine to our loop."""
        tool_name = spec["name"]
        exposed_name = f"{self.name_prefix}{tool_name}" if self.name_prefix else tool_name

        def execute(args: dict[str, Any]) -> ToolResult:
            assert self._loop is not None
            fut = asyncio.run_coroutine_threadsafe(
                self._call_tool(tool_name, args), self._loop
            )
            try:
                content_blocks, structured = fut.result(timeout=120)
            except Exception as e:  # noqa: BLE001
                return ToolResult.fail(f"{type(e).__name__}: {e}")

            # Convert MCP content blocks (TextContent, ImageContent, etc.) to
            # the harness ToolContent shape: [{"type": "text", "text": "..."}, ...].
            converted: list[dict[str, Any]] = []
            for blk in content_blocks:
                btype = getattr(blk, "type", "text")
                if btype == "text":
                    converted.append({"type": "text", "text": getattr(blk, "text", "")})
                elif btype == "image":
                    converted.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": getattr(blk, "mimeType", "image/png"),
                                "data": getattr(blk, "data", ""),
                            },
                        }
                    )
                else:
                    # Unknown block — stringify defensively.
                    converted.append({"type": "text", "text": json.dumps(blk, default=str)})

            return ToolResult(
                content=converted or [{"type": "text", "text": ""}],
                details=structured,
            )

        return AgentTool(
            name=exposed_name,
            description=spec["description"],
            parameters=spec["input_schema"],
            execute=execute,
        )

    async def _call_tool(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[list[Any], Any]:
        assert self._session is not None
        result = await self._session.call_tool(tool_name, args)
        # MCP call_tool returns a CallToolResult with `content` list.
        content = list(getattr(result, "content", []))
        structured = getattr(result, "structuredContent", None)
        return content, structured


__all__ = ["MCPToolset"]
