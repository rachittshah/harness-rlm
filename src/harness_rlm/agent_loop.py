"""Pi-style agent loop with hook seams.

Implements the loop documented in pi-mono's `agent-loop.ts` at the conceptual
level: one outer loop, one inner turn loop, five hook seams. Anthropic's
tool-use is the transport; tools are `AgentTool` instances.

Loop semantics:

    while True (outer):
        while True (inner):
            steering = getSteeringMessages()  if any -> append
            msg = call LM with tool schemas
            if msg has no tool_calls: break inner
            run tool_calls -> tool_results
            if any tool returned terminate=True: end outer
            if shouldStopAfterTurn(ctx): end outer
            prepareNextTurn(ctx) -> may swap model / instruction
        followUps = getFollowUpMessages()  if none: end outer
        append(followUps)

All five hooks are optional. The default loop with no hooks is the simplest
useful tool-using agent (~50 LOC of real work). Hooks let callers plug in:

    transformContext   — compaction before each LM call
    shouldStopAfterTurn — early-stop on a custom condition
    prepareNextTurn    — swap model / instruction mid-run
    beforeToolCall     — veto a tool call
    afterToolCall      — rewrite a tool result

Note: this requires anthropic SDK tool-use support. Stub LMs for tests can
emit a synthetic Message-shaped object via `_StubMessage`.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable

import anthropic

from harness_rlm.tools import AgentTool, ToolResult


@dataclass
class AgentLoopConfig:
    """Knobs + hook seams for the agent loop.

    All hooks are optional. Defaults run a straightforward tool-use loop.
    """

    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 4096
    max_turns: int = 20
    system: str | None = None
    # Hooks (Pi-style)
    transform_context: Callable[[list[dict]], list[dict]] | None = None
    should_stop_after_turn: Callable[[dict], bool] | None = None
    prepare_next_turn: Callable[[dict], dict | None] | None = None
    before_tool_call: Callable[[str, dict], bool] | None = None  # return True to block
    after_tool_call: Callable[[str, dict, ToolResult], ToolResult] | None = None
    get_steering_messages: Callable[[], list[dict]] | None = None
    get_follow_up_messages: Callable[[], list[dict]] | None = None


@dataclass
class AgentLoopResult:
    """Output of one agent loop run."""

    final_text: str
    messages: list[dict]
    turns: int
    terminated_by_tool: bool
    tool_call_count: int = 0
    cost_usd: float = 0.0
    events: list[dict] = field(default_factory=list)


class AgentLoop:
    """Tool-using agent loop with Pi-style hook seams."""

    def __init__(
        self,
        tools: list[AgentTool],
        config: AgentLoopConfig | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        if not tools:
            raise ValueError("AgentLoop needs at least one tool.")
        names = [t.name for t in tools]
        if len(set(names)) != len(names):
            raise ValueError(f"Tool names must be unique. Got {names}.")
        self.tools = {t.name: t for t in tools}
        self.config = config or AgentLoopConfig()
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # ---- public ----------------------------------------------------------
    def run(self, user_message: str) -> AgentLoopResult:
        cfg = self.config
        messages: list[dict] = [{"role": "user", "content": user_message}]
        tool_schemas = self._tool_schemas()
        events: list[dict] = [{"kind": "agent_start", "tool_count": len(self.tools)}]
        turns = 0
        terminated = False
        tool_calls_total = 0
        cost_total = 0.0
        final_text = ""

        outer_done = False
        while not outer_done:
            # Inner loop — one turn = one LM call + zero or more tool calls.
            while True:
                turns += 1
                if turns > cfg.max_turns:
                    events.append({"kind": "max_turns_reached"})
                    outer_done = True
                    break

                # Steering: pull pending user input.
                if cfg.get_steering_messages:
                    steering = cfg.get_steering_messages()
                    if steering:
                        messages.extend(steering)
                        events.append({"kind": "steering_added", "n": len(steering)})

                # Transform context (compaction etc.) before LM call.
                send_messages = messages
                if cfg.transform_context:
                    send_messages = cfg.transform_context(messages)

                msg = self._client.messages.create(
                    model=cfg.model,
                    max_tokens=cfg.max_tokens,
                    system=cfg.system or "",
                    messages=send_messages,
                    tools=tool_schemas,
                )
                cost_total += _approx_cost(cfg.model, msg.usage.input_tokens, msg.usage.output_tokens)

                # Capture the assistant message verbatim for the next turn.
                content_blocks: list[dict] = []
                text_blocks: list[str] = []
                tool_calls: list[dict] = []
                for blk in msg.content:
                    if blk.type == "text":
                        content_blocks.append({"type": "text", "text": blk.text})
                        text_blocks.append(blk.text)
                    elif blk.type == "tool_use":
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": blk.id,
                                "name": blk.name,
                                "input": blk.input,
                            }
                        )
                        tool_calls.append({"id": blk.id, "name": blk.name, "input": blk.input})
                messages.append({"role": "assistant", "content": content_blocks})
                events.append(
                    {
                        "kind": "turn",
                        "turn": turns,
                        "text_chars": sum(len(t) for t in text_blocks),
                        "tool_calls": len(tool_calls),
                        "stop_reason": msg.stop_reason,
                    }
                )
                final_text = "\n".join(text_blocks) if text_blocks else final_text

                if not tool_calls:
                    # Model is done with this turn — exit inner.
                    break

                # Execute tools (parallel if any tool requests it; else sequential).
                tool_results = self._execute_tools(tool_calls, events)
                tool_calls_total += len(tool_calls)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tc["id"],
                                "content": tr.content,
                                "is_error": bool(tr.error),
                            }
                            for tc, tr in zip(tool_calls, tool_results)
                        ],
                    }
                )

                # Did any tool ask to terminate?
                if any(tr.terminate for tr in tool_results):
                    terminated = True
                    outer_done = True
                    # Carry the terminating tool's content as final_text.
                    for tc, tr in zip(tool_calls, tool_results):
                        if tr.terminate:
                            final_text = _first_text(tr.content) or final_text
                    break

                # Should we stop after this turn?
                if cfg.should_stop_after_turn:
                    ctx_snapshot = {
                        "messages": messages,
                        "turns": turns,
                        "events": events,
                    }
                    if cfg.should_stop_after_turn(ctx_snapshot):
                        outer_done = True
                        break

                # Apply mid-run reconfigurations (model swap, etc.).
                if cfg.prepare_next_turn:
                    update = cfg.prepare_next_turn(
                        {"messages": messages, "turns": turns, "events": events}
                    )
                    if update:
                        if "model" in update:
                            cfg.model = update["model"]
                        if "system" in update:
                            cfg.system = update["system"]
                        events.append({"kind": "turn_update", "applied": list(update.keys())})

            # Outer-loop follow-ups (e.g. queued user replies).
            if not outer_done and cfg.get_follow_up_messages:
                follow = cfg.get_follow_up_messages()
                if follow:
                    messages.extend(follow)
                    events.append({"kind": "followups_added", "n": len(follow)})
                else:
                    outer_done = True
            elif not outer_done:
                outer_done = True

        events.append(
            {
                "kind": "agent_end",
                "terminated_by_tool": terminated,
                "tool_calls": tool_calls_total,
            }
        )
        return AgentLoopResult(
            final_text=final_text.strip(),
            messages=messages,
            turns=turns,
            terminated_by_tool=terminated,
            tool_call_count=tool_calls_total,
            cost_usd=round(cost_total, 6),
            events=events,
        )

    # ---- helpers ---------------------------------------------------------
    def _tool_schemas(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in self.tools.values()
        ]

    def _execute_tools(
        self, tool_calls: list[dict], events: list[dict]
    ) -> list[ToolResult]:
        any_parallel = any(
            self.tools[tc["name"]].execution_mode == "parallel" for tc in tool_calls
        )

        def call_one(tc: dict) -> ToolResult:
            tool = self.tools.get(tc["name"])
            if tool is None:
                return ToolResult.fail(f"unknown tool {tc['name']!r}")
            if self.config.before_tool_call:
                blocked = self.config.before_tool_call(tc["name"], tc["input"])
                if blocked:
                    return ToolResult.fail(f"tool {tc['name']!r} blocked by policy")
            try:
                result = tool.execute(tc["input"])
            except Exception as e:  # noqa: BLE001
                result = ToolResult.fail(f"{type(e).__name__}: {e}")
            if self.config.after_tool_call:
                result = self.config.after_tool_call(tc["name"], tc["input"], result)
            events.append(
                {
                    "kind": "tool_call",
                    "name": tc["name"],
                    "terminate": result.terminate,
                    "error": result.error,
                }
            )
            return result

        if any_parallel and len(tool_calls) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(8, len(tool_calls))
            ) as ex:
                return list(ex.map(call_one, tool_calls))
        return [call_one(tc) for tc in tool_calls]


def _first_text(content: list[dict]) -> str:
    for b in content:
        if b.get("type") == "text":
            return b.get("text", "")
    return ""


def _approx_cost(model: str, in_tok: int, out_tok: int) -> float:
    from harness_rlm.llm import compute_cost

    return compute_cost(model, in_tok, out_tok)


__all__ = ["AgentLoop", "AgentLoopConfig", "AgentLoopResult"]
