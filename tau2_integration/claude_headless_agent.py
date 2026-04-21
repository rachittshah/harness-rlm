"""tau2 custom agent that delegates each turn to ``claude -p`` (Claude Code headless).

This is the faithful harness-RLM pattern: the Claude Code harness is the
system-under-test. Each tau2 turn, we serialize the conversation so far into
a single prompt string, hand it to the ``claude`` CLI via ``subprocess.run``,
and parse the response back into a tau2 ``AssistantMessage``.

If the response contains a JSON ``tool_call`` line (our thin convention),
we convert it to a tau2 ``ToolCall`` on the returned message. Otherwise the
response is used as plain text.

Design notes:
    * Tau2 expects each ``AssistantMessage`` to be *either* plain text *or*
      tool calls (validator ``validate_message_format_default`` in
      ``tau2.agent.base_agent``). We enforce that here.
    * Every claude invocation is logged to ``/tmp/rlm/tau2_invocations.jsonl``
      so we can audit what was sent / received end-to-end.
    * Timeouts yield a friendly assistant message so the eval finishes
      cleanly rather than crashing mid-trajectory.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
INVOCATIONS_LOG = Path("/tmp/rlm/tau2_invocations.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_log(entry: dict) -> None:
    """Append one JSON line to /tmp/rlm/tau2_invocations.jsonl."""
    try:
        INVOCATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with INVOCATIONS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        # Never let logging failures break the eval.
        pass


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ClaudeHeadlessAgentState:
    """Simple mutable state for the Claude-headless agent.

    We do not need Pydantic here — tau2 treats the state as an opaque object
    that we pass back on each turn. Using a plain class also means we can
    stash non-serialisable things (like a session id) freely.
    """

    def __init__(
        self,
        system_prompt: str,
        transcript: Optional[list[dict]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.system_prompt = system_prompt
        # transcript: list of {role: "user"|"assistant"|"tool", content: str}
        self.transcript: list[dict] = list(transcript) if transcript else []
        self.session_id = session_id or str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ClaudeHeadlessAgent(HalfDuplexAgent[ClaudeHeadlessAgentState]):
    """tau2 agent that proxies each turn to the ``claude`` CLI in ``-p`` mode."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        claude_bin: str = "claude",
        timeout: int = 120,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.claude_bin = claude_bin
        self.timeout = timeout

    # -- state init ---------------------------------------------------------

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> ClaudeHeadlessAgentState:
        """Construct the system prompt and seed the transcript."""
        system_prompt = self._build_system_prompt()
        transcript: list[dict] = []
        if message_history:
            for msg in message_history:
                entry = self._tau2_message_to_transcript_entry(msg)
                if entry is not None:
                    transcript.append(entry)
        return ClaudeHeadlessAgentState(
            system_prompt=system_prompt,
            transcript=transcript,
        )

    # -- main turn ----------------------------------------------------------

    def generate_next_message(
        self,
        message: UserMessage | ToolMessage | MultiToolMessage,
        state: ClaudeHeadlessAgentState,
    ) -> tuple[AssistantMessage, ClaudeHeadlessAgentState]:
        """Append the incoming message, call ``claude -p``, return the reply."""
        # 1) Append whatever the orchestrator just handed us.
        incoming_entries = self._tau2_message_to_transcript_entries(message)
        state.transcript.extend(incoming_entries)

        # 2) Build the single flat prompt string.
        prompt = self._build_prompt(state)

        # 3) Invoke claude headless and parse.
        raw, elapsed_ms, invocation_err = self._invoke_claude(prompt, state.session_id)

        text, tool_call = self._parse_response(raw)

        # 4) Convert parsed response into a tau2 AssistantMessage.
        if tool_call is not None:
            assistant_msg = AssistantMessage.text(
                content=None,
                tool_calls=[tool_call],
            )
            transcript_content = f"[tool_call] {tool_call.name}({json.dumps(tool_call.arguments)})"
        else:
            # tau2 rejects empty messages, so guarantee non-empty content.
            safe_text = text if text and text.strip() else "[claude returned empty response]"
            assistant_msg = AssistantMessage.text(content=safe_text)
            transcript_content = safe_text

        state.transcript.append({"role": "assistant", "content": transcript_content})

        # 5) Audit log.
        _append_log(
            {
                "timestamp": _now_iso(),
                "session_id": state.session_id,
                "prompt_chars": len(prompt),
                "prompt_preview": prompt[:400],
                "response_chars": len(raw) if raw else 0,
                "response_preview": (raw or "")[:400],
                "elapsed_ms": elapsed_ms,
                "tool_call": tool_call.model_dump() if tool_call else None,
                "error": invocation_err,
            }
        )

        return assistant_msg, state

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Compose system prompt = persona + policy + tools schema + output contract."""
        tool_schemas: list[dict[str, Any]] = []
        for t in self.tools:
            try:
                tool_schemas.append(t.openai_schema)
            except Exception:  # noqa: BLE001 — defensive; bad schema shouldn't kill the run
                tool_schemas.append({"name": getattr(t, "name", "unknown")})

        tools_block = json.dumps(tool_schemas, indent=2) if tool_schemas else "[]"

        return (
            "You are a customer service agent. Follow the domain policy below "
            "strictly. You will see the conversation transcript and must produce "
            "exactly ONE next assistant message.\n\n"
            "## Output contract\n"
            "If you need to call a tool, output a single JSON line of the form:\n"
            '  {"tool_call": {"name": "<tool_name>", "arguments": {...}}}\n'
            "and nothing else — no prose, no markdown fences around it.\n"
            "Otherwise, output plain text as your reply to the user (no JSON, "
            "no tool_call key).\n\n"
            "## Tools available\n"
            f"{tools_block}\n\n"
            "## Domain policy\n"
            f"{self.domain_policy}\n"
        )

    @staticmethod
    def _tau2_message_to_transcript_entry(msg: Message) -> Optional[dict]:
        """Best-effort conversion of one tau2 message into a {role, content} dict.

        Returns ``None`` for message types we don't want in the transcript
        (e.g. bare ``MultiToolMessage`` wrappers — we unpack those in
        ``_tau2_message_to_transcript_entries``).
        """
        if isinstance(msg, UserMessage):
            return {"role": "user", "content": msg.content or ""}
        if isinstance(msg, AssistantMessage):
            if msg.is_tool_call() and msg.tool_calls:
                names = ", ".join(tc.name for tc in msg.tool_calls)
                return {"role": "assistant", "content": f"[tool_call] {names}"}
            return {"role": "assistant", "content": msg.content or ""}
        if isinstance(msg, ToolMessage):
            return {
                "role": "tool",
                "content": f"[tool_result id={msg.id}] {msg.content or ''}",
            }
        return None

    def _tau2_message_to_transcript_entries(
        self, msg: UserMessage | ToolMessage | MultiToolMessage
    ) -> list[dict]:
        """Handle ``MultiToolMessage`` by flattening into multiple entries."""
        if isinstance(msg, MultiToolMessage):
            out: list[dict] = []
            for tm in msg.tool_messages:
                entry = self._tau2_message_to_transcript_entry(tm)
                if entry:
                    out.append(entry)
            return out
        entry = self._tau2_message_to_transcript_entry(msg)
        return [entry] if entry else []

    @staticmethod
    def _build_prompt(state: ClaudeHeadlessAgentState) -> str:
        """Flatten (system_prompt + transcript) into the single string claude -p takes."""
        lines: list[str] = [state.system_prompt, "", "## Conversation so far"]
        if not state.transcript:
            lines.append("(no prior turns)")
        else:
            for turn in state.transcript:
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "")
                lines.append(f"[{role}] {content}")
        lines.append("")
        lines.append(
            "Respond with the next assistant message now. Follow the output "
            "contract exactly."
        )
        return "\n".join(lines)

    def _invoke_claude(
        self, prompt: str, session_id: str
    ) -> tuple[str, int, Optional[str]]:
        """Call ``claude -p`` and return (raw_text, elapsed_ms, error_or_None)."""
        if shutil.which(self.claude_bin) is None:
            return (
                "",
                0,
                f"claude binary not found on PATH (looked for '{self.claude_bin}')",
            )

        cmd = [
            self.claude_bin,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--permission-mode",
            "bypassPermissions",
        ]

        t0 = time.monotonic()
        try:
            proc = subprocess.run(  # noqa: S603 — trusted local CLI
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            return "", elapsed_ms, "timeout"
        except FileNotFoundError as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            return "", elapsed_ms, f"FileNotFoundError: {e}"

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if proc.returncode != 0:
            # Fall back to plain-text output if --output-format json wasn't
            # accepted — retry once without the JSON flag.
            stderr = (proc.stderr or "").strip()
            if "output-format" in stderr.lower() or "json" in stderr.lower():
                retry_cmd = [
                    self.claude_bin,
                    "-p",
                    prompt,
                    "--permission-mode",
                    "bypassPermissions",
                ]
                try:
                    retry = subprocess.run(  # noqa: S603
                        retry_cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                    )
                except subprocess.TimeoutExpired:
                    return "", elapsed_ms, "timeout (retry plain text)"
                if retry.returncode == 0:
                    return retry.stdout or "", elapsed_ms, None
                return (
                    "",
                    elapsed_ms,
                    f"claude exited {retry.returncode}: {(retry.stderr or '').strip()[:400]}",
                )
            return (
                "",
                elapsed_ms,
                f"claude exited {proc.returncode}: {stderr[:400]}",
            )

        return proc.stdout or "", elapsed_ms, None

    @staticmethod
    def _parse_response(raw: str) -> tuple[str, Optional[ToolCall]]:
        """Extract assistant text + optional tool_call from ``claude -p`` stdout.

        ``--output-format json`` returns an envelope like::

            {"type": "result", "subtype": "success", "result": "<text>", ...}

        We first try to unwrap that envelope; if that fails we fall through to
        treating stdout as plain text. Then we look for a ``{"tool_call": ...}``
        line inside the text — if present, we build a ``ToolCall`` from it
        and drop the text.
        """
        if not raw or not raw.strip():
            return "", None

        text = raw.strip()

        # Step 1: try JSON envelope (output-format=json).
        try:
            envelope = json.loads(text)
            if isinstance(envelope, dict):
                # Claude Code's --output-format=json uses "result" for the
                # final text. Older/newer builds have varied this slightly,
                # so accept a small set of keys.
                for key in ("result", "text", "content", "response", "message"):
                    if isinstance(envelope.get(key), str):
                        text = envelope[key]
                        break
        except json.JSONDecodeError:
            pass  # plain text — use `text` as-is

        # Step 2: look for a tool_call JSON anywhere in the text.
        tool_call = ClaudeHeadlessAgent._extract_tool_call(text)
        if tool_call is not None:
            return "", tool_call

        return text.strip(), None

    @staticmethod
    def _extract_tool_call(text: str) -> Optional[ToolCall]:
        """Find and parse a ``{"tool_call": {...}}`` object inside text."""
        # Fast path: a line that is itself a JSON object with "tool_call".
        for line in text.splitlines():
            line = line.strip()
            if not (line.startswith("{") and '"tool_call"' in line):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tc = ClaudeHeadlessAgent._tool_call_from_obj(obj)
            if tc is not None:
                return tc

        # Slow path: regex-scan for the first {"tool_call": { ... }} block.
        match = re.search(r'\{\s*"tool_call"\s*:\s*\{.*?\}\s*\}', text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                return ClaudeHeadlessAgent._tool_call_from_obj(obj)
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _tool_call_from_obj(obj: Any) -> Optional[ToolCall]:
        if not isinstance(obj, dict):
            return None
        tc = obj.get("tool_call")
        if not isinstance(tc, dict):
            return None
        name = tc.get("name")
        arguments = tc.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            name=name,
            arguments=arguments,
            requestor="assistant",
        )


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


def create_claude_headless_agent(tools, domain_policy, **kwargs) -> ClaudeHeadlessAgent:
    """tau2 registry factory for :class:`ClaudeHeadlessAgent`.

    Recognised kwargs:
        llm_args (dict): may contain ``claude_bin`` (str) and ``timeout`` (int).
    All other kwargs are ignored for API compatibility with tau2's factory
    protocol (``llm``, ``task``, etc.).
    """
    llm_args = kwargs.get("llm_args") or {}
    return ClaudeHeadlessAgent(
        tools=tools,
        domain_policy=domain_policy,
        claude_bin=llm_args.get("claude_bin", "claude"),
        timeout=int(llm_args.get("timeout", 120)),
    )
