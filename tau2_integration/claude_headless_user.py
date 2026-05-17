"""tau2 user simulator that delegates each turn to ``claude -p``.

Mirror of ClaudeHeadlessAgent for the user side. Lets you run tau²-bench
end-to-end on `claude -p` Enterprise OAuth — no OPENAI_API_KEY, no
ANTHROPIC_API_KEY. Both agent and user are the same backend.

Caveat: the user simulator's role is to play the *customer* against the
agent's policy. We pass the task scenario / persona / instructions as the
system prompt and have claude -p produce the next user utterance.

Sentinels (from tau2.user.user_simulator_base):
  - ###STOP###          end the conversation
  - ###TRANSFER###      transfer to another agent
  - ###OUT-OF-SCOPE###  decline as out of scope

We instruct claude to use these verbatim when appropriate.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.user.user_simulator_base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    HalfDuplexUser,
    UserState,
    is_valid_user_history_message,
)

INVOCATIONS_LOG = Path("/tmp/rlm/tau2_user_invocations.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_log(entry: dict) -> None:
    try:
        INVOCATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with INVOCATIONS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


class ClaudeHeadlessUserState(UserState):
    """Inherits UserState; adds a session id so audit log can correlate turns."""

    session_id: str = ""

    def model_post_init(self, _context) -> None:
        if not self.session_id:
            self.session_id = str(uuid.uuid4())


class ClaudeHeadlessUserSimulator(HalfDuplexUser[ClaudeHeadlessUserState]):
    """User simulator that shells out to ``claude -p`` for each turn.

    Compatible with tau2's HalfDuplexUser protocol. Accepts an `llm_args`
    dict from the registry / runner with:

        claude_bin   (str, optional, default 'claude')
        timeout      (int, optional, default 300 seconds)
        model        (str, optional, e.g. 'claude-opus-4-7[1m]')
        effort       (str, optional, 'low'|'medium'|'high'|'xhigh'|'max')
        mcp_config_path (str, optional, path to MCP server config JSON)

    Args:
        instructions: tau2 task instructions/persona injected as system prompt.
        tools: tau2 user-side tools (rare; usually None).
        llm: a string slug. Currently ignored — we always use the local claude CLI.
        llm_args: see above.
    """

    def __init__(
        self,
        instructions: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ) -> None:
        super().__init__(instructions=instructions, tools=tools)
        llm_args = llm_args or {}
        self.claude_bin = llm_args.get("claude_bin", "claude")
        self.timeout = int(llm_args.get("timeout", 300))
        self.model = llm_args.get("model") or llm  # accept either
        self.effort = llm_args.get("effort")
        self.mcp_config_path = llm_args.get("mcp_config_path")

    # ---- init state -------------------------------------------------------
    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> ClaudeHeadlessUserState:
        if message_history is None:
            message_history = []
        assert all(is_valid_user_history_message(m) for m in message_history), (
            "Invalid user message history."
        )
        return ClaudeHeadlessUserState(
            system_messages=[SystemMessage(role="system", content=self._build_system_prompt())],
            messages=list(message_history),
        )

    # ---- one turn ---------------------------------------------------------
    def generate_next_message(
        self,
        message: AssistantMessage | ToolMessage | MultiToolMessage,
        state: ClaudeHeadlessUserState,
    ) -> tuple[UserMessage, ClaudeHeadlessUserState]:
        # Append the agent's message into the user-side state.
        state.messages.append(message)

        # Build a single flat prompt: system + flipped (user-view) history.
        # tau2's flip_roles converts assistant<->user from the user simulator's POV.
        flipped = state.flip_roles()
        prompt = self._build_prompt(state.system_messages[0].content, flipped)

        raw, elapsed_ms, err = self._invoke_claude(prompt, state.session_id)
        text = self._parse_response(raw)
        safe_text = text if text and text.strip() else "[claude returned empty response]"

        user_msg = UserMessage(role="user", content=safe_text)
        state.messages.append(user_msg)

        _append_log({
            "timestamp": _now_iso(),
            "session_id": state.session_id,
            "side": "user",
            "prompt_chars": len(prompt),
            "prompt_preview": prompt[:400],
            "response_chars": len(raw) if raw else 0,
            "response_preview": (raw or "")[:400],
            "elapsed_ms": elapsed_ms,
            "error": err,
        })

        return user_msg, state

    # ---- helpers ----------------------------------------------------------
    def _build_system_prompt(self) -> str:
        """Compose the user-side system prompt (persona + sentinels guide)."""
        return (
            "You are simulating a CUSTOMER talking to a customer-service AGENT. "
            "Follow your persona / scenario below strictly. Produce exactly ONE next "
            "customer message per turn — first-person, conversational, no markdown.\n\n"
            "Special sentinels (use VERBATIM, alone on a line, when the situation calls for it):\n"
            f"  {STOP}            — end the conversation (issue resolved or unresolvable)\n"
            f"  {TRANSFER}        — request transfer to another agent\n"
            f"  {OUT_OF_SCOPE}    — the agent's request is outside what you, the customer, would do\n\n"
            "## Persona / scenario\n"
            f"{self.instructions or '(no scenario provided)'}\n"
        )

    @staticmethod
    def _build_prompt(system: str, flipped: list) -> str:
        lines: list[str] = [system, "", "## Conversation so far (your view)"]
        if not flipped:
            lines.append("(no prior turns — the agent will speak first; respond with your opening request)")
        for m in flipped:
            role = m.role.capitalize() if hasattr(m.role, "capitalize") else str(m.role).capitalize()
            content = getattr(m, "content", "") or ""
            if not isinstance(content, str):
                content = str(content)
            lines.append(f"[{role}] {content}")
        lines.append("")
        lines.append("Respond as the customer with exactly ONE next message. "
                     f"Use {STOP} alone on a line when the issue is resolved or hopeless.")
        return "\n".join(lines)

    def _invoke_claude(self, prompt: str, session_id: str) -> tuple[str, int, Optional[str]]:
        if shutil.which(self.claude_bin) is None:
            return "", 0, f"claude binary not found: {self.claude_bin}"
        cmd: list[str] = [
            self.claude_bin, "-p",
            "--output-format", "text",
            "--permission-mode", "bypassPermissions",
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.effort:
            cmd.extend(["--effort", self.effort])
        if self.mcp_config_path:
            cmd.extend(["--mcp-config", self.mcp_config_path])

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                timeout=self.timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            return "", int((time.monotonic() - t0) * 1000), "timeout"
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if proc.returncode != 0:
            return "", elapsed_ms, f"claude exit {proc.returncode}: {(proc.stderr or '').strip()[:300]}"
        return proc.stdout or "", elapsed_ms, None

    @staticmethod
    def _parse_response(raw: str) -> str:
        if not raw or not raw.strip():
            return ""
        text = raw.strip()
        # Strip code fences if model returned them.
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines() if not line.strip().startswith("```")
            )
        return text.strip()


__all__ = ["ClaudeHeadlessUserSimulator", "ClaudeHeadlessUserState"]
