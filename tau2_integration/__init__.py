"""tau2-bench integration for harness-rlm.

Provides two tau2 agents that exercise the harness-RLM pattern:

- :class:`ClaudeHeadlessAgent` — delegates each tau2 turn to ``claude -p``
  (Claude Code in headless / ``--print`` mode). This is literally the Claude
  Code harness being used as the evaluated agent. If the tau2 policy is
  large enough that the ``/rlm`` skill triggers, the skill will internally
  decompose via the harness-rlm MCP server.

- :class:`RLMAgent` — a *direct* root/sub LLM implementation (no Claude CLI
  in the loop). Uses Anthropic's SDK for the root model and the
  ``harness_rlm.mcp_server.run_llm_query`` helper for cheap sub-LLM calls.
  Falls through to a straight root call when the prompt is small.

Use :func:`register.register` to expose them to ``tau2 run``.
"""

from tau2_integration.claude_headless_agent import (
    ClaudeHeadlessAgent,
    ClaudeHeadlessAgentState,
    create_claude_headless_agent,
)
from tau2_integration.register import register
from tau2_integration.rlm_agent import (
    RLMAgent,
    RLMAgentState,
    create_rlm_agent,
)

__all__ = [
    "ClaudeHeadlessAgent",
    "ClaudeHeadlessAgentState",
    "RLMAgent",
    "RLMAgentState",
    "create_claude_headless_agent",
    "create_rlm_agent",
    "register",
]
