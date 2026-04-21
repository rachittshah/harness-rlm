"""Register harness-rlm's tau2 agents with the tau2 registry.

Usage:

    # At program start, before constructing a TextRunConfig:
    from tau2_integration.register import register
    register()

    # Then the agents are available under these names:
    #   "harness-rlm/claude-headless"   -> ClaudeHeadlessAgent
    #   "harness-rlm/rlm"               -> RLMAgent
    # e.g.
    #   tau2 run --domain airline --agent harness-rlm/claude-headless ...
    #
    # NOTE: tau2's CLI imports ``tau2.registry`` at startup, so to use these
    # agents via ``tau2 run ...`` you must make sure your registration runs
    # before the CLI dispatches. Easiest: run the programmatic entry point
    # in ``examples/run_tau2_py.py`` instead.

The factory functions this module registers are defined alongside the agent
classes in :mod:`tau2_integration.claude_headless_agent` and
:mod:`tau2_integration.rlm_agent`.
"""

from __future__ import annotations

from tau2.registry import registry

from tau2_integration.claude_headless_agent import create_claude_headless_agent
from tau2_integration.rlm_agent import create_rlm_agent

CLAUDE_HEADLESS_AGENT_NAME = "harness-rlm/claude-headless"
RLM_AGENT_NAME = "harness-rlm/rlm"


def register(*, overwrite: bool = False) -> list[str]:
    """Register both harness-rlm agents with the tau2 registry.

    Args:
        overwrite: If True, silently skip agents that are already registered
            (useful in notebooks / re-import scenarios). If False and an
            agent name is already taken, ``registry.register_agent_factory``
            raises ``ValueError``.

    Returns:
        The list of agent names now registered by this call.
    """
    registered: list[str] = []

    existing = set(registry.get_agents())
    pairs = [
        (CLAUDE_HEADLESS_AGENT_NAME, create_claude_headless_agent),
        (RLM_AGENT_NAME, create_rlm_agent),
    ]
    for name, factory in pairs:
        if name in existing:
            if overwrite:
                # Registry API has no unregister; mutate the private dict.
                # This is safe as long as overwrite=True is opt-in.
                registry._agent_factories.pop(name, None)  # noqa: SLF001
            else:
                continue
        registry.register_agent_factory(factory, name)
        registered.append(name)

    return registered


__all__ = [
    "CLAUDE_HEADLESS_AGENT_NAME",
    "RLM_AGENT_NAME",
    "register",
]
