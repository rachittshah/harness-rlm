"""Codex-style declarative subagents — TOML files in `~/.harness-rlm/agents/`.

Each subagent is one TOML file with name, description, model, instructions,
and optional reasoning effort. Loaded at runtime, dispatched by name. The
parent passes a task; the subagent runs an isolated agent loop with its own
model and instructions, then returns text + cost.

File layout:
    ~/.harness-rlm/agents/<name>.toml             # personal
    <repo>/.harness-rlm/agents/<name>.toml        # project (takes precedence)

Schema (one file per agent):

    name = "explorer"
    description = "Read-only codebase explorer."
    model = "claude-haiku-4-5-20251001"
    reasoning_effort = "medium"            # low | medium | high (advisory)
    sandbox_mode = "read-only"             # read-only | workspace-write | danger
    instructions = '''
    Stay in exploration mode. Trace the real execution path,
    cite files and symbols, avoid proposing fixes.
    '''
    tools = ["read", "bash"]               # subset of registered tool names

The `dispatch` function loads the spec, builds an `AgentLoop` with the
declared tools, runs it on the task message, and returns the result.

Inheritance / sandbox tiers follow Codex: subagents inherit the parent's
sandbox unless their TOML says otherwise — and can only DOWNGRADE, never
escalate. Read-only < workspace-write < danger.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from harness_rlm.agent_loop import AgentLoop, AgentLoopConfig, AgentLoopResult
from harness_rlm.tools import (
    BASH_TOOL,
    EDIT_TOOL,
    FINISH_TOOL,
    READ_TOOL,
    WRITE_TOOL,
    AgentTool,
)

SANDBOX_TIERS = {"read-only": 0, "workspace-write": 1, "danger": 2}


@dataclass
class SubagentSpec:
    """One declarative subagent loaded from a TOML file."""

    name: str
    description: str
    model: str = "claude-haiku-4-5-20251001"
    instructions: str = ""
    reasoning_effort: str = "medium"
    sandbox_mode: str = "read-only"
    tools: list[str] = field(default_factory=lambda: ["read", "bash", "finish_task"])
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Subagent name must be safe identifier (got {self.name!r})")
        if self.sandbox_mode not in SANDBOX_TIERS:
            raise ValueError(
                f"sandbox_mode must be one of {sorted(SANDBOX_TIERS)} (got {self.sandbox_mode!r})"
            )


# ---------------------------------------------------------------------------
# Built-in tool registry for resolving names in `tools = [...]`.
# Callers can pass their own tool registry to dispatch().
# ---------------------------------------------------------------------------
BUILTIN_TOOL_REGISTRY: dict[str, AgentTool] = {
    READ_TOOL.name: READ_TOOL,
    WRITE_TOOL.name: WRITE_TOOL,
    EDIT_TOOL.name: EDIT_TOOL,
    BASH_TOOL.name: BASH_TOOL,
    FINISH_TOOL.name: FINISH_TOOL,
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def default_search_paths() -> list[Path]:
    """Return the standard subagent search paths, in priority order.

    Project-local wins over personal — matches Codex's behaviour.
    """
    return [
        Path.cwd() / ".harness-rlm" / "agents",
        Path.home() / ".harness-rlm" / "agents",
    ]


def discover(search_paths: list[Path] | None = None) -> dict[str, SubagentSpec]:
    """Find all *.toml subagent specs across the search paths.

    Returns a name → spec map. Later entries with the same name are ignored
    so project-local files override personal ones.
    """
    found: dict[str, SubagentSpec] = {}
    for base in search_paths or default_search_paths():
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.toml")):
            try:
                spec = load_spec(path)
            except Exception as e:  # noqa: BLE001
                # Don't let one corrupt file kill discovery — record + skip.
                print(f"[subagents] skipping {path}: {e}")
                continue
            if spec.name not in found:
                found[spec.name] = spec
    return found


def load_spec(path: Path) -> SubagentSpec:
    """Read one TOML file and validate it into a SubagentSpec."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    required = {"name", "description"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"{path}: missing required fields {sorted(missing)}")
    return SubagentSpec(
        name=raw["name"],
        description=raw["description"],
        model=raw.get("model", "claude-haiku-4-5-20251001"),
        instructions=raw.get("instructions", "").strip(),
        reasoning_effort=raw.get("reasoning_effort", "medium"),
        sandbox_mode=raw.get("sandbox_mode", "read-only"),
        tools=list(raw.get("tools", ["read", "bash", "finish_task"])),
        source_path=path,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def dispatch(
    spec: SubagentSpec,
    task: str,
    *,
    parent_sandbox: str = "danger",
    tool_registry: dict[str, AgentTool] | None = None,
    max_turns: int = 12,
    api_key: str | None = None,
) -> AgentLoopResult:
    """Run a subagent on a task and return its loop result.

    The subagent's tools are looked up in `tool_registry` (defaults to
    BUILTIN_TOOL_REGISTRY). Unknown tool names raise — fail fast.

    Sandbox enforcement: the subagent's effective sandbox is the *more
    restrictive* of (parent_sandbox, spec.sandbox_mode). Codex behaviour:
    a permissive parent never lets a child escalate.
    """
    registry = tool_registry or BUILTIN_TOOL_REGISTRY
    unknown = [t for t in spec.tools if t not in registry]
    if unknown:
        raise ValueError(
            f"Subagent {spec.name!r} requests unknown tools: {unknown}. "
            f"Available: {sorted(registry)}"
        )
    tools = [registry[t] for t in spec.tools]

    # Sandbox: take the more restrictive of parent and spec.
    parent_tier = SANDBOX_TIERS.get(parent_sandbox, 2)
    spec_tier = SANDBOX_TIERS[spec.sandbox_mode]
    effective_tier = min(parent_tier, spec_tier)
    effective_mode = next(k for k, v in SANDBOX_TIERS.items() if v == effective_tier)

    # If read-only, drop write/edit/bash tools — best-effort enforcement.
    # (A real sandbox would also chroot / restrict syscalls; we just remove
    # the tools from the loop's surface.)
    if effective_mode == "read-only":
        tools = [t for t in tools if t.name not in {"write", "edit", "bash"}]
        if not tools:
            # Always keep at least one tool — finish_task.
            tools.append(FINISH_TOOL)

    cfg = AgentLoopConfig(
        model=spec.model,
        max_turns=max_turns,
        system=_compose_system(spec, effective_mode),
    )
    loop = AgentLoop(tools, cfg, api_key=api_key)
    return loop.run(task)


def _compose_system(spec: SubagentSpec, sandbox: str) -> str:
    """Compose the system prompt for a subagent."""
    parts = [
        f"You are subagent '{spec.name}'.",
        spec.description,
    ]
    if spec.instructions:
        parts.append("Instructions:")
        parts.append(spec.instructions)
    parts.append(f"Sandbox: {sandbox}.")
    parts.append("Call finish_task with your final answer when done.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# AGENTS.md hierarchical instruction discovery (Codex pattern).
# ---------------------------------------------------------------------------
def load_agents_md(start_dir: Path | None = None, max_bytes: int = 32 * 1024) -> str:
    """Walk from ~/.harness-rlm/ down to `start_dir`, concatenating AGENTS.md files.

    Override files (`AGENTS.override.md`) replace the lower-priority file at
    that level. Returns "" if no files found. Total output capped at `max_bytes`.
    """
    start = (start_dir or Path.cwd()).resolve()
    layers: list[tuple[Path, str]] = []
    # Personal override first.
    home_override = Path.home() / ".harness-rlm" / "AGENTS.override.md"
    home_base = Path.home() / ".harness-rlm" / "AGENTS.md"
    if home_override.is_file():
        layers.append((home_override, home_override.read_text(encoding="utf-8")))
    elif home_base.is_file():
        layers.append((home_base, home_base.read_text(encoding="utf-8")))

    # Walk from filesystem root to `start`, collecting per-dir AGENTS.md.
    chain = list(reversed([start, *start.parents]))
    for d in chain:
        override = d / "AGENTS.override.md"
        base = d / "AGENTS.md"
        if override.is_file():
            layers.append((override, override.read_text(encoding="utf-8")))
        elif base.is_file():
            layers.append((base, base.read_text(encoding="utf-8")))

    if not layers:
        return ""

    blocks: list[str] = []
    total = 0
    for path, text in layers:
        header = f"<!-- {path} -->\n"
        block = header + text.rstrip() + "\n"
        if total + len(block) > max_bytes:
            blocks.append(f"<!-- truncated at {max_bytes} bytes -->\n")
            break
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks)


__all__ = [
    "SubagentSpec",
    "discover",
    "load_spec",
    "dispatch",
    "load_agents_md",
    "BUILTIN_TOOL_REGISTRY",
    "default_search_paths",
    "SANDBOX_TIERS",
]
