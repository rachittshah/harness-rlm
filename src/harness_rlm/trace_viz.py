"""Trace visualization — ASCII tree + Mermaid diagrams.

Two outputs:

  format_trace(trace, *, color=False) → str
      Indented text dump with cost/latency totals. Good for `print()`.

  trace_to_mermaid(trace) → str
      Mermaid `flowchart` source. Paste into a Markdown renderer (GitHub,
      Notion, mermaid.live) to get a labelled execution diagram with
      cost per node.
"""

from __future__ import annotations

from typing import Any

from harness_rlm.modules import Trace


# ---------------------------------------------------------------------------
# ASCII format
# ---------------------------------------------------------------------------
def format_trace(trace: Trace, *, indent: int = 0, color: bool = False) -> str:
    """Pretty-print a Trace with cost/latency totals + per-event lines.

    Args:
        trace:  The Trace to format.
        indent: Number of spaces to indent the block (used in recursive calls).
        color: If True, wraps key fields in ANSI colour codes.

    Returns:
        Human-readable multi-line string.
    """
    pad = " " * indent
    bold = _BOLD if color else ""
    reset = _RESET if color else ""
    dim = _DIM if color else ""
    green = _GREEN if color else ""
    yellow = _YELLOW if color else ""

    lines: list[str] = []
    lines.append(
        f"{pad}{bold}{trace.module}{reset}  "
        f"{green}calls={trace.calls}{reset}  "
        f"{yellow}cost=${trace.cost_usd:.4f}{reset}  "
        f"{dim}lat={trace.latency_s:.2f}s{reset}  "
        f"{dim}tokens={trace.input_tokens}/{trace.output_tokens}{reset}"
    )
    for evt in trace.events:
        label = evt.get("label", "?")
        bits: list[str] = []
        for k in ("model", "input_tokens", "output_tokens", "cost_usd", "latency_s"):
            if k in evt:
                v = evt[k]
                if isinstance(v, float):
                    v = f"{v:.4f}" if "cost" in k else f"{v:.2f}"
                bits.append(f"{k}={v}")
        # Tool-call events.
        for k in ("name", "terminate", "error"):
            if k in evt and evt[k] not in (None, False, ""):
                bits.append(f"{k}={evt[k]}")
        line = f"{pad}  • {label}"
        if bits:
            line += f"  {dim}({' '.join(bits)}){reset}"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mermaid flowchart
# ---------------------------------------------------------------------------
def trace_to_mermaid(trace: Trace, *, title: str | None = None) -> str:
    """Produce a Mermaid flowchart source from a Trace.

    Nodes:
        start → e1 → e2 → ... → end

    Each event becomes one node labelled with its kind + (model, cost).
    The summary footer carries totals.
    """
    safe_module = _safe_node_id(trace.module)
    lines = ["```mermaid", "flowchart TD"]
    if title:
        lines.append(f"    %% {title}")
    lines.append(f'    start(["▶ {trace.module}"])')

    prev = "start"
    for i, evt in enumerate(trace.events, start=1):
        node_id = f"{safe_module}_e{i}"
        label = _node_label(evt)
        lines.append(f"    {node_id}[\"{label}\"]")
        lines.append(f"    {prev} --> {node_id}")
        prev = node_id

    footer = (
        f"calls={trace.calls}  "
        f"cost=${trace.cost_usd:.4f}  "
        f"latency={trace.latency_s:.2f}s"
    )
    lines.append(f'    end_node(["■ done<br/>{footer}"])')
    lines.append(f"    {prev} --> end_node")
    lines.append("```")
    return "\n".join(lines)


def _node_label(evt: dict[str, Any]) -> str:
    """Build a one-line Mermaid label for one trace event."""
    label = evt.get("label", "?")
    parts = [label]
    if "model" in evt:
        parts.append(str(evt["model"]))
    if "cost_usd" in evt:
        parts.append(f"${evt['cost_usd']:.4f}")
    elif "calls" in evt:
        parts.append(f"calls={evt['calls']}")
    if "name" in evt:  # tool call
        parts.append(f"tool: {evt['name']}")
    # Mermaid disallows certain chars in labels.
    safe = " ".join(parts).replace('"', "'").replace("\n", " ")
    return safe


def _safe_node_id(s: str) -> str:
    """Make a Mermaid-safe node id (alphanumeric + underscore)."""
    return "".join(c if c.isalnum() else "_" for c in s)[:32] or "n"


# ---------------------------------------------------------------------------
# ANSI codes (used only when color=True)
# ---------------------------------------------------------------------------
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"


__all__ = ["format_trace", "trace_to_mermaid"]
