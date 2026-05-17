"""LM provider that shells out to `claude -p` (Claude Code headless).

Useful when:
  - ANTHROPIC_API_KEY isn't set but the user has `claude` installed and
    authenticated (OAuth / keychain).
  - You want each LM call to go through the full Claude Code harness
    (skills, hooks, etc.) — e.g. to test that harness-rlm interoperates
    with an existing Claude Code session.

Caveat: each invocation pays Claude Code's ~50K-token init tax (skill +
MCP descriptions + CLAUDE.md). For tight RLM loops, prefer the direct
`LM` client when an API key is available — claude_cli_lm is a fallback,
not the optimal path.

The interface matches `LM.__call__(prompt, *, system=None, model=None,
max_tokens=None)` so it's drop-in for harness modules.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

from harness_rlm.llm import LMResult, compute_cost


@dataclass
class ClaudeCLILM:
    """Shell-out LM provider — calls `claude -p <prompt>` for each completion.

    Args:
        model:      Optional model override passed as `--model`. If None, lets
                    Claude Code use its configured default.
        bin:        Path to the `claude` binary (defaults to PATH lookup).
        timeout_s:  Hard timeout per call. Default 120s.
        bare:       If True, pass `--bare` to skip hooks/skills/auto-memory.
                    Strongly recommended for sub-LLM calls — drops the per-call
                    overhead from ~50K tokens to ~2K.
        extra_args: Extra args appended verbatim (e.g. `["--effort", "low"]`).
    """

    model: str | None = None
    bin: str = "claude"
    timeout_s: int = 120
    # --bare strips skills/hooks/CLAUDE.md but REQUIRES ANTHROPIC_API_KEY
    # or apiKeyHelper. Default False so OAuth users can use this provider.
    bare: bool = False
    extra_args: tuple[str, ...] = ()
    # Same cumulative counters as `LM` so callers can mix providers.
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    max_tokens: int = 1024  # advertised for compatibility; CLI ignores it.

    def __post_init__(self) -> None:
        if shutil.which(self.bin) is None:
            raise RuntimeError(
                f"`{self.bin}` not found on PATH. Install Claude Code or set "
                f"bin=... to a full path."
            )
        self._lock = threading.Lock()

    def __call__(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LMResult:
        used_model = model or self.model
        cmd: list[str] = [self.bin, "-p"]
        if self.bare:
            cmd.append("--bare")
        if used_model:
            cmd.extend(["--model", used_model])
        if system:
            cmd.extend(["--append-system-prompt", system])
        cmd.extend(self.extra_args)
        # Pass the prompt on stdin (not argv) so we don't confuse argparse with
        # prompts that begin with `-` or `---`.

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"claude -p timed out after {self.timeout_s}s. cmd={shlex.join(cmd[:5])}..."
            ) from e
        latency = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude -p exited {proc.returncode}: {proc.stderr.strip()[:500]}"
            )

        text = proc.stdout.strip()
        # CLI doesn't surface token counts — approximate by chars (~4 chars/token).
        in_tok = max(1, len(prompt) // 4)
        out_tok = max(1, len(text) // 4)
        cost = compute_cost(used_model or "claude-haiku-4-5", in_tok, out_tok)

        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += in_tok
            self.total_output_tokens += out_tok
            self.total_cost_usd += cost

        return LMResult(
            text=text,
            model=used_model or "claude-cli",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            latency_s=latency,
        )

    def stats(self) -> dict:
        with self._lock:
            return {
                "calls": self.total_calls,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "cost_usd": round(self.total_cost_usd, 6),
            }


__all__ = ["ClaudeCLILM"]
