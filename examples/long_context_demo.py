#!/usr/bin/env python3
"""End-to-end demo of the harness-rlm adapter on a synthetic long document.

Exercises the full RLM pattern WITHOUT going through tau2 or Claude Code:

    1. Build a deterministic ~100KB document with three planted facts.
    2. Decompose it into N ~10KB chunks.
    3. For each chunk, ask the cheap sub-LLM whether it contains info about
       the target fact (using `harness_rlm.mcp_server.run_llm_query` directly,
       as a library import — NOT through MCP transport).
    4. Aggregate hits and print a final synthesized answer.
    5. Print total cost, token usage, sub-call count, and wall-clock time.

This is the hello-world proof the adapter works end-to-end. It does not
depend on Claude Code, MCP stdio, or tau2.

USAGE
-----
    # Real run (hits Anthropic API; requires ANTHROPIC_API_KEY):
    python examples/long_context_demo.py --chunks 10 --verbose

    # Dry run (no API calls; mocked responses, prints cost *estimate*):
    python examples/long_context_demo.py --chunks 10 --dry-run --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

# Make `src/` importable when the script is run directly from a checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harness_rlm.mcp_server import compute_cost, run_llm_query  # noqa: E402
from harness_rlm.models import DEFAULT_MODEL, LLMQueryRequest, LLMQueryResponse  # noqa: E402

# ---------------------------------------------------------------------------
# Document synthesis
# ---------------------------------------------------------------------------
# Three planted "facts" — one per paragraph variant. These are the needles the
# query is designed to find.
PLANTED_FACTS = {
    "capital": "The capital of the fictional nation of Zephyria is Marquel City.",
    "river": "The longest river in Zephyria is the Indris, which spans 2,847 km.",
    "year": "Zephyria declared independence in the year 1847.",
}

BASE_PARAGRAPH = (
    "Zephyria is a fictional nation located in a temperate archipelago. "
    "Its economy is primarily based on maritime trade and precision agriculture. "
    "The cultural traditions include the annual Solstice Festival and the spring Lantern Parade. "
    "Education is publicly funded from primary school through tertiary level. "
    "Administrative governance follows a bicameral parliamentary model with elected regional councils. "
)


def build_document(num_paragraphs: int = 200) -> str:
    """Build a deterministic ~100KB document with three planted facts.

    Facts are distributed across the document (positions 17, 88, 163) so the
    chunker doesn't co-locate them. Repeating the base paragraph in between
    provides realistic filler without network dependencies.
    """
    fact_positions = {
        17: PLANTED_FACTS["capital"],
        88: PLANTED_FACTS["river"],
        163: PLANTED_FACTS["year"],
    }
    paragraphs: list[str] = []
    for i in range(num_paragraphs):
        if i in fact_positions:
            paragraphs.append(f"{BASE_PARAGRAPH}{fact_positions[i]}")
        else:
            paragraphs.append(BASE_PARAGRAPH)
    return "\n\n".join(paragraphs)


def chunk_document(doc: str, num_chunks: int) -> list[str]:
    """Split `doc` into `num_chunks` roughly-equal pieces."""
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    # Split on paragraph boundaries so facts stay intact.
    paragraphs = doc.split("\n\n")
    per_chunk = max(1, len(paragraphs) // num_chunks)
    chunks: list[str] = []
    for i in range(num_chunks):
        start = i * per_chunk
        # Last chunk grabs the remainder so nothing is dropped.
        end = (i + 1) * per_chunk if i < num_chunks - 1 else len(paragraphs)
        chunks.append("\n\n".join(paragraphs[start:end]))
    return chunks


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------
@dataclass
class ChunkResult:
    chunk_index: int
    chunk_chars: int
    response: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float


@dataclass
class DemoSummary:
    question: str
    num_chunks: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    wall_clock_s: float = 0.0
    sub_call_count: int = 0
    chunk_results: list[ChunkResult] = field(default_factory=list)
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Sub-LLM dispatch — real or mocked
# ---------------------------------------------------------------------------
SubLLM = Callable[[LLMQueryRequest], LLMQueryResponse]


def _real_sub_llm_factory(api_key: str) -> SubLLM:
    """Return a callable that invokes run_llm_query with the supplied key."""

    def call(req: LLMQueryRequest) -> LLMQueryResponse:
        return run_llm_query(req, api_key=api_key)

    return call


def _dry_run_sub_llm_factory() -> SubLLM:
    """Return a callable that fabricates a plausible-looking response.

    Uses heuristic token estimates (~4 chars per token) so the final cost
    printout is at least order-of-magnitude correct. No network access.
    """

    def call(req: LLMQueryRequest) -> LLMQueryResponse:
        # Heuristic: 1 token ~= 4 chars. We do not actually tokenise.
        input_tokens = max(1, len(req.prompt) // 4)
        # Inspect the chunk body (between <<< and >>>) for the planted-fact
        # sentinels. Mimics what a real sub-LLM would do: only fire when the
        # fact is actually present in the chunk, not just in the question.
        body = req.prompt
        try:
            body = req.prompt.split("<<<", 1)[1].rsplit(">>>", 1)[0]
        except (IndexError, ValueError):
            pass
        if "Marquel City" in body:
            content = "Yes — this chunk mentions the capital Marquel City."
        elif "Indris" in body:
            content = "Yes — this chunk mentions the river Indris (2,847 km)."
        elif "1847" in body:
            content = "Yes — this chunk mentions independence in 1847."
        else:
            content = "NOT_IN_CHUNK"
        output_tokens = max(1, len(content) // 4)
        return LLMQueryResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=req.model,
            cost_usd=compute_cost(req.model, input_tokens, output_tokens),
        )

    return call


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a precise extraction sub-LLM inside an RLM. "
    "Read the provided text chunk and answer the question in one sentence. "
    "If the chunk does not contain the answer, reply 'NOT_IN_CHUNK'. "
    "Never hallucinate facts."
)


def build_chunk_prompt(chunk: str, question: str, chunk_index: int, total: int) -> str:
    return (
        f"Chunk {chunk_index + 1}/{total} follows between <<< and >>>.\n"
        f"Question: {question}\n\n"
        f"<<<\n{chunk}\n>>>"
    )


def run_rlm_loop(
    document: str,
    question: str,
    num_chunks: int,
    sub_llm: SubLLM,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 256,
    verbose: bool = False,
) -> DemoSummary:
    """Decompose → dispatch → aggregate."""
    chunks = chunk_document(document, num_chunks)
    summary = DemoSummary(question=question, num_chunks=len(chunks))

    t0 = time.perf_counter()
    for idx, chunk in enumerate(chunks):
        prompt = build_chunk_prompt(chunk, question, idx, len(chunks))
        req = LLMQueryRequest(
            prompt=prompt, model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT
        )
        cstart = time.perf_counter()
        resp = sub_llm(req)
        latency = time.perf_counter() - cstart

        result = ChunkResult(
            chunk_index=idx,
            chunk_chars=len(chunk),
            response=resp.content,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
            latency_s=latency,
        )
        summary.chunk_results.append(result)
        summary.total_input_tokens += resp.input_tokens
        summary.total_output_tokens += resp.output_tokens
        summary.total_cost_usd += resp.cost_usd
        summary.sub_call_count += 1

        if verbose:
            preview = resp.content.replace("\n", " ")[:80]
            print(
                f"  chunk {idx + 1:02d}/{len(chunks)}  "
                f"in={resp.input_tokens:>5}  out={resp.output_tokens:>4}  "
                f"${resp.cost_usd:.5f}  {latency:.2f}s  -> {preview}"
            )

    summary.wall_clock_s = time.perf_counter() - t0
    return summary


def synthesize_answer(summary: DemoSummary) -> str:
    """Aggregate per-chunk responses into a final answer.

    "Hit" = the sub-LLM did NOT say NOT_IN_CHUNK. Everything else is a miss.
    This is deliberately simple; a real RLM would do another synthesis pass.
    """
    hits = [r for r in summary.chunk_results if "NOT_IN_CHUNK" not in r.response]
    if not hits:
        return "No chunks appear to contain the requested information."
    # Format as a bulleted summary.
    lines = [f"Found {len(hits)} relevant chunk(s):"]
    for h in hits:
        lines.append(f"  - chunk {h.chunk_index + 1}: {h.response.strip().splitlines()[0][:200]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end harness-rlm demo on a synthetic long document."
    )
    p.add_argument(
        "--chunks",
        type=int,
        default=10,
        help="Number of chunks to decompose the document into (default: 10).",
    )
    p.add_argument(
        "--paragraphs",
        type=int,
        default=200,
        help="Number of paragraphs in the synthetic doc (default: 200 ~= 100KB).",
    )
    p.add_argument(
        "--question",
        type=str,
        default=(
            "Does this chunk contain any information about Zephyria's capital, "
            "its longest river, or its year of independence?"
        ),
        help="Question asked of each chunk.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Sub-LLM model ID (default: {DEFAULT_MODEL}).",
    )
    p.add_argument("--max-tokens", type=int, default=256, help="Max output tokens per sub-call.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls; use mocked responses and print cost estimate only.",
    )
    p.add_argument("--verbose", action="store_true", help="Print per-chunk progress.")
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON summary at the end.",
    )
    return p.parse_args(argv)


def print_summary(summary: DemoSummary, final_answer: str) -> None:
    print()
    print("=" * 72)
    print("FINAL ANSWER")
    print("=" * 72)
    print(final_answer)
    print()
    print("=" * 72)
    print("RLM RUN METRICS")
    print("=" * 72)
    print(f"  question         : {summary.question}")
    print(f"  chunks           : {summary.num_chunks}")
    print(f"  sub-calls        : {summary.sub_call_count}")
    print(f"  input tokens     : {summary.total_input_tokens:,}")
    print(f"  output tokens    : {summary.total_output_tokens:,}")
    tag = "ESTIMATE (dry-run)" if summary.dry_run else "ACTUAL"
    print(f"  total cost ({tag}): ${summary.total_cost_usd:.6f}")
    print(f"  wall-clock       : {summary.wall_clock_s:.2f}s")
    if summary.sub_call_count:
        print(f"  cost per chunk   : ${summary.total_cost_usd / summary.sub_call_count:.6f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        print(f"building synthetic document ({args.paragraphs} paragraphs)...")
    doc = build_document(num_paragraphs=args.paragraphs)
    if args.verbose:
        print(f"  document size: {len(doc):,} chars ({len(doc) / 1024:.1f} KB)")

    if args.dry_run:
        sub_llm = _dry_run_sub_llm_factory()
        if args.verbose:
            print("DRY RUN — no API calls will be made; responses are mocked.")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "ERROR: ANTHROPIC_API_KEY is not set. Either export it or pass --dry-run.",
                file=sys.stderr,
            )
            return 2
        sub_llm = _real_sub_llm_factory(api_key)

    summary = run_rlm_loop(
        document=doc,
        question=args.question,
        num_chunks=args.chunks,
        sub_llm=sub_llm,
        model=args.model,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )
    summary.dry_run = args.dry_run
    final_answer = synthesize_answer(summary)

    if args.json:
        payload = {
            "final_answer": final_answer,
            "question": summary.question,
            "num_chunks": summary.num_chunks,
            "sub_call_count": summary.sub_call_count,
            "total_input_tokens": summary.total_input_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "total_cost_usd": summary.total_cost_usd,
            "wall_clock_s": summary.wall_clock_s,
            "dry_run": summary.dry_run,
        }
        print(json.dumps(payload, indent=2))
    else:
        print_summary(summary, final_answer)

    return 0


# Expose the mock factory so the smoke test can verify it in isolation.
__all__ = [
    "build_document",
    "chunk_document",
    "run_rlm_loop",
    "synthesize_answer",
    "_dry_run_sub_llm_factory",
    "main",
]


if __name__ == "__main__":
    # Guard: if someone accidentally wires an Anthropic() call through
    # a MagicMock, MagicMock quacks like everything and we don't want that.
    assert not isinstance(run_llm_query, MagicMock)
    sys.exit(main())
