"""End-to-end demo: harness-rlm driven by `claude -p` (no API key needed).

What this exercises:
    1. `ClaudeCLILM`  — shell out to `claude -p` for each LM call.
    2. `Predict`      — simple signature-typed Q→A.
    3. `RLM`          — long-context map-reduce with parallel sub-LM dispatch.
    4. `Orchestrator` — composes Predict → RLM with state forwarding.
    5. Trace + cost rollup across providers.

Run:
    uv run python examples/e2e_claude_p.py

Cost envelope (Haiku 4.5 sub, ~5 calls): well under $0.01.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness_rlm.claude_cli_lm import ClaudeCLILM
from harness_rlm.modules import Predict
from harness_rlm.orchestrator import Orchestrator, SessionStore, Step
from harness_rlm.rlm import RLM, RLMConfig


def build_long_context() -> str:
    """Build a ~50K-char document with one hidden fact at chunk index 2."""
    # Three "chunks" of filler + one fact-bearing line in chunk 2.
    chunks: list[str] = []
    filler = "This is filler line about company policy. " * 200  # ~8K chars
    for i in range(6):
        if i == 2:
            # Embed the fact in the middle of the second chunk.
            chunks.append(
                filler[:4000]
                + "\nIMPORTANT: The product launch date is 2027-03-14.\n"
                + filler[4000:]
            )
        else:
            chunks.append(filler)
    return "\n".join(chunks)


def main() -> int:
    t0 = time.perf_counter()
    print("[setup] building 50K-char document with one hidden fact...")
    doc = build_long_context()
    print(f"[setup] doc chars: {len(doc):,}")

    # Use the same model for root and sub via claude -p. In practice you'd use
    # Opus root + Haiku sub for cost. Here both go through claude -p / Haiku.
    root_lm = ClaudeCLILM(model="claude-haiku-4-5")
    sub_lm = ClaudeCLILM(model="claude-haiku-4-5")

    print("\n=== Step 1: Predict (flat) — short context, single call ===")
    short = Predict("question -> answer", lm=root_lm)
    short_pred = short(question="What is 7 * 6? Reply with just the number.")
    print(f"answer: {short_pred.answer}")
    print(f"calls: {short_pred.trace.calls}  cost: ${short_pred.trace.cost_usd:.6f}")

    print("\n=== Step 2: RLM (decomposed) — long context, fan-out + synth ===")
    cfg = RLMConfig(
        flat_char_threshold=10_000,
        chunk_size=20_000,
        overlap=200,
        max_parallel=3,
        max_llm_calls=10,
        sub_max_tokens=256,
        synth_max_tokens=512,
    )
    rlm = RLM(
        "question, document -> answer",
        long_context_field="document",
        config=cfg,
        root_lm=root_lm,
        sub_lm=sub_lm,
    )
    rlm_pred = rlm(
        question="What is the product launch date mentioned in the document?",
        document=doc,
    )
    print(f"answer: {rlm_pred.answer}")
    print(
        f"calls: {rlm_pred.trace.calls}  cost: ${rlm_pred.trace.cost_usd:.6f}  "
        f"latency: {rlm_pred.trace.latency_s:.1f}s"
    )

    print("\n=== Step 3: Orchestrator — chain Predict + RLM with state ===")
    store = SessionStore(name="e2e_claude_p", base_dir=Path("/tmp/rlm"))
    store.clear()
    orchestrator = Orchestrator(
        [
            Step(
                name="rephrase",
                module=Predict("question -> rephrased", lm=root_lm),
                input_builder=lambda s: {"question": s["raw_question"]},
            ),
            Step(
                name="answer",
                module=rlm,
                input_builder=lambda s: {
                    "question": s["rephrase"]["rephrased"],
                    "document": s["raw_question_doc"],
                },
            ),
        ]
    )
    orch_result = orchestrator.run(
        initial_state={
            "raw_question": "When does the product launch?",
            "raw_question_doc": doc,
        },
        session_store=store,
    )
    print(f"final answer: {orch_result.state['answer']['answer'][:200]}...")
    print(f"total calls: {orch_result.trace.calls}  total cost: ${orch_result.trace.cost_usd:.6f}")
    print(f"session events on disk: {len(store.read())}")

    elapsed = time.perf_counter() - t0
    total_cost = root_lm.stats()["cost_usd"] + sub_lm.stats()["cost_usd"]
    total_calls = root_lm.stats()["calls"] + sub_lm.stats()["calls"]
    print("\n=== Summary ===")
    print(
        f"wall_clock: {elapsed:.1f}s   total_calls: {total_calls}   total_cost: ${total_cost:.6f}"
    )

    # Smoke-test acceptance: RLM must find the fact (date 2027-03-14).
    if "2027-03-14" in rlm_pred.answer:
        print("[PASS] RLM correctly extracted the embedded date.")
        return 0
    print(f"[FAIL] RLM did not surface the date 2027-03-14. Got: {rlm_pred.answer[:200]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
