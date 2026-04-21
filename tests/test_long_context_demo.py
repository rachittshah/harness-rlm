"""Smoke test for examples/long_context_demo.py.

Runs the demo in --dry-run mode via subprocess and verifies it exits cleanly
and produces the expected final-answer/metrics output. No real API calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

DEMO_PATH = Path(__file__).resolve().parent.parent / "examples" / "long_context_demo.py"


def _skip_if_demo_missing():
    if not DEMO_PATH.exists():
        pytest.skip(f"demo script missing: {DEMO_PATH}")


def test_demo_script_file_exists():
    """The demo file should be present in examples/."""
    assert DEMO_PATH.exists(), f"demo script missing: {DEMO_PATH}"


def test_demo_dry_run_exits_zero():
    """Dry-run with small chunk count should exit 0 and print final answer."""
    _skip_if_demo_missing()

    result = subprocess.run(
        [sys.executable, str(DEMO_PATH), "--chunks", "5", "--dry-run"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"expected exit 0, got {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    out = result.stdout.decode("utf-8")
    assert "FINAL ANSWER" in out
    assert "RLM RUN METRICS" in out
    assert "sub-calls        : 5" in out
    assert "ESTIMATE (dry-run)" in out


def test_demo_dry_run_verbose():
    """Verbose dry-run logs per-chunk progress lines."""
    _skip_if_demo_missing()

    result = subprocess.run(
        [sys.executable, str(DEMO_PATH), "--chunks", "3", "--dry-run", "--verbose"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    out = result.stdout.decode("utf-8")
    # Verbose mode prints per-chunk progress.
    assert "chunk 01/3" in out or "chunk 1/3" in out.replace("01", "1")
    assert "DRY RUN" in out


def test_demo_dry_run_json_output():
    """--json flag emits a parseable JSON summary."""
    _skip_if_demo_missing()

    result = subprocess.run(
        [sys.executable, str(DEMO_PATH), "--chunks", "4", "--dry-run", "--json"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    out = result.stdout.decode("utf-8")
    # The script prints a JSON blob at the end. Find the outermost JSON object.
    start = out.find("{")
    assert start >= 0, f"no JSON found in output: {out!r}"
    payload = json.loads(out[start:])
    assert payload["num_chunks"] == 4
    assert payload["sub_call_count"] == 4
    assert payload["dry_run"] is True
    assert payload["total_cost_usd"] >= 0.0
    assert "final_answer" in payload


def test_demo_no_api_key_without_dry_run_exits_nonzero(monkeypatch):
    """Real-run mode without ANTHROPIC_API_KEY should exit non-zero quickly."""
    _skip_if_demo_missing()

    env = {"PATH": "/usr/bin:/bin"}
    # Do NOT set ANTHROPIC_API_KEY — the script should bail out.
    result = subprocess.run(
        [sys.executable, str(DEMO_PATH), "--chunks", "2"],
        capture_output=True,
        timeout=15,
        check=False,
        env=env,
    )
    assert result.returncode != 0
    assert b"ANTHROPIC_API_KEY" in result.stderr


# --- In-process library tests of helper functions ---------------------------
# These don't spawn subprocesses but verify the internals of the demo module.
def _import_demo_module():
    """Import examples/long_context_demo.py as a module for in-process tests."""
    import importlib.util
    import sys as _sys

    mod_name = "_harness_rlm_demo"
    if mod_name in _sys.modules:
        return _sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(mod_name, DEMO_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so dataclass annotation resolution
    # (which looks up cls.__module__ in sys.modules) works on Python 3.14.
    _sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_document_contains_all_planted_facts():
    _skip_if_demo_missing()

    demo = _import_demo_module()
    doc = demo.build_document(num_paragraphs=200)
    for fact in demo.PLANTED_FACTS.values():
        assert fact in doc
    # Size sanity: ~100KB is what the spec asks for.
    assert 50_000 < len(doc) < 200_000


def test_chunk_document_covers_entire_document():
    _skip_if_demo_missing()

    demo = _import_demo_module()
    doc = demo.build_document(num_paragraphs=60)
    chunks = demo.chunk_document(doc, num_chunks=6)
    assert len(chunks) == 6
    # Concatenating chunks with paragraph separator should recover the document.
    rejoined = "\n\n".join(chunks)
    assert rejoined == doc


def test_chunk_document_raises_on_zero_chunks():
    _skip_if_demo_missing()

    demo = _import_demo_module()
    with pytest.raises(ValueError):
        demo.chunk_document("abc", num_chunks=0)


def test_dry_run_sub_llm_never_hits_network():
    """The dry-run factory returns a callable that never imports anthropic."""
    _skip_if_demo_missing()

    demo = _import_demo_module()
    from harness_rlm.models import LLMQueryRequest

    sub_llm = demo._dry_run_sub_llm_factory()
    # Prompt frames the chunk body between <<< >>> — this is the shape
    # build_chunk_prompt() produces in the real loop.
    req = LLMQueryRequest(
        prompt=(
            "Question: does this mention the capital?\n"
            "<<<\nThe capital is Marquel City in Zephyria.\n>>>"
        )
    )
    resp = sub_llm(req)
    assert resp.cost_usd >= 0.0
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert isinstance(resp.content, str)
    # Response should recognize the capital fact from the chunk body.
    assert "Marquel" in resp.content or "capital" in resp.content.lower()


def test_dry_run_sub_llm_returns_not_in_chunk_when_absent():
    """Dry-run mock returns NOT_IN_CHUNK when no planted fact is in the body."""
    _skip_if_demo_missing()

    demo = _import_demo_module()
    from harness_rlm.models import LLMQueryRequest

    sub_llm = demo._dry_run_sub_llm_factory()
    req = LLMQueryRequest(prompt="Question: anything?\n<<<\nGeneric paragraph.\n>>>")
    resp = sub_llm(req)
    assert "NOT_IN_CHUNK" in resp.content
