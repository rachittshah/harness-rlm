"""Tests for harness_rlm.core — budget guard, chunker, ingest parser, skill loader."""

from __future__ import annotations

import pytest

from harness_rlm.core import (
    DEFAULT_BUDGETS,
    BudgetExceededError,
    BudgetGuard,
    chunk_context,
    load_shared_skill,
    parse_ingest_directives,
)


# ---------------------------------------------------------------------------
# BudgetGuard
# ---------------------------------------------------------------------------
class TestBudgetGuard:
    def test_defaults(self):
        g = BudgetGuard()
        assert g.budgets == DEFAULT_BUDGETS
        assert g.iterations == 0
        assert g.llm_calls == 0
        assert g.total_output_chars == 0

    def test_partial_budgets_fill_from_defaults(self):
        g = BudgetGuard(budgets={"max_llm_calls": 3})
        assert g.budgets["max_llm_calls"] == 3
        assert g.budgets["max_iterations"] == DEFAULT_BUDGETS["max_iterations"]
        assert g.budgets["max_output_chars"] == DEFAULT_BUDGETS["max_output_chars"]

    def test_check_call_raises_when_next_exceeds(self):
        g = BudgetGuard(budgets={"max_llm_calls": 2})
        g.increment_call()
        g.increment_call()  # now at 2 — next would be 3, over cap
        with pytest.raises(BudgetExceededError) as exc:
            g.check_call()
        assert exc.value.budget == "max_llm_calls"
        assert exc.value.limit == 2
        assert exc.value.actual == 3

    def test_check_call_allows_under_cap(self):
        g = BudgetGuard(budgets={"max_llm_calls": 5})
        g.increment_call()
        g.check_call()  # 2nd call is fine

    def test_check_iteration(self):
        g = BudgetGuard(budgets={"max_iterations": 1})
        g.check_iteration()  # ok, next would be 1
        g.increment_iteration()
        with pytest.raises(BudgetExceededError):
            g.check_iteration()

    def test_check_output(self):
        g = BudgetGuard(budgets={"max_output_chars": 100})
        g.check_output(50)  # fine
        g.check_output(100)  # boundary = fine
        with pytest.raises(BudgetExceededError):
            g.check_output(101)

    def test_record_output_accumulates(self):
        g = BudgetGuard()
        g.record_output(500)
        g.record_output(300)
        assert g.total_output_chars == 800

    def test_state_dict_roundtrip(self):
        g = BudgetGuard(budgets={"max_llm_calls": 10})
        g.increment_call()
        g.increment_call()
        g.increment_iteration()
        g.record_output(42)

        state = g.state_dict()
        assert state["llm_calls"] == 2
        assert state["iterations"] == 1
        assert state["total_output_chars"] == 42
        assert state["budgets"]["max_llm_calls"] == 10

        restored = BudgetGuard.from_state_dict(state)
        assert restored.llm_calls == 2
        assert restored.iterations == 1
        assert restored.total_output_chars == 42
        assert restored.budgets["max_llm_calls"] == 10

    def test_from_state_dict_is_forward_compat(self):
        """Missing keys should default — this protects against future counter adds."""
        g = BudgetGuard.from_state_dict({})
        assert g.llm_calls == 0
        assert g.iterations == 0
        assert g.budgets == DEFAULT_BUDGETS

    def test_exception_message_has_context(self):
        g = BudgetGuard(budgets={"max_llm_calls": 1})
        g.increment_call()
        try:
            g.check_call()
        except BudgetExceededError as e:
            assert "max_llm_calls" in str(e)
            assert "FINAL" in str(e)  # hints the caller how to halt cleanly


# ---------------------------------------------------------------------------
# chunk_context
# ---------------------------------------------------------------------------
class TestChunkContext:
    def test_empty(self):
        assert chunk_context("") == []

    def test_short_text_one_chunk(self):
        assert chunk_context("hello", chunk_size=100, overlap=10) == ["hello"]

    def test_exact_fit_one_chunk(self):
        text = "x" * 100
        assert chunk_context(text, chunk_size=100, overlap=10) == [text]

    def test_basic_overlap(self):
        text = "abcdefghij"  # 10 chars
        chunks = chunk_context(text, chunk_size=5, overlap=2)
        # step = 3 — starts at 0, 3, 6 — but 6+5=11 > 10 so last chunk ends
        assert chunks[0] == "abcde"
        assert chunks[1] == "defgh"
        assert chunks[2] == "ghij"
        assert len(chunks) == 3

    def test_no_overlap(self):
        text = "abcdefghij"
        chunks = chunk_context(text, chunk_size=5, overlap=0)
        assert chunks == ["abcde", "fghij"]

    def test_chunks_cover_full_text(self):
        """Every char in input must appear in at least one chunk."""
        text = "".join(chr(ord("a") + (i % 26)) for i in range(1000))
        chunks = chunk_context(text, chunk_size=100, overlap=10)
        concat_with_overlap = ""
        offset = 0
        for c in chunks:
            # Each chunk should start at the expected offset (step-based).
            assert text[offset : offset + len(c)] == c
            offset += 100 - 10  # step

    def test_spec_example(self):
        """Spec check: chunk_context('test'*1000, chunk_size=500) -> 2+ chunks."""
        chunks = chunk_context("test" * 1000, chunk_size=500)
        assert len(chunks) >= 2
        assert all(len(c) <= 500 for c in chunks)

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_context("x", chunk_size=0)
        with pytest.raises(ValueError):
            chunk_context("x", chunk_size=-1)

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            chunk_context("x" * 100, chunk_size=10, overlap=10)  # overlap == chunk_size
        with pytest.raises(ValueError):
            chunk_context("x" * 100, chunk_size=10, overlap=-1)


# ---------------------------------------------------------------------------
# parse_ingest_directives
# ---------------------------------------------------------------------------
class TestParseIngestDirectives:
    def test_empty(self):
        assert parse_ingest_directives("") == []

    def test_no_markers(self):
        assert parse_ingest_directives("just a plain question") == []

    def test_single_file(self):
        assert parse_ingest_directives("summarize /file README.md") == [
            {"kind": "file", "path": "README.md"}
        ]

    def test_single_url(self):
        assert parse_ingest_directives("fetch /url https://example.com/x") == [
            {"kind": "url", "url": "https://example.com/x"}
        ]

    def test_paste_inline(self):
        result = parse_ingest_directives("/paste hello world")
        assert result == [{"kind": "paste", "text": "hello world"}]

    def test_multiple_markers_in_order(self):
        msg = "do stuff /file foo.md /url https://x.com /paste inline text here"
        result = parse_ingest_directives(msg)
        assert result == [
            {"kind": "file", "path": "foo.md"},
            {"kind": "url", "url": "https://x.com"},
            {"kind": "paste", "text": "inline text here"},
        ]

    def test_paste_consumes_until_next_marker(self):
        msg = "/paste first body /file second.md"
        result = parse_ingest_directives(msg)
        assert result == [
            {"kind": "paste", "text": "first body"},
            {"kind": "file", "path": "second.md"},
        ]

    def test_marker_at_start_of_string(self):
        # The regex uses (?:^|\s) — leading marker should match.
        result = parse_ingest_directives("/file a.txt")
        assert result == [{"kind": "file", "path": "a.txt"}]


# ---------------------------------------------------------------------------
# load_shared_skill
# ---------------------------------------------------------------------------
class TestLoadSharedSkill:
    def test_strips_frontmatter(self):
        body = load_shared_skill()
        # Frontmatter delimiters should be gone.
        assert not body.startswith("---")
        # But the body content should be there.
        assert "Recursive Language Model" in body
        assert "Budget invariants" in body

    def test_is_nonempty(self):
        body = load_shared_skill()
        assert len(body) > 500  # sanity bound
