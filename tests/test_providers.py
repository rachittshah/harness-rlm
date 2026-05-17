"""Tests for harness_rlm.providers — OpenAILM, LiteLLMLM (no network)."""

from __future__ import annotations

import pytest

from harness_rlm.providers import _openai_cost


class TestOpenAICost:
    def test_gpt5_mini_rate(self):
        # 1M input @ $1, 1M output @ $4 → $5 total
        cost = _openai_cost("gpt-5-mini", 1_000_000, 1_000_000)
        assert cost == pytest.approx(5.0, rel=0.01)

    def test_o3_rate(self):
        # 1M input @ $15, 1M output @ $60 → $75 total
        cost = _openai_cost("o3", 1_000_000, 1_000_000)
        assert cost == pytest.approx(75.0, rel=0.01)

    def test_unknown_falls_back(self):
        # Falls back to gpt-5-mini pricing.
        cost = _openai_cost("nonexistent-model", 1_000_000, 1_000_000)
        assert cost == pytest.approx(5.0, rel=0.01)


class TestImportSurface:
    def test_openai_lm_import(self):
        from harness_rlm.providers import OpenAILM

        assert callable(OpenAILM)

    def test_litellm_lm_import(self):
        from harness_rlm.providers import LiteLLMLM

        assert callable(LiteLLMLM)
