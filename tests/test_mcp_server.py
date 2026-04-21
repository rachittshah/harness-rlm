"""Tests for harness_rlm.mcp_server — cost calc, fallback, sub-call logging.

Never hits the real Anthropic API. All network paths are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# compute_cost
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model, input_tokens, output_tokens, expected_usd",
    [
        # Haiku 4.5: $1/M in, $5/M out. 1M in + 1M out = $6.00.
        ("claude-haiku-4-5-20251001", 1_000_000, 1_000_000, 6.0),
        # Sonnet 4.6: $3/M in, $15/M out. 1M in + 1M out = $18.00.
        ("claude-sonnet-4-6-20250101", 1_000_000, 1_000_000, 18.0),
        # Opus 4.7: $5/M in, $25/M out. 1M in + 1M out = $30.00.
        ("claude-opus-4-7-20260301", 1_000_000, 1_000_000, 30.0),
        # Sub-million realistic case on Haiku: 2000 in + 500 out.
        # = (2000 * 1 + 500 * 5) / 1e6 = 0.0045.
        ("claude-haiku-4-5-20251001", 2000, 500, 0.0045),
        # Zero tokens — cost is zero regardless of model.
        ("claude-opus-4-7", 0, 0, 0.0),
    ],
)
def test_cost_calculation(model, input_tokens, output_tokens, expected_usd):
    from harness_rlm.mcp_server import compute_cost

    got = compute_cost(model, input_tokens, output_tokens)
    assert got == pytest.approx(expected_usd, rel=1e-9, abs=1e-12)


def test_cost_fallback_unknown_model():
    """Unknown model strings fall back to Haiku rates silently — no raise."""
    from harness_rlm.mcp_server import compute_cost

    # Haiku rates: $1/M in + $5/M out. 1000 in + 200 out = (1000 + 1000) / 1e6 = 0.002.
    got = compute_cost("totally-unknown-model-xyz", 1000, 200)
    assert got == pytest.approx(0.002)

    # Empty string: same fallback.
    assert compute_cost("", 1000, 200) == pytest.approx(0.002)


def test_cost_prefix_match_dated_variants():
    """Dated IDs like claude-sonnet-4-6-20250315 resolve via prefix match."""
    from harness_rlm.mcp_server import compute_cost

    # Sonnet rates: $3/M in + $15/M out. 1000 in + 100 out = (3000 + 1500) / 1e6 = 0.0045.
    assert compute_cost("claude-sonnet-4-6-20250315", 1000, 100) == pytest.approx(0.0045)


# ---------------------------------------------------------------------------
# run_llm_query: end-to-end with mocked Anthropic client + isolated log path
# ---------------------------------------------------------------------------
def _build_fake_anthropic_response(text: str, input_tokens: int, output_tokens: int):
    """Construct a fake anthropic.Messages.create() return value."""
    block = MagicMock()
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    msg.usage = MagicMock()
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    return msg


def _redirect_log(monkeypatch: pytest.MonkeyPatch, log_path: Path) -> None:
    """Redirect sub-call log writes into `log_path`.

    We can't just monkeypatch `mcp_server.SUB_CALLS_LOG` because
    `_append_sub_call_log(path: Path = SUB_CALLS_LOG)` captures the
    original value as a default-argument binding. Patch the function itself.
    """
    from harness_rlm import mcp_server

    monkeypatch.setattr(mcp_server, "SUB_CALLS_LOG", log_path)

    original = mcp_server._append_sub_call_log

    def redirected(log_entry, path=log_path):
        return original(log_entry, path=path)

    monkeypatch.setattr(mcp_server, "_append_sub_call_log", redirected)


def test_sub_call_log_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Calling run_llm_query appends a JSON line to SUB_CALLS_LOG."""
    from harness_rlm import mcp_server
    from harness_rlm.models import LLMQueryRequest

    # Redirect the log target into tmp_path so we don't pollute /tmp/rlm.
    log_path = tmp_path / "sub_calls.jsonl"
    _redirect_log(monkeypatch, log_path)

    # Build a fake anthropic.Anthropic client whose messages.create returns canned data.
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _build_fake_anthropic_response(
        text="hello from mock",
        input_tokens=123,
        output_tokens=45,
    )
    fake_constructor = MagicMock(return_value=fake_client)
    monkeypatch.setattr(mcp_server.anthropic, "Anthropic", fake_constructor)

    req = LLMQueryRequest(prompt="ping", max_tokens=32)
    resp = mcp_server.run_llm_query(req, api_key="fake-key")

    # Verify Anthropic was called with expected args (api_key passed through).
    fake_constructor.assert_called_once_with(api_key="fake-key")
    create_kwargs = fake_client.messages.create.call_args.kwargs
    assert create_kwargs["model"] == req.model
    assert create_kwargs["max_tokens"] == 32
    assert create_kwargs["messages"] == [{"role": "user", "content": "ping"}]
    # system is None in the request → should NOT be forwarded as a kwarg.
    assert "system" not in create_kwargs

    # Verify the response object.
    assert resp.content == "hello from mock"
    assert resp.input_tokens == 123
    assert resp.output_tokens == 45
    # Haiku default rates: (123 * 1 + 45 * 5) / 1e6 = 0.000348.
    assert resp.cost_usd == pytest.approx(0.000348)

    # Verify a line was appended to the log.
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["prompt_preview"] == "ping"
    assert entry["response_chars"] == len("hello from mock")
    assert entry["model"] == req.model
    assert entry["cost_usd"] == pytest.approx(0.000348)
    assert "timestamp" in entry


def test_sub_call_log_forwards_system_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """If system is set on the request, it's forwarded to messages.create."""
    from harness_rlm import mcp_server
    from harness_rlm.models import LLMQueryRequest

    _redirect_log(monkeypatch, tmp_path / "sub_calls.jsonl")

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _build_fake_anthropic_response(
        text="ok", input_tokens=5, output_tokens=2
    )
    monkeypatch.setattr(mcp_server.anthropic, "Anthropic", MagicMock(return_value=fake_client))

    req = LLMQueryRequest(prompt="x", system="be terse")
    mcp_server.run_llm_query(req, api_key="fake-key")

    create_kwargs = fake_client.messages.create.call_args.kwargs
    assert create_kwargs["system"] == "be terse"


def test_sub_call_log_multiple_appends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Multiple calls append multiple lines (not overwrite)."""
    from harness_rlm import mcp_server
    from harness_rlm.models import LLMQueryRequest

    log_path = tmp_path / "sub_calls.jsonl"
    _redirect_log(monkeypatch, log_path)

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _build_fake_anthropic_response(
        text="x", input_tokens=1, output_tokens=1
    )
    monkeypatch.setattr(mcp_server.anthropic, "Anthropic", MagicMock(return_value=fake_client))

    for i in range(3):
        mcp_server.run_llm_query(LLMQueryRequest(prompt=f"p{i}"), api_key="k")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    previews = [json.loads(line)["prompt_preview"] for line in lines]
    assert previews == ["p0", "p1", "p2"]


def test_prompt_preview_truncated_to_200_chars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """prompt_preview stores only first 200 chars of the prompt."""
    from harness_rlm import mcp_server
    from harness_rlm.models import LLMQueryRequest

    log_path = tmp_path / "sub_calls.jsonl"
    _redirect_log(monkeypatch, log_path)

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _build_fake_anthropic_response(
        text="r", input_tokens=1, output_tokens=1
    )
    monkeypatch.setattr(mcp_server.anthropic, "Anthropic", MagicMock(return_value=fake_client))

    long_prompt = "a" * 500
    mcp_server.run_llm_query(LLMQueryRequest(prompt=long_prompt), api_key="k")

    entry = json.loads(log_path.read_text().splitlines()[0])
    assert len(entry["prompt_preview"]) == 200
    assert entry["prompt_preview"] == "a" * 200


def test_require_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch):
    """_require_api_key raises RuntimeError when env var is unset."""
    from harness_rlm import mcp_server

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        mcp_server._require_api_key()


def test_require_api_key_returns_env_value(monkeypatch: pytest.MonkeyPatch):
    from harness_rlm import mcp_server

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-xyz")
    assert mcp_server._require_api_key() == "sk-test-xyz"
