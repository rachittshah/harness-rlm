"""Tests for harness_rlm.signatures."""

from __future__ import annotations

import pytest

from harness_rlm.signatures import Field_, Signature, SignatureParseError


class TestShorthand:
    def test_basic(self):
        sig = Signature("question -> answer")
        assert sig.input_names() == ["question"]
        assert sig.output_names() == ["answer"]

    def test_multiple_inputs_outputs(self):
        sig = Signature("question, context -> reasoning, answer")
        assert sig.input_names() == ["question", "context"]
        assert sig.output_names() == ["reasoning", "answer"]

    def test_no_inputs_allowed(self):
        sig = Signature(" -> hello")
        assert sig.input_names() == []
        assert sig.output_names() == ["hello"]

    def test_missing_arrow_raises(self):
        with pytest.raises(ValueError, match="Shorthand must look like"):
            Signature("just an input")

    def test_missing_outputs_raises(self):
        # Either "no outputs" (regex matched but RHS empty) or shorthand format
        # error — both are acceptable failure modes for "->" with nothing after.
        with pytest.raises(ValueError):
            Signature("a, b ->")


class TestExplicit:
    def test_list_inputs(self):
        sig = Signature(inputs=["a", "b"], outputs=["c"])
        assert sig.input_names() == ["a", "b"]
        assert sig.output_names() == ["c"]

    def test_dict_inputs_carry_desc(self):
        sig = Signature(
            inputs={"q": "the question"},
            outputs={"a": "the answer"},
            instruction="Be brief.",
        )
        assert sig.inputs[0].desc == "the question"
        assert sig.outputs[0].desc == "the answer"
        assert sig.instruction == "Be brief."

    def test_no_outputs_raises(self):
        with pytest.raises(ValueError, match="at least one output"):
            Signature(inputs=["a"], outputs=[])

    def test_both_forms_rejected(self):
        with pytest.raises(ValueError, match="either a shorthand"):
            Signature("a -> b", inputs=["x"])

    def test_invalid_field_name_raises(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            Field_("bad name!")


class TestPromptRender:
    def test_basic_layout(self):
        sig = Signature("question -> answer", instruction="Be brief.")
        sig.instruction = "Be brief."
        prompt = sig.render_prompt({"question": "What is 2+2?"})
        assert "Be brief." in prompt
        assert "question" in prompt
        assert "What is 2+2?" in prompt
        # Output schema must be present so the parser is unambiguous.
        assert "answer:" in prompt

    def test_missing_input_raises(self):
        sig = Signature("question, context -> answer")
        with pytest.raises(KeyError, match="missing required inputs"):
            sig.render_prompt({"question": "..."})


class TestResponseParse:
    def test_simple(self):
        sig = Signature("q -> answer")
        parsed = sig.parse_response("answer: forty-two")
        assert parsed == {"answer": "forty-two"}

    def test_multiline_value(self):
        sig = Signature("q -> answer")
        parsed = sig.parse_response("answer: line one\nline two\nline three")
        assert parsed["answer"].startswith("line one")
        assert "line three" in parsed["answer"]

    def test_two_outputs(self):
        sig = Signature("q -> reasoning, answer")
        text = "reasoning: think think\nanswer: 42"
        parsed = sig.parse_response(text)
        assert parsed == {"reasoning": "think think", "answer": "42"}

    def test_markdown_bold_labels(self):
        sig = Signature("q -> answer")
        parsed = sig.parse_response("**answer**: forty-two")
        assert parsed == {"answer": "forty-two"}

    def test_missing_field_raises(self):
        sig = Signature("q -> answer")
        with pytest.raises(SignatureParseError, match="missing fields"):
            sig.parse_response("there is no label here")

    def test_case_insensitive_label(self):
        sig = Signature("q -> Answer")
        parsed = sig.parse_response("answer: 42")
        assert parsed == {"Answer": "42"}


class TestMutation:
    def test_with_instruction_returns_copy(self):
        sig = Signature("q -> a", instruction="old")
        sig2 = sig.with_instruction("new")
        assert sig.instruction == "old"
        assert sig2.instruction == "new"
        assert sig.input_names() == sig2.input_names()
