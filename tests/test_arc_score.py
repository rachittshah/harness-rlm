"""Tests for arc_integration.score — strict grid scoring."""

from __future__ import annotations


from arc_integration.score import (
    grids_equal,
    score_run,
    task_passes_at_k,
)


class TestGridsEqual:
    def test_exact_match(self):
        assert grids_equal([[1, 2], [3, 4]], [[1, 2], [3, 4]])

    def test_one_cell_diff(self):
        assert not grids_equal([[1, 2], [3, 4]], [[1, 2], [3, 5]])

    def test_different_shape(self):
        assert not grids_equal([[1, 2]], [[1, 2], [3, 4]])

    def test_different_row_width(self):
        assert not grids_equal([[1, 2]], [[1, 2, 3]])

    def test_none_inputs(self):
        assert not grids_equal(None, [[1]])
        assert not grids_equal([[1]], None)
        assert not grids_equal(None, None)


class TestTaskPassesAtK:
    def test_first_attempt_passes(self):
        gold = [[1, 2]]
        attempts = [gold, None]
        assert task_passes_at_k(attempts, gold, k=2)

    def test_second_attempt_passes(self):
        gold = [[1, 2]]
        attempts = [None, gold]
        assert task_passes_at_k(attempts, gold, k=2)

    def test_third_attempt_outside_k(self):
        gold = [[1, 2]]
        attempts = [None, None, gold]
        # Only first 2 counted.
        assert not task_passes_at_k(attempts, gold, k=2)

    def test_no_pass(self):
        attempts = [[[9]], [[8]]]
        assert not task_passes_at_k(attempts, [[1]], k=2)


class TestScoreRun:
    def test_all_pass(self):
        predictions = {
            "task_a": {"attempt_1": [[1, 2]], "attempt_2": None},
            "task_b": {"attempt_1": [[3]], "attempt_2": None},
        }
        gold = {"task_a": [[1, 2]], "task_b": [[3]]}
        result = score_run(predictions, gold, k=2)
        assert result["passed_at_k"] == 2
        assert result["pass_rate_pct"] == 100.0

    def test_half_pass(self):
        predictions = {
            "a": {"attempt_1": [[1]], "attempt_2": None},
            "b": {"attempt_1": [[9]], "attempt_2": None},
        }
        gold = {"a": [[1]], "b": [[2]]}
        result = score_run(predictions, gold, k=2)
        assert result["passed_at_k"] == 1
        assert result["pass_rate_pct"] == 50.0

    def test_missing_prediction(self):
        # Task in gold but no prediction → 0
        predictions: dict = {}
        gold = {"a": [[1]]}
        result = score_run(predictions, gold, k=2)
        assert result["passed_at_k"] == 0

    def test_second_attempt_credited(self):
        predictions = {"a": {"attempt_1": [[9]], "attempt_2": [[1]]}}
        gold = {"a": [[1]]}
        result = score_run(predictions, gold, k=2)
        assert result["passed_at_k"] == 1
        per = result["per_task"][0]
        assert per["best_attempt"] == 2

    def test_per_task_records(self):
        predictions = {"a": {"attempt_1": [[1]], "attempt_2": None}}
        gold = {"a": [[1]]}
        result = score_run(predictions, gold, k=2)
        per = result["per_task"][0]
        assert per["task_id"] == "a"
        assert per["passed"] is True
        assert per["best_attempt"] == 1
        assert per["num_attempts"] == 2


class TestRunnerHelpers:
    """Tests for runner-side helpers that don't invoke claude."""

    def test_build_prompt_renders_examples(self):
        from arc_integration.runner import build_prompt

        task = {
            "train": [
                {"input": [[1, 2]], "output": [[2, 1]]},
            ],
            "test": [{"input": [[3, 4]], "output": [[4, 3]]}],
        }
        prompt = build_prompt(task)
        assert "Example 1" in prompt
        assert "1 2" in prompt
        assert "2 1" in prompt
        assert "3 4" in prompt
        # Gold output for the test must NOT appear in the prompt (no leakage).
        assert "4 3" not in prompt

    def test_parse_grid_simple_json(self):
        from arc_integration.runner import parse_grid

        result = parse_grid('{"output": [[1, 2], [3, 4]]}')
        assert result == [[1, 2], [3, 4]]

    def test_parse_grid_with_code_fence(self):
        from arc_integration.runner import parse_grid

        result = parse_grid('```json\n{"output": [[1, 2]]}\n```')
        assert result == [[1, 2]]

    def test_parse_grid_with_commentary(self):
        from arc_integration.runner import parse_grid

        result = parse_grid('I think the answer is: {"output": [[5]]}')
        assert result == [[5]]

    def test_parse_grid_invalid_cell_value(self):
        from arc_integration.runner import parse_grid

        # 11 is out of the 0-9 range.
        assert parse_grid('{"output": [[1, 11]]}') is None

    def test_parse_grid_no_output_key(self):
        from arc_integration.runner import parse_grid

        assert parse_grid('{"answer": [[1]]}') is None

    def test_parse_grid_empty_response(self):
        from arc_integration.runner import parse_grid

        assert parse_grid("") is None
        assert parse_grid(None) is None  # type: ignore[arg-type]
