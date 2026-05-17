"""ARC-AGI evaluation harness for harness-rlm.

Two pieces:
  runner.py — calls `claude -p` per task with grid prompt, parses response
  score.py  — pass@k strict-grid-match scoring + result aggregation

ARC-AGI scoring is pass@k (default k=2): the model gets up to k attempts per
test input. Pass iff any attempt's output grid is bit-exact-equal to the gold.
"""
