from __future__ import annotations

from movierec.metrics import coverage_at_k, ndcg_at_k, recall_at_k


def test_recall_at_k() -> None:
    assert recall_at_k([1, 2, 3], {2, 5}, k=3) == 0.5


def test_ndcg_at_k() -> None:
    score = ndcg_at_k([3, 2, 1], {2, 3}, k=3)
    assert 0.9 < score <= 1.0


def test_coverage_at_k() -> None:
    score = coverage_at_k([[1, 2], [2, 3]], catalog_size=5, k=2)
    assert score == 0.6
