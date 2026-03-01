from __future__ import annotations

import numpy as np
from scipy import sparse

from movierec.recommend import explain, popularity_recommend, recommend, similar_movies


def sample_artifacts() -> dict:
    item_factors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    return {
        "item_factors": item_factors,
        "user_factors": np.array([[1.0, 0.0]], dtype=np.float32),
        "popularity_scores": np.array([10.0, 8.0, 2.0, 6.0], dtype=np.float32),
        "movie_id_to_idx": {10: 0, 11: 1, 12: 2, 13: 3},
        "idx_to_movie_id": {0: 10, 1: 11, 2: 12, 3: 13},
        "movie_titles": {10: "Alpha", 11: "Beta", 12: "Gamma", 13: "Delta"},
        "interaction_matrix": sparse.csr_matrix([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        "user_id_to_idx": {1: 0},
        "movies": None,
    }


def test_popularity_recommend_excludes_rated_movies() -> None:
    results = popularity_recommend(sample_artifacts(), rated_movie_ids={10}, k=2)
    assert [row["movie_id"] for row in results] == [11, 13]


def test_recommend_returns_embedding_based_results() -> None:
    results = recommend(sample_artifacts(), user_ratings={10: 5}, model="als", k=2)
    assert [row["movie_id"] for row in results] == [11, 13]
    assert results[0]["score"] > results[1]["score"]


def test_similar_movies_uses_factor_similarity() -> None:
    results = similar_movies(sample_artifacts(), movie_id=10, k=2)
    assert [row["movie_id"] for row in results] == [11, 13]


def test_explain_references_positive_ratings() -> None:
    message = explain(sample_artifacts(), recommended_movie_id=11, user_ratings={10: 5, 12: 2})
    assert "Because you liked" in message
    assert "Alpha" in message
