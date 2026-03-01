from __future__ import annotations

import numpy as np


def ratings_to_targets(ratings: dict[int, float]) -> tuple[list[int], np.ndarray]:
    movie_ids = list(ratings.keys())
    targets = np.array([float(score) - 3.0 for score in ratings.values()], dtype=np.float32)
    return movie_ids, targets


def infer_user_vector(
    ratings: dict[int, float],
    item_factors: np.ndarray,
    movie_id_to_idx: dict[int, int],
    regularization: float = 0.1,
) -> np.ndarray:
    rated_movie_ids, targets = ratings_to_targets(ratings)
    known_movie_ids = [movie_id for movie_id in rated_movie_ids if movie_id in movie_id_to_idx]
    if not known_movie_ids:
        raise ValueError("No rated movies exist in the trained catalog.")

    rows = np.array([movie_id_to_idx[movie_id] for movie_id in known_movie_ids], dtype=int)
    factor_matrix = item_factors[rows]
    target_vector = np.array([ratings[movie_id] - 3.0 for movie_id in known_movie_ids], dtype=np.float32)

    lhs = factor_matrix.T @ factor_matrix + regularization * np.eye(item_factors.shape[1], dtype=np.float32)
    rhs = factor_matrix.T @ target_vector
    return np.linalg.solve(lhs, rhs)
