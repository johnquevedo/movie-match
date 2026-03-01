from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from .user_profile import infer_user_vector


def _movie_title(artifacts: dict[str, Any], movie_id: int) -> str:
    return artifacts["movie_titles"].get(movie_id, f"Movie {movie_id}")


def _build_result(artifacts: dict[str, Any], movie_id: int, score: float) -> dict[str, Any]:
    return {"movie_id": movie_id, "title": _movie_title(artifacts, movie_id), "score": float(score)}


def popularity_recommend(
    artifacts: dict[str, Any],
    rated_movie_ids: list[int] | set[int],
    k: int = 10,
) -> list[dict[str, Any]]:
    excluded = set(rated_movie_ids)
    scores = artifacts["popularity_scores"]
    ranked_indices = np.argsort(scores)[::-1]
    results: list[dict[str, Any]] = []
    for index in ranked_indices:
        movie_id = artifacts["idx_to_movie_id"][int(index)]
        if movie_id in excluded or scores[index] <= 0:
            continue
        results.append(_build_result(artifacts, movie_id, float(scores[index])))
        if len(results) >= k:
            break
    return results


def explain(
    artifacts: dict[str, Any],
    recommended_movie_id: int,
    user_ratings: dict[int, float],
    max_items: int = 3,
) -> str:
    item_factors = artifacts.get("item_factors")
    movie_id_to_idx = artifacts["movie_id_to_idx"]
    positive_movies = [movie_id for movie_id, rating in user_ratings.items() if rating >= 4 and movie_id in movie_id_to_idx]
    if item_factors is None or recommended_movie_id not in movie_id_to_idx or not positive_movies:
        return "Because it matches your rating history."

    rec_vector = item_factors[movie_id_to_idx[recommended_movie_id]]
    rec_norm = np.linalg.norm(rec_vector)
    if rec_norm == 0:
        return "Because it matches your rating history."

    scored: list[tuple[float, int]] = []
    for movie_id in positive_movies:
        movie_vector = item_factors[movie_id_to_idx[movie_id]]
        denom = rec_norm * np.linalg.norm(movie_vector)
        similarity = float(np.dot(rec_vector, movie_vector) / denom) if denom else 0.0
        scored.append((similarity, movie_id))

    scored.sort(reverse=True)
    titles = [_movie_title(artifacts, movie_id) for _, movie_id in scored[:max_items]]
    return "Because you liked: " + ", ".join(titles)


def recommend(
    artifacts: dict[str, Any],
    user_ratings: dict[int, float],
    model: str = "als",
    k: int = 10,
    regularization: float = 0.1,
) -> list[dict[str, Any]]:
    rated_movie_ids = set(user_ratings.keys())
    item_factors = artifacts.get("item_factors")
    if model == "popularity" or item_factors is None:
        results = popularity_recommend(artifacts, rated_movie_ids, k=k)
        for row in results:
            row["explanation"] = explain(artifacts, row["movie_id"], user_ratings)
        return results

    user_vector = infer_user_vector(
        user_ratings,
        item_factors=item_factors,
        movie_id_to_idx=artifacts["movie_id_to_idx"],
        regularization=regularization,
    )
    scores = item_factors @ user_vector
    ranked_indices = np.argsort(scores)[::-1]
    results: list[dict[str, Any]] = []
    for index in ranked_indices:
        movie_id = artifacts["idx_to_movie_id"][int(index)]
        if movie_id in rated_movie_ids:
            continue
        result = _build_result(artifacts, movie_id, float(scores[index]))
        result["explanation"] = explain(artifacts, movie_id, user_ratings)
        results.append(result)
        if len(results) >= k:
            break
    return results


def _sparse_cosine_scores(interaction_matrix: sparse.csr_matrix, movie_index: int) -> np.ndarray:
    item_user = interaction_matrix.T.tocsr()
    target = item_user[movie_index]
    scores = item_user @ target.T
    scores = np.asarray(scores.todense()).ravel().astype(np.float32)
    norms = np.sqrt(item_user.multiply(item_user).sum(axis=1)).A1
    target_norm = norms[movie_index]
    denom = norms * target_norm
    valid = denom > 0
    cosine = np.zeros_like(scores, dtype=np.float32)
    cosine[valid] = scores[valid] / denom[valid]
    return cosine


def similar_movies(
    artifacts: dict[str, Any],
    movie_id: int,
    k: int = 10,
) -> list[dict[str, Any]]:
    movie_id_to_idx = artifacts["movie_id_to_idx"]
    if movie_id not in movie_id_to_idx:
        return []

    movie_index = movie_id_to_idx[movie_id]
    item_factors = artifacts.get("item_factors")
    if item_factors is not None:
        query = item_factors[movie_index]
        norms = np.linalg.norm(item_factors, axis=1)
        denom = norms * np.linalg.norm(query)
        scores = np.zeros(item_factors.shape[0], dtype=np.float32)
        valid = denom > 0
        scores[valid] = (item_factors[valid] @ query) / denom[valid]
    else:
        scores = _sparse_cosine_scores(artifacts["interaction_matrix"], movie_index)

    scores[movie_index] = -np.inf
    ranked_indices = np.argsort(scores)[::-1]
    results: list[dict[str, Any]] = []
    for index in ranked_indices[:k]:
        candidate_movie_id = artifacts["idx_to_movie_id"][int(index)]
        results.append(_build_result(artifacts, candidate_movie_id, float(scores[index])))
    return results


def recommend_for_known_user(
    artifacts: dict[str, Any],
    user_id: int,
    k: int = 10,
    model: str = "als",
) -> list[int]:
    user_index = artifacts["user_id_to_idx"].get(user_id)
    if user_index is None:
        return []

    seen_indices = set(artifacts["interaction_matrix"][user_index].indices.tolist())
    if model == "als" and artifacts.get("user_factors") is not None and artifacts.get("item_factors") is not None:
        user_vector = artifacts["user_factors"][user_index]
        scores = artifacts["item_factors"] @ user_vector
    else:
        scores = artifacts["popularity_scores"].copy()

    ranked_indices = np.argsort(scores)[::-1]
    recommendations: list[int] = []
    for index in ranked_indices:
        if int(index) in seen_indices:
            continue
        recommendations.append(artifacts["idx_to_movie_id"][int(index)])
        if len(recommendations) >= k:
            break
    return recommendations
