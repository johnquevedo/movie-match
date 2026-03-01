from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import save_model_artifacts
from .data import DEFAULT_POSITIVE_THRESHOLD, build_positive_interactions

try:
    from implicit.als import AlternatingLeastSquares
except ImportError:  # pragma: no cover - exercised when implicit is not installed
    AlternatingLeastSquares = None


@dataclass
class TrainingConfig:
    positive_threshold: int = DEFAULT_POSITIVE_THRESHOLD
    factors: int = 32
    regularization: float = 0.05
    iterations: int = 20
    alpha: float = 20.0
    random_state: int = 42


def _build_id_maps(ids: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    id_to_idx = {value: index for index, value in enumerate(ids)}
    idx_to_id = {index: value for value, index in id_to_idx.items()}
    return id_to_idx, idx_to_id


def build_interaction_matrix(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    positive_threshold: int = DEFAULT_POSITIVE_THRESHOLD,
) -> tuple[sparse.csr_matrix, dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    positives = build_positive_interactions(ratings, positive_threshold=positive_threshold)
    user_ids = sorted(positives["user_id"].unique().tolist())
    movie_ids = sorted(movies["movie_id"].unique().tolist())
    user_id_to_idx, idx_to_user_id = _build_id_maps(user_ids)
    movie_id_to_idx, idx_to_movie_id = _build_id_maps(movie_ids)

    filtered = positives.loc[positives["movie_id"].isin(movie_id_to_idx)].copy()
    rows = filtered["user_id"].map(user_id_to_idx).to_numpy()
    cols = filtered["movie_id"].map(movie_id_to_idx).to_numpy()
    data = filtered["implicit_strength"].to_numpy(dtype=np.float32)

    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32,
    )
    return matrix, user_id_to_idx, idx_to_user_id, movie_id_to_idx, idx_to_movie_id


def train_als_model(
    interaction_matrix: sparse.csr_matrix,
    config: TrainingConfig,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if AlternatingLeastSquares is None:
        return None, None

    model = AlternatingLeastSquares(
        factors=config.factors,
        regularization=config.regularization,
        iterations=config.iterations,
        random_state=config.random_state,
    )
    confidence_matrix = interaction_matrix * config.alpha
    model.fit(confidence_matrix)
    return model.user_factors.astype(np.float32), model.item_factors.astype(np.float32)


def popularity_scores(interaction_matrix: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(interaction_matrix.sum(axis=0)).ravel().astype(np.float32)


def build_movie_titles(movies: pd.DataFrame) -> dict[int, str]:
    return dict(zip(movies["movie_id"], movies["title"], strict=False))


def train_and_save(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    artifacts_dir: Path | str,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    config = config or TrainingConfig()
    interaction_matrix, user_id_to_idx, idx_to_user_id, movie_id_to_idx, idx_to_movie_id = build_interaction_matrix(
        ratings,
        movies,
        positive_threshold=config.positive_threshold,
    )
    user_factors, item_factors = train_als_model(interaction_matrix, config)
    artifacts: dict[str, Any] = {
        "config": {
            "positive_threshold": config.positive_threshold,
            "factors": config.factors,
            "regularization": config.regularization,
            "iterations": config.iterations,
            "alpha": config.alpha,
            "random_state": config.random_state,
        },
        "als_available": item_factors is not None,
        "interaction_matrix": interaction_matrix,
        "popularity_scores": popularity_scores(interaction_matrix),
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_id_to_idx": user_id_to_idx,
        "idx_to_user_id": idx_to_user_id,
        "movie_id_to_idx": movie_id_to_idx,
        "idx_to_movie_id": idx_to_movie_id,
        "movie_titles": build_movie_titles(movies),
        "movies": movies[["movie_id", "title"]].copy(),
    }
    save_model_artifacts(artifacts, artifacts_dir=artifacts_dir)
    return artifacts
