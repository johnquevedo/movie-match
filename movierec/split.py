from __future__ import annotations

import random

import pandas as pd

from .data import DEFAULT_POSITIVE_THRESHOLD, build_positive_interactions


def per_user_holdout_split(
    ratings: pd.DataFrame,
    positive_threshold: int = DEFAULT_POSITIVE_THRESHOLD,
    min_positive_items: int = 2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positives = build_positive_interactions(ratings, positive_threshold=positive_threshold)
    rng = random.Random(random_state)
    holdout_indices: list[int] = []

    for _, group in positives.groupby("user_id"):
        if len(group) < min_positive_items:
            continue
        sampled_index = rng.choice(group.index.tolist())
        holdout_indices.append(sampled_index)

    test = positives.loc[holdout_indices].copy()
    train = positives.drop(index=holdout_indices).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)
