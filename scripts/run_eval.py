from __future__ import annotations

from pathlib import Path

import pandas as pd

from movierec.data import DEFAULT_DATA_DIR, load_movies, load_ratings
from movierec.metrics import coverage_at_k, ndcg_at_k, recall_at_k
from movierec.recommend import recommend_for_known_user
from movierec.split import per_user_holdout_split
from movierec.train import TrainingConfig, train_and_save


def evaluate_model(artifacts: dict, test: pd.DataFrame, model: str, k: int = 10) -> dict[str, float]:
    grouped = test.groupby("user_id")["movie_id"].apply(set)
    recalls: list[float] = []
    ndcgs: list[float] = []
    recommendation_lists: list[list[int]] = []

    for user_id, relevant_ids in grouped.items():
        recommended_ids = recommend_for_known_user(artifacts, int(user_id), k=k, model=model)
        recommendation_lists.append(recommended_ids)
        recalls.append(recall_at_k(recommended_ids, relevant_ids, k=k))
        ndcgs.append(ndcg_at_k(recommended_ids, relevant_ids, k=k))

    return {
        "Recall@10": sum(recalls) / len(recalls) if recalls else 0.0,
        "NDCG@10": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "Coverage@10": coverage_at_k(recommendation_lists, catalog_size=len(artifacts["movie_id_to_idx"]), k=k),
    }


def print_metrics_table(results: dict[str, dict[str, float]]) -> None:
    header = f"{'Model':<12} {'Recall@10':>10} {'NDCG@10':>10} {'Coverage@10':>12}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in results.items():
        print(
            f"{model_name:<12} "
            f"{metrics['Recall@10']:>10.4f} "
            f"{metrics['NDCG@10']:>10.4f} "
            f"{metrics['Coverage@10']:>12.4f}"
        )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ratings = load_ratings(DEFAULT_DATA_DIR)
    movies = load_movies(DEFAULT_DATA_DIR)
    train, test = per_user_holdout_split(ratings)
    artifacts = train_and_save(train, movies, artifacts_dir=root / "artifacts", config=TrainingConfig())
    results = {
        "ALS": evaluate_model(artifacts, test, model="als", k=10),
        "Popularity": evaluate_model(artifacts, test, model="popularity", k=10),
    }
    print_metrics_table(results)


if __name__ == "__main__":
    main()
