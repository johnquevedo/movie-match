from __future__ import annotations

from pathlib import Path

from movierec.data import DEFAULT_DATA_DIR, load_movies, load_ratings
from movierec.train import TrainingConfig, train_and_save


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts_dir = root / "artifacts"
    ratings = load_ratings(DEFAULT_DATA_DIR)
    movies = load_movies(DEFAULT_DATA_DIR)
    artifacts = train_and_save(ratings, movies, artifacts_dir=artifacts_dir, config=TrainingConfig())
    model_status = "ALS" if artifacts["als_available"] else "popularity-only fallback"
    print(f"Saved model artifacts to: {artifacts_dir}")
    print(f"Model status: {model_status}")


if __name__ == "__main__":
    main()
