from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from .utils import ensure_dir, project_root

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEFAULT_DATA_DIR = project_root() / "artifacts" / "data"
DEFAULT_POSITIVE_THRESHOLD = 4


def download_movielens_100k(data_dir: Path | str = DEFAULT_DATA_DIR, force: bool = False) -> Path:
    data_dir = ensure_dir(data_dir)
    dataset_dir = data_dir / "ml-100k"
    archive_path = data_dir / "ml-100k.zip"

    if dataset_dir.exists() and not force:
        return dataset_dir

    if archive_path.exists() and force:
        archive_path.unlink()
    if dataset_dir.exists() and force:
        shutil.rmtree(dataset_dir)

    urlretrieve(MOVIELENS_100K_URL, archive_path)
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(data_dir)
    return dataset_dir


def load_ratings(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    ratings_path = Path(data_dir) / "ml-100k" / "u.data"
    columns = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=columns,
        engine="python",
    )
    return ratings.astype({"user_id": int, "movie_id": int, "rating": int, "timestamp": int})


def load_movies(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    movies_path = Path(data_dir) / "ml-100k" / "u.item"
    columns = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "action",
        "adventure",
        "animation",
        "children",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "fantasy",
        "film_noir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sci_fi",
        "thriller",
        "war",
        "western",
    ]
    movies = pd.read_csv(
        movies_path,
        sep="|",
        names=columns,
        encoding="latin-1",
        engine="python",
    )
    return movies[["movie_id", "title"]].astype({"movie_id": int})


def build_positive_interactions(
    ratings: pd.DataFrame,
    positive_threshold: int = DEFAULT_POSITIVE_THRESHOLD,
) -> pd.DataFrame:
    positives = ratings.loc[ratings["rating"] >= positive_threshold, ["user_id", "movie_id", "rating"]].copy()
    positives["implicit_strength"] = 1.0
    return positives
