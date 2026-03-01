"""MovieMatch package."""

from .data import DEFAULT_DATA_DIR, download_movielens_100k, load_movies, load_ratings

__all__ = [
    "DEFAULT_DATA_DIR",
    "download_movielens_100k",
    "load_movies",
    "load_ratings",
]
