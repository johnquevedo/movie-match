from __future__ import annotations

from movierec.data import DEFAULT_DATA_DIR, download_movielens_100k


def main() -> None:
    dataset_dir = download_movielens_100k(DEFAULT_DATA_DIR)
    print(f"Downloaded MovieLens 100K to: {dataset_dir}")


if __name__ == "__main__":
    main()
