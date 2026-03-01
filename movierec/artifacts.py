from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from .utils import ensure_dir, project_root


ARTIFACTS_DIR = project_root() / "artifacts"
MODEL_ARTIFACTS_FILENAME = "model_artifacts.pkl"


def artifact_path(filename: str, artifacts_dir: Path | str | None = None) -> Path:
    base_dir = ensure_dir(artifacts_dir or ARTIFACTS_DIR)
    return base_dir / filename


def save_pickle(obj: Any, filename: str, artifacts_dir: Path | str | None = None) -> Path:
    path = artifact_path(filename, artifacts_dir=artifacts_dir)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)
    return path


def load_pickle(filename: str, artifacts_dir: Path | str | None = None) -> Any:
    path = artifact_path(filename, artifacts_dir=artifacts_dir)
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_model_artifacts(artifacts: dict[str, Any], artifacts_dir: Path | str | None = None) -> Path:
    return save_pickle(artifacts, MODEL_ARTIFACTS_FILENAME, artifacts_dir=artifacts_dir)


def load_model_artifacts(artifacts_dir: Path | str | None = None) -> dict[str, Any]:
    return load_pickle(MODEL_ARTIFACTS_FILENAME, artifacts_dir=artifacts_dir)
