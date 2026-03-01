from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def unique_preserving_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
