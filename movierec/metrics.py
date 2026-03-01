from __future__ import annotations

import math
from collections.abc import Iterable


def recall_at_k(recommended_ids: Iterable[int], relevant_ids: set[int], k: int = 10) -> float:
    if not relevant_ids:
        return 0.0
    recommended = list(recommended_ids)[:k]
    hits = sum(1 for movie_id in recommended if movie_id in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(recommended_ids: Iterable[int], relevant_ids: set[int], k: int = 10) -> float:
    recommended = list(recommended_ids)[:k]
    if not relevant_ids:
        return 0.0

    dcg = 0.0
    for rank, movie_id in enumerate(recommended, start=1):
        if movie_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def coverage_at_k(recommendation_lists: Iterable[Iterable[int]], catalog_size: int, k: int = 10) -> float:
    if catalog_size == 0:
        return 0.0
    covered: set[int] = set()
    for recs in recommendation_lists:
        covered.update(list(recs)[:k])
    return len(covered) / catalog_size
