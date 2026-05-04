from __future__ import annotations

"""
Сборка pairwise feature-таблицы (публичная reference-версия).

В production-эксперименте эта таблица считалась из:
- эмбеддингов Postgres (semantic cosine)
- времени постов (temporal proximity)
- эвристического entity extraction (entity Jaccard)
- фич внутреннего graph pipeline (Neo4j / neighborhood / relation types)

Публичный репозиторий оставляет *интерфейс* и логику оценки; graph features здесь — placeholders.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class PairwiseRow:
    post_id_a: int
    post_id_b: int
    semantic_score: float
    temporal_score: float
    entity_jaccard: float
    graph_neighborhood_jaccard: float = 0.0
    graph_relation_type_jaccard: float = 0.0
    near_duplicate_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "post_id_a": self.post_id_a,
            "post_id_b": self.post_id_b,
            "semantic_score": self.semantic_score,
            "temporal_score": self.temporal_score,
            "entity_jaccard": self.entity_jaccard,
            "graph_neighborhood_jaccard": self.graph_neighborhood_jaccard,
            "graph_relation_type_jaccard": self.graph_relation_type_jaccard,
            "near_duplicate_score": self.near_duplicate_score,
        }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def temporal_score(ts_a: datetime, ts_b: datetime, half_life_hours: float = 36.0) -> float:
    dt_h = abs((ts_a - ts_b).total_seconds()) / 3600.0
    # экспоненциальное затухание
    return float(2 ** (-dt_h / max(1e-6, half_life_hours)))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / max(1, uni)


def graph_features_hook(post_id_a: int, post_id_b: int) -> dict[str, float]:
    """
    В production-эксперименте эти фичи считались внутренним TeleHub graph pipeline.
    В публичном репозитории оставлен только интерфейс и логика оценки.
    """
    return {
        "graph_neighborhood_jaccard": 0.0,
        "graph_relation_type_jaccard": 0.0,
    }

