import math
import random
from typing import Iterable


FEATURE_NAMES = (
    "keyword_affinity",
    "topic_affinity",
    "source_affinity",
    "post_popularity",
    "freshness",
    "engagement",
    "has_summary",
    "has_media",
    "embedding_similarity",
)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def baseline_score(features: dict[str, float]) -> float:
    weighted = (
        0.26 * features.get("keyword_affinity", 0.0)
        + 0.14 * features.get("topic_affinity", 0.0)
        + 0.22 * features.get("source_affinity", 0.0)
        + 0.12 * features.get("post_popularity", 0.0)
        + 0.10 * features.get("freshness", 0.0)
        + 0.08 * features.get("engagement", 0.0)
        + 0.03 * features.get("has_summary", 0.0)
        + 0.02 * features.get("has_media", 0.0)
        + 0.03 * features.get("embedding_similarity", 0.0)
    )
    return clamp01(weighted)


def logistic_probability(weights: dict[str, float], bias: float, features: dict[str, float]) -> float:
    logit = float(bias)
    for name in FEATURE_NAMES:
        logit += float(weights.get(name, 0.0)) * float(features.get(name, 0.0))
    return sigmoid(logit)


def blend_scores(baseline: float, logistic_prob: float, alpha: float) -> float:
    a = clamp01(alpha)
    return clamp01((a * baseline) + ((1.0 - a) * logistic_prob))


def decision_for_mode(
    *,
    mode: str,
    score: float,
    threshold_not_interesting: float,
    threshold_only_fire: float,
    cold_start: bool,
    epsilon: float,
) -> tuple[bool, str]:
    if mode == "digest_only":
        return False, "digest_only"
    if mode == "all":
        return True, "all_mode"

    if cold_start:
        return True, "safe_default_all"

    if mode not in {"not_interesting", "only_fire"}:
        return True, "unknown_mode_fallback"

    explore = random.random() < epsilon
    if explore:
        return True, "explore"

    if mode == "not_interesting":
        return score >= threshold_not_interesting, "exploit"
    return score >= threshold_only_fire, "exploit"


def reco_error_fallback_decision() -> tuple[bool, str]:
    return True, "reco_error_fallback"


def shadow_mode_decisions(
    *,
    score: float,
    threshold_not_interesting: float,
    threshold_only_fire: float,
    cold_start: bool,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for mode in ("all", "not_interesting", "only_fire", "digest_only"):
        send, reason = decision_for_mode(
            mode=mode,
            score=score,
            threshold_not_interesting=threshold_not_interesting,
            threshold_only_fire=threshold_only_fire,
            cold_start=cold_start,
            epsilon=0.0,
        )
        result[mode] = {"send": send, "reason": reason}
    return result


def online_logistic_update(
    *,
    weights: dict[str, float],
    bias: float,
    features: dict[str, float],
    label: int,
    learning_rate: float,
    l2: float,
) -> tuple[dict[str, float], float]:
    pred = logistic_probability(weights, bias, features)
    error = pred - float(label)
    lr = max(1e-5, learning_rate)
    reg = max(0.0, l2)

    next_weights = dict(weights)
    for name in FEATURE_NAMES:
        w = float(next_weights.get(name, 0.0))
        x = float(features.get(name, 0.0))
        grad = error * x + reg * w
        next_weights[name] = w - (lr * grad)

    next_bias = float(bias) - (lr * error)
    return next_weights, next_bias


def cosine_similarity(vec1: Iterable[float], vec2: Iterable[float]) -> float:
    a = list(vec1)
    b = list(vec2)
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    n1 = math.sqrt(sum(x * x for x in a))
    n2 = math.sqrt(sum(y * y for y in b))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return clamp01((dot / (n1 * n2) + 1.0) / 2.0)


def ema_update_vector(current: list[float], sample: list[float], beta: float) -> list[float]:
    if not sample:
        return current or []
    if not current or len(current) != len(sample):
        return list(sample)
    b = clamp01(beta)
    return [(b * c) + ((1.0 - b) * s) for c, s in zip(current, sample)]
