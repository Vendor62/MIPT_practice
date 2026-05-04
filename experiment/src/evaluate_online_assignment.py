from __future__ import annotations

"""
Оценка online assignment (reference-реализация).

Ключевая идея:
- Посты приходят в хронологическом порядке.
- Для каждого поста: либо прикрепляем к существующему story cluster, если similarity >= threshold, либо создаём новую story.
- Threshold тюнится только на dev с anti-merge ограничениями.
- Gold labels используются только для расчёта метрик и тюнинга порогов; никогда — внутри решения assignment.
"""

from dataclasses import dataclass
from typing import Any, Callable

from .metrics import evaluate


ScoreFn = Callable[[int, int], float]


@dataclass
class OnlineCfg:
    threshold: float


def simulate_online(order: list[int], score: ScoreFn, cfg: OnlineCfg) -> dict[int, str]:
    next_story = 1
    story_of: dict[int, str] = {}
    members: dict[str, list[int]] = {}
    for pid in order:
        best_sid = None
        best_sc = -1.0
        for sid, mids in members.items():
            # representative = последний пост в истории (простая online-эвристика)
            rep = mids[-1]
            sc = float(score(pid, rep))
            if sc > best_sc:
                best_sc, best_sid = sc, sid
        if best_sid is not None and best_sc >= cfg.threshold:
            story_of[pid] = best_sid
            members[best_sid].append(pid)
        else:
            sid = f"PRED_{next_story:05d}"
            next_story += 1
            story_of[pid] = sid
            members[sid] = [pid]
    return story_of


def tune_threshold_dev(
    *,
    order_train_dev: list[int],
    order_full: list[int],
    score: ScoreFn,
    gold: dict[int, str],
    dev_ids: set[int],
    candidate_thresholds: list[float],
    min_pred_story_ratio: float = 0.5,
    max_false_merge_rate: float = 0.5,
    min_pairwise_precision: float = 0.10,
) -> tuple[float, dict[str, Any]]:
    gold_story_count_dev = len({gold[p] for p in dev_ids})
    min_pred_stories = max(2, int(min_pred_story_ratio * gold_story_count_dev))

    best_thr = candidate_thresholds[0]
    best_obj = -1e9
    best_dev_metrics: dict[str, Any] = {}
    n_ok = 0

    for thr in candidate_thresholds:
        pred = simulate_online(order_train_dev, score, OnlineCfg(threshold=thr))
        m = evaluate(pred, gold, dev_ids, order_train_dev).as_dict()
        # ограничения (anti-merge и др.)
        if int(m["predicted_story_count"]) < min_pred_stories:
            continue
        if float(m["false_merge_rate"]) > max_false_merge_rate:
            continue
        if float(m["pairwise_cluster_precision"]) < min_pairwise_precision:
            continue
        n_ok += 1
        obj = (
            float(m["pairwise_cluster_f1"])
            + 0.5 * float(m["adjusted_rand_index"])
            + 0.25 * float(m["nmi"])
            - 0.5 * float(m["false_merge_rate"])
            - 0.25 * float(m["false_split_rate"])
        )
        if obj > best_obj:
            best_obj = obj
            best_thr = thr
            best_dev_metrics = m

    info = {"n_constraint_candidates": n_ok, "dev_metrics_at_best": best_dev_metrics, "objective": best_obj}
    return best_thr, info

