from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class Metrics:
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    ari: float
    nmi: float
    false_merge_rate: float
    false_split_rate: float
    predicted_story_count: int
    gold_story_count: int
    post_assignment_accuracy: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairwise_cluster_precision": self.pairwise_precision,
            "pairwise_cluster_recall": self.pairwise_recall,
            "pairwise_cluster_f1": self.pairwise_f1,
            "adjusted_rand_index": self.ari,
            "nmi": self.nmi,
            "false_merge_rate": self.false_merge_rate,
            "false_split_rate": self.false_split_rate,
            "predicted_story_count": float(self.predicted_story_count),
            "gold_story_count": float(self.gold_story_count),
            "post_assignment_accuracy": self.post_assignment_accuracy,
        }


def _pairs(posts: list[int]) -> list[tuple[int, int]]:
    out = []
    for i in range(len(posts)):
        for j in range(i + 1, len(posts)):
            out.append((posts[i], posts[j]))
    return out


def _clusters_from_pred(pred_story_id: dict[int, str], universe: set[int]) -> dict[str, list[int]]:
    by = defaultdict(list)
    for pid, sid in pred_story_id.items():
        if pid in universe:
            by[str(sid)].append(pid)
    return dict(by)


def evaluate(pred_story_id: dict[int, str], gold_story_id: dict[int, str], universe: set[int], order: list[int]) -> Metrics:
    posts = [p for p in order if p in universe]
    if not posts:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Assignment accuracy (secondary)
    correct = 0
    for p in posts:
        correct += 1 if pred_story_id.get(p) == gold_story_id.get(p) else 0
    acc = correct / max(1, len(posts))

    # Pairwise P/R/F1
    tp = fp = fn = 0
    for a, b in _pairs(posts):
        g = gold_story_id[a] == gold_story_id[b]
        y = pred_story_id.get(a) == pred_story_id.get(b)
        if y and g:
            tp += 1
        elif y and not g:
            fp += 1
        elif (not y) and g:
            fn += 1
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))

    # ARI/NMI
    u_g = sorted({gold_story_id[p] for p in posts})
    g2i = {g: i for i, g in enumerate(u_g)}
    u_p = sorted({pred_story_id.get(p, "") for p in posts})
    p2i = {g: i for i, g in enumerate(u_p)}
    y_true = [g2i[gold_story_id[p]] for p in posts]
    y_pred = [p2i[pred_story_id.get(p, "")] for p in posts]
    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))

    # Predicted/gold story counts + singleton stats to proxy split/merge rates
    gold_stories = {gold_story_id[p] for p in posts}
    pred_stories = {pred_story_id.get(p, "") for p in posts}

    gold_by = _clusters_from_pred(gold_story_id, set(posts))
    pred_by = _clusters_from_pred(pred_story_id, set(posts))

    # false merge: predicted cluster contains >=2 gold stories
    fm = 0
    for members in pred_by.values():
        gs = {gold_story_id[p] for p in members}
        if len(gs) >= 2:
            fm += 1
    fmr = fm / max(1, len(pred_by))

    # false split: gold story split across >=2 predicted clusters
    fs = 0
    pred_of = {p: pred_story_id.get(p, "") for p in posts}
    for members in gold_by.values():
        ps = {pred_of[p] for p in members}
        if len(ps) >= 2:
            fs += 1
    fsr = fs / max(1, len(gold_by))

    return Metrics(
        pairwise_precision=prec,
        pairwise_recall=rec,
        pairwise_f1=f1,
        ari=ari,
        nmi=nmi,
        false_merge_rate=fmr,
        false_split_rate=fsr,
        predicted_story_count=len(pred_stories),
        gold_story_count=len(gold_stories),
        post_assignment_accuracy=acc,
    )

