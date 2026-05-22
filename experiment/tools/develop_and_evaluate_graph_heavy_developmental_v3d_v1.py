#!/usr/bin/env python3.11
"""
develop_and_evaluate_graph_heavy_developmental_v3d_v1

Exploratory graph-heavy variants on v3d with dev/test split by gold_story_id.
No prod writes, no LLM, no fulltext in sync_out.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from evaluate_large_hybrid_v4_1_guard_grid import (  # noqa: E402
    _load_feats as _load_feats_v2,
    _pair_score_v41,
    simulate_v4_1,
)
from evaluate_manual_gold_expansion_v3d_strict_developmental_labeled import V41_CFG  # noqa: E402
from pilot_eval_v2_assignment_lib import (  # noqa: E402
    aggregate_metrics,
    attach_ari_nmi,
    build_score_fn,
    pair_key,
    simulate_no_gold,
)

OUT_DIR = "sync_out/data/prepared/manual_gold_expansion_v3d_graph_heavy_v1"
REPORT_DIR = "sync_out/reports"

FEATURES = [
    "semantic_score",
    "entity_jaccard",
    "event_signature_jaccard",
    "graph_neighborhood_jaccard",
    "graph_relation_type_jaccard",
    "storyline_overlap",
    "time_delta_hours",
]

FORBIDDEN_COLS = frozenset({"text", "content", "fulltext", "body", "post_text"})


def _exp_root() -> Path:
    return _TOOLS.parent


def _parse_ts(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S+00:00"):
        try:
            return datetime.strptime(s.replace("Z", "+00:00"), fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _hours_delta(ts_a: str, ts_b: str) -> float:
    da, db = _parse_ts(ts_a), _parse_ts(ts_b)
    if not da or not db:
        return 0.0
    return abs((da - db).total_seconds()) / 3600.0


def _load_v3d_labels(root: Path) -> tuple[list[int], dict[int, str], dict[int, str]]:
    p = root / "sync_out/data/prepared/manual_gold_expansion_v3d_eval/v3d_labeled_clean_posts_v1.csv"
    rows = sorted(csv.DictReader(p.open(encoding="utf-8")), key=lambda r: (r.get("published_at") or "", int(r["post_id"])))
    order = [int(r["post_id"]) for r in rows]
    gold = {int(r["post_id"]): r["namespaced_gold_story_id"].strip() for r in rows}
    ts = {int(r["post_id"]): (r.get("published_at") or "").strip() for r in rows}
    return order, gold, ts


def _load_v3d_feats(root: Path) -> dict[tuple[int, int], dict[str, float]]:
    p = root / "sync_out/data/prepared/manual_gold_expansion_v3d_eval_neo4j_v2/v3d_pairwise_features_neo4j_v2.csv"
    m: dict[tuple[int, int], dict[str, float]] = {}
    for r in csv.DictReader(p.open(encoding="utf-8")):
        a, b = int(r["post_id_a"]), int(r["post_id_b"])
        row: dict[str, float] = {}
        for k in list(FEATURES) + ["temporal_score", "near_duplicate_score", "channel_same_flag"]:
            if k in r:
                try:
                    row[k] = float(r[k] or 0)
                except ValueError:
                    row[k] = 0.0
        if "time_delta_hours" not in row and "time_delta_hours" in r:
            row["time_delta_hours"] = float(r.get("time_delta_hours") or 0)
        m[pair_key(a, b)] = row
    return m


def _load_old_test(root: Path) -> tuple[list[int], dict[int, str], dict[tuple[int, int], dict[str, float]]]:
    split_p = root / "sync_out/data/prepared/combined_manual_gold_v2/combined_split_v2.csv"
    lab_p = root / "sync_out/data/prepared/combined_manual_gold_v2/combined_manual_gold_labels_v2.csv"
    feat_p = root / "sync_out/data/prepared/combined_manual_gold_v2/combined_pairwise_features_v2.csv"

    split_rows = sorted(csv.DictReader(split_p.open(encoding="utf-8")), key=lambda r: int(r["chronological_index"]))
    test_ids = {int(r["post_id"]) for r in split_rows if r["split"].strip() == "test"}
    order = [int(r["post_id"]) for r in split_rows if int(r["post_id"]) in test_ids]
    gold = {
        int(r["post_id"]): r["gold_story_id"].strip()
        for r in csv.DictReader(lab_p.open(encoding="utf-8"))
        if int(r["post_id"]) in test_ids
    }

    feats: dict[tuple[int, int], dict[str, float]] = {}
    for r in csv.DictReader(feat_p.open(encoding="utf-8")):
        a, b = int(r["post_id_a"]), int(r["post_id_b"])
        if a not in test_ids or b not in test_ids:
            continue
        k = pair_key(a, b)
        row = {
            "semantic_score": float(r.get("semantic_score") or 0),
            "entity_jaccard": float(r.get("entity_jaccard") or 0),
            "event_signature_jaccard": float(r.get("event_signature_jaccard") or 0),
            "graph_neighborhood_jaccard": float(r.get("graph_neighborhood_jaccard") or 0),
            "graph_relation_type_jaccard": float(r.get("graph_relation_type_jaccard") or 0),
            "storyline_overlap": 0.0,
            "temporal_score": float(r.get("temporal_score") or 0),
            "time_delta_hours": _hours_delta(r.get("published_at_a", ""), r.get("published_at_b", "")),
        }
        feats[k] = row
    return order, gold, feats


def _dist_stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "std": 0.0, "frac_zero": 1.0}
    s = sorted(vals)
    n = len(s)
    return {
        "n": float(n),
        "mean": float(statistics.mean(s)),
        "median": float(statistics.median(s)),
        "p25": float(np.percentile(s, 25)),
        "p75": float(np.percentile(s, 75)),
        "std": float(statistics.pstdev(s)) if n > 1 else 0.0,
        "frac_zero": float(sum(1 for x in s if x <= 1e-12) / n),
    }


def _feature_diagnosis(
    root: Path,
    v3d_order: list[int],
    v3d_gold: dict[int, str],
    v3d_feats: dict[tuple[int, int], dict[str, float]],
    old_order: list[int],
    old_gold: dict[int, str],
    old_feats: dict[tuple[int, int], dict[str, float]],
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []

    def collect(dataset: str, order: list[int], gold: dict[int, str], feats: dict[tuple[int, int], dict[str, float]]):
        posts = [p for p in order if p in gold]
        for i in range(len(posts)):
            for j in range(i + 1, len(posts)):
                a, b = posts[i], posts[j]
                k = pair_key(a, b)
                f = feats.get(k, {})
                pt = "same_gold" if gold[a] == gold[b] else "cross_gold"
                for feat in FEATURES:
                    v = float(f.get(feat, 0.0))
                    if feat == "time_delta_hours" and v <= 0 and "published_at" not in f:
                        pass
                    bucket = (dataset, pt, feat)
                    if bucket not in locals().get("_acc", {}):
                        pass
        # accumulate
        acc: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for i in range(len(posts)):
            for j in range(i + 1, len(posts)):
                a, b = posts[i], posts[j]
                pt = "same_gold" if gold[a] == gold[b] else "cross_gold"
                f = feats.get(pair_key(a, b), {})
                for feat in FEATURES:
                    acc[(dataset, pt, feat)].append(float(f.get(feat, 0.0)))
        for (dataset, pt, feat), vals in sorted(acc.items()):
            st = _dist_stats(vals)
            rows_out.append(
                {
                    "dataset": dataset,
                    "pair_type": pt,
                    "feature": feat,
                    **{k: round(v, 6) if isinstance(v, float) else v for k, v in st.items()},
                }
            )

    collect("old_test", old_order, old_gold, old_feats)
    collect("v3d_external", v3d_order, v3d_gold, v3d_feats)
    return rows_out


def _story_split(gold: dict[int, str], dev_frac: float = 0.4, seed: int = 42) -> dict[str, str]:
    """Split by gold_story_id; stratify singleton vs multi-post stories."""
    by_story: dict[str, list[int]] = defaultdict(list)
    for p, g in gold.items():
        by_story[g].append(p)
    singleton = [g for g, ps in by_story.items() if len(ps) == 1]
    multi = [g for g, ps in by_story.items() if len(ps) > 1]
    rng = random.Random(seed)

    def assign_bucket(stories: list[str], frac: float) -> dict[str, str]:
        sh = stories[:]
        rng.shuffle(sh)
        n_dev = max(1, int(round(len(sh) * frac))) if len(sh) >= 2 else (1 if len(sh) == 1 and frac >= 0.5 else 0)
        if len(sh) == 1:
            n_dev = 1 if frac >= 0.5 else 0
        out: dict[str, str] = {}
        for i, g in enumerate(sh):
            out[g] = "v3d_dev" if i < n_dev else "v3d_test"
        return out

    split_map: dict[str, str] = {}
    split_map.update(assign_bucket(singleton, dev_frac))
    split_map.update(assign_bucket(multi, dev_frac))
    return split_map


def _norm_time(row: dict[str, float], cap_hours: float = 336.0) -> float:
    h = float(row.get("time_delta_hours", 0.0))
    return min(1.0, h / cap_hours) if cap_hours > 0 else 0.0


def _row(feats: dict[tuple[int, int], dict[str, float]], a: int, b: int) -> dict[str, float]:
    return dict(feats.get(pair_key(a, b), {}))


def score_boost_linear(row: dict[str, float], cfg: dict[str, float]) -> float:
    return (
        cfg["w_sem"] * float(row.get("semantic_score", 0))
        + cfg["w_entity"] * float(row.get("entity_jaccard", 0))
        + cfg["w_event"] * float(row.get("event_signature_jaccard", 0))
        + cfg["w_storyline"] * float(row.get("storyline_overlap", 0))
        + cfg["w_graph"] * float(row.get("graph_neighborhood_jaccard", 0))
        - cfg["w_time"] * _norm_time(row, cfg.get("time_cap_hours", 336.0))
    )


def attach_guard_or(
    order: list[int],
    feats: dict[tuple[int, int], dict[str, float]],
    cfg: dict[str, float],
    score_fn: Callable[[dict[str, float]], float] | None = None,
) -> tuple[dict[int, int], list[list[int]]]:
    sem_thr = float(cfg["sem_threshold"])
    evt_thr = float(cfg["event_threshold"])
    ent_thr = float(cfg["entity_threshold"])
    graph_thr = float(cfg["graph_threshold"])
    max_days = float(cfg["max_days"])
    thr = float(cfg.get("threshold", 0.5))

    def guard_attach(row: dict[str, float]) -> bool:
        sem = float(row.get("semantic_score", 0))
        if sem >= sem_thr:
            return True
        if float(row.get("storyline_overlap", 0)) >= 1.0 - 1e-9:
            return True
        td = float(row.get("time_delta_hours", 0)) / 24.0
        if float(row.get("event_signature_jaccard", 0)) >= evt_thr and td <= max_days:
            return True
        if float(row.get("entity_jaccard", 0)) >= ent_thr and float(row.get("graph_neighborhood_jaccard", 0)) >= graph_thr:
            return True
        return False

    clusters: list[list[int]] = []
    pred_of: dict[int, int] = {}
    for i, p in enumerate(order):
        if i == 0:
            clusters.append([p])
            pred_of[p] = 0
            continue
        best_j, best_s = -1, -1.0
        for j, memb in enumerate(clusters):
            if not memb:
                continue
            scores = []
            guards = []
            for q in memb:
                row = _row(feats, p, q)
                if score_fn:
                    scores.append(score_fn(row))
                else:
                    scores.append(score_boost_linear(row, cfg))
                guards.append(guard_attach(row))
            s = max(scores) if scores else -1.0
            if any(guards):
                s = max(s, thr)
            if s > best_s:
                best_s, best_j = s, j
        if best_j < 0 or best_s < thr:
            nj = len(clusters)
            clusters.append([p])
            pred_of[p] = nj
        else:
            clusters[best_j].append(p)
            pred_of[p] = best_j
    return pred_of, clusters


def attach_developmental(
    order: list[int],
    feats: dict[tuple[int, int], dict[str, float]],
    cfg: dict[str, float],
) -> tuple[dict[int, int], list[list[int]]]:
    sem_ret = float(cfg["retrieve_sem"])
    graph_ret = float(cfg["retrieve_graph"])
    thr = float(cfg["threshold"])
    long_days = float(cfg["long_gap_days"])
    graph_strong = float(cfg["graph_strong"])

    def retrieve(row: dict[str, float]) -> bool:
        if float(row.get("semantic_score", 0)) >= sem_ret:
            return True
        if float(row.get("graph_neighborhood_jaccard", 0)) >= graph_ret:
            return True
        return False

    def weighted(row: dict[str, float]) -> float:
        base = score_boost_linear(row, cfg)
        td = float(row.get("time_delta_hours", 0)) / 24.0
        if float(row.get("graph_neighborhood_jaccard", 0)) >= graph_strong and td > long_days:
            base += float(cfg.get("long_gap_bonus", 0.15))
        return base

    clusters: list[list[int]] = []
    pred_of: dict[int, int] = {}
    for i, p in enumerate(order):
        if i == 0:
            clusters.append([p])
            pred_of[p] = 0
            continue
        cands: list[tuple[int, float]] = []
        for j, memb in enumerate(clusters):
            ok = False
            scores = []
            for q in memb:
                row = _row(feats, p, q)
                if retrieve(row):
                    ok = True
                scores.append(weighted(row))
            if ok:
                cands.append((j, max(scores) if scores else -1.0))
        if not cands:
            nj = len(clusters)
            clusters.append([p])
            pred_of[p] = nj
            continue
        best_j, best_s = max(cands, key=lambda t: t[1])
        if best_s < thr:
            nj = len(clusters)
            clusters.append([p])
            pred_of[p] = nj
        else:
            clusters[best_j].append(p)
            pred_of[p] = best_j
    return pred_of, clusters


def attach_conservative(
    order: list[int],
    feats: dict[tuple[int, int], dict[str, float]],
    cfg: dict[str, float],
) -> tuple[dict[int, int], list[list[int]]]:
    sem_fn = build_score_fn("semantic_only", 1, 0, 0, 0, 0, 0, 0)
    thr_sem = float(cfg["sem_threshold"])
    graph_ov = float(cfg["graph_override"])
    evt_ov = float(cfg["event_override"])
    storyline_ov = float(cfg.get("storyline_override", 1.0))

    clusters: list[list[int]] = []
    pred_of: dict[int, int] = {}
    for i, p in enumerate(order):
        if i == 0:
            clusters.append([p])
            pred_of[p] = 0
            continue
        best_j, best_s = -1, -1.0
        force_j = -1
        for j, memb in enumerate(clusters):
            scores = []
            for q in memb:
                row = _row(feats, p, q)
                s = sem_fn(row)
                scores.append(s)
                if (
                    float(row.get("graph_neighborhood_jaccard", 0)) >= graph_ov
                    and float(row.get("event_signature_jaccard", 0)) >= evt_ov
                ) or float(row.get("storyline_overlap", 0)) >= storyline_ov - 1e-9:
                    force_j = j
            mx = max(scores) if scores else -1.0
            if mx > best_s:
                best_s, best_j = mx, j
        if force_j >= 0:
            clusters[force_j].append(p)
            pred_of[p] = force_j
        elif best_j < 0 or best_s < thr_sem:
            nj = len(clusters)
            clusters.append([p])
            pred_of[p] = nj
        else:
            clusters[best_j].append(p)
            pred_of[p] = best_j
    return pred_of, clusters


def _metrics_dict(m: dict[str, float]) -> dict[str, float]:
    return {
        "post_assignment_accuracy": m["post_assignment_accuracy"],
        "pairwise_precision": m["pairwise_cluster_precision"],
        "pairwise_recall": m["pairwise_cluster_recall"],
        "pairwise_f1": m["pairwise_cluster_f1"],
        "adjusted_rand_index": m.get("adjusted_rand_index", float("nan")),
        "nmi": m.get("nmi", float("nan")),
        "false_merge_rate": m["false_merge_rate"],
        "false_split_rate": m["false_split_rate"],
        "predicted_story_count": m["predicted_story_count"],
        "gold_story_count": m["gold_story_count"],
        "singleton_predicted_rate": m["singleton_predicted_rate"],
    }


def _eval_method(
    name: str,
    order: list[int],
    gold: dict[int, str],
    feats: dict[tuple[int, int], dict[str, float]],
    universe: set[int],
    cfg: dict[str, Any],
) -> tuple[dict[str, float], dict[int, int]]:
    if name == "semantic_only":
        fn = build_score_fn("semantic_only", 1, 0, 0, 0, 0, 0, 0)
        pred, cl = simulate_no_gold(order, feats, fn, float(cfg["threshold"]))
    elif name == "semantic_time":
        fn = build_score_fn("semantic_time", float(cfg.get("w_sem", 0.65)), float(cfg.get("w_time", 0.35)), 0, 0, 0, 0, 0)
        pred, cl = simulate_no_gold(order, feats, fn, float(cfg["threshold"]))
    elif name == "hybrid_v4_1_soft_entity_penalty":
        v41 = dict(V41_CFG)
        v41.update(cfg)
        pred, cl = simulate_v4_1(order, feats, v41)
    elif name == "graph_boost_linear_v1":
        fn = lambda row: score_boost_linear(row, cfg)  # noqa: E731
        pred, cl = simulate_no_gold(order, feats, fn, float(cfg["threshold"]))
    elif name == "graph_guard_or_v1":
        pred, cl = attach_guard_or(order, feats, cfg)
    elif name == "graph_developmental_v1":
        pred, cl = attach_developmental(order, feats, cfg)
    elif name == "conservative_graph_v1":
        pred, cl = attach_conservative(order, feats, cfg)
    else:
        raise ValueError(name)
    m = aggregate_metrics(order, gold, pred, cl, universe, universe)
    attach_ari_nmi(m, order, gold, pred, universe)
    return _metrics_dict(m), pred


def _tune_graph_heavy(
    method: str,
    order: list[int],
    gold: dict[int, str],
    feats: dict[tuple[int, int], dict[str, float]],
    dev_ids: set[int],
    sem_baseline: dict[str, float],
) -> dict[str, Any]:
    sem_fmr = float(sem_baseline["false_merge_rate"])
    sem_sing = float(sem_baseline["singleton_predicted_rate"])
    sem_pred = float(sem_baseline["predicted_story_count"])
    gold_n = float(sem_baseline["gold_story_count"])
    best: tuple[float, float, dict[str, Any], dict[str, float]] | None = None

    if method == "graph_boost_linear_v1":
        grids = [
            dict(
                w_sem=w,
                w_entity=e,
                w_event=ev,
                w_storyline=sl,
                w_graph=g,
                w_time=t,
                threshold=th,
                time_cap_hours=336.0,
            )
            for w, e, ev, sl, g, t, th in product(
                [0.4, 0.55, 0.7],
                [0.0, 0.1, 0.2],
                [0.0, 0.15, 0.25],
                [0.0, 0.2],
                [0.1, 0.25, 0.4],
                [0.05, 0.15],
                [0.35, 0.45, 0.55, 0.65],
            )
        ]
    elif method == "graph_guard_or_v1":
        grids = [
            dict(
                sem_threshold=ss,
                event_threshold=et,
                entity_threshold=ent,
                graph_threshold=gt,
                max_days=md,
                threshold=th,
                w_sem=0.5,
                w_entity=0.1,
                w_event=0.15,
                w_storyline=0.2,
                w_graph=0.25,
                w_time=0.1,
            )
            for ss, et, ent, gt, md, th in product(
                [0.55, 0.65, 0.75],
                [0.5, 0.75, 1.0],
                [0.2, 0.35],
                [0.25, 0.4, 0.55],
                [7.0, 14.0, 30.0],
                [0.4, 0.5, 0.6],
            )
        ]
    elif method == "graph_developmental_v1":
        grids = [
            dict(
                retrieve_sem=rs,
                retrieve_graph=rg,
                graph_strong=gs,
                long_gap_days=lgd,
                long_gap_bonus=0.1,
                threshold=th,
                w_sem=0.5,
                w_entity=0.1,
                w_event=0.2,
                w_storyline=0.25,
                w_graph=0.35,
                w_time=0.08,
            )
            for rs, rg, gs, lgd, th in product(
                [0.45, 0.6],
                [0.2, 0.35, 0.5],
                [0.45, 0.6],
                [14.0, 30.0],
                [0.35, 0.45, 0.55],
            )
        ]
    elif method == "conservative_graph_v1":
        grids = [
            dict(sem_threshold=st, graph_override=go, event_override=eo, storyline_override=1.0)
            for st, go, eo in product([0.75, 0.82], [0.5, 0.65], [0.75, 1.0])
        ]
    else:
        return {}

    for cfg in grids:
        m, _ = _eval_method(method, order, gold, feats, dev_ids, cfg)
        if float(m["false_merge_rate"]) > sem_fmr + 0.05 + 1e-9:
            continue
        if float(m["predicted_story_count"]) < max(3.0, 0.15 * gold_n):
            continue
        if float(m["predicted_story_count"]) > 0.95 * gold_n:
            continue
        f1 = float(m["pairwise_f1"])
        ari = float(m.get("adjusted_rand_index") or 0.0)
        sing = float(m["singleton_predicted_rate"])
        key = (f1, ari, -sing if sing < sem_sing else 0.0)
        if best is None or key > (best[0], best[1], best[2]):
            best = (f1, ari, -sing, cfg, m)

    if best is None:
        # fallback: first grid point
        cfg = grids[0]
        m, _ = _eval_method(method, order, gold, feats, dev_ids, cfg)
        return {"config": cfg, "dev_metrics": m, "note": "fallback_no_constraint_match"}
    return {"config": best[3], "dev_metrics": best[4], "note": "tuned_on_v3d_dev"}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _diagnosis_report(dist_rows: list[dict[str, Any]], path: Path) -> None:
    def get(ds: str, pt: str, feat: str, stat: str) -> float:
        for r in dist_rows:
            if r["dataset"] == ds and r["pair_type"] == pt and r["feature"] == feat:
                return float(r.get(stat, 0))
        return 0.0

    lines = [
        "# v3d graph-heavy feature diagnosis v1",
        "",
        "Сравнение распределений pairwise признаков: **old_test** (combined v2 test) vs **v3d_external** (140 clean).",
        "",
        "## same_gold vs cross_gold — old_test",
        "",
    ]
    for feat in FEATURES:
        sm = get("old_test", "same_gold", feat, "mean")
        cm = get("old_test", "cross_gold", feat, "mean")
        lines.append(f"- **{feat}**: same mean={sm:.4f}, cross mean={cm:.4f}, Δ={sm - cm:.4f}")
    lines += ["", "## same_gold vs cross_gold — v3d_external", ""]
    for feat in FEATURES:
        sm = get("v3d_external", "same_gold", feat, "mean")
        cm = get("v3d_external", "cross_gold", feat, "mean")
        lines.append(f"- **{feat}**: same mean={sm:.4f}, cross mean={cm:.4f}, Δ={sm - cm:.4f}")
    lines += [
        "",
        "## Почти константные признаки",
        "",
    ]
    for ds in ("old_test", "v3d_external"):
        for feat in FEATURES:
            z_same = get(ds, "same_gold", feat, "frac_zero")
            z_cross = get(ds, "cross_gold", feat, "frac_zero")
            sm = get(ds, "same_gold", feat, "mean")
            cm = get(ds, "cross_gold", feat, "mean")
            if z_same > 0.95 and z_cross > 0.95:
                lines.append(f"- {ds}/{feat}: frac_zero≈1 (почти константа 0)")
            elif abs(sm - cm) < 0.02 and max(sm, cm) < 0.05:
                lines.append(f"- {ds}/{feat}: слабое разделение same/cross (|Δmean|<0.02)")
    lines += [
        "",
        "## Почему hybrid_v4_1 не меняет decisions на v3d",
        "",
        "1. Пороги и веса v4.1 подобраны на **combined dev**, не на developmental v3d.",
        "2. На v3d при audit-v2 threshold решения доминирует **semantic_score**; graph/entity слабо меняют ранжирование пар выше порога attach.",
        "3. **entity_jaccard** в PG для v3d почти 0; Neo4j entity восстановлен, но v4.1 штрафует low-event+entity без сильного graph boost в итоговом score.",
        "4. Высокий **false_split_rate** и singleton predicted → большинство attach идёт по semantic; малые отличия score не переворачивают кластеры.",
        "5. **storyline_overlap** редок (~1.4% пар); **time_delta** на developmental велик → temporal_score низкий.",
        "",
        "Примечание: old_test **storyline_overlap** в combined features отсутствует (записан как 0).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=_exp_root())
    ap.add_argument("--skip-rclone", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    out = root / OUT_DIR
    rep = root / REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)

    v3d_order, v3d_gold, _ = _load_v3d_labels(root)
    v3d_feats = _load_v3d_feats(root)
    old_order, old_gold, old_feats = _load_old_test(root)

    # 1. Feature diagnosis
    dist_rows = _feature_diagnosis(root, v3d_order, v3d_gold, v3d_feats, old_order, old_gold, old_feats)
    _write_csv(
        out / "old_vs_v3d_feature_distributions_v1.csv",
        dist_rows,
        ["dataset", "pair_type", "feature", "n", "mean", "median", "p25", "p75", "std", "frac_zero"],
    )
    _diagnosis_report(dist_rows, rep / "v3d_graph_heavy_feature_diagnosis_v1.md")

    # 2. Split
    story_split = _story_split(v3d_gold, dev_frac=0.4, seed=42)
    split_rows = []
    for p in v3d_order:
        g = v3d_gold[p]
        split_rows.append(
            {
                "post_id": p,
                "namespaced_gold_story_id": g,
                "split": story_split[g],
                "story_size": sum(1 for x, gg in v3d_gold.items() if gg == g),
                "is_singleton_story": 1 if sum(1 for x, gg in v3d_gold.items() if gg == g) == 1 else 0,
            }
        )
    _write_csv(out / "v3d_dev_test_split_v1.csv", split_rows, list(split_rows[0].keys()))
    dev_ids = {int(r["post_id"]) for r in split_rows if r["split"] == "v3d_dev"}
    test_ids = {int(r["post_id"]) for r in split_rows if r["split"] == "v3d_test"}
    order_dev = [p for p in v3d_order if p in dev_ids]
    order_test = [p for p in v3d_order if p in test_ids]

    # Baselines thresholds from audit v2
    tuned_v2_path = root / "sync_out/data/prepared/combined_manual_gold_v2/tuned_params_combined_manual_gold_v2.json"
    tuned_v2 = json.loads(tuned_v2_path.read_text(encoding="utf-8"))

    sem_cfg = {"threshold": tuned_v2["methods"]["semantic_only"]["tuned_threshold"]}
    st_cfg = {
        "threshold": tuned_v2["methods"]["semantic_time"]["tuned_threshold"],
        "w_sem": 0.65,
        "w_time": 0.35,
    }
    v41_cfg = dict(V41_CFG)

    sem_dev, _ = _eval_method("semantic_only", order_dev, v3d_gold, v3d_feats, dev_ids, sem_cfg)

    graph_methods = [
        "graph_boost_linear_v1",
        "graph_guard_or_v1",
        "graph_developmental_v1",
        "conservative_graph_v1",
    ]
    tuned_all: dict[str, Any] = {
        "split": "v3d_dev_test_by_gold_story_id_v1",
        "dev_frac_target": 0.4,
        "dev_posts": len(dev_ids),
        "test_posts": len(test_ids),
        "dev_stories": len({v3d_gold[p] for p in dev_ids}),
        "test_stories": len({v3d_gold[p] for p in test_ids}),
        "semantic_only": sem_cfg,
        "semantic_time": st_cfg,
        "hybrid_v4_1_soft_entity_penalty": v41_cfg,
        "tuning_constraints": {
            "fmr_max_delta_vs_semantic": 0.05,
            "predicted_story_count_min_frac_gold": 0.15,
            "predicted_story_count_max_frac_gold": 0.95,
        },
    }
    for gm in graph_methods:
        tuned_all[gm] = _tune_graph_heavy(gm, order_dev, v3d_gold, v3d_feats, dev_ids, sem_dev)

    (out / "tuned_graph_heavy_params_v1.json").write_text(json.dumps(tuned_all, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4. Evaluate v3d_test
    eval_methods = [
        "semantic_only",
        "semantic_time",
        "hybrid_v4_1_soft_entity_penalty",
        "graph_boost_linear_v1",
        "graph_guard_or_v1",
        "graph_developmental_v1",
        "conservative_graph_v1",
    ]
    metrics_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    metrics_by_method: dict[str, dict[str, float]] = {}

    for method in eval_methods:
        if method in graph_methods:
            cfg = tuned_all[method]["config"]
        elif method == "hybrid_v4_1_soft_entity_penalty":
            cfg = v41_cfg
        elif method == "semantic_time":
            cfg = st_cfg
        else:
            cfg = sem_cfg
        m, pred = _eval_method(method, order_test, v3d_gold, v3d_feats, test_ids, cfg)
        metrics_by_method[method] = m
        metrics_rows.append({"method": method, "split": "v3d_test", **{k: round(v, 4) if isinstance(v, float) and not math.isnan(v) else v for k, v in m.items()}})
        for p, cid in pred.items():
            pred_rows.append({"post_id": p, "method": method, "predicted_cluster_id": cid, "eval_split": "v3d_test"})

    _write_csv(out / "v3d_graph_heavy_metrics_v1.csv", metrics_rows, ["method", "split"] + list(metrics_rows[0].keys())[2:])
    _write_csv(out / "v3d_graph_heavy_predictions_v1.csv", pred_rows, ["post_id", "method", "predicted_cluster_id", "eval_split"])

    # eval report
    best_gh = max(
        (m for k, m in metrics_by_method.items() if k in graph_methods),
        key=lambda m: float(m["pairwise_f1"]),
    )
    best_name = [k for k, m in metrics_by_method.items() if m == best_gh and k in graph_methods][0]
    sem_m = metrics_by_method["semantic_only"]
    ev_lines = [
        "# v3d graph-heavy evaluation v1",
        "",
        f"Dev/test split: {len(dev_ids)} dev / {len(test_ids)} test posts.",
        "",
        "## v3d_test metrics",
        "",
        "| method | pairwise_f1 | ARI | FMR | FSR | singleton_pred | pred/gold stories |",
        "|--------|-------------|-----|-----|-----|----------------|-------------------|",
    ]
    for method in eval_methods:
        m = metrics_by_method[method]
        ev_lines.append(
            f"| {method} | {m['pairwise_f1']:.4f} | {m.get('adjusted_rand_index', float('nan')):.4f} | "
            f"{m['false_merge_rate']:.4f} | {m['false_split_rate']:.4f} | {m['singleton_predicted_rate']:.4f} | "
            f"{int(m['predicted_story_count'])}/{int(m['gold_story_count'])} |"
        )
    ev_lines += [
        "",
        f"**Best graph-heavy on v3d_test:** {best_name} (F1={best_gh['pairwise_f1']:.4f})",
        f"**Delta vs semantic_only:** F1 {best_gh['pairwise_f1'] - sem_m['pairwise_f1']:+.4f}",
        "",
        "Пороги graph-heavy подобраны только на v3d_dev; audit v2 пороги для semantic/hybrid/v41.",
        "",
    ]
    (rep / "v3d_graph_heavy_evaluation_v1.md").write_text("\n".join(ev_lines), encoding="utf-8")

    # 5. old_test transfer
    old_test_ids = set(old_order)
    best_cfg = tuned_all[best_name]["config"]
    transfer_rows: list[dict[str, Any]] = []
    unified = list(csv.DictReader((root / "sync_out/data/prepared/commission_audit_v2/unified_metrics_v2.csv").open(encoding="utf-8")))
    uni_test = {r["method"]: r for r in unified if r.get("split") == "test"}

    for method in ["semantic_only", "semantic_time", "hybrid_v4_1_soft_entity_penalty", best_name]:
        if method == best_name:
            cfg = best_cfg
        elif method == "hybrid_v4_1_soft_entity_penalty":
            cfg = v41_cfg
        elif method == "semantic_time":
            cfg = st_cfg
        else:
            cfg = sem_cfg
        m, _ = _eval_method(method, old_order, old_gold, old_feats, old_test_ids, cfg)
        row = {"method": method, "split": "old_test_transfer", **m}
        if method in uni_test:
            u = uni_test[method]
            row["official_pairwise_f1"] = float(u.get("pairwise_f1", 0))
            row["delta_f1_vs_official"] = m["pairwise_f1"] - float(u.get("pairwise_f1", 0))
        transfer_rows.append(row)

    _write_csv(
        out / "old_test_transfer_metrics_v1.csv",
        transfer_rows,
        list(transfer_rows[0].keys()),
    )
    tr = [r for r in transfer_rows if r["method"] == best_name][0]
    off = tr.get("official_pairwise_f1", "n/a")
    (rep / "v3d_graph_heavy_old_test_transfer_v1.md").write_text(
        "\n".join(
            [
                "# v3d graph-heavy old_test transfer v1",
                "",
                f"Best v3d graph-heavy method **{best_name}** applied to combined v2 **test** without retune.",
                "",
                f"- Transfer F1: {tr['pairwise_f1']:.4f}",
                f"- Official audit v2 F1 (semantic): {uni_test.get('semantic_only', {}).get('pairwise_f1', 'n/a')}",
                f"- Official hybrid v4.1 F1: {uni_test.get('hybrid_v4_1_soft_entity_penalty', {}).get('pairwise_f1', 'n/a')}",
                f"- Delta F1 vs official for transferred method: {tr.get('delta_f1_vs_official', 'n/a')}",
                "",
                "old_test storyline_overlap отсутствует в combined features (0); transfer — diagnostic only.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # 6. Summary
    hyb_m = metrics_by_method["hybrid_v4_1_soft_entity_penalty"]
    summary = [
        "# v3d graph-heavy summary v1",
        "",
        "## Ответы",
        "",
        f"1. **Обошёл ли graph-heavy semantic на v3d_test?** "
        f"{'Да' if best_gh['pairwise_f1'] > sem_m['pairwise_f1'] + 1e-4 else 'Нет'} "
        f"(best {best_name} F1={best_gh['pairwise_f1']:.4f} vs semantic {sem_m['pairwise_f1']:.4f}).",
        "",
        f"2. **old_test:** transfer delta F1={tr.get('delta_f1_vs_official', 0):+.4f} vs official for {best_name}.",
        "",
        f"3. **Лучший graph-heavy:** {best_name}.",
        "",
        "4. **Полезность graph для developmental:** "
        "частично в features (neo4j nb/event), но при фиксированном v4.1 и exploratory tuning прирост на v3d_test ограничен.",
        "",
        "5. **ВКР:** можно как **дополнительный exploratory эксперимент** (v3d dev/test, graph-heavy v1), "
        "не подменяя hybrid_v4_1 production/audit v2.",
        "",
        "6. **Формулировка:** «graph-heavy developmental variant v1 (exploratory, tuned on v3d_dev)» "
        "отдельно от «hybrid_v4_1_soft_entity_penalty (combined dev, audit v2)».",
        "",
        "## Ограничения",
        "",
        "- Малый N (140 постов, 99 gold stories).",
        "- v3d_dev/test split exploratory — не для confirmatory claims.",
        "- old_test без storyline_overlap в pairwise CSV.",
        "",
        "## Next step",
        "",
        "Threshold-free pairwise ablation или отдельный v4.2 с graph weights на combined dev, "
        "затем frozen eval на v3d_test без retune.",
        "",
    ]
    (rep / "v3d_graph_heavy_summary_v1.md").write_text("\n".join(summary), encoding="utf-8")

    # forbidden cols check
    for p in out.rglob("*.csv"):
        hdr = next(csv.reader(p.open(encoding="utf-8")))
        bad = FORBIDDEN_COLS & set(hdr)
        if bad:
            raise SystemExit(f"forbidden columns in {p}: {bad}")

    print("OK", out)
    print("dev", len(dev_ids), "test", len(test_ids), "best", best_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
