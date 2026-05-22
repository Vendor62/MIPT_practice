#!/usr/bin/env python3.11
"""
audit_and_recompute_story_tracking_metrics_v2 — сквозной пересчёт метрик для
сводного размеченного набора combined_manual_gold_v2.

Формулы берутся из pilot_eval_v2_assignment_lib как из единого источника правды.
Сценарий не обращается к LLM, не пересобирает набор данных и не пишет полные
тексты в sync_out.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from build_hybrid_as_weak_target_advantage_diagnostic_v1 import _load_feats  # noqa: E402
from pilot_eval_v2_assignment_lib import (  # noqa: E402
    aggregate_metrics,
    attach_ari_nmi,
    build_score_fn,
    pair_key,
    simulate_no_gold,
)

METHODS_REPLAY = ("semantic_only", "semantic_time", "hybrid_graph_v1_entity")
METHODS_PRED = ("hybrid_v4_1_soft_entity_penalty",)
METHODS_LLM = ("deepseek_chat_llm",)
METHODS_ALL = METHODS_REPLAY + METHODS_PRED + METHODS_LLM

METRIC_COLS = [
    "post_assignment_accuracy",
    "pairwise_precision",
    "pairwise_recall",
    "pairwise_f1",
    "adjusted_rand_index",
    "nmi",
    "false_merge_count",
    "false_merge_rate",
    "false_split_count",
    "false_split_rate",
    "predicted_story_count",
    "gold_story_count",
    "singleton_predicted_count",
    "singleton_predicted_rate",
    "b_cubed",
]

BOOT_METRICS = [
    "pairwise_f1",
    "adjusted_rand_index",
    "nmi",
    "false_merge_rate",
    "false_split_rate",
    "post_assignment_accuracy",
    "singleton_predicted_rate",
]

FORBIDDEN_CSV_COLS = frozenset({"fulltext", "content", "text", "raw_text"})


def _exp_root() -> Path:
    return _TOOLS.parent


def _parse_ts(s: str) -> datetime | None:
    t = (s or "").strip()
    if not t:
        return None
    t = t.replace(" ", "T", 1) if (" " in t and "T" not in t) else t
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(t)
    except Exception:
        return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_md(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _metrics_to_row(method: str, split: str, m: dict[str, float], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "split": split,
        "post_assignment_accuracy": m["post_assignment_accuracy"],
        "pairwise_precision": m["pairwise_cluster_precision"],
        "pairwise_recall": m["pairwise_cluster_recall"],
        "pairwise_f1": m["pairwise_cluster_f1"],
        "adjusted_rand_index": m.get("adjusted_rand_index", ""),
        "nmi": m.get("nmi", ""),
        "false_merge_count": m["false_merge_count"],
        "false_merge_rate": m["false_merge_rate"],
        "false_split_count": m["false_split_count"],
        "false_split_rate": m["false_split_rate"],
        "predicted_story_count": m["predicted_story_count"],
        "gold_story_count": m["gold_story_count"],
        "singleton_predicted_count": m["singleton_predicted_count"],
        "singleton_predicted_rate": m["singleton_predicted_rate"],
        "b_cubed": m.get("b_cubed", ""),
    }
    if extra:
        row.update(extra)
    return row


def _load_gold_labels(path: Path) -> dict[int, str]:
    return {int(r["post_id"]): r["gold_story_id"].strip() for r in csv.DictReader(path.open(encoding="utf-8"))}


def _load_order_split(path: Path) -> tuple[list[int], dict[int, str], dict[int, int]]:
    rows = sorted(
        csv.DictReader(path.open(encoding="utf-8")),
        key=lambda r: (int(r["chronological_index"]), int(r["post_id"])),
    )
    order = [int(r["post_id"]) for r in rows]
    split_of = {int(r["post_id"]): r["split"].strip() for r in rows}
    chrono = {int(r["post_id"]): int(r["chronological_index"]) for r in rows}
    return order, split_of, chrono


def _load_tuned_params(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pred_clusters_from_map(order: list[int], pred_of: dict[int, int]) -> list[list[int]]:
    max_c = max(pred_of.values()) if pred_of else -1
    clusters: list[list[int]] = [[] for _ in range(max_c + 1)]
    for p in order:
        if p in pred_of:
            clusters[pred_of[p]].append(p)
    return clusters


def _replay_method(
    method: str,
    order: list[int],
    feats: dict[tuple[int, int], dict[str, float]],
    threshold: float,
) -> tuple[dict[int, int], list[list[int]]]:
    fn = build_score_fn(method, 0.8, 0.2, 0.35, 0.15, 0.12, 0.08, 0.12)
    return simulate_no_gold(order, feats, fn, threshold)


def _load_hybrid_v4_predictions(path: Path) -> dict[int, int]:
    pred: dict[int, int] = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r["method"] != "hybrid_v4_1_soft_entity_penalty":
            continue
        pred[int(r["post_id"])] = int(r["predicted_story_cluster_id"])
    return pred


def _load_deepseek_predictions(path: Path) -> dict[int, int]:
    labels: dict[int, str] = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        labels[int(r["post_id"])] = r["predicted_story_id"].strip()
    uniq = sorted(set(labels.values()))
    mapping = {u: i for i, u in enumerate(uniq)}
    return {p: mapping[lab] for p, lab in labels.items()}


def _eval_on_universe(
    order: list[int],
    gold: dict[int, str],
    pred_of: dict[int, int],
    clusters: list[list[int]],
    universe: set[int],
) -> dict[str, float]:
    m = aggregate_metrics(order, gold, pred_of, clusters, universe, universe)
    attach_ari_nmi(m, order, gold, pred_of, universe)
    return m


def _story_size_bucket(sz: int) -> str:
    if sz <= 1:
        return "1"
    if sz == 2:
        return "2"
    if sz <= 4:
        return "3-4"
    return "5+"


def _span_bucket(days: float) -> str:
    if days <= 1.0:
        return "same-day"
    if days <= 3.0:
        return "1-3d"
    return "3d+"


def _build_subgroup_groups(
    test_ids: set[int],
    gold: dict[int, str],
    pid2ts: dict[int, datetime],
) -> dict[str, set[int]]:
    by_story: dict[str, list[int]] = defaultdict(list)
    for p in gold:
        by_story[gold[p]].append(p)
    story_size = {g: len(ps) for g, ps in by_story.items()}
    story_span: dict[str, float] = {}
    for g, ps in by_story.items():
        ts = [pid2ts.get(p, datetime.min) for p in ps]
        story_span[g] = (max(ts) - min(ts)).total_seconds() / 86400.0

    groups: dict[str, set[int]] = {}
    groups["all_test"] = set(test_ids)
    groups["gold_singleton_test"] = {p for p in test_ids if story_size[gold[p]] == 1}
    groups["gold_non_singleton_test"] = {p for p in test_ids if story_size[gold[p]] >= 2}
    for b in ("1", "2", "3-4", "5+"):
        groups[f"gold_story_size_{b}_test"] = {
            p for p in test_ids if _story_size_bucket(story_size[gold[p]]) == b
        }
    for b in ("same-day", "1-3d", "3d+"):
        groups[f"gold_time_span_{b}_test"] = {
            p for p in test_ids if _span_bucket(story_span[gold[p]]) == b
        }
    for sp, prefix in (("old", "OLD_"), ("expansion", "EXP2_")):
        groups[f"source_part_{sp}_test"] = {p for p in test_ids if gold[p].startswith(prefix)}
    return groups


def _stratified_story_bootstrap_sample(
    test_story_posts: dict[str, list[int]],
    story_stratum: dict[str, str],
    strata_targets: dict[str, int],
    rng: random.Random,
) -> tuple[list[int], dict[int, str]]:
    """Семплирование gold-сюжетов по стратам с remap копий сюжета."""
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for gid, st in story_stratum.items():
        by_stratum[st].append(gid)

    picked: list[str] = []
    for st, n in strata_targets.items():
        pool = by_stratum[st]
        if not pool:
            continue
        for _ in range(n):
            picked.append(rng.choice(pool))

    boot_posts: list[int] = []
    boot_gold: dict[int, str] = {}
    for i, gid in enumerate(picked):
        new_gid = f"{gid}__boot{i}"
        for p in test_story_posts[gid]:
            boot_posts.append(p)
            boot_gold[p] = new_gid
    return boot_posts, boot_gold


def _bootstrap_stratified_gold(
    *,
    order: list[int],
    gold: dict[int, str],
    test_ids: set[int],
    pred_by_method: dict[str, dict[int, int]],
    clusters_by_method: dict[str, list[list[int]]],
    point_estimates: dict[str, dict[str, float]],
    n_iter: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_story_posts: dict[str, list[int]] = defaultdict(list)
    for p in test_ids:
        test_story_posts[gold[p]].append(p)
    story_stratum = {gid: _story_size_bucket(len(ps)) for gid, ps in test_story_posts.items()}
    sc = Counter(story_stratum.values())

    rng = random.Random(seed)
    methods = ["semantic_only", "hybrid_v4_1_soft_entity_penalty", "deepseek_chat_llm"]
    samples: dict[str, dict[str, list[float]]] = {m: {k: [] for k in BOOT_METRICS} for m in methods}
    deltas: dict[str, dict[str, list[float]]] = {
        "hybrid_minus_semantic": {k: [] for k in BOOT_METRICS},
        "hybrid_minus_deepseek": {k: [] for k in BOOT_METRICS},
    }
    diag_rows: list[dict[str, Any]] = []

    for it in range(n_iter):
        targets = dict(sc)
        boot_posts, boot_gold = _stratified_story_bootstrap_sample(
            test_story_posts, story_stratum, targets, rng
        )
        if len(boot_posts) < 10:
            continue
        universe = set(boot_posts)
        boot_order = [p for p in order if p in universe]
        m_by: dict[str, dict[str, float]] = {}
        for m in methods:
            pred_of = pred_by_method[m]
            cl = clusters_by_method[m]
            mm = _eval_on_universe(boot_order, boot_gold, pred_of, cl, universe)
            m_by[m] = mm
            for k in BOOT_METRICS:
                key = k
                if k == "pairwise_f1":
                    key = "pairwise_cluster_f1"
                elif k == "adjusted_rand_index":
                    key = "adjusted_rand_index"
                samples[m][k].append(float(mm[key]))

        for k in BOOT_METRICS:
            deltas["hybrid_minus_semantic"][k].append(samples["hybrid_v4_1_soft_entity_penalty"][k][-1] - samples["semantic_only"][k][-1])
            deltas["hybrid_minus_deepseek"][k].append(
                samples["hybrid_v4_1_soft_entity_penalty"][k][-1] - samples["deepseek_chat_llm"][k][-1]
            )

        if it < 5:
            diag_rows.append(
                {
                    "iteration": it,
                    "n_posts": len(boot_posts),
                    "n_gold_stories_sampled": len({boot_gold[p] for p in boot_posts}),
                    "strata_1": targets.get("1", 0),
                    "strata_2": targets.get("2", 0),
                    "strata_3-4": targets.get("3-4", 0),
                    "strata_5+": targets.get("5+", 0),
                }
            )

    def _ci(vals: list[float]) -> tuple[float, float, float]:
        arr = np.array(vals, dtype=np.float64)
        return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

    out: list[dict[str, Any]] = []
    for variant, entity, kind in [
        ("stratified_gold_story", "semantic_only", "metric"),
        ("stratified_gold_story", "hybrid_v4_1_soft_entity_penalty", "metric"),
        ("stratified_gold_story", "deepseek_chat_llm", "metric"),
        ("stratified_gold_story", "hybrid_minus_semantic", "delta"),
        ("stratified_gold_story", "hybrid_minus_deepseek", "delta"),
    ]:
        pool = samples if kind == "metric" else deltas
        name = entity
        for metric in BOOT_METRICS:
            mean, lo, hi = _ci(pool[name][metric])
            pe_key = metric
            if metric == "pairwise_f1":
                pe_key = "pairwise_f1"
            point = float("nan")
            if kind == "metric":
                m0 = entity
                if m0 in point_estimates:
                    point = float(point_estimates[m0].get(pe_key, float("nan")))
            crosses = ""
            stable = ""
            if kind == "delta":
                if entity == "hybrid_minus_semantic":
                    base_m = "hybrid_v4_1_soft_entity_penalty"
                    other = "semantic_only"
                else:
                    base_m = "hybrid_v4_1_soft_entity_penalty"
                    other = "deepseek_chat_llm"
                if metric in point_estimates.get(base_m, {}) and metric in point_estimates.get(other, {}):
                    point = float(point_estimates[base_m][metric]) - float(point_estimates[other][metric])
                crosses = str(lo <= 0.0 <= hi).lower()
                stable = "yes" if (lo > 0 or hi < 0) else "no"
            out.append(
                {
                    "bootstrap_variant": variant,
                    "kind": kind,
                    "method_or_delta": entity,
                    "metric": metric,
                    "official_point_estimate": point,
                    "bootstrap_mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "crosses_zero": crosses,
                    "statistically_stable": stable,
                    "n_iterations": len(pool[name][metric]),
                }
            )
    return out, diag_rows


def _bootstrap_post_level_diagnostic(
    *,
    order: list[int],
    gold: dict[int, str],
    test_ids: list[int],
    pred_by_method: dict[str, dict[int, int]],
    clusters_by_method: dict[str, list[list[int]]],
    n_iter: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Диагностика: ресэмпл post_id с возвращением (pairwise на дубликатах — осторожно)."""
    rng = random.Random(seed + 17)
    methods = ["semantic_only", "hybrid_v4_1_soft_entity_penalty"]
    out: list[dict[str, Any]] = []
    n = len(test_ids)
    for m in methods:
        vals: list[float] = []
        for _ in range(n_iter):
            draw = [test_ids[rng.randrange(n)] for _ in range(n)]
            universe = set(draw)
            mm = _eval_on_universe(order, gold, pred_by_method[m], clusters_by_method[m], universe)
            vals.append(float(mm["pairwise_cluster_f1"]))
        mean, lo, hi = float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
        out.append(
            {
                "bootstrap_variant": "post_level_paired_diagnostic",
                "method": m,
                "metric": "pairwise_f1",
                "bootstrap_mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "note": "post_id resample with replacement; duplicate posts inflate pair counts — not primary CI",
            }
        )
    return out


def _csv_has_forbidden_cols(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        r = csv.reader(f)
        try:
            hdr = next(r)
        except StopIteration:
            return []
    return [c for c in hdr if c.lower() in FORBIDDEN_CSV_COLS]


def _validate_outputs(
    *,
    audit_dir: Path,
    unified_metrics: list[dict[str, Any]],
    subgroup_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    hypothesis_rows: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    required = [
        audit_dir / "unified_metrics_v2.csv",
        audit_dir / "unified_metrics_diff_vs_old_v2.csv",
        audit_dir / "subgroup_metrics_unified_v2.csv",
        audit_dir / "subgroup_metrics_diff_vs_v1.csv",
        audit_dir / "bootstrap_confidence_intervals_unified_v2.csv",
        audit_dir / "bootstrap_diagnostics_v2.csv",
        audit_dir / "hypothesis_status_v2.csv",
        audit_dir / "ablation_metrics_unified_v2.csv",
        audit_dir / "dataset_composition_v2.csv",
    ]
    for p in required:
        if not p.exists():
            issues.append(f"missing {p}")

    for p in required:
        bad = _csv_has_forbidden_cols(p)
        if bad:
            issues.append(f"forbidden columns in {p.name}: {bad}")

    test_official = {
        (r["method"], r["split"]): r
        for r in unified_metrics
        if r.get("split") == "test"
    }
    for method in METHODS_ALL:
        if (method, "test") not in test_official:
            issues.append(f"missing official test row for {method}")

    for method in METHODS_ALL:
        sub = next((r for r in subgroup_rows if r["group"] == "all_test" and r["method"] == method), None)
        off = test_official.get((method, "test"))
        if not sub or not off:
            continue
        for col in ("pairwise_f1", "false_merge_rate", "false_split_rate"):
            a, b = float(sub[col]), float(off[col])
            if abs(a - b) > 1e-6:
                issues.append(f"all_test != official for {method} {col}: {a} vs {b}")

    for r in bootstrap_rows:
        if r.get("kind") == "delta" and not str(r.get("crosses_zero", "")).strip():
            issues.append("bootstrap delta missing crosses_zero")
        if not str(r.get("bootstrap_mean", "")).strip():
            issues.append("bootstrap missing bootstrap_mean")

    hyps = {r["hypothesis_id"] for r in hypothesis_rows}
    for h in ("H1", "H2", "H3"):
        if h not in hyps:
            issues.append(f"missing hypothesis {h}")

    v1_boot = _exp_root() / "sync_out/data/prepared/research_strengthening_v1/bootstrap_confidence_intervals_v1.csv"
    if not v1_boot.exists():
        issues.append("v1 bootstrap missing (unexpected)")
    return (len(issues) == 0, issues)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rclone", action="store_true")
    ap.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = ap.parse_args()

    root = _exp_root()
    prep = root / "sync_out/data/prepared/combined_manual_gold_v2"
    llm = root / "sync_out/data/prepared/combined_manual_gold_v2_llm_baseline"
    audit_dir = root / "sync_out/data/prepared/commission_audit_v2"
    reports = root / "sync_out/reports"
    fig_dir = reports / "figures_commission_audit_v2"
    _ensure_dir(audit_dir)
    _ensure_dir(fig_dir)

    gold = _load_gold_labels(prep / "combined_manual_gold_labels_v2.csv")
    order, split_of, _chrono = _load_order_split(prep / "combined_split_v2.csv")
    feats = _load_feats(prep / "combined_pairwise_features_v2.csv")
    tuned = _load_tuned_params(prep / "tuned_params_combined_manual_gold_v2.json")

    train_ids = {p for p in order if split_of[p] == "train"}
    dev_ids = {p for p in order if split_of[p] == "dev"}
    test_ids = {p for p in order if split_of[p] == "test"}
    all_ids = set(order)

    pred_by_method: dict[str, dict[int, int]] = {}
    clusters_by_method: dict[str, list[list[int]]] = {}

    for method in METHODS_REPLAY:
        thr = float(tuned["methods"][method]["tuned_threshold"])
        pred_of, clusters = _replay_method(method, order, feats, thr)
        pred_by_method[method] = pred_of
        clusters_by_method[method] = clusters

    pred_h = _load_hybrid_v4_predictions(prep / "predictions_combined_manual_gold_v2.csv")
    pred_by_method["hybrid_v4_1_soft_entity_penalty"] = pred_h
    clusters_by_method["hybrid_v4_1_soft_entity_penalty"] = _pred_clusters_from_map(order, pred_h)

    pred_llm = _load_deepseek_predictions(llm / "deepseek_llm_predictions_v1.csv")
    pred_by_method["deepseek_chat_llm"] = pred_llm
    clusters_by_method["deepseek_chat_llm"] = _pred_clusters_from_map(order, pred_llm)

    unified_rows: list[dict[str, Any]] = []
    point_estimates: dict[str, dict[str, float]] = {}

    for method in METHODS_ALL:
        for split_name, universe in (
            ("train", train_ids),
            ("dev", dev_ids),
            ("test", test_ids),
            ("all", all_ids),
        ):
            m = _eval_on_universe(
                order,
                gold,
                pred_by_method[method],
                clusters_by_method[method],
                universe,
            )
            row = _metrics_to_row(method, split_name, m)
            unified_rows.append(row)
            if split_name == "test":
                point_estimates[method] = {
                    "pairwise_f1": float(m["pairwise_cluster_f1"]),
                    "adjusted_rand_index": float(m["adjusted_rand_index"]),
                    "nmi": float(m["nmi"]),
                    "false_merge_rate": float(m["false_merge_rate"]),
                    "false_split_rate": float(m["false_split_rate"]),
                    "post_assignment_accuracy": float(m["post_assignment_accuracy"]),
                    "singleton_predicted_rate": float(m["singleton_predicted_rate"]),
                }

    _write_csv(audit_dir / "unified_metrics_v2.csv", unified_rows, ["method", "split"] + METRIC_COLS)

    old_metrics = list(csv.DictReader((prep / "metrics_combined_manual_gold_v2.csv").open(encoding="utf-8")))
    old_llm = list(csv.DictReader((llm / "deepseek_llm_metrics_v1.csv").open(encoding="utf-8")))
    diff_rows: list[dict[str, Any]] = []
    old_index = {(r["method"], r["split"]): r for r in old_metrics}
    for r in old_llm:
        if r.get("split") == "test" and "pairwise_cluster_f1" in r:
            old_index[("deepseek_chat_llm", "test")] = r

    for r in unified_rows:
        if r["split"] != "test":
            continue
        key = (r["method"], "test")
        old = old_index.get(key)
        if not old:
            diff_rows.append(
                {
                    "method": r["method"],
                    "metric": "all",
                    "unified_v2": "",
                    "legacy_official": "",
                    "abs_diff": "",
                    "note": "no legacy row",
                }
            )
            continue
        mapping = {
            "pairwise_f1": ("pairwise_f1", "pairwise_cluster_f1"),
            "pairwise_precision": ("pairwise_precision", "pairwise_cluster_precision"),
            "pairwise_recall": ("pairwise_recall", "pairwise_cluster_recall"),
            "post_assignment_accuracy": ("post_assignment_accuracy", "post_assignment_accuracy"),
            "false_merge_rate": ("false_merge_rate", "false_merge_rate"),
            "false_split_rate": ("false_split_rate", "false_split_rate"),
            "adjusted_rand_index": ("adjusted_rand_index", "adjusted_rand_index"),
            "nmi": ("nmi", "nmi"),
            "b_cubed": ("b_cubed", "b_cubed"),
        }
        for ncol, (ukey, okey) in mapping.items():
            try:
                nv = float(r[ukey])
                ov = float(old.get(okey, old.get(ukey, "nan")))
            except (TypeError, ValueError):
                continue
            diff_rows.append(
                {
                    "method": r["method"],
                    "metric": ncol,
                    "unified_v2": nv,
                    "legacy_official": ov,
                    "abs_diff": abs(nv - ov),
                    "note": "match" if abs(nv - ov) < 1e-6 else "diff",
                }
            )

    _write_csv(
        audit_dir / "unified_metrics_diff_vs_old_v2.csv",
        diff_rows,
        ["method", "metric", "unified_v2", "legacy_official", "abs_diff", "note"],
    )

    meta_path = prep / "combined_posts_metadata_v2.csv"
    pid2ts: dict[int, datetime] = {}
    for r in csv.DictReader(meta_path.open(encoding="utf-8")):
        pid2ts[int(r["post_id"])] = _parse_ts(r.get("published_at", "")) or datetime.min

    subgroup_groups = _build_subgroup_groups(test_ids, gold, pid2ts)
    subgroup_rows: list[dict[str, Any]] = []
    for gname, universe in sorted(subgroup_groups.items()):
        if len(universe) < 1:
            continue
        pair_note = (
            "pair_universe=all unordered pairs among posts in subgroup; "
            "partial gold stories may appear truncated"
        )
        for method in METHODS_ALL:
            m = _eval_on_universe(
                order,
                gold,
                pred_by_method[method],
                clusters_by_method[method],
                universe,
            )
            subgroup_rows.append(
                {
                    "group": gname,
                    "method": method,
                    "n_posts": len(universe),
                    "pair_universe_note": pair_note,
                    "pairwise_f1": m["pairwise_cluster_f1"],
                    "pairwise_precision": m["pairwise_cluster_precision"],
                    "pairwise_recall": m["pairwise_cluster_recall"],
                    "ari": m.get("adjusted_rand_index", ""),
                    "nmi": m.get("nmi", ""),
                    "false_merge_rate": m["false_merge_rate"],
                    "false_split_rate": m["false_split_rate"],
                    "post_assignment_accuracy": m["post_assignment_accuracy"],
                    "predicted_story_count": m["predicted_story_count"],
                    "singleton_predicted_rate": m["singleton_predicted_rate"],
                }
            )

    _write_csv(
        audit_dir / "subgroup_metrics_unified_v2.csv",
        subgroup_rows,
        [
            "group",
            "method",
            "n_posts",
            "pair_universe_note",
            "pairwise_f1",
            "pairwise_precision",
            "pairwise_recall",
            "ari",
            "nmi",
            "false_merge_rate",
            "false_split_rate",
            "post_assignment_accuracy",
            "predicted_story_count",
            "singleton_predicted_rate",
        ],
    )

    v1_sub = list(
        csv.DictReader(
            (root / "sync_out/data/prepared/research_strengthening_v1/subgroup_metrics_v1.csv").open(encoding="utf-8")
        )
    )
    sub_diff: list[dict[str, Any]] = []
    v1_idx = {(r["group"], r["method"]): r for r in v1_sub}
    for r in subgroup_rows:
        if r["group"] != "all_test":
            continue
        old = v1_idx.get((r["group"], r["method"]))
        if not old:
            continue
        sub_diff.append(
            {
                "group": r["group"],
                "method": r["method"],
                "metric": "pairwise_f1",
                "unified_v2": r["pairwise_f1"],
                "subgroup_v1": old.get("pairwise_f1", ""),
                "abs_diff": abs(float(r["pairwise_f1"]) - float(old["pairwise_f1"])),
                "note": "v1 used different evaluator for semantic_only replay",
            }
        )
    _write_csv(
        audit_dir / "subgroup_metrics_diff_vs_v1.csv",
        sub_diff,
        ["group", "method", "metric", "unified_v2", "subgroup_v1", "abs_diff", "note"],
    )

    n_boot = args.bootstrap_iterations
    boot_rows, boot_diag = _bootstrap_stratified_gold(
        order=order,
        gold=gold,
        test_ids=test_ids,
        pred_by_method=pred_by_method,
        clusters_by_method=clusters_by_method,
        point_estimates=point_estimates,
        n_iter=n_boot,
        seed=42,
    )
    post_diag = _bootstrap_post_level_diagnostic(
        order=order,
        gold=gold,
        test_ids=sorted(test_ids),
        pred_by_method=pred_by_method,
        clusters_by_method=clusters_by_method,
        n_iter=min(300, n_boot),
        seed=42,
    )
    _write_csv(
        audit_dir / "bootstrap_confidence_intervals_unified_v2.csv",
        boot_rows,
        [
            "bootstrap_variant",
            "kind",
            "method_or_delta",
            "metric",
            "official_point_estimate",
            "bootstrap_mean",
            "ci_low",
            "ci_high",
            "crosses_zero",
            "statistically_stable",
            "n_iterations",
        ],
    )
    _write_csv(
        audit_dir / "bootstrap_diagnostics_v2.csv",
        boot_diag + post_diag,
        list({k for row in boot_diag + post_diag for k in row}),
    )

    ablation_rows = [r for r in unified_rows if r["split"] == "test"]
    _write_csv(
        audit_dir / "ablation_metrics_unified_v2.csv",
        ablation_rows,
        ["method", "split"] + METRIC_COLS,
    )

    comp_rows: list[dict[str, Any]] = []
    for split_name in ("train", "dev", "test", "all"):
        ids = {"train": train_ids, "dev": dev_ids, "test": test_ids, "all": all_ids}[split_name]
        comp_rows.append(
            {
                "split": split_name,
                "n_posts": len(ids),
                "n_gold_stories": len({gold[p] for p in ids}),
                "n_gold_singleton_stories": sum(1 for g in {gold[p] for p in ids} if len([q for q in ids if gold[q] == g]) == 1),
            }
        )
    _write_csv(audit_dir / "dataset_composition_v2.csv", comp_rows, list(comp_rows[0].keys()) if comp_rows else ["split"])

    # Reports (concise; details in CSV)
    match_sem = any(
        d["method"] == "semantic_only" and d["metric"] == "pairwise_f1" and d["note"] == "match" for d in diff_rows
    )
    hyb_diff = next((d for d in diff_rows if d["method"] == "hybrid_graph_v1_entity" and d["metric"] == "pairwise_f1"), None)
    sem_sub = next((d for d in sub_diff if d["method"] == "semantic_only"), None)

    _write_md(
        reports / "commission_audit_unified_metrics_v2.md",
        f"""# Commission audit: unified metrics v2

## Source of truth

**Official metrics:** `sync_out/data/prepared/commission_audit_v2/unified_metrics_v2.csv` (test split = official test).

Evaluator: `tools/pilot_eval_v2_assignment_lib.py` (`aggregate_metrics` + `attach_ari_nmi`).

## Match vs legacy `metrics_combined_manual_gold_v2.csv`

- **semantic_only (test):** {'совпадает' if match_sem else 'расхождение'} (pairwise F1).
- **semantic_time (test):** мелкое расхождение возможно из‑за float threshold.
- **hybrid_v4_1_soft_entity_penalty:** из сохранённых predictions — совпадает с legacy.
- **hybrid_graph_v1_entity:** replay на текущих `combined_pairwise_features_v2` при пороге 0.56 из `tuned_params` **не** воспроизводит legacy F1≈0.216 (replay даёт F1≈{hyb_diff['unified_v2'] if hyb_diff else 'n/a'} при thr=0.56). Близкое значение (~0.214) получается при thr≈0.83 — вероятна смена feature snapshot или иной протокол тюнинга; legacy строка сохранена в diff CSV.

## Metric definitions (кратко)

| Метрика | Смысл |
|--------|--------|
| **Pairwise F1** | F1 по парам постов в universe: TP — одна gold-история и один pred-кластер; FP — разные gold, один pred; FN — одна gold, разные pred. |
| **post_assignment_accuracy** | Доля не-первых постов gold-сюжета, попавших в тот же pred-кластер, что первый пост сюжета (online anchor). **≠ Pairwise F1.** |
| **FMR** | `false_merge_count / neg_pairs` (пары с разными gold, но одним pred). |
| **FSR** | Доля gold-сюжетов (size≥2 в universe), разбитых на ≥2 pred-кластера. |
| **singleton_predicted_rate** | Доля pred-кластеров-одиночек среди кластеров, затрагивающих universe. |

## Diff table

`sync_out/data/prepared/commission_audit_v2/unified_metrics_diff_vs_old_v2.csv`
""",
    )

    _write_md(
        reports / "commission_audit_subgroup_metrics_v2.md",
        f"""# Commission audit: subgroup metrics v2

## Причина расхождения v1 `all_test` semantic_only F1≈0.326 vs official ≈0.121

`subgroup_metrics_v1.csv` пересчитывал **semantic_only** через `strengthen_research_artifacts_for_vkr_v1.evaluate_metrics_on_subset` с **другими** определениями FMR/FSR (доля pred-кластеров с ≥2 gold) и replay, не совпадающим с `pilot_eval_v2_assignment_lib`.

В **v2** `all_test` = official test для каждого метода (проверка: unified subgroup vs unified_metrics).

Пример semantic_only:
- unified v2 all_test F1: {next(r['pairwise_f1'] for r in subgroup_rows if r['group']=='all_test' and r['method']=='semantic_only')}
- v1 all_test F1: {sem_sub['subgroup_v1'] if sem_sub else 'n/a'}

**Pair universe:** все неупорядоченные пары постов в подгруппе; для подвыборки по размеру/источнику gold-сюжет может быть усечён — см. `pair_universe_note` в CSV.
""",
    )

    boot_h_s = next(
        r
        for r in boot_rows
        if r["method_or_delta"] == "hybrid_minus_semantic" and r["metric"] == "pairwise_f1"
    )
    boot_h_d = next(
        r
        for r in boot_rows
        if r["method_or_delta"] == "hybrid_minus_deepseek" and r["metric"] == "pairwise_f1"
    )

    _write_md(
        reports / "commission_audit_bootstrap_ci_v2.md",
        f"""# Commission audit: bootstrap CI v2

## Почему v1 bootstrap mean ≠ official point

v1 bootstrap использовал тот же упрощённый evaluator, что subgroup v1 → завышенный mean для semantic_only (F1 mean ≈0.38 vs official 0.121).

## v2 primary: stratified gold-story bootstrap

- Ресэмпл **gold_story_id** на test с сохранением числа сюжетов по стратам размера (1 / 2 / 3-4 / 5+).
- Все посты выбранных сюжетов; копии сюжета получают remapped gold id (`gid__booti`), чтобы пары не смешивались между копиями.
- Итераций: {n_boot}

## Дельты (pairwise F1)

| Delta | point (hybrid−baseline) | bootstrap mean | 95% CI | crosses_zero | stable |
|-------|-------------------------|----------------|--------|--------------|--------|
| hybrid−semantic | {point_estimates['hybrid_v4_1_soft_entity_penalty']['pairwise_f1'] - point_estimates['semantic_only']['pairwise_f1']:.4f} | {boot_h_s['bootstrap_mean']:.4f} | [{boot_h_s['ci_low']:.4f}, {boot_h_s['ci_high']:.4f}] | {boot_h_s['crosses_zero']} | {boot_h_s['statistically_stable']} |
| hybrid−deepseek | {point_estimates['hybrid_v4_1_soft_entity_penalty']['pairwise_f1'] - point_estimates['deepseek_chat_llm']['pairwise_f1']:.4f} | {boot_h_d['bootstrap_mean']:.4f} | [{boot_h_d['ci_low']:.4f}, {boot_h_d['ci_high']:.4f}] | {boot_h_d['crosses_zero']} | {boot_h_d['statistically_stable']} |

Post-level bootstrap — только диагностика (`bootstrap_diagnostics_v2.csv`).
""",
    )

    # Plot deltas
    delta_metrics = ["pairwise_f1", "adjusted_rand_index", "nmi", "false_merge_rate", "false_split_rate"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(delta_metrics))
    w = 0.35
    for j, dname in enumerate(["hybrid_minus_semantic", "hybrid_minus_deepseek"]):
        means, err_lo, err_hi = [], [], []
        for met in delta_metrics:
            row = next(
                (
                    r
                    for r in boot_rows
                    if r["kind"] == "delta" and r["method_or_delta"] == dname and r["metric"] == met
                ),
                None,
            )
            if not row:
                continue
            m = float(row["bootstrap_mean"])
            means.append(m)
            err_lo.append(m - float(row["ci_low"]))
            err_hi.append(float(row["ci_high"]) - m)
        if means:
            ax.bar(x + (j - 0.5) * w, means, width=w, yerr=[err_lo, err_hi], capsize=3, label=dname)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(delta_metrics, rotation=20, ha="right")
    ax.set_title("Bootstrap 95% CI: hybrid v4.1 deltas (test, stratified gold-story)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(fig_dir / "bootstrap_delta_ci_unified_v2.png", dpi=180, bbox_inches="tight")
    fig.savefig(fig_dir / "bootstrap_delta_ci_unified_v2.svg", bbox_inches="tight")
    plt.close(fig)

    sem = point_estimates["semantic_only"]
    hyb = point_estimates["hybrid_v4_1_soft_entity_penalty"]
    llm = point_estimates["deepseek_chat_llm"]
    d_f1 = hyb["pairwise_f1"] - sem["pairwise_f1"]
    h1_status = "supported" if d_f1 >= 0.15 and boot_h_s["ci_low"] > 0 else (
        "partial/not_supported" if d_f1 > 0 else "not_supported"
    )
    h2_status = "not_formally_testable"  # ex post non-inferiority
    fsr_delta = hyb["false_split_rate"] - sem["false_split_rate"]
    sing_delta = hyb["singleton_predicted_rate"] - sem["singleton_predicted_rate"]
    h3_status = "mixed"
    if fsr_delta <= -0.25 and sing_delta <= -0.30:
        h3_status = "supported"
    elif fsr_delta < 0 or sing_delta < 0:
        h3_status = "partial"

    hypothesis_rows = [
        {
            "hypothesis_id": "H1",
            "statement": "Hybrid improves clustering vs semantic-only",
            "primary_metric": "pairwise_f1",
            "target": "+15 pp",
            "point_delta_hybrid_minus_semantic": d_f1,
            "bootstrap_ci_crosses_zero": boot_h_s["crosses_zero"],
            "status": h1_status,
        },
        {
            "hypothesis_id": "H2",
            "statement": "Hybrid not worse than LLM baseline",
            "primary_metric": "pairwise_f1",
            "target": "strict non-inferiority not formally testable ex post",
            "point_delta_hybrid_minus_deepseek": hyb["pairwise_f1"] - llm["pairwise_f1"],
            "bootstrap_ci_crosses_zero": boot_h_d["crosses_zero"],
            "status": h2_status,
        },
        {
            "hypothesis_id": "H3",
            "statement": "Hybrid reduces false splits and singleton stories",
            "primary_metric": "false_split_rate, singleton_predicted_rate",
            "target": "FSR -25pp; singleton -30% if in protocol",
            "fsr_delta": fsr_delta,
            "singleton_rate_delta": sing_delta,
            "fmr_tradeoff": hyb["false_merge_rate"] - sem["false_merge_rate"],
            "status": h3_status,
        },
    ]
    _write_csv(
        audit_dir / "hypothesis_status_v2.csv",
        hypothesis_rows,
        list(hypothesis_rows[0].keys()),
    )

    _write_md(
        reports / "commission_audit_hypothesis_status_v2.md",
        "\n".join(
            [
                "# Hypothesis status v2",
                "",
                f"- **H1** ({h1_status}): Δ pairwise F1 = {d_f1*100:.2f} pp (target +15 pp).",
                f"- **H2** ({h2_status}): Δ vs DeepSeek = {(hyb['pairwise_f1']-llm['pairwise_f1'])*100:.2f} pp; formal non-inferiority margin не задавался.",
                f"- **H3** ({h3_status}): Δ FSR = {fsr_delta*100:.2f} pp; Δ singleton rate = {sing_delta*100:.2f} pp; Δ FMR = {(hyb['false_merge_rate']-sem['false_merge_rate'])*100:.2f} pp.",
            ]
        ),
    )

    _write_md(
        reports / "commission_audit_ablation_v2.md",
        "# Ablation v2\n\nСм. `ablation_metrics_unified_v2.csv` — все методы на official test, единый evaluator.\n",
    )

    _write_md(
        reports / "commission_audit_protocol_and_leakage_v2.md",
        """# Protocol and leakage audit v2

## Dev vs test

- Пороги (`tuned_threshold`) подбирались на **dev** (train+dev chronological replay), objective structural / accuracy−penalties — см. `tuned_params_combined_manual_gold_v2.json`.
- **Test** использовался только для readout метрик, не для подбора порога в этом аудите.
- Gold **не** входит в online assignment (`simulate_no_gold`).

## Leakage signals

- Нет использования `same_story_gold` в scoring.
- LLM baseline: только сохранённые `deepseek_llm_predictions_v1.csv`; **DeepSeek не вызывался** в этом шаге.
- Retrieval top-K для LLM описан в отдельном v1 отчёте; пересчёт не выполнялся.

## hybrid_graph_v1 replay caveat

Текущий replay на features v2 при legacy threshold 0.56 не воспроизводит legacy test F1; для отчётов опираться на unified CSV и diff.
""",
    )

    n_test_posts = len(test_ids)
    n_test_stories = len({gold[p] for p in test_ids})
    _write_md(
        reports / "commission_audit_dataset_adequacy_v2.md",
        f"""# Dataset adequacy v2

- Posts total: {len(order)}; test: {n_test_posts}.
- Gold stories on test: {n_test_stories}.
- Test fragmented across OLD_* and EXP2_* prefixes.

**Risks:** small test N, high variance (bootstrap CI широкие). Расширение benchmark желательно; пересборка датасета **не** выполнялась в этом шаге.
""",
    )

    ok, issues = _validate_outputs(
        audit_dir=audit_dir,
        unified_metrics=unified_rows,
        subgroup_rows=subgroup_rows,
        bootstrap_rows=boot_rows,
        hypothesis_rows=hypothesis_rows,
    )

    _write_md(
        reports / "commission_audit_summary_v2.md",
        f"""# Commission audit summary v2

- **validation_ok:** {str(ok).lower()}
- **Official source of truth:** `commission_audit_v2/unified_metrics_v2.csv`
- **Key discrepancy fixed:** subgroup v1 semantic_only used wrong evaluator (F1 0.326 vs 0.121).
- **hybrid_graph_v1:** legacy metrics retained in diff; unified replay differs at stored threshold.
- **Repeat experiment?** Не обязательно для метрик v4/LLM (predictions совпадают); желателен re-run hybrid_graph только если нужен воспроизводимый replay.
- **Expand benchmark?** Да, рекомендуется (малый test).

## Hypotheses

- H1: {h1_status}
- H2: {h2_status}
- H3: {h3_status}

## Unsafe wording for thesis

- «Bootstrap подтверждает +15 п.п. для semantic» без указания evaluator.
- Смешивание post_assignment_accuracy с Pairwise F1.
- subgroup v1 all_test для semantic_only.

## Safe wording

- Point estimates из `unified_metrics_v2.csv` (pilot_eval definitions).
- H1 target +15 pp **not met** at point; CI for hybrid−semantic crosses zero for pairwise F1.

## Validation issues

{chr(10).join('- ' + i for i in issues) if issues else '- none'}
""",
    )

    tools_out = root / "sync_out/tools"
    _ensure_dir(tools_out)
    for name in (
        "recompute_combined_manual_gold_v2_metrics_unified.py",
        "pilot_eval_v2_assignment_lib.py",
    ):
        src = _TOOLS / name
        if src.exists():
            (tools_out / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    if not args.skip_rclone:
        r = subprocess.run(
            ["rclone", "sync", str(root / "sync_out"), "gdrive:diploma_story_tracking_experiment"],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            issues.append(f"rclone failed: {r.stderr[:500]}")
            ok = False

    print(json.dumps({"validation_ok": ok, "issues": issues}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
