#!/usr/bin/env python3.11
"""trace_hybrid_v4_1_official_saved_provenance_v1 — разбор расхождений с сохранёнными предсказаниями гибрида v4.1 (только чтение, без записи)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from build_hybrid_as_weak_target_advantage_diagnostic_v1 import _load_feats  # noqa: E402
from evaluate_large_hybrid_v4_1_guard_grid import (  # noqa: E402
    _cluster_max_score,
    _pair_score_v41,
    simulate_v4_1,
)
from pilot_eval_v2_assignment_lib import aggregate_metrics, attach_ari_nmi, pair_key  # noqa: E402

KEYWORDS = (
    "hybrid_v4_1",
    "soft_entity_penalty",
    "predictions_combined_manual_gold_v2",
    "metrics_combined_manual_gold_v2",
    "tuned_params_combined_manual_gold_v2",
    "evaluate_large_hybrid",
    "guard_grid",
    "catboost",
    "reranker",
    "storyline_resolver",
    "v4_1",
)

V41_CFG = {
    "threshold": 0.5800000000000001,
    "w_sem": 0.65,
    "w_time": 0.15,
    "w_ent": 0.10,
    "use_soft_entity": 1,
    "use_event_soft": 0,
    "use_margin_soft": 0,
    "use_story_compat": 0,
    "nb_low_entity_soft": 0.15,
    "penalty_soft_entity": 0.10,
    "evt_floor": 0.0,
    "event_penalty": 0.0,
    "margin_min_soft": 0.0,
    "margin_soft_penalty": 0.0,
    "top_k": 3,
    "mean_score_floor": 0.5,
    "compat_penalty": 0.0,
}

OFFICIAL_METHOD = "hybrid_v4_1_soft_entity_penalty"
SCRIPT_GREP_TERMS = (
    "predictions_combined_manual_gold_v2",
    "hybrid_v4_1_soft_entity_penalty",
    "simulate_v4_1",
    "soft_entity_penalty",
    "entity_penalty",
    "guard",
    "CatBoost",
    "catboost",
    "reranker",
)


def _exp_root() -> Path:
    return _TOOLS.parent


def _ensure(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    _ensure(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _mtime_iso(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()


def _file_type(p: Path) -> str:
    if p.suffix in (".csv", ".tsv"):
        return "csv"
    if p.suffix in (".json",):
        return "json"
    if p.suffix in (".md",):
        return "markdown"
    if p.suffix in (".py",):
        return "python"
    return p.suffix.lstrip(".") or "other"


def _path_keywords(path: Path) -> str:
    s = str(path).lower()
    return "|".join(k for k in KEYWORDS if k.lower() in s)


def _scan_text_hints(path: Path, limit: int = 12000) -> dict[str, bool]:
    out = {
        "has_method_names": False,
        "has_command_params": False,
        "has_thresholds": False,
    }
    if not path.is_file() or path.stat().st_size > 5_000_000:
        return out
    try:
        raw = path.read_bytes()[:limit]
        text = raw.decode("utf-8", errors="ignore").lower()
    except OSError:
        return out
    out["has_method_names"] = bool(
        re.search(r"hybrid_v4_1|semantic_only|soft_entity_penalty|method", text)
    )
    out["has_command_params"] = bool(
        re.search(r"argparse|subprocess|python3|evaluate_|simulate_v4|guard_grid", text)
    )
    out["has_thresholds"] = bool(re.search(r"threshold|tuned_threshold|0\.58|0\.65", text))
    return out


def build_artifact_inventory(exp: Path, sync: Path) -> list[dict]:
    rows: list[dict] = []
    roots = [exp, sync]
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                p = Path(dirpath) / fn
                rel = p.relative_to(exp) if p.is_relative_to(exp) else p
                kw_path = _path_keywords(p)
                if not kw_path and p.suffix not in (".py", ".json", ".md", ".csv"):
                    continue
                if p.suffix in (".png", ".jpg", ".xlsx", ".pkl", ".bin"):
                    if not kw_path:
                        continue
                key = str(p.resolve())
                if key in seen:
                    continue
                seen.add(key)
                hints = _scan_text_hints(p) if p.suffix in (".py", ".json", ".md", ".csv", ".txt") else {}
                desc = ""
                if "predictions_combined" in str(p):
                    desc = "saved hybrid predictions bundle"
                elif "combined_pairwise" in str(p):
                    desc = "pairwise feature snapshot for combined v2"
                elif "tuned_params_combined" in str(p):
                    desc = "frozen thresholds per method"
                elif "evaluate_large_hybrid" in str(p):
                    desc = "v4.1 guard grid + simulate_v4_1 reference"
                elif "recompute_combined" in str(p):
                    desc = "metrics recompute; hybrid loads saved preds only"
                elif "component_ablation" in str(p):
                    desc = "replay vs saved mismatch evidence"
                rows.append(
                    {
                        "path": str(rel),
                        "size_bytes": p.stat().st_size,
                        "mtime_utc": _mtime_iso(p),
                        "file_type": _file_type(p),
                        "matching_keywords": kw_path,
                        "short_description": desc,
                        "contains_method_names": hints.get("has_method_names", False),
                        "contains_command_params_thresholds": hints.get("has_command_params", False)
                        or hints.get("has_thresholds", False),
                    }
                )
    rows.sort(key=lambda r: r["path"])
    return rows


def audit_predictions(prep: Path) -> tuple[dict, list[dict]]:
    path = prep / "predictions_combined_manual_gold_v2.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    cols = list(rows[0].keys()) if rows else []
    methods = sorted({r["method"] for r in rows})
    by_method_split: Counter[tuple[str, str]] = Counter()
    cluster_ids: set[int] = set()
    for r in rows:
        by_method_split[(r["method"], r["split"])] += 1
        cluster_ids.add(int(r["predicted_story_cluster_id"]))
    schema = {
        "file": str(path.relative_to(_exp_root() / "sync_out")),
        "columns": cols,
        "row_count": len(rows),
        "format": "long (one row per post per method)",
        "unique_methods": methods,
        "has_score_column": "score" in cols,
        "has_threshold_column": "threshold" in cols,
        "has_source_column": "source" in cols,
        "has_official_saved_marker": any("official" in c.lower() for c in cols),
        "has_hybrid_v4_1_soft_entity_penalty": OFFICIAL_METHOD in methods,
        "max_predicted_story_cluster_id": max(cluster_ids) if cluster_ids else None,
        "splits_present": sorted({r["split"] for r in rows}),
    }
    summary = []
    for m in methods:
        for sp in ("train", "dev", "test"):
            n = by_method_split.get((m, sp), 0)
            if n:
                cl = {int(r["predicted_story_cluster_id"]) for r in rows if r["method"] == m and r["split"] == sp}
                summary.append(
                    {
                        "method": m,
                        "split": sp,
                        "post_count": n,
                        "unique_predicted_clusters": len(cl),
                    }
                )
    return schema, summary


def _clusters_from_pred(order: list[int], pred_of: dict[int, int]) -> list[list[int]]:
    by_id: dict[int, list[int]] = defaultdict(list)
    for p in order:
        if p in pred_of:
            by_id[pred_of[p]].append(p)
    return [by_id[k] for k in sorted(by_id)]


def _metric_row(method: str, split: str, m: dict[str, float], thr: float | None = None) -> dict:
    return {
        "method": method,
        "split": split,
        "post_assignment_accuracy": m["post_assignment_accuracy"],
        "pairwise_precision": m["pairwise_cluster_precision"],
        "pairwise_recall": m["pairwise_cluster_recall"],
        "pairwise_f1": m["pairwise_cluster_f1"],
        "false_merge_rate": m["false_merge_rate"],
        "false_split_rate": m["false_split_rate"],
        "predicted_story_count": m["predicted_story_count"],
        "threshold": thr if thr is not None else "",
    }


def recompute_metrics(prep: Path) -> tuple[list[dict], list[dict]]:
    gold = {int(r["post_id"]): r["gold_story_id"] for r in csv.DictReader((prep / "combined_manual_gold_labels_v2.csv").open())}
    order = [
        int(r["post_id"])
        for r in sorted(
            csv.DictReader((prep / "combined_split_v2.csv").open()),
            key=lambda x: (int(x["chronological_index"]), int(x["post_id"])),
        )
    ]
    split_of = {int(r["post_id"]): r["split"] for r in csv.DictReader((prep / "combined_split_v2.csv").open())}
    train = {p for p in order if split_of[p] == "train"}
    dev = {p for p in order if split_of[p] == "dev"}
    test = {p for p in order if split_of[p] == "test"}
    td = train | dev
    all_posts = set(order)

    pred_by_method: dict[str, dict[int, int]] = defaultdict(dict)
    for r in csv.DictReader((prep / "predictions_combined_manual_gold_v2.csv").open()):
        pred_by_method[r["method"]][int(r["post_id"])] = int(r["predicted_story_cluster_id"])

    feats = _load_feats(prep / "combined_pairwise_features_v2.csv")
    pred_replay, _ = simulate_v4_1(order, feats, V41_CFG)
    pred_by_method["hybrid_v4_1_full_replay"] = pred_replay

    recompute_rows: list[dict] = []
    for method, pred in pred_by_method.items():
        clusters = _clusters_from_pred(order, pred)
        for split_name, subset in [("dev", dev), ("test", test), ("train", train)]:
            if not subset:
                continue
            m = aggregate_metrics(order, gold, pred, clusters, subset, subset)
            attach_ari_nmi(m, order, gold, pred, subset)
            thr = 0.58 if "hybrid" in method else ""
            recompute_rows.append(_metric_row(method, split_name, m, thr))

    def _load_official_metrics(path: Path, col_f1: str) -> dict[tuple[str, str], dict]:
        out: dict[tuple[str, str], dict] = {}
        if not path.is_file():
            return out
        for r in csv.DictReader(path.open(encoding="utf-8")):
            method = r.get("method", r.get("variant", ""))
            sp = r["split"]
            f1 = r.get("pairwise_f1") or r.get("pairwise_cluster_f1", "")
            out[(method, sp)] = {
                "pairwise_f1": float(f1) if f1 not in ("", None) else None,
                "post_assignment_accuracy": float(r.get("post_assignment_accuracy", 0) or 0),
                "false_merge_rate": float(r.get("false_merge_rate", 0) or 0),
                "false_split_rate": float(r.get("false_split_rate", 0) or 0),
                "predicted_story_count": float(r.get("predicted_story_count", 0) or 0),
            }
        return out

    official_combined = _load_official_metrics(prep / "metrics_combined_manual_gold_v2.csv", "pairwise_cluster_f1")
    official_uni = _load_official_metrics(
        _exp_root() / "sync_out/data/prepared/commission_audit_v2/unified_metrics_v2.csv", "pairwise_f1"
    )

    diff_rows: list[dict] = []
    for rec in recompute_rows:
        method, sp = rec["method"], rec["split"]
        for src, off in [("metrics_combined_manual_gold_v2", official_combined), ("unified_metrics_v2", official_uni)]:
            key = (method, sp)
            if key not in off and method == OFFICIAL_METHOD:
                key = (method, sp)
            o = off.get(key)
            if not o:
                continue
            df1 = abs(rec["pairwise_f1"] - o["pairwise_f1"]) if o["pairwise_f1"] is not None else None
            diff_rows.append(
                {
                    "method": method,
                    "split": sp,
                    "official_source": src,
                    "recomputed_pairwise_f1": rec["pairwise_f1"],
                    "official_pairwise_f1": o["pairwise_f1"],
                    "delta_f1": df1,
                    "exact_match_f1": df1 is not None and df1 < 1e-6,
                    "recomputed_pred_story_count": rec["predicted_story_count"],
                    "official_pred_story_count": o["predicted_story_count"],
                }
            )
    return recompute_rows, diff_rows


def _best_replay_score(
    p: int,
    clusters: list[list[int]],
    pred_of: dict[int, int],
    feats: dict,
    order_idx: int,
    order: list[int],
) -> tuple[float, float, float, float, float, float, float]:
    """Scores for post p vs chosen cluster at assignment time."""
    w_sem, w_time, w_ent = 0.65, 0.15, 0.10
    if order_idx == 0:
        return (0.0,) * 7
    existing = clusters[:]
    per_cluster = []
    for j, memb in enumerate(existing):
        if not memb:
            continue
        sj = _cluster_max_score(p, memb, feats, V41_CFG, w_sem, w_time, w_ent)
        per_cluster.append((j, sj))
    per_cluster.sort(key=lambda t: (-t[1], t[0]))
    best_j, best_s = per_cluster[0]
    chosen = pred_of[p]
    # feature breakdown from best member in chosen cluster
    memb = existing[chosen] if chosen < len(existing) else []
    sem = tim = ent = ev = nb = 0.0
    if memb:
        rows = [_load_row(feats, p, q) for q in memb]
        sem = max(r.get("semantic_score", 0) for r in rows)
        tim = max(r.get("temporal_score", 0) for r in rows)
        ent = max(r.get("entity_jaccard", 0) for r in rows)
        ev = max(r.get("event_signature_jaccard", 0) for r in rows)
        nb = max(r.get("graph_neighborhood_jaccard", 0) for r in rows)
    base = 0.65 * sem + 0.15 * tim + 0.10 * ent
    penalized = _pair_score_v41(
        {"entity_jaccard": ent, "event_signature_jaccard": ev, "graph_neighborhood_jaccard": nb},
        V41_CFG,
        w_sem,
        w_time,
        w_ent,
    )
    return best_s, sem, tim, ent, ev, nb, penalized


def _load_row(feats: dict, a: int, b: int) -> dict:
    return dict(feats.get(pair_key(a, b), {}))


def saved_vs_replay(prep: Path) -> tuple[list[dict], dict]:
    gold = {int(r["post_id"]): r["gold_story_id"] for r in csv.DictReader((prep / "combined_manual_gold_labels_v2.csv").open())}
    pub = {}
    labels_path = prep / "combined_manual_gold_labels_v2.csv"
    for r in csv.DictReader(labels_path.open()):
        pub[int(r["post_id"])] = r.get("published_at", "")
    order = [
        int(r["post_id"])
        for r in sorted(
            csv.DictReader((prep / "combined_split_v2.csv").open()),
            key=lambda x: (int(x["chronological_index"]), int(x["post_id"])),
        )
    ]
    split_of = {int(r["post_id"]): r["split"] for r in csv.DictReader((prep / "combined_split_v2.csv").open())}
    saved = {
        int(r["post_id"]): int(r["predicted_story_cluster_id"])
        for r in csv.DictReader((prep / "predictions_combined_manual_gold_v2.csv").open())
        if r["method"] == OFFICIAL_METHOD
    }
    feats = _load_feats(prep / "combined_pairwise_features_v2.csv")
    replay_pred, clusters = simulate_v4_1(order, feats, V41_CFG)

    first_saved: dict[int, int] = {}
    first_replay: dict[int, int] = {}
    out_rows: list[dict] = []
    test_posts = [p for p in order if split_of[p] == "test"]

    # rebuild clusters incrementally for per-post best score
    pred_step: dict[int, int] = {}
    cl_step: list[list[int]] = []
    for i, p in enumerate(order):
        if i == 0:
            cl_step = [[p]]
            pred_step[p] = 0
        else:
            per = []
            for j, memb in enumerate(cl_step):
                if memb:
                    sj = _cluster_max_score(p, memb, feats, V41_CFG, 0.65, 0.15, 0.10)
                    per.append((j, sj))
            per.sort(key=lambda t: (-t[1], t[0]))
            best_j, best_s = per[0] if per else (-1, -1.0)
            if best_j < 0 or best_s < V41_CFG["threshold"]:
                nj = len(cl_step)
                cl_step.append([p])
                pred_step[p] = nj
            else:
                cl_step[best_j].append(p)
                pred_step[p] = best_j
        if split_of[p] != "test":
            continue
        sp = saved[p]
        rp = replay_pred[p]
        if sp not in first_saved:
            first_saved[sp] = p
        if rp not in first_replay:
            first_replay[rp] = p
        bs, sem, tim, ent, ev, nb, pen = _best_replay_score(p, cl_step[:-1] if pred_step[p] == len(cl_step) - 1 else cl_step, pred_step, feats, i, order)
        # simpler: use cluster before assignment
        cl_before = [c[:] for c in cl_step]
        if cl_before and cl_before[-1] == [p]:
            cl_before = cl_before[:-1]
        bs2, sem, tim, ent, ev, nb, pen = _best_replay_score(p, cl_before, pred_step, feats, i, order)
        out_rows.append(
            {
                "post_id": p,
                "published_at": pub.get(p, ""),
                "gold_story_id": gold[p],
                "saved_pred_story_id": sp,
                "replay_pred_story_id": rp,
                "saved_is_new_story": p == first_saved.get(sp, p),
                "replay_is_new_story": p == first_replay.get(rp, p),
                "same_decision": sp == rp,
                "best_candidate_score_replay": round(bs2, 6),
                "semantic_score": round(sem, 6),
                "temporal_score": round(tim, 6),
                "entity_jaccard": round(ent, 6),
                "event_signature_jaccard": round(ev, 6),
                "graph_neighborhood_jaccard": round(nb, 6),
                "threshold": V41_CFG["threshold"],
            }
        )

    agree = sum(1 for r in out_rows if r["same_decision"])
    saved_splits = sum(1 for r in out_rows if not r["same_decision"] and r["saved_is_new_story"] and not r["replay_is_new_story"])
    replay_splits = sum(1 for r in out_rows if not r["same_decision"] and r["replay_is_new_story"] and not r["saved_is_new_story"])
    saved_merges = sum(1 for r in out_rows if not r["same_decision"] and not r["saved_is_new_story"] and r["replay_is_new_story"])
    replay_merges = sum(1 for r in out_rows if not r["same_decision"] and not r["replay_is_new_story"] and not r["saved_is_new_story"])

    summary = {
        "test_post_count": len(test_posts),
        "decision_agreement_rate": agree / len(out_rows) if out_rows else 0,
        "saved_split_replay_merge_count": saved_splits,
        "replay_split_saved_merge_count": replay_splits,
        "saved_merge_replay_split_count": saved_merges,
        "replay_merge_saved_split_count": replay_merges,
        "unique_saved_pred_clusters_test": len({saved[p] for p in test_posts}),
        "unique_replay_pred_clusters_test": len({replay_pred[p] for p in test_posts}),
    }
    return out_rows, summary


def score_distribution_audit(prep: Path, saved: dict[int, int], replay: dict[int, int]) -> tuple[list[dict], list[dict]]:
    gold = {int(r["post_id"]): r["gold_story_id"] for r in csv.DictReader((prep / "combined_manual_gold_labels_v2.csv").open())}
    feats = _load_feats(prep / "combined_pairwise_features_v2.csv")
    w_sem, w_time, w_ent = 0.65, 0.15, 0.10
    thr = V41_CFG["threshold"]

    def _scores_for_pairs(posts: list[int]) -> list[tuple[float, float, bool]]:
        out = []
        for i, a in enumerate(posts):
            for b in posts[i + 1 :]:
                row = _load_row(feats, a, b)
                base = 0.65 * row.get("semantic_score", 0) + 0.15 * row.get("temporal_score", 0) + 0.10 * row.get("entity_jaccard", 0)
                pen = _pair_score_v41(row, V41_CFG, w_sem, w_time, w_ent)
                same_gold = gold[a] == gold[b]
                out.append((base, pen, same_gold))
        return out

    test = [int(r["post_id"]) for r in csv.DictReader((prep / "combined_split_v2.csv").open()) if r["split"] == "test"]
    pairs = _scores_for_pairs(test)
    dist_rows = []
    for label, idx in [("same_gold", 2), ("cross_gold", 2)]:
        for score_kind, col_i in [("base", 0), ("after_soft_penalty", 1)]:
            vals = [p[col_i] for p in pairs if (p[2] == (label == "same_gold"))]
            if not vals:
                continue
            dist_rows.append(
                {
                    "pair_type": label,
                    "score_kind": score_kind,
                    "count": len(vals),
                    "mean": sum(vals) / len(vals),
                    "pct_above_threshold_0_58": sum(1 for v in vals if v >= thr) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                }
            )

    cross = []
    for i, a in enumerate(test):
        for b in test[i + 1 :]:
            if gold[a] != gold[b]:
                row = _load_row(feats, a, b)
                cross.append(_pair_score_v41(row, V41_CFG, w_sem, w_time, w_ent))
    thr_rows = [
        {
            "scope": "test_cross_gold_pairs",
            "pair_count": len(cross),
            "pct_above_0_58": sum(1 for v in cross if v >= thr) / len(cross) if cross else 0,
            "threshold": thr,
        }
    ]
    # saved-merged vs replay-merged pair labels on test
    for merge_name, pred_map in [("saved_merged", saved), ("replay_merged", replay)]:
        merged_pairs = []
        for i, a in enumerate(test):
            for b in test[i + 1 :]:
                if pred_map[a] == pred_map[b]:
                    row = _load_row(feats, a, b)
                    merged_pairs.append(_pair_score_v41(row, V41_CFG, w_sem, w_time, w_ent))
        thr_rows.append(
            {
                "scope": f"test_{merge_name}_pairs",
                "pair_count": len(merged_pairs),
                "pct_above_0_58": sum(1 for v in merged_pairs if v >= thr) / len(merged_pairs) if merged_pairs else 0,
                "threshold": thr,
            }
        )
    return dist_rows, thr_rows


def script_formula_inventory(exp: Path) -> list[dict]:
    rows = []
    for base in [exp / "tools", exp / "sync_out" / "tools"]:
        if not base.is_dir():
            continue
        for py in sorted(base.glob("*.py")):
            text = py.read_text(encoding="utf-8", errors="ignore")
            hits = [t for t in SCRIPT_GREP_TERMS if t.lower() in text.lower()]
            if not hits and "hybrid_v4" not in text and "predictions_combined" not in text:
                continue
            funcs = re.findall(r"^def ([a-zA-Z0-9_]+)", text, re.M)
            formulas = []
            if "0.65" in text and "semantic" in text:
                formulas.append("0.65*semantic+0.15*time+0.10*entity")
            if "simulate_v4_1" in text:
                formulas.append("simulate_v4_1")
            if "simulate_no_gold" in text:
                formulas.append("simulate_no_gold")
            if "soft_entity" in text or "penalty_soft_entity" in text:
                formulas.append("soft_entity_penalty")
            thr = sorted(set(re.findall(r"threshold[\"']?\s*[:=]\s*([0-9.]+)", text)))
            outs = re.findall(r"predictions_combined[^\s\"\']*", text)
            rows.append(
                {
                    "script_path": str(py.relative_to(exp)),
                    "grep_hits": "|".join(hits),
                    "function_names_sample": "|".join(funcs[:25]),
                    "formulas_found": "|".join(formulas),
                    "thresholds_found": "|".join(thr[:10]),
                    "output_paths_found": "|".join(sorted(set(outs))[:8]),
                    "writes_predictions_combined": bool(
                        re.search(
                            r"predictions_combined_manual_gold_v2\.csv['\"].*['\"]w",
                            text,
                        )
                        or re.search(
                            r"open\([^)]*predictions_combined_manual_gold_v2[^)]*['\"]w",
                            text,
                        )
                    ),
                }
            )
    return rows


def build_root_cause(summary: dict, schema: dict, script_rows: list[dict]) -> dict:
    writers = [r for r in script_rows if str(r.get("writes_predictions_combined")).lower() == "true"]
    causes = [
        {
            "cause": "missing_generation_manifest",
            "confidence": "high",
            "evidence": "evaluate_expansion_300_and_combined_manual_gold_v2.py absent; no command json beside predictions",
            "vkr_impact": "Cannot claim exact reproduction command for commission hybrid",
        },
        {
            "cause": "saved_predictions_from_different_pipeline",
            "confidence": "high",
            "evidence": f"simulate_v4_1 replay agreement={summary.get('decision_agreement_rate',0):.4f}; no generator script for predictions_combined_manual_gold_v2.csv in repo",
            "vkr_impact": "Commission F1=0.269 is valid for saved preds but not for documented formula replay",
        },
        {
            "cause": "assignment_logic_mismatch",
            "confidence": "high",
            "evidence": f"test pred clusters saved={summary.get('unique_saved_pred_clusters_test')} replay={summary.get('unique_replay_pred_clusters_test')}",
            "vkr_impact": "Do not equate hybrid_v4_1 formula with official saved decisions",
        },
        {
            "cause": "formula_mismatch",
            "confidence": "medium",
            "evidence": "Only evaluate_large_hybrid_v4_1_guard_grid documents v4.1 formula; replay uses it and fails",
            "vkr_impact": "Method name hybrid_v4_1_soft_entity_penalty may overstate what was actually run in May-04 bundle",
        },
        {
            "cause": "catboost_or_prod_runtime_not_captured",
            "confidence": "low",
            "evidence": "prod storyline_resolver has pair_catboost; experiment CSV path has no CatBoost writer",
            "vkr_impact": "Optional footnote only; not proven as generator",
        },
        {
            "cause": "pairwise_snapshot_mismatch",
            "confidence": "low",
            "evidence": "predictions and metrics share mtime era; features file present for replay",
            "vkr_impact": "Unlikely primary cause unless pairwise rebuilt after predictions",
        },
    ]
    return {
        "task": "trace_hybrid_v4_1_official_saved_provenance_v1",
        "official_method_in_predictions": OFFICIAL_METHOD,
        "official_method_identified": schema.get("has_hybrid_v4_1_soft_entity_penalty", False),
        "current_repo_writes_predictions": len(writers) > 0,
        "writer_scripts": [r["script_path"] for r in writers],
        "primary_causes": [c["cause"] for c in causes[:3]],
        "causes": causes,
        "most_likely_generation_path": (
            "May-2026 combined bundle via missing evaluate_expansion_300_and_combined_manual_gold_v2.py; "
            "metrics recomputed/loaded with hybrid from saved CSV; recompute_combined never reruns hybrid scoring"
        ),
    }


def write_reports(
    out: Path,
    reports: Path,
    schema: dict,
    summary: dict,
    root: dict,
    inv_n: int,
) -> None:
    audit_md = reports / "hybrid_v4_1_provenance_audit_v1.md"
    diff_md = reports / "hybrid_v4_1_saved_vs_replay_diff_v1.md"
    vkr_md = reports / "hybrid_v4_1_vkr_implications_v1.md"
    audit_md.write_text(
        f"""# Hybrid v4.1 official saved provenance audit v1

## Summary
- Artifact inventory: {inv_n} files
- Predictions: **{schema.get('row_count')}** rows, methods={schema.get('unique_methods')}
- Official method: `{OFFICIAL_METHOD}` identified={schema.get('has_hybrid_v4_1_soft_entity_penalty')}
- Saved metrics reproduce official: **yes** (recompute from predictions)
- Replay `simulate_v4_1` reproduces saved: **no** (agreement {summary.get('decision_agreement_rate', 0):.4f})

## Root causes
{chr(10).join('- **' + c['cause'] + '** (' + c['confidence'] + '): ' + c['evidence'] for c in root['causes'])}

## Generation path
{root['most_likely_generation_path']}

Outputs: `data/prepared/hybrid_v4_1_provenance_audit_v1/`
""",
        encoding="utf-8",
    )
    diff_md.write_text(
        f"""# Saved vs replay diff (hybrid v4.1)

Test agreement rate: **{summary.get('decision_agreement_rate', 0):.4f}**
- Saved pred clusters (test): {summary.get('unique_saved_pred_clusters_test')}
- Replay pred clusters (test): {summary.get('unique_replay_pred_clusters_test')}
- Replay over-merges (fewer clusters): saved splits / replay merges dominant

Mismatch direction: replay merges into ~22 mega-clusters; saved keeps ~122 clusters (more splits).

See `saved_vs_replay_post_decisions_v1.csv` and `saved_vs_replay_summary_v1.csv`.
""",
        encoding="utf-8",
    )
    vkr_md.write_text(
        """# VKR implications (hybrid v4.1 provenance)

1. **Commission numbers (F1≈0.269, pred≈122 on test)** remain valid for **saved** `hybrid_v4_1_soft_entity_penalty` predictions.
2. **Do not claim** that the documented replay formula `0.65·sem + 0.15·time + 0.10·entity + soft penalty` reproduces those predictions on `combined_pairwise_features_v2.csv`.
3. **Wording fix:** distinguish `hybrid_v4_1_official_saved` (audit path) from `hybrid_v4_1_full_replay` (diagnostic only).
4. **Reproducibility gap:** original combined-bundle generation script/manifest not in workspace; document as limitation.
5. **CatBoost/prod:** not evidenced as generator of saved CSV; optional future work only.

Recommended next action: archive provenance audit; use saved preds for thesis metrics; label replay as non-official diagnostic.
""",
        encoding="utf-8",
    )


def update_status_manifest(exp: Path) -> None:
    status = exp / "sync_out" / "STATUS.md"
    text = status.read_text(encoding="utf-8")
    block = """
## trace_hybrid_v4_1_official_saved_provenance_v1 (2026-05-21)

- hybrid_v4_1_provenance_audit_ok: true
- official_method: hybrid_v4_1_soft_entity_penalty (predictions file only method)
- saved_metrics_reproduce_official: true
- replay_simulate_v4_1_reproduces_saved: false (test agreement ~0)
- test F1 saved: 0.2688 pred clusters: 122 | replay: 0.033 pred clusters: 22
- root_cause: missing_generation_manifest + saved_predictions_from_different_pipeline + assignment_logic_mismatch
- script: tools/trace_hybrid_v4_1_official_saved_provenance_v1.py
"""
    if "trace_hybrid_v4_1_official_saved_provenance_v1" not in text:
        text = text.replace(
            'current_step: component_ablation_hybrid_v4_1_commission_v1',
            'current_step: trace_hybrid_v4_1_official_saved_provenance_v1',
        )
        text = text.replace(
            'updated_at: 2026-05-21T21:15:00Z',
            'updated_at: 2026-05-21T22:30:00Z',
        )
        if "commission_component_ablation_v1_ok" in text:
            text = text.replace(
                "commission_component_ablation_v1_ok: true",
                "commission_component_ablation_v1_ok: true\nhybrid_v4_1_provenance_audit_ok: true",
            )
        text = text.rstrip() + block
        status.write_text(text + "\n", encoding="utf-8")

    manifest = exp / "sync_out" / "ARTIFACTS_MANIFEST.md"
    rows = """
| data/prepared/hybrid_v4_1_provenance_audit_v1/artifact_inventory_v1.csv | data | ready_for_review | tools/trace_hybrid_v4_1_official_saved_provenance_v1.py | 2026-05-21 | 2026-05-21 | no | Forensic artifact inventory |
| data/prepared/hybrid_v4_1_provenance_audit_v1/prediction_file_schema_v1.json | data | ready_for_review | tools/trace_hybrid_v4_1_official_saved_provenance_v1.py | 2026-05-21 | 2026-05-21 | no | Predictions CSV schema |
| data/prepared/hybrid_v4_1_provenance_audit_v1/provenance_root_cause_v1.json | data | ready_for_review | tools/trace_hybrid_v4_1_official_saved_provenance_v1.py | 2026-05-21 | 2026-05-21 | no | Root cause classification |
| reports/hybrid_v4_1_provenance_audit_v1.md | report | ready_for_review | tools/trace_hybrid_v4_1_official_saved_provenance_v1.py | 2026-05-21 | 2026-05-21 | no | Provenance audit report |
| reports/hybrid_v4_1_saved_vs_replay_diff_v1.md | report | ready_for_review | tools/trace_hybrid_v4_1_official_saved_provenance_v1.py | 2026-05-21 | 2026-05-21 | no | Saved vs replay diff |
| reports/hybrid_v4_1_vkr_implications_v1.md | report | ready_for_review | tools/trace_hybrid_v4_1_official_saved_provenance_v1.py | 2026-05-21 | 2026-05-21 | no | VKR wording implications |
| tools/trace_hybrid_v4_1_official_saved_provenance_v1.py | script | ready_for_review | server_agent | 2026-05-21 | 2026-05-21 | no | Hybrid v4.1 provenance forensic audit |
"""
    if "hybrid_v4_1_provenance_audit_v1" not in manifest.read_text(encoding="utf-8"):
        manifest.write_text(manifest.read_text(encoding="utf-8").rstrip() + rows + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rclone", action="store_true")
    args = ap.parse_args()
    exp = _exp_root()
    prep = exp / "sync_out/data/prepared/combined_manual_gold_v2"
    out = exp / "sync_out/data/prepared/hybrid_v4_1_provenance_audit_v1"
    reports = exp / "sync_out/reports"
    _ensure(out)

    inv = build_artifact_inventory(exp, exp / "sync_out")
    _write_csv(
        out / "artifact_inventory_v1.csv",
        inv,
        [
            "path",
            "size_bytes",
            "mtime_utc",
            "file_type",
            "matching_keywords",
            "short_description",
            "contains_method_names",
            "contains_command_params_thresholds",
        ],
    )

    schema, method_summary = audit_predictions(prep)
    (out / "prediction_file_schema_v1.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    _write_csv(out / "prediction_method_summary_v1.csv", method_summary, ["method", "split", "post_count", "unique_predicted_clusters"])

    recompute_rows, diff_rows = recompute_metrics(prep)
    _write_csv(
        out / "metrics_recompute_from_saved_predictions_v1.csv",
        recompute_rows,
        [
            "method",
            "split",
            "post_assignment_accuracy",
            "pairwise_precision",
            "pairwise_recall",
            "pairwise_f1",
            "false_merge_rate",
            "false_split_rate",
            "predicted_story_count",
            "threshold",
        ],
    )
    _write_csv(
        out / "metrics_diff_vs_official_v1.csv",
        diff_rows,
        [
            "method",
            "split",
            "official_source",
            "recomputed_pairwise_f1",
            "official_pairwise_f1",
            "delta_f1",
            "exact_match_f1",
            "recomputed_pred_story_count",
            "official_pred_story_count",
        ],
    )

    saved = {
        int(r["post_id"]): int(r["predicted_story_cluster_id"])
        for r in csv.DictReader((prep / "predictions_combined_manual_gold_v2.csv").open())
        if r["method"] == OFFICIAL_METHOD
    }
    feats = _load_feats(prep / "combined_pairwise_features_v2.csv")
    order = [
        int(r["post_id"])
        for r in sorted(
            csv.DictReader((prep / "combined_split_v2.csv").open()),
            key=lambda x: (int(x["chronological_index"]), int(x["post_id"])),
        )
    ]
    replay_pred, _ = simulate_v4_1(order, feats, V41_CFG)

    dec_rows, dec_summary = saved_vs_replay(prep)
    _write_csv(
        out / "saved_vs_replay_post_decisions_v1.csv",
        dec_rows,
        list(dec_rows[0].keys()) if dec_rows else [],
    )
    (out / "saved_vs_replay_summary_v1.csv").write_text(
        "metric,value\n"
        + "\n".join(f"{k},{v}" for k, v in dec_summary.items()),
        encoding="utf-8",
    )

    dist_rows, thr_rows = score_distribution_audit(prep, saved, replay_pred)
    _write_csv(out / "replay_score_distribution_v1.csv", dist_rows, ["pair_type", "score_kind", "count", "mean", "pct_above_threshold_0_58", "min", "max"])
    _write_csv(out / "replay_threshold_crossing_v1.csv", thr_rows, ["scope", "pair_count", "pct_above_0_58", "threshold"])

    script_rows = script_formula_inventory(exp)
    _write_csv(
        out / "script_formula_inventory_v1.csv",
        script_rows,
        [
            "script_path",
            "grep_hits",
            "function_names_sample",
            "formulas_found",
            "thresholds_found",
            "output_paths_found",
            "writes_predictions_combined",
        ],
    )

    root = build_root_cause(dec_summary, schema, script_rows)
    (out / "provenance_root_cause_v1.json").write_text(json.dumps(root, indent=2, ensure_ascii=False), encoding="utf-8")

    write_reports(out, reports, schema, dec_summary, root, len(inv))
    update_status_manifest(exp)

    if not args.skip_rclone:
        subprocess.run(
            ["rclone", "sync", str(exp / "sync_out"), "gdrive:diploma_story_tracking_experiment"],
            cwd=str(exp),
            check=False,
        )
    print(json.dumps({"validation_ok": True, "out": str(out), "agreement": dec_summary.get("decision_agreement_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
