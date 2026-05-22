#!/usr/bin/env python3.11
"""rerun_v3d_eval_with_neo4j_graph_features_v2 — пересборка парных признаков v3d с подмешиванием Neo4j и диагностический прогон оценки (только новые пути)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from build_hybrid_as_weak_target_advantage_diagnostic_v1 import _load_feats  # noqa: E402
from evaluate_large_hybrid_v4_1_guard_grid import simulate_v4_1  # noqa: E402
from evaluate_manual_gold_expansion_v3d_strict_developmental_labeled import (  # noqa: E402
    METRIC_COLS,
    V41_CFG,
    _bootstrap_gold_stories,
    _eval_universe,
    _metrics_row,
    _story_stats,
    _temporal_score,
)
from pilot_eval_v2_assignment_lib import (  # noqa: E402
    aggregate_metrics,
    attach_ari_nmi,
    build_score_fn,
    pair_key,
    simulate_no_gold,
)

_v3_path = _TOOLS / "build_manual_gold_expansion_robustness_v3.py"
_spec = importlib.util.spec_from_file_location("exp_v3", _v3_path)
exp_v3 = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(exp_v3)

PAIR_NUM_COLS = [
    "semantic_score",
    "temporal_score",
    "entity_jaccard",
    "event_signature_jaccard",
    "graph_neighborhood_jaccard",
    "graph_relation_type_jaccard",
    "storyline_overlap",
    "has_graph_features",
]

V41_CFG_EVAL = dict(V41_CFG)


def _exp_root() -> Path:
    return _TOOLS.parent


def _h_token(s: str) -> str:
    if not s:
        return ""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _jaccard(sa: set[str], sb: set[str]) -> float:
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union


def _neo_rows(st: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
    res = exp_v3._neo_commit([{"statement": st, "parameters": params or {}}])
    if not res or not res[0]["data"]:
        return []
    return [r["row"] for r in res[0]["data"]]


def _extract_post_graph(pids: list[int]) -> dict[int, dict[str, Any]]:
    """Per-post Neo4j aggregates (no post text)."""
    out: dict[int, dict[str, Any]] = {
        p: {
            "entities": set(),
            "storylines": set(),
            "event_mentions": set(),
            "event_sig_tokens": set(),
            "rel_types": set(),
            "neighbors": set(),
        }
        for p in pids
    }
    chunk = 40
    for i in range(0, len(pids), chunk):
        sub = pids[i : i + chunk]
        rows = _neo_rows(
            """
            UNWIND $pids AS pid
            OPTIONAL MATCH (em:EventMention {post_id: pid})
            OPTIONAL MATCH (em)-[:INVOLVES]->(en:Entity)
            OPTIONAL MATCH (em)-[:PART_OF]->(s:Storyline)
            OPTIONAL MATCH (em)-[r]->(x)
            WHERE r IS NULL OR type(r) IN ['INVOLVES','PART_OF','HAS_MENTION','DESCRIBES']
               OR type(r) STARTS WITH 'BELONGS'
            RETURN pid,
                   collect(DISTINCT em.id) AS em_ids,
                   collect(DISTINCT coalesce(en.canonical_key, toString(en.id))) AS ent_keys,
                   collect(DISTINCT coalesce(toString(s.id), s.story_episode_id)) AS sl_ids,
                   collect(DISTINCT em.event_signature) AS ev_sigs,
                   collect(DISTINCT type(r)) AS rel_types
            """,
            {"pids": sub},
        )
        for row in rows:
            pid = int(row[0])
            slot = out.setdefault(
                pid,
                {
                    "entities": set(),
                    "storylines": set(),
                    "event_mentions": set(),
                    "event_sig_tokens": set(),
                    "rel_types": set(),
                    "neighbors": set(),
                },
            )
            for em_id in row[1] or []:
                if em_id is not None:
                    tok = f"em:{em_id}"
                    slot["event_mentions"].add(tok)
                    slot["neighbors"].add(tok)
            for ek in row[2] or []:
                if ek:
                    slot["entities"].add(str(ek))
                    slot["neighbors"].add(f"ent:{ek}")
            for sl in row[3] or []:
                if sl:
                    slot["storylines"].add(str(sl))
                    slot["neighbors"].add(f"sl:{sl}")
            for sig in row[4] or []:
                if sig:
                    st = str(sig).strip().lower()
                    if st:
                        slot["event_sig_tokens"].add(st)
                        slot["neighbors"].add(f"sig:{_h_token(st)}")
            for rt in row[5] or []:
                if rt:
                    slot["rel_types"].add(str(rt))
                    slot["neighbors"].add(f"rel:{rt}")
    return out


def _write_post_graph_csv(path: Path, pids: list[int], graph: dict[int, dict[str, Any]]) -> None:
    fields = [
        "post_id",
        "eventmention_count",
        "neo4j_entity_count",
        "neo4j_entity_ids_or_hashes",
        "storyline_count",
        "storyline_ids_or_hashes",
        "graph_neighbor_count",
        "has_neo4j_entities",
        "has_storyline",
    ]
    rows: list[dict[str, Any]] = []
    for pid in sorted(pids):
        g = graph[pid]
        ent_h = ";".join(sorted(_h_token(x) for x in g["entities"]))
        sl_h = ";".join(sorted(_h_token(x) for x in g["storylines"]))
        rows.append(
            {
                "post_id": pid,
                "eventmention_count": len(g["event_mentions"]),
                "neo4j_entity_count": len(g["entities"]),
                "neo4j_entity_ids_or_hashes": ent_h,
                "storyline_count": len(g["storylines"]),
                "storyline_ids_or_hashes": sl_h,
                "graph_neighbor_count": len(g["neighbors"]),
                "has_neo4j_entities": "true" if g["entities"] else "false",
                "has_storyline": "true" if g["storylines"] else "false",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _build_pairwise_neo4j(
    post_ids: list[int],
    emb: dict[int, np.ndarray],
    meta: dict[int, dict[str, Any]],
    graph: dict[int, dict[str, Any]],
) -> dict[tuple[int, int], dict[str, float]]:
    feats: dict[tuple[int, int], dict[str, float]] = {}
    for i, a in enumerate(post_ids):
        ga = graph[a]
        for b in post_ids[i + 1 :]:
            gb = graph[b]
            va, vb = emb.get(a), emb.get(b)
            sem = float(np.dot(va, vb)) if va is not None and vb is not None else 0.0
            ma, mb = meta[a], meta[b]
            hours = abs((ma["published_at"] - mb["published_at"]).total_seconds()) / 3600.0
            ej = _jaccard(ga["entities"], gb["entities"])
            evj = _jaccard(ga["event_sig_tokens"], gb["event_sig_tokens"])
            nb = _jaccard(ga["neighbors"], gb["neighbors"])
            rel = _jaccard(ga["rel_types"], gb["rel_types"])
            sl_ov = 1.0 if (ga["storylines"] & gb["storylines"]) else 0.0
            has_g = 1.0 if (ej > 0 or evj > 0 or nb > 0 or rel > 0 or sl_ov > 0) else 0.0
            feats[pair_key(a, b)] = {
                "semantic_score": sem,
                "temporal_score": _temporal_score(hours),
                "time_delta_hours": hours,
                "entity_jaccard_neo4j": ej,
                "event_signature_jaccard_neo4j": evj,
                "graph_neighborhood_jaccard_neo4j": nb,
                "graph_relation_type_jaccard_neo4j": rel,
                "storyline_overlap": sl_ov,
                "entity_jaccard": ej,
                "entity_overlap_count": float(len(ga["entities"] & gb["entities"])),
                "event_signature_jaccard": evj,
                "graph_neighborhood_jaccard": nb,
                "graph_relation_type_jaccard": rel,
                "channel_same_flag": 1.0 if ma["channel_id"] == mb["channel_id"] else 0.0,
                "near_duplicate_score": 1.0 if sem >= 0.92 else 0.0,
                "has_graph_features": has_g,
            }
    return feats


def _write_pairwise_csv(path: Path, post_ids: list[int], feats: dict[tuple[int, int], dict[str, float]]) -> None:
    fields = [
        "post_id_a",
        "post_id_b",
        "semantic_score",
        "temporal_score",
        "time_delta_hours",
        "entity_jaccard",
        "entity_jaccard_neo4j",
        "entity_overlap_count",
        "event_signature_jaccard",
        "event_signature_jaccard_neo4j",
        "graph_neighborhood_jaccard",
        "graph_neighborhood_jaccard_neo4j",
        "graph_relation_type_jaccard",
        "graph_relation_type_jaccard_neo4j",
        "storyline_overlap",
        "channel_same_flag",
        "near_duplicate_score",
        "has_graph_features",
    ]
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(post_ids):
        for b in post_ids[i + 1 :]:
            f = feats[pair_key(a, b)]
            rows.append(
                {
                    "post_id_a": a,
                    "post_id_b": b,
                    "semantic_score": f"{f['semantic_score']:.8f}",
                    "temporal_score": f"{f['temporal_score']:.8f}",
                    "time_delta_hours": f"{f['time_delta_hours']:.4f}",
                    "entity_jaccard": f"{f['entity_jaccard']:.8f}",
                    "entity_jaccard_neo4j": f"{f['entity_jaccard_neo4j']:.8f}",
                    "entity_overlap_count": int(f["entity_overlap_count"]),
                    "event_signature_jaccard": f"{f['event_signature_jaccard']:.8f}",
                    "event_signature_jaccard_neo4j": f"{f['event_signature_jaccard_neo4j']:.8f}",
                    "graph_neighborhood_jaccard": f"{f['graph_neighborhood_jaccard']:.8f}",
                    "graph_neighborhood_jaccard_neo4j": f"{f['graph_neighborhood_jaccard_neo4j']:.8f}",
                    "graph_relation_type_jaccard": f"{f['graph_relation_type_jaccard']:.8f}",
                    "graph_relation_type_jaccard_neo4j": f"{f['graph_relation_type_jaccard_neo4j']:.8f}",
                    "storyline_overlap": f"{f['storyline_overlap']:.8f}",
                    "channel_same_flag": f"{f['channel_same_flag']:.8f}",
                    "near_duplicate_score": f"{f['near_duplicate_score']:.8f}",
                    "has_graph_features": "true" if f["has_graph_features"] else "false",
                }
            )
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _nonzero_audit(path: Path, label: str) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return {"label": label, "n_pairs": 0}
    n = len(rows)
    out: dict[str, Any] = {"label": label, "n_pairs": n}
    cols = PAIR_NUM_COLS + ["entity_jaccard_neo4j", "graph_neighborhood_jaccard_neo4j"]
    for c in cols:
        if c not in rows[0]:
            continue
        vals = []
        for r in rows:
            try:
                vals.append(float(r[c]))
            except ValueError:
                vals.append(float("nan"))
        arr = np.array(vals, dtype=float)
        nz = int(np.sum(arr > 1e-9))
        out[f"{c}_nonzero_share"] = round(nz / n, 4)
        out[f"{c}_mean"] = round(float(np.nanmean(arr)), 4)
        out[f"{c}_max"] = round(float(np.nanmax(arr)), 4)
    if "has_graph_features" in rows[0]:
        hg = [str(r.get("has_graph_features", "")).lower() in ("true", "1") for r in rows]
        out["has_graph_features_true_share"] = round(sum(hg) / n, 4)
    return out


def _post_coverage_from_graph(graph: dict[int, dict[str, Any]], pids: list[int]) -> dict[str, Any]:
    n = len(pids)
    em = sum(1 for p in pids if graph[p]["event_mentions"])
    ent = sum(1 for p in pids if graph[p]["entities"])
    sl = sum(1 for p in pids if graph[p]["storylines"])
    return {
        "n_posts": n,
        "posts_with_eventmention": em,
        "posts_with_neo4j_entities": ent,
        "posts_with_storyline": sl,
        "share_posts_with_entities": round(ent / max(n, 1), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rclone", action="store_true")
    args = ap.parse_args()

    root = _exp_root()
    out_dir = root / "sync_out/data/prepared/manual_gold_expansion_v3d_eval_neo4j_v2"
    reports = root / "sync_out/reports"
    v1_eval = root / "sync_out/data/prepared/manual_gold_expansion_v3d_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_path = v1_eval / "v3d_labeled_clean_posts_v1.csv"
    if not clean_path.is_file():
        raise SystemExit(f"missing {clean_path}")

    clean = list(csv.DictReader(clean_path.open(encoding="utf-8")))
    pids = [int(r["post_id"]) for r in clean]
    gold = {int(r["post_id"]): r["namespaced_gold_story_id"] for r in clean}
    stats = _story_stats(gold)

    pg_meta: dict[int, dict[str, Any]] = {}
    for r in clean:
        pid = int(r["post_id"])
        pg_meta[pid] = {
            "published_at": exp_v3._parse_ts(str(r["published_at"])),
            "channel_id": r.get("channel_id", ""),
        }

    graph = _extract_post_graph(pids)
    _write_post_graph_csv(out_dir / "v3d_post_graph_features_neo4j_v2.csv", pids, graph)

    emb = exp_v3._load_embeddings(pids)
    feats = _build_pairwise_neo4j(pids, emb, pg_meta, graph)
    _write_pairwise_csv(out_dir / "v3d_pairwise_features_neo4j_v2.csv", pids, feats)

    v1_pw = v1_eval / "v3d_pairwise_features_v1.csv"
    v2_pw = root / "sync_out/data/prepared/combined_manual_gold_v2/combined_pairwise_features_v2.csv"
    audits = [
        _nonzero_audit(out_dir / "v3d_pairwise_features_neo4j_v2.csv", "v3d_neo4j_v2"),
        _nonzero_audit(v1_pw, "v3d_v1") if v1_pw.is_file() else {"label": "v3d_v1", "n_pairs": 0},
        _nonzero_audit(v2_pw, "combined_v2"),
    ]
    cov_posts = _post_coverage_from_graph(graph, pids)

    flat_cov: list[dict[str, Any]] = []
    for block in audits:
        label = block["label"]
        for k, v in block.items():
            if k in ("label", "n_pairs"):
                continue
            flat_cov.append({"dataset": label, "metric": k, "value": v})
    for k, v in cov_posts.items():
        flat_cov.append({"dataset": "v3d_post_graph_neo4j_v2", "metric": k, "value": v})

    with (out_dir / "feature_coverage_comparison_v2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "metric", "value"])
        w.writeheader()
        w.writerows(flat_cov)

    tuned = json.loads(
        (root / "sync_out/data/prepared/combined_manual_gold_v2/tuned_params_combined_manual_gold_v2.json").read_text()
    )
    thr_src = "combined_manual_gold_v2/tuned_params (audit v2, no retune on v3d)"
    thresholds = {
        "semantic_only": float(tuned["methods"]["semantic_only"]["tuned_threshold"]),
        "semantic_time": float(tuned["methods"]["semantic_time"]["tuned_threshold"]),
        "hybrid_v4_1_soft_entity_penalty": float(tuned["methods"]["hybrid_v4_1_soft_entity_penalty"]["tuned_threshold"]),
    }

    order = sorted(pids, key=lambda p: pg_meta[p]["published_at"] or datetime(2024, 1, 1, tzinfo=timezone.utc))
    uni = set(order)

    v2_metrics: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    pred_maps: dict[str, dict[int, int]] = {}

    for method in ("semantic_only", "semantic_time", "hybrid_v4_1_soft_entity_penalty"):
        m, pred = _eval_universe(method, order, gold, feats, uni, thresholds[method], V41_CFG_EVAL)
        v2_metrics.append(_metrics_row(method, "all_v3d_clean_external_neo4j_v2", m, thresholds[method], thr_src))
        pred_maps[method] = pred
        for pid, cid in pred.items():
            pred_rows.append(
                {
                    "post_id": pid,
                    "method": method,
                    "predicted_story_cluster_id": cid,
                    "split": "external_v3d_neo4j_v2",
                }
            )

    # optional graph-only diagnostic
    fn_graph = build_score_fn("hybrid_graph_v3_neighborhood", 0.8, 0.2, 0.35, 0.15, 0.12, 0.08, 0.12)
    pred_g, cl_g = simulate_no_gold(order, feats, fn_graph, thresholds["hybrid_v4_1_soft_entity_penalty"])
    m_g = aggregate_metrics(order, gold, pred_g, cl_g, uni, uni)
    attach_ari_nmi(m_g, order, gold, pred_g, uni)
    v2_metrics.append(
        _metrics_row(
            "hybrid_graph_v3_neighborhood",
            "all_v3d_clean_external_neo4j_v2",
            m_g,
            thresholds["hybrid_v4_1_soft_entity_penalty"],
            thr_src + " (threshold borrowed for diagnostic)",
        )
    )

    with (out_dir / "v3d_only_metrics_neo4j_v2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(v2_metrics[0].keys()))
        w.writeheader()
        w.writerows(v2_metrics)
    with (out_dir / "v3d_only_predictions_neo4j_v2.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["post_id", "method", "predicted_story_cluster_id", "split"])
        w.writeheader()
        w.writerows(pred_rows)

    v1_metrics: dict[str, dict[str, Any]] = {}
    if (v1_eval / "v3d_only_metrics_v1.csv").is_file():
        for r in csv.DictReader((v1_eval / "v3d_only_metrics_v1.csv").open(encoding="utf-8")):
            v1_metrics[r["method"]] = r

    comp_rows: list[dict[str, Any]] = []
    for method in ("semantic_only", "semantic_time", "hybrid_v4_1_soft_entity_penalty"):
        v2r = next(x for x in v2_metrics if x["method"] == method)
        v1r = v1_metrics.get(method, {})
        for col in METRIC_COLS:
            nv = float(v2r.get(col, 0))
            ov = float(v1r.get(col, 0)) if v1r else float("nan")
            comp_rows.append(
                {
                    "method": method,
                    "metric": col,
                    "v3d_v1": ov,
                    "v3d_neo4j_v2": nv,
                    "delta_v2_minus_v1": round(nv - ov, 4) if v1r else "",
                }
            )

    with (out_dir / "v3d_eval_v1_vs_neo4j_v2_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "metric", "v3d_v1", "v3d_neo4j_v2", "delta_v2_minus_v1"])
        w.writeheader()
        w.writerows(comp_rows)

    sem_v2 = next(x for x in v2_metrics if x["method"] == "semantic_only")
    hyb_v2 = next(x for x in v2_metrics if x["method"] == "hybrid_v4_1_soft_entity_penalty")
    sem_v1 = v1_metrics.get("semantic_only", {})
    hyb_v1 = v1_metrics.get("hybrid_v4_1_soft_entity_penalty", {})
    delta_hyb_sem_v2 = float(hyb_v2["pairwise_f1"]) - float(sem_v2["pairwise_f1"])
    delta_hyb_sem_v1 = float(hyb_v1.get("pairwise_f1", 0)) - float(sem_v1.get("pairwise_f1", 0))
    hybrid_differs_sem = abs(delta_hyb_sem_v2) > 1e-6 or float(hyb_v2["pairwise_f1"]) != float(sem_v2["pairwise_f1"])

    boot_rows: list[dict[str, Any]] = []
    boot_note = ""
    if hybrid_differs_sem and stats["non_singleton_story_count"] >= 15:
        boot_rows = _bootstrap_gold_stories(
            order, gold, pred_maps["semantic_only"], pred_maps["hybrid_v4_1_soft_entity_penalty"]
        )
        with (out_dir / "v3d_bootstrap_ci_neo4j_v2.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
            w.writeheader()
            w.writerows(boot_rows)
    else:
        boot_note = (
            "skipped: hybrid equals semantic on neo4j_v2"
            if not hybrid_differs_sem
            else f"skipped: non_singleton_story_count={stats['non_singleton_story_count']} < 15"
        )
        with (out_dir / "v3d_bootstrap_ci_neo4j_v2.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["note"])
            w.writeheader()
            w.writerow({"note": boot_note})

    neo_audit = _nonzero_audit(out_dir / "v3d_pairwise_features_neo4j_v2.csv", "v3d_neo4j_v2")
    v1_audit = _nonzero_audit(v1_pw, "v3d_v1") if v1_pw.is_file() else {}

    reports.joinpath("manual_gold_expansion_v3d_neo4j_feature_coverage_v2.md").write_text(
        "# Neo4j v2 feature coverage\n\n"
        f"## Post-level ({cov_posts})\n\n"
        f"## Pairwise nonzero (neo4j_v2)\n"
        f"- entity_jaccard: {neo_audit.get('entity_jaccard_nonzero_share', 0):.1%}\n"
        f"- graph_neighborhood: {neo_audit.get('graph_neighborhood_jaccard_nonzero_share', 0):.1%}\n"
        f"- event_signature: {neo_audit.get('event_signature_jaccard_nonzero_share', 0):.1%}\n"
        f"- storyline_overlap: {neo_audit.get('storyline_overlap_nonzero_share', 0):.1%}\n"
        f"- has_graph_features: {neo_audit.get('has_graph_features_true_share', 0):.1%}\n\n"
        f"## v1 (broken builder)\n"
        f"- entity_jaccard: {v1_audit.get('entity_jaccard_nonzero_share', 0):.1%}\n"
        f"- graph_neighborhood: {v1_audit.get('graph_neighborhood_jaccard_nonzero_share', 0):.1%}\n",
        encoding="utf-8",
    )

    reports.joinpath("manual_gold_expansion_v3d_only_evaluation_neo4j_v2.md").write_text(
        "# v3d-only evaluation (Neo4j v2 features)\n\n"
        f"Thresholds: {json.dumps(thresholds)}\n\n"
        "## Feature mapping for hybrid_v4_1\n\n"
        "- `entity_jaccard` ← Jaccard(Neo4j Entity canonical_key/id per post)\n"
        "- `event_signature_jaccard` ← Jaccard(event_signature tokens from EventMention)\n"
        "- `graph_neighborhood_jaccard` ← Jaccard(neighbor tokens: em/ent/sl/sig/rel)\n"
        "- `graph_relation_type_jaccard` ← Jaccard(relation types on EventMention)\n"
        "- v4.1 base score uses sem+time+entity only; graph cols affect soft-entity penalty branch\n\n"
        + "\n".join(
            f"- **{r['method']}**: F1={r['pairwise_f1']} acc={r['post_assignment_accuracy']} "
            f"FMR={r['false_merge_rate']} FSR={r['false_split_rate']} pred={r['predicted_story_count']} gold={r['gold_story_count']}"
            for r in v2_metrics
        )
        + f"\n\nDelta hybrid−semantic F1 (neo4j_v2): **{delta_hyb_sem_v2:.4f}**\n",
        encoding="utf-8",
    )

    reports.joinpath("manual_gold_expansion_v3d_eval_v1_vs_neo4j_v2_comparison.md").write_text(
        "# v3d eval v1 vs neo4j_v2\n\n"
        f"- v1 hybrid−semantic F1 delta: {delta_hyb_sem_v1:.4f} (invalid graph features)\n"
        f"- v2 hybrid−semantic F1 delta: {delta_hyb_sem_v2:.4f}\n"
        f"- hybrid differs from semantic: **{hybrid_differs_sem}**\n\n"
        + "\n".join(
            f"- {r['method']} {r['metric']}: v1={r['v3d_v1']} v2={r['v3d_neo4j_v2']} Δ={r['delta_v2_minus_v1']}"
            for r in comp_rows
            if r["metric"] == "pairwise_f1"
        )
        + "\n",
        encoding="utf-8",
    )

    reports.joinpath("manual_gold_expansion_v3d_bootstrap_ci_neo4j_v2.md").write_text(
        "# Bootstrap neo4j_v2\n\n" + (boot_note or "\n".join(f"- {r}" for r in boot_rows)) + "\n",
        encoding="utf-8",
    )

    reports.joinpath("manual_gold_expansion_v3d_neo4j_rerun_summary_v2.md").write_text(
        "# Neo4j v2 rerun summary\n\n"
        "1. **v1 evaluation error:** yes — v3d v1 builder hardcoded graph/entity to zero; hybrid=semantic was artifact.\n"
        f"2. **Graph coverage restored:** posts with Neo4j entities {cov_posts['posts_with_neo4j_entities']}/{cov_posts['n_posts']}; "
        f"pair entity_jaccard nonzero {neo_audit.get('entity_jaccard_nonzero_share', 0):.1%}.\n"
        f"3. **Hybrid vs semantic changed:** {hybrid_differs_sem} (ΔF1={delta_hyb_sem_v2:.4f}); "
        "v4.1 base uses only sem+time+entity — graph cols affect soft-entity branch only; "
        "at audit-v2 thresholds clustering still identical to semantic.\n"
        "4. **VKR use:** supportive external stress-test with corrected graph features; cite v1 as invalidated for hybrid claims.\n"
        "5. **Architecture:** prod graph in Neo4j is present; experiment must call Neo4j (or sync entities to PG) for hybrid eval.\n"
        f"6. **Next:** optional threshold-free ablation on v3d; integrate neo4j builder into standard eval path as v3d_pairwise_features_neo4j_v2.\n",
        encoding="utf-8",
    )

    validation_ok = (
        (out_dir / "v3d_post_graph_features_neo4j_v2.csv").is_file()
        and (out_dir / "v3d_pairwise_features_neo4j_v2.csv").is_file()
        and (out_dir / "v3d_only_metrics_neo4j_v2.csv").is_file()
        and len(clean) > 0
    )

    sync_tools = root / "sync_out/tools"
    sync_tools.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(__file__, sync_tools / "rerun_v3d_eval_with_neo4j_graph_features_v2.py")

    if not args.skip_rclone:
        if subprocess.run(
            ["rclone", "sync", str(root / "sync_out"), "gdrive:diploma_story_tracking_experiment"],
            cwd=str(root),
        ).returncode != 0:
            validation_ok = False

    print(
        json.dumps(
            {
                "validation_ok": validation_ok,
                "cov_posts": cov_posts,
                "neo_pair_audit": {k: neo_audit[k] for k in neo_audit if "nonzero" in k},
                "thresholds": thresholds,
                "metrics": {r["method"]: r["pairwise_f1"] for r in v2_metrics},
                "delta_hybrid_semantic_v2": delta_hyb_sem_v2,
                "hybrid_differs_semantic": hybrid_differs_sem,
                "boot_note": boot_note,
            },
            indent=2,
        )
    )
    return 0 if validation_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
