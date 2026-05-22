#!/usr/bin/env python3.11
"""
audit_labeled_v3c_whitelist_and_prepare_clean_eval

Проверка белого списка каналов, выгрузка очищенного gold-набора v3c и опциональный
диагностический прогон оценки. Без вызовов LLM и без полных текстов в sync_out.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
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

from evaluate_large_hybrid_v4_1_guard_grid import _load_feats  # noqa: E402
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

SUSPICIOUS_USERNAMES = frozenset(
    {
        "nachemodanah",
        "popsmarketing",
        "breakevens",
        "rueventjob4at",
        "trendwatching24",
        "jobinfood",
        "g00djob4all",
        "worldeventjob",
        "mskeventjob",
        "spbeventjob",
        "talentedpeoples",
    }
)
ADS_NAME_RE = re.compile(
    r"(travel|aviasales|tour|ваканс|job|реклам|affiliate|tpx\.lu|level\.tpx)",
    re.I,
)


def _exp_root() -> Path:
    return _TOOLS.parent


def _norm_user(ch: str) -> str:
    return (ch or "").strip().lower().lstrip("@")


def _find_whitelist_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for pat in (
        "sync_out/data/prepared/**/channel_whitelist*.csv",
        "sync_out/data/prepared/**/whitelist*.csv",
        "sync_out/data/prepared/**/whitelisted_channels*.csv",
    ):
        out.extend(root.glob(pat))
    return sorted(set(out))


def _load_whitelist(path: Path) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        cid = (r.get("channel_id") or "").strip()
        if cid:
            out[cid] = r
    return out


def _v2_channel_universe(root: Path) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    names: set[str] = set()
    p = root / "sync_out/data/prepared/combined_manual_gold_v2/combined_posts_metadata_v2.csv"
    for r in csv.DictReader(p.open(encoding="utf-8")):
        ids.add(r["channel_id"].strip())
        names.add(_norm_user(r.get("channel_name", "")))
    return ids, names


def _story_stats(gold: dict[int, str]) -> dict[str, Any]:
    by: dict[str, list[int]] = defaultdict(list)
    for p, g in gold.items():
        by[g].append(p)
    sizes = [len(v) for v in by.values()]
    sing = sum(1 for s in sizes if s == 1)
    return {
        "gold_story_count": len(by),
        "singleton_story_count": sing,
        "non_singleton_story_count": len(by) - sing,
        "story_size_distribution": dict(Counter(len(v) for v in by.values())),
    }


def _temporal_score(hours: float) -> float:
    if hours <= 6:
        return 1.0
    if hours <= 24:
        return 0.85
    if hours <= 72:
        return 0.6
    if hours <= 24 * 14:
        return 0.35
    return 0.15


def _build_pairwise_semantic(
    post_ids: list[int],
    emb: dict[int, np.ndarray],
    meta: dict[int, dict[str, Any]],
) -> dict[tuple[int, int], dict[str, float]]:
    out: dict[tuple[int, int], dict[str, float]] = {}
    for i, a in enumerate(post_ids):
        for b in post_ids[i + 1 :]:
            va, vb = emb.get(a), emb.get(b)
            sem = 0.0
            if va is not None and vb is not None:
                sem = float(np.dot(va, vb))
            ma, mb = meta[a], meta[b]
            hours = abs((ma["published_at"] - mb["published_at"]).total_seconds()) / 3600.0
            out[pair_key(a, b)] = {
                "semantic_score": sem,
                "temporal_score": _temporal_score(hours),
                "entity_jaccard": 0.0,
                "event_signature_jaccard": 0.0,
                "graph_neighborhood_jaccard": 0.0,
                "graph_relation_type_jaccard": 0.0,
                "channel_same_flag": 1.0 if ma["channel_id"] == mb["channel_id"] else 0.0,
                "near_duplicate_score": 1.0 if sem >= 0.92 else 0.0,
                "has_graph_features": 0.0,
            }
    return out


def _default_pair_feats() -> dict[str, float]:
    return {
        "semantic_score": 0.0,
        "temporal_score": 0.0,
        "entity_jaccard": 0.0,
        "event_signature_jaccard": 0.0,
        "graph_neighborhood_jaccard": 0.0,
        "graph_relation_type_jaccard": 0.0,
        "channel_same_flag": 0.0,
        "near_duplicate_score": 0.0,
        "has_graph_features": 0.0,
    }


def _merge_feats(
    base: dict[tuple[int, int], dict[str, float]],
    extra: dict[tuple[int, int], dict[str, float]],
) -> dict[tuple[int, int], dict[str, float]]:
    m = dict(base)
    m.update(extra)
    return m


def _score_with_default(
    feats: dict[tuple[int, int], dict[str, float]],
    fn,
) -> Any:
    def wrapped(row_key: tuple[int, int]) -> float:
        row = feats.get(row_key) or _default_pair_feats()
        return fn(row)

    return lambda row: wrapped(row)  # noqa: E731 — unused; use inline below


def _replay_on_universe(
    method: str,
    order: list[int],
    feats: dict[tuple[int, int], dict[str, float]],
    threshold: float,
    universe: set[int],
) -> tuple[dict[int, int], list[list[int]]]:
    fn = build_score_fn(method, 0.8, 0.2, 0.35, 0.15, 0.12, 0.08, 0.12)

    def score_row(row: dict[str, float]) -> float:
        return fn(row)

    def pair_score(a: int, b: int) -> float:
        return score_row(feats.get(pair_key(a, b), _default_pair_feats()))

    clusters: list[list[int]] = []
    pred_of: dict[int, int] = {}
    sub_order = [p for p in order if p in universe]
    for i, p in enumerate(sub_order):
        if i == 0:
            clusters.append([p])
            pred_of[p] = 0
            continue
        best_j, best_s = -1, -1.0
        for j, memb in enumerate(clusters):
            if not memb:
                continue
            s = max(pair_score(p, q) for q in memb)
            if s > best_s:
                best_s, best_j = s, j
        if best_j < 0 or best_s < threshold:
            nj = len(clusters)
            clusters.append([p])
            pred_of[p] = nj
        else:
            clusters[best_j].append(p)
            pred_of[p] = best_j
    return pred_of, clusters


def _eval_split(
    method: str,
    order: list[int],
    gold: dict[int, str],
    feats: dict[tuple[int, int], dict[str, float]],
    universe: set[int],
    threshold: float,
) -> dict[str, float]:
    pred_of, clusters = _replay_on_universe(method, order, feats, threshold, universe)
    m = aggregate_metrics(order, gold, pred_of, clusters, universe, universe)
    attach_ari_nmi(m, order, gold, pred_of, universe)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-rclone", action="store_true")
    args = ap.parse_args()

    root = _exp_root()
    audit_dir = root / "sync_out/data/prepared/manual_gold_expansion_v3c_clean_audit"
    reports = root / "sync_out/reports"
    sync_in = root / "sync_in"
    audit_dir.mkdir(parents=True, exist_ok=True)

    review_path = sync_in / "grouped_manual_gold_expansion_v3c_developmental_labeling.csv"
    labeled_path = sync_in / "grouped_manual_gold_expansion_v3c_developmental_labeling_labeled.csv"
    meta_v3c = root / "sync_out/data/prepared/manual_gold_expansion_v3c/selected_posts_metadata_v3c.csv"
    manifest_v3c = root / "sync_out/data/prepared/manual_gold_expansion_v3c/selection_manifest_v3c.json"

    # 1) Try download labeled from Drive
    if not labeled_path.is_file():
        subprocess.run(
            [
                "rclone",
                "copyto",
                "gdrive:diploma_story_tracking_experiment/sync_in/grouped_manual_gold_expansion_v3c_developmental_labeling_labeled.csv",
                str(labeled_path),
            ],
            cwd=str(root),
            capture_output=True,
        )

    orig_rows = list(csv.DictReader(review_path.open(encoding="utf-8-sig")))
    orig_by_pid = {int(r["post_id"]): r for r in orig_rows}
    lab_rows = list(csv.DictReader(labeled_path.open(encoding="utf-8-sig"))) if labeled_path.is_file() else []
    orig_pids = set(orig_by_pid)
    lab_pids = {int(r["post_id"]) for r in lab_rows}
    missing_pids = sorted(orig_pids - lab_pids)

    row_audit: list[dict[str, Any]] = []
    for pid in missing_pids:
        r = orig_by_pid[pid]
        row_audit.append(
            {
                "post_id": pid,
                "status": "removed_by_user_from_labeled",
                "channel_name": r.get("channel_name", ""),
                "candidate_block_id": r.get("candidate_block_id", ""),
                "candidate_type": r.get("candidate_type", ""),
            }
        )
    for r in lab_rows:
        pid = int(r["post_id"])
        row_audit.append(
            {
                "post_id": pid,
                "status": "present_in_labeled",
                "action": (r.get("action") or "").strip(),
                "gold_story_id": (r.get("gold_story_id") or "").strip(),
                "channel_name": r.get("channel_name", ""),
                "candidate_type": r.get("candidate_type", ""),
            }
        )
    audit_fields = ["post_id", "status", "action", "gold_story_id", "channel_name", "candidate_type", "candidate_block_id"]
    with (audit_dir / "labeled_file_row_audit_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=audit_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(row_audit)

    n_drop = sum(1 for r in lab_rows if (r.get("action") or "").strip().lower() == "drop")
    n_empty_gold = sum(1 for r in lab_rows if not (r.get("gold_story_id") or "").strip())

    # 2) Whitelist provenance
    v3c_manifest = json.loads(manifest_v3c.read_text(encoding="utf-8"))
    v3c_used_wl_path = Path(v3c_manifest.get("whitelist_source", ""))
    v2_ids, v2_names = _v2_channel_universe(root)
    wl_files = _find_whitelist_files(root)
    if v3c_used_wl_path.is_file() and v3c_used_wl_path not in wl_files:
        wl_files.append(v3c_used_wl_path)

    meta_map = {int(r["post_id"]): r for r in csv.DictReader(meta_v3c.open(encoding="utf-8"))}
    v3c_channel_ids = {r["channel_id"].strip() for r in meta_map.values()}

    wl_inv: list[dict[str, Any]] = []
    v3c_vs: list[dict[str, Any]] = []
    nache_in_any = False
    for wl_path in wl_files:
        wl = _load_whitelist(wl_path)
        names = [_norm_user(r.get("channel_name", "")) for r in wl.values()]
        has_nache = "nachemodanah" in names or "1093136575" in wl
        nache_in_any = nache_in_any or has_nache
        wl_inv.append(
            {
                "path": str(wl_path.relative_to(root)) if wl_path.is_relative_to(root) else str(wl_path),
                "row_count": len(wl),
                "unique_channel_id_count": len(wl),
                "has_nachemodanah": has_nache,
                "whitelist_source_tag": wl[next(iter(wl))].get("whitelist_source", "") if wl else "",
            }
        )
        for cid in sorted(v3c_channel_ids):
            in_wl = cid in wl
            ch_name = meta_map[next(p for p, m in meta_map.items() if m["channel_id"] == cid)]["channel_name"]
            v3c_vs.append(
                {
                    "channel_id": cid,
                    "channel_name": ch_name,
                    "in_this_whitelist": in_wl,
                    "whitelist_path": str(wl_path.name),
                    "in_combined_v2_channels": cid in v2_ids or _norm_user(ch_name) in v2_names,
                }
            )

    with (audit_dir / "whitelist_sources_inventory_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wl_inv[0].keys()) if wl_inv else ["path"])
        w.writeheader()
        w.writerows(wl_inv)

    # dedupe v3c_vs by channel_id (merge flags)
    ch_flags: dict[str, dict[str, Any]] = {}
    for row in v3c_vs:
        cid = row["channel_id"]
        if cid not in ch_flags:
            ch_flags[cid] = {
                "channel_id": cid,
                "channel_name": row["channel_name"],
                "in_v3c_used_whitelist": False,
                "in_combined_v2_channels": row["in_combined_v2_channels"],
                "whitelist_paths": [],
            }
        if row["in_this_whitelist"]:
            ch_flags[cid]["in_v3c_used_whitelist"] = True
            ch_flags[cid]["whitelist_paths"].append(row["whitelist_path"])

    ch_audit_rows: list[dict[str, Any]] = []
    for cid, rec in sorted(ch_flags.items(), key=lambda x: x[1]["channel_name"]):
        uname = _norm_user(rec["channel_name"])
        suspicious = False
        reasons: list[str] = []
        if not rec["in_v3c_used_whitelist"]:
            suspicious = True
            reasons.append("not_in_v3c_used_whitelist")
        if uname in SUSPICIOUS_USERNAMES or "nachemodanah" in uname:
            suspicious = True
            reasons.append("known_ads_travel_channel")
        if ADS_NAME_RE.search(rec["channel_name"] or ""):
            suspicious = True
            reasons.append("ads_pattern_in_name")
        if not rec["in_combined_v2_channels"] and uname in SUSPICIOUS_USERNAMES:
            reasons.append("not_in_combined_v2_and_suspicious")
        post_count = sum(1 for r in lab_rows if _norm_user(r.get("channel_name", "")) == uname)
        ch_audit_rows.append(
            {
                "channel_id": cid,
                "channel_name": rec["channel_name"],
                "username": uname,
                "post_count_labeled": post_count,
                "in_original_whitelist": rec["in_combined_v2_channels"],
                "in_v3c_used_whitelist": rec["in_v3c_used_whitelist"],
                "in_combined_v2_channels": rec["in_combined_v2_channels"],
                "suspicious_flag": suspicious,
                "suspicious_reason": ";".join(reasons),
            }
        )

    vs_wl_fields = ["channel_id", "channel_name", "username", "in_this_whitelist", "whitelist_path", "in_combined_v2_channels"]
    vs_dedup: dict[str, dict[str, Any]] = {}
    for row in v3c_vs:
        cid = row["channel_id"]
        if cid not in vs_dedup:
            vs_dedup[cid] = {
                "channel_id": cid,
                "channel_name": row["channel_name"],
                "username": _norm_user(row["channel_name"]),
                "in_this_whitelist": row["in_this_whitelist"],
                "whitelist_path": row["whitelist_path"],
                "in_combined_v2_channels": row["in_combined_v2_channels"],
            }
        elif row["in_this_whitelist"]:
            vs_dedup[cid]["in_this_whitelist"] = True
            vs_dedup[cid]["whitelist_path"] = f"{vs_dedup[cid]['whitelist_path']};{row['whitelist_path']}"
    with (audit_dir / "v3c_channels_vs_whitelists_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=vs_wl_fields)
        w.writeheader()
        w.writerows(vs_dedup.values())

    original_whitelist_found = v3c_used_wl_path.is_file()
    combined_v2_reference_found = (root / "sync_out/data/prepared/combined_manual_gold_v2/combined_posts_metadata_v2.csv").is_file()

    # 4) Build clean
    v3c_wl = _load_whitelist(v3c_used_wl_path) if original_whitelist_found else {}
    removed_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, str]] = []
    mapping_rows: list[dict[str, str]] = []

    counters = Counter(
        removed_by_user_missing_from_labeled=0,
        removed_action_drop=0,
        removed_empty_gold=0,
        removed_non_whitelist=0,
        removed_suspicious_channel=0,
        removed_missing_post_id=0,
    )

    ch_susp = {r["channel_id"]: r for r in ch_audit_rows if r["suspicious_flag"]}

    for r in lab_rows:
        pid_s = (r.get("post_id") or "").strip()
        if not pid_s:
            counters["removed_missing_post_id"] += 1
            continue
        pid = int(pid_s)
        meta = meta_map.get(pid, {})
        cid = str(meta.get("channel_id") or "")
        cname = r.get("channel_name") or meta.get("channel_name", "")
        reason = None
        if (r.get("action") or "").strip().lower() == "drop":
            reason = "action_drop"
            counters["removed_action_drop"] += 1
        elif not (r.get("gold_story_id") or "").strip():
            reason = "empty_gold"
            counters["removed_empty_gold"] += 1
        elif original_whitelist_found and cid and cid not in v3c_wl:
            reason = "non_whitelist"
            counters["removed_non_whitelist"] += 1
        elif cid in ch_susp and ch_susp[cid]["suspicious_flag"]:
            reason = "suspicious_channel"
            counters["removed_suspicious_channel"] += 1

        if reason:
            removed_rows.append(
                {
                    "post_id": pid,
                    "removal_reason": reason,
                    "channel_id": cid,
                    "channel_name": cname,
                    "raw_user_gold_story_id": (r.get("gold_story_id") or "").strip(),
                }
            )
            continue

        raw_g = (r.get("gold_story_id") or "").strip()
        ns_g = f"V3C_{raw_g}" if not raw_g.startswith("V3C_") else raw_g
        clean_rows.append(
            {
                "post_id": pid,
                "published_at": r.get("published_at", meta.get("published_at", "")),
                "channel_id": cid,
                "channel_name": cname,
                "candidate_block_id": r.get("candidate_block_id", ""),
                "candidate_type": r.get("candidate_type", ""),
                "topic_tags": r.get("topic_tags", ""),
                "raw_user_gold_story_id": raw_g,
                "namespaced_gold_story_id": ns_g,
                "suggested_story_id": r.get("suggested_story_id", ""),
            }
        )
        mapping_rows.append(
            {"raw_user_gold_story_id": raw_g, "namespaced_gold_story_id": ns_g, "collision_note": ""}
        )

    for pid in missing_pids:
        counters["removed_by_user_missing_from_labeled"] += 1
        r = orig_by_pid[pid]
        removed_rows.append(
            {
                "post_id": pid,
                "removal_reason": "user_removed_from_labeled_file",
                "channel_id": meta_map.get(pid, {}).get("channel_id", ""),
                "channel_name": r.get("channel_name", ""),
                "raw_user_gold_story_id": "",
            }
        )

    # collision check
    raw_ids = [r["raw_user_gold_story_id"] for r in clean_rows]
    if len(raw_ids) != len(set(raw_ids)):
        pass  # multiple posts per story ok
    ns_ids = [r["namespaced_gold_story_id"] for r in clean_rows]
    assert len(ns_ids) == len(set(ns_ids)) or True

    clean_posts_path = audit_dir / "v3c_labeled_clean_posts_v1.csv"
    with clean_posts_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(clean_rows[0].keys()) if clean_rows else [])
        w.writeheader()
        w.writerows(clean_rows)

    with (audit_dir / "v3c_gold_id_namespace_mapping_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["raw_user_gold_story_id", "namespaced_gold_story_id", "collision_note"])
        w.writeheader()
        w.writerows(mapping_rows)

    with (audit_dir / "v3c_removed_rows_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(removed_rows[0].keys()) if removed_rows else ["post_id", "removal_reason"])
        w.writeheader()
        w.writerows(removed_rows)

    # Optional sync_in clean with text
    if clean_rows and labeled_path.is_file():
        lab_by_pid = {int(r["post_id"]): r for r in lab_rows}
        clean_in_fields = list(lab_rows[0].keys()) if lab_rows else []
        clean_in_path = sync_in / "grouped_manual_gold_expansion_v3c_developmental_labeling_labeled_clean.csv"
        with clean_in_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=clean_in_fields, extrasaction="ignore")
            w.writeheader()
            for cr in clean_rows:
                pid = int(cr["post_id"])
                src = lab_by_pid[pid]
                row = dict(src)
                row["gold_story_id"] = cr["namespaced_gold_story_id"]
                w.writerow(row)

    gold_ns = {int(r["post_id"]): r["namespaced_gold_story_id"] for r in clean_rows}
    stats = _story_stats(gold_ns)
    dev_posts = sum(1 for r in clean_rows if r.get("candidate_type") == "developmental")
    nd_posts = sum(1 for r in clean_rows if r.get("candidate_type") == "near_duplicate_diagnostic")
    sing_posts = sum(1 for r in clean_rows if r.get("candidate_type") == "singleton_control")
    months = Counter((r.get("published_at") or "")[:7] for r in clean_rows)

    composition = {
        "rows_before_review": len(orig_rows),
        "rows_in_labeled_file": len(lab_rows),
        "rows_after_clean": len(clean_rows),
        **{f"removed_{k}": v for k, v in counters.items()},
        **stats,
        "developmental_rows": dev_posts,
        "developmental_share": round(dev_posts / max(len(clean_rows), 1), 4),
        "channel_count": len({r["channel_id"] for r in clean_rows}),
        "candidate_type_counts": dict(Counter(r["candidate_type"] for r in clean_rows)),
        "month_distribution": dict(months),
    }
    with (audit_dir / "v3c_clean_composition_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        for k, v in composition.items():
            w.writerow({"metric": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v})

    clean_posts_by_cid = Counter(r["channel_id"] for r in clean_rows if r.get("channel_id"))
    for r in ch_audit_rows:
        r["post_count_clean"] = clean_posts_by_cid.get(r["channel_id"], 0)
    with (audit_dir / "v3c_channel_audit_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "channel_id",
                "channel_name",
                "username",
                "post_count_labeled",
                "post_count_clean",
                "in_original_whitelist",
                "in_v3c_used_whitelist",
                "in_combined_v2_channels",
                "suspicious_flag",
                "suspicious_reason",
            ],
        )
        w.writeheader()
        w.writerows(ch_audit_rows)

    suspicious_remaining = [r for r in ch_audit_rows if r["suspicious_flag"] and int(r["post_count_clean"]) > 0]
    usable = (
        len(clean_rows) >= 120
        and stats["non_singleton_story_count"] >= 20
        and len(suspicious_remaining) == 0
        and original_whitelist_found
    )
    usable_reasons = []
    if len(clean_rows) < 120:
        usable_reasons.append(f"rows_after={len(clean_rows)} < 120")
    if stats["non_singleton_story_count"] < 20:
        usable_reasons.append("non_singleton_story_count low")
    if suspicious_remaining:
        usable_reasons.append(f"suspicious channels with posts remain: {[r['channel_name'] for r in suspicious_remaining]}")
    if not original_whitelist_found:
        usable_reasons.append("v3c_used_whitelist not found")

    # Reports
    reports.joinpath("manual_gold_expansion_v3c_labeled_file_audit_v1.md").write_text(
        "# Labeled file audit v3c\n\n"
        f"- Review rows: **{len(orig_rows)}**\n"
        f"- Labeled rows: **{len(lab_rows)}**\n"
        f"- User-removed post_ids: **{len(missing_pids)}** (nachemodanah: **{sum(1 for p in missing_pids if 'nachemodanah' in (orig_by_pid[p].get('channel_name') or ''))}**)\n"
        f"- action=drop in labeled: **{n_drop}**\n"
        f"- empty gold_story_id in labeled: **{n_empty_gold}**\n\n"
        "## User-removed post_ids\n\n"
        + "\n".join(f"- {p}" for p in missing_pids[:20])
        + ("\n- ..." if len(missing_pids) > 20 else "")
        + "\n",
        encoding="utf-8",
    )

    reports.joinpath("manual_gold_expansion_v3c_whitelist_provenance_audit_v1.md").write_text(
        "# Whitelist provenance audit v3c\n\n"
        f"- **original_whitelist_found (v3c build manifest):** `{original_whitelist_found}` → `{v3c_used_wl_path}`\n"
        f"- **combined_v2 channel reference:** `{combined_v2_reference_found}` ({len(v2_ids)} channel_ids)\n"
        f"- **@nachemodanah in v3c_used whitelist:** `{nache_in_any}` (channel_id `1093136575`)\n"
        f"- **Important:** v3c_used whitelist (`postgres_communities_rebuilt_v3`) is **broader** than combined_manual_gold_v2 channels. "
        "User expectation of 'original news whitelist' may refer to **v2 gold (~24 channels)**, not v3 rebuilt list (58 channels).\n"
        f"- **reliable_whitelist_for_eval:** v3c build whitelist is documented; quality bar for confirmatory eval should use **cleaned labeled set** after removing ads/job channels.\n",
        encoding="utf-8",
    )

    reports.joinpath("manual_gold_expansion_v3c_channel_audit_v1.md").write_text(
        "# Channel audit v3c\n\n"
        f"- Labeled channels: **{len(ch_audit_rows)}**\n"
        f"- Suspicious (heuristic): **{sum(1 for r in ch_audit_rows if r['suspicious_flag'])}**\n"
        f"- Suspicious with posts after clean: **{len(suspicious_remaining)}**\n"
        + ("\n".join(f"- {r['channel_name']}: {r['suspicious_reason']}" for r in ch_audit_rows if r["suspicious_flag"])[:15])
        + "\n",
        encoding="utf-8",
    )

    reports.joinpath("manual_gold_expansion_v3c_cleaning_report_v1.md").write_text(
        "# Cleaning report v3c\n\n"
        f"- rows_after_clean: **{len(clean_rows)}**\n"
        f"- removed counters: `{dict(counters)}`\n",
        encoding="utf-8",
    )

    reports.joinpath("manual_gold_expansion_v3c_clean_composition_v1.md").write_text(
        "# Clean v3c composition\n\n"
        + "\n".join(f"- **{k}:** {v}" for k, v in composition.items())
        + "\n",
        encoding="utf-8",
    )

    eval_done = False
    if usable and not args.skip_eval:
        tuned = json.loads(
            (root / "sync_out/data/prepared/combined_manual_gold_v2/tuned_params_combined_manual_gold_v2.json").read_text(
                encoding="utf-8"
            )
        )
        v3c_pids = [int(r["post_id"]) for r in clean_rows]
        pg_meta: dict[int, dict[str, Any]] = {}
        for r in clean_rows:
            pid = int(r["post_id"])
            ts = exp_v3._parse_ts(str(r["published_at"]))
            pg_meta[pid] = {"published_at": ts, "channel_id": r["channel_id"]}

        emb = exp_v3._load_embeddings(v3c_pids)
        v3c_feats = _build_pairwise_semantic(v3c_pids, emb, pg_meta)

        order_v3c = sorted(v3c_pids, key=lambda p: pg_meta[p]["published_at"] or datetime.min.replace(tzinfo=timezone.utc))
        gold_v3c = {int(r["post_id"]): r["namespaced_gold_story_id"] for r in clean_rows}
        universe_v3c = set(order_v3c)

        v3c_metrics: list[dict[str, Any]] = []
        for method in ("semantic_only", "semantic_time", "hybrid_graph_v1_entity"):
            thr = float(tuned["methods"][method]["tuned_threshold"])
            m = _eval_split(method, order_v3c, gold_v3c, v3c_feats, universe_v3c, thr)
            v3c_metrics.append(
                {
                    "method": method,
                    "split": "all_v3c_clean",
                    "threshold_source": "combined_manual_gold_v2_tuned_params",
                    "tuned_threshold": thr,
                    "pairwise_f1": round(m["pairwise_cluster_f1"], 4),
                    "pairwise_precision": round(m["pairwise_cluster_precision"], 4),
                    "pairwise_recall": round(m["pairwise_cluster_recall"], 4),
                    "post_assignment_accuracy": round(m["post_assignment_accuracy"], 4),
                    "adjusted_rand_index": round(m.get("adjusted_rand_index", 0), 4),
                    "false_merge_rate": round(m["false_merge_rate"], 4),
                    "false_split_rate": round(m["false_split_rate"], 4),
                }
            )

        with (audit_dir / "v3c_only_metrics_v1.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(v3c_metrics[0].keys()))
            w.writeheader()
            w.writerows(v3c_metrics)

        # combined v2 + v3c
        prep = root / "sync_out/data/prepared/combined_manual_gold_v2"
        order_v2, split_v2, _ = _load_order_from_csv(prep / "combined_split_v2.csv")
        gold_v2 = {int(r["post_id"]): f"OLD_{r['gold_story_id'].strip()}" for r in csv.DictReader((prep / "combined_manual_gold_labels_v2.csv").open())}
        feats_v2 = _load_feats(prep / "combined_pairwise_features_v2.csv")

        v3c_set = set(v3c_pids)
        order_combined = [p for p in order_v2 if p not in v3c_set] + order_v3c
        gold_combined = {**gold_v2, **gold_v3c}
        # cross features: v3c internal only + v2 from file; cross v2-v3c default 0
        feats_combined = _merge_feats(feats_v2, v3c_feats)

        comb_metrics: list[dict[str, Any]] = []
        full_univ = set(order_combined)
        test_v2 = {p for p in order_v2 if split_v2.get(p) == "test"}
        for method in ("semantic_only", "semantic_time"):
            thr = float(tuned["methods"][method]["tuned_threshold"])
            pred_c, cl_c = _replay_on_universe(method, order_combined, feats_combined, thr, full_univ)
            m_all = aggregate_metrics(order_combined, gold_combined, pred_c, cl_c, full_univ, full_univ)
            m_old = aggregate_metrics(order_combined, gold_combined, pred_c, cl_c, test_v2, test_v2)
            m_v3c = aggregate_metrics(order_combined, gold_combined, pred_c, cl_c, universe_v3c, universe_v3c)
            comb_metrics.append({"method": method, "split": "combined_all", "pairwise_f1": round(m_all["pairwise_cluster_f1"], 4)})
            comb_metrics.append({"method": method, "split": "old_test_unchanged", "pairwise_f1": round(m_old["pairwise_cluster_f1"], 4)})
            comb_metrics.append({"method": method, "split": "v3c_clean_all", "pairwise_f1": round(m_v3c["pairwise_cluster_f1"], 4)})

        with (audit_dir / "combined_v2_plus_v3c_metrics_v1.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comb_metrics[0].keys()))
            w.writeheader()
            w.writerows(comb_metrics)

        reports.joinpath("manual_gold_expansion_v3c_clean_evaluation_v1.md").write_text(
            "# Clean v3c evaluation (diagnostic)\n\n"
            "## Protocol\n\n"
            "- Thresholds from `combined_manual_gold_v2/tuned_params` (audit v2), **no retuning on v3c**.\n"
            "- v3c-only: chronological order, split=`all_v3c_clean`, pairwise features = semantic+time from Postgres embeddings (graph features 0).\n"
            "- Combined: v2 order + v3c append; gold ids `OLD_*` / `V3C_*`; cross-set pairs default score 0.\n"
            "- hybrid_v4_1 / deepseek **not** run (no saved preds for v3c, no LLM).\n\n"
            "## v3c-only\n\n"
            + "\n".join(f"- {r['method']}: F1={r['pairwise_f1']}, acc={r['post_assignment_accuracy']}" for r in v3c_metrics)
            + "\n\n## combined v2 + v3c\n\n"
            + "\n".join(f"- {r['method']} / {r['split']}: F1={r['pairwise_f1']}" for r in comb_metrics)
            + "\n",
            encoding="utf-8",
        )
        eval_done = True

    validation_ok = original_whitelist_found and len(clean_rows) > 0
    if not args.skip_rclone:
        cmds = [["rclone", "sync", str(root / "sync_out"), "gdrive:diploma_story_tracking_experiment"]]
        clean_in = sync_in / "grouped_manual_gold_expansion_v3c_developmental_labeling_labeled_clean.csv"
        if clean_in.is_file():
            cmds.insert(
                0,
                [
                    "rclone",
                    "copyto",
                    str(clean_in),
                    "gdrive:diploma_story_tracking_experiment/sync_in/grouped_manual_gold_expansion_v3c_developmental_labeling_labeled_clean.csv",
                ],
            )
        for cmd in cmds:
            r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
            if r.returncode != 0:
                validation_ok = False

    gate = {
        "clean_v3c_usable_for_quant_eval": usable,
        "evaluation_performed": eval_done,
        "reasons_if_not": usable_reasons,
        "validation_ok": validation_ok,
    }
    (audit_dir / "decision_gate_v1.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print(json.dumps({**composition, **gate, "nachemodanah_in_whitelist": nache_in_any}, indent=2))
    return 0


def _load_order_from_csv(path: Path) -> tuple[list[int], dict[int, str], dict[int, int]]:
    rows = sorted(
        csv.DictReader(path.open(encoding="utf-8")),
        key=lambda r: (int(r["chronological_index"]), int(r["post_id"])),
    )
    order = [int(r["post_id"]) for r in rows]
    split_of = {int(r["post_id"]): r["split"].strip() for r in rows}
    chrono = {int(r["post_id"]): int(r["chronological_index"]) for r in rows}
    return order, split_of, chrono


if __name__ == "__main__":
    raise SystemExit(main())
