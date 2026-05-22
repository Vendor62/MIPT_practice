#!/usr/bin/env python3.11
"""
audit_graph_advantage_extension_preview_v1_labeled

Import manual labeling for graph-advantage preview, safety/overlap audit,
weak-block survival, expansion decision gate. No LLM, no fulltext in sync_out.
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

from evaluate_large_hybrid_v4_1_guard_grid import simulate_v4_1  # noqa: E402
from evaluate_manual_gold_expansion_v3d_strict_developmental_labeled import (  # noqa: E402
    V41_CFG,
    _build_pairwise_features,
    _load_entity_sets,
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

NAMESPACE = "GADV1"
FORBIDDEN_USERS = frozenset(
    {
        "nachemodanah",
        "trendwatching24",
        "rueventjob4at",
        "popsmarketing",
        "breakevens",
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

GATE_THRESHOLDS = {
    "clean_rows_min": 50,
    "non_singleton_gold_story_rate_min": 0.30,
    "blocks_with_any_multi_gold_rate_min": 0.40,
    "split_to_singletons_rate_max": 0.50,
}


def _exp_root() -> Path:
    return _TOOLS.parent


def _norm_user(ch: str) -> str:
    return (ch or "").strip().lower().lstrip("@")


def _size_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "3-4"
    return "5+"


def _load_exclude(root: Path) -> tuple[set[int], list[dict[str, str]]]:
    exclude: set[int] = set()
    summary: list[dict[str, str]] = []
    paths = [
        ("combined_v2", "sync_out/data/prepared/combined_manual_gold_v2/combined_manual_gold_labels_v2.csv"),
        ("v3", "sync_out/data/prepared/manual_gold_expansion_v3/selected_posts_metadata_v3.csv"),
        ("v3b", "sync_out/data/prepared/manual_gold_expansion_v3b/selected_posts_metadata_v3b.csv"),
        ("v3c", "sync_out/data/prepared/manual_gold_expansion_v3c/selected_posts_metadata_v3c.csv"),
        ("v3c_clean", "sync_out/data/prepared/manual_gold_expansion_v3c_clean_audit/v3c_labeled_clean_posts_v1.csv"),
        ("v3d", "sync_out/data/prepared/manual_gold_expansion_v3d/selected_posts_metadata_v3d.csv"),
        ("v3d_clean", "sync_out/data/prepared/manual_gold_expansion_v3d_eval/v3d_labeled_clean_posts_v1.csv"),
        ("same_protocol", "sync_out/data/prepared/same_protocol_extension_v1/selected_posts_metadata_v1.csv"),
        ("same_protocol_labeled_60", "sync_in/same_protocol_extension_v1_labeling_labeled_60.csv"),
    ]
    for tag, rel in paths:
        p = root / rel
        n = 0
        if p.is_file():
            ids = {int(r["post_id"]) for r in csv.DictReader(p.open(encoding="utf-8"))}
            n = len(ids)
            exclude |= ids
        summary.append({"dataset": tag, "path": rel, "present": str(p.is_file()), "post_ids": n})
    return exclude, summary


def _strict_whitelist_ids(root: Path) -> set[str]:
    p = root / "sync_out/data/prepared/graph_advantage_extension_preview_v1/channel_audit_v1.csv"
    ids: set[str] = set()
    for r in csv.DictReader(p.open(encoding="utf-8")):
        src = (r.get("whitelist_source") or r.get("in_strict_whitelist") or "").strip()
        if src == "combined_manual_gold_v2" or src.lower() in ("true", "1", "yes"):
            ids.add(r["channel_id"].strip())
    return ids


def _block_survival(
    weak_block_id: str,
    block_size_input: int,
    clean_golds: list[str],
) -> dict[str, Any]:
    clean = len(clean_golds)
    counts = Counter(clean_golds)
    n_stories = len(counts)
    largest = max(counts.values()) if counts else 0
    any_multi = largest >= 2
    purity = round(largest / clean, 4) if clean else 0.0
    fragmentation = round(n_stories / clean, 4) if clean else 0.0
    incomplete = clean < block_size_input

    if clean == 0 or clean / max(block_size_input, 1) < 0.5:
        status = "dropped_or_invalid"
    elif n_stories == 1 and largest >= 2:
        status = "full_survived"
    elif n_stories == clean and largest == 1 and clean > 0:
        status = "split_to_singletons"
    elif any_multi and n_stories > 1:
        status = "partial_survived"
    else:
        status = "mixed_other"

    return {
        "weak_block_id": weak_block_id,
        "block_size_input": block_size_input,
        "clean_rows_in_block": clean,
        "incomplete_labeling": incomplete,
        "number_of_gold_stories_inside_block": n_stories,
        "largest_gold_story_size_inside_block": largest,
        "survival_status": status,
        "block_purity": purity,
        "block_fragmentation": fragmentation,
        "has_any_multi_gold_story": any_multi,
    }


def _eval_method(
    method: str,
    order: list[int],
    gold: dict[int, str],
    feats: dict[tuple[int, int], dict[str, float]],
    threshold: float,
) -> dict[str, Any]:
    uni = set(order)
    if method == "hybrid_v4_1_soft_entity_penalty":
        cfg = dict(V41_CFG)
        cfg["threshold"] = threshold
        pred, clusters = simulate_v4_1(order, feats, cfg)
    else:
        fn = build_score_fn(method, 0.8, 0.2, 0.35, 0.15, 0.12, 0.08, 0.12)
        pred, clusters = simulate_no_gold(order, feats, fn, threshold)
    m = aggregate_metrics(order, gold, pred, clusters, uni, uni)
    attach_ari_nmi(m, order, gold, pred, uni)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rclone", action="store_true")
    ap.add_argument("--skip-diagnostic", action="store_true")
    args = ap.parse_args()

    root = _exp_root()
    labeled_path = root / "sync_in/graph_advantage_extension_preview_v1_labeling_labeled.csv"
    preview_dir = root / "sync_out/data/prepared/graph_advantage_extension_preview_v1"
    audit_dir = root / "sync_out/data/prepared/graph_advantage_extension_preview_v1_labeled_audit"
    reports = root / "sync_out/reports"
    audit_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    validation_issues: list[str] = []
    if not labeled_path.is_file():
        validation_issues.append("labeled_file_missing")
        labeled_path.write_text("post_id\n", encoding="utf-8")

    preview_posts = {
        int(r["post_id"]): r
        for r in csv.DictReader((preview_dir / "selected_posts_metadata_v1.csv").open(encoding="utf-8"))
    }
    blocks_meta = {
        r["weak_block_id"]: int(r["post_count"])
        for r in csv.DictReader((preview_dir / "selected_blocks_metadata_v1.csv").open(encoding="utf-8"))
    }

    lab_rows: list[dict[str, str]] = []
    if labeled_path.is_file():
        lab_rows = list(csv.DictReader(labeled_path.open(encoding="utf-8-sig")))

    dup_pids = [pid for pid, c in Counter(r.get("post_id", "").strip() for r in lab_rows).items() if pid and c > 1]
    empty_pid = sum(1 for r in lab_rows if not (r.get("post_id") or "").strip())
    extra_pids: list[int] = []
    missing_pids: list[int] = []
    for r in lab_rows:
        if not (r.get("post_id") or "").strip():
            continue
        pid = int(r["post_id"])
        if pid not in preview_posts:
            extra_pids.append(pid)
    missing_pids = sorted(set(preview_posts) - {int(r["post_id"]) for r in lab_rows if (r.get("post_id") or "").strip()})

    if dup_pids:
        validation_issues.append(f"duplicate_post_id:{dup_pids[:5]}")
    if empty_pid:
        validation_issues.append(f"empty_post_id_rows:{empty_pid}")
    if extra_pids:
        validation_issues.append(f"post_id_not_in_preview:{extra_pids[:5]}")
    if missing_pids:
        validation_issues.append(f"preview_posts_not_labeled:{len(missing_pids)}")

    n_drop = sum(1 for r in lab_rows if (r.get("action") or "").strip().lower() == "drop")
    n_empty_gold = sum(1 for r in lab_rows if not (r.get("gold_story_id") or "").strip())

    wl_ids = _strict_whitelist_ids(root)
    exclude_prev, excl_summary = _load_exclude(root)

    removed: list[dict[str, Any]] = []
    clean: list[dict[str, Any]] = []
    clean_counters = Counter(
        action_drop=0,
        empty_gold=0,
        non_whitelist=0,
        forbidden_channel=0,
    )

    for r in lab_rows:
        pid_s = (r.get("post_id") or "").strip()
        if not pid_s:
            continue
        pid = int(pid_s)
        meta = preview_posts.get(pid, {})
        cid = (r.get("channel_id") or meta.get("channel_id") or "").strip()
        user = _norm_user(r.get("channel_username") or r.get("channel_title") or meta.get("channel_name", ""))
        title = (r.get("channel_title") or meta.get("channel_name") or "").strip()

        reason = None
        if (r.get("action") or "").strip().lower() == "drop":
            reason = "action_drop"
            clean_counters["action_drop"] += 1
        elif not (r.get("gold_story_id") or "").strip():
            reason = "empty_gold"
            clean_counters["empty_gold"] += 1
        elif cid and wl_ids and cid not in wl_ids:
            reason = "non_whitelist"
            clean_counters["non_whitelist"] += 1
        elif user in FORBIDDEN_USERS or any(ADS_NAME_RE.search(x or "") for x in (user, title)):
            reason = "forbidden_channel"
            clean_counters["forbidden_channel"] += 1

        if reason:
            removed.append(
                {
                    "post_id": pid,
                    "removal_reason": reason,
                    "weak_block_id": r.get("weak_block_id", ""),
                    "channel_id": cid,
                    "channel_username": user,
                }
            )
            continue

        raw_g = (r.get("gold_story_id") or "").strip()
        if raw_g.startswith(f"{NAMESPACE}_"):
            ns_g = raw_g
            raw_out = raw_g[len(NAMESPACE) + 1 :]
        else:
            raw_out = raw_g
            ns_g = f"{NAMESPACE}_{raw_g}"

        clean.append(
            {
                "post_id": pid,
                "weak_block_id": r.get("weak_block_id", meta.get("weak_block_id", "")),
                "row_in_block": r.get("row_in_block", meta.get("row_in_block", "")),
                "block_size": r.get("block_size", meta.get("block_size", "")),
                "channel_id": cid,
                "channel_username": user,
                "channel_title": title,
                "published_at": r.get("published_at", meta.get("published_at", "")),
                "gold_story_id": raw_out,
                "namespaced_gold_story_id": ns_g,
                "action": (r.get("action") or "").strip(),
                "graph_entity_count": r.get("graph_entity_count", meta.get("graph_entity_count", "")),
                "graph_storyline_count": r.get("graph_storyline_count", meta.get("graph_storyline_count", "")),
                "semantic_summary_fields": r.get("semantic_summary_fields", ""),
            }
        )

    clean_pids = {int(r["post_id"]) for r in clean}
    overlap_rows: list[dict[str, Any]] = []
    overlap_count = 0
    for row in excl_summary:
        ds = row["dataset"]
        rel = row["path"]
        p = root / rel
        hits: list[int] = []
        if p.is_file():
            for r in csv.DictReader(p.open(encoding="utf-8")):
                pid = int(r["post_id"])
                if pid in clean_pids:
                    hits.append(pid)
        if hits:
            overlap_count += len(hits)
        overlap_rows.append(
            {
                "dataset": ds,
                "path": rel,
                "overlap_post_ids": ";".join(str(x) for x in sorted(hits)),
                "overlap_count": len(hits),
            }
        )

    # channel audit
    ch_posts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in clean:
        ch_posts[r["channel_id"]].append(r)

    ch_audit: list[dict[str, Any]] = []
    forbidden_present = False
    for cid, posts in sorted(ch_posts.items(), key=lambda x: x[0]):
        user = posts[0].get("channel_username", "")
        title = posts[0].get("channel_title", "")
        suspicious = user in FORBIDDEN_USERS or bool(ADS_NAME_RE.search(f"{user} {title}"))
        if suspicious:
            forbidden_present = True
        ch_audit.append(
            {
                "channel_id": cid,
                "channel_username": user,
                "channel_title": title,
                "post_count_clean": len(posts),
                "in_strict_whitelist": cid in wl_ids if wl_ids else "",
                "forbidden_flag": suspicious,
            }
        )

    whitelist_only = all(r["channel_id"] in wl_ids for r in clean if r["channel_id"]) if wl_ids else False
    if clean_counters["non_whitelist"]:
        whitelist_only = False

    gold_map = {int(r["post_id"]): r["namespaced_gold_story_id"] for r in clean}
    by_story: dict[str, list[int]] = defaultdict(list)
    for p, g in gold_map.items():
        by_story[g].append(p)
    stats = _story_stats(gold_map)
    story_buckets = Counter(_size_bucket(len(v)) for v in by_story.values())
    post_by_bucket: Counter[str] = Counter()
    for pids in by_story.values():
        post_by_bucket[_size_bucket(len(pids))] += len(pids)

    # weak block survival (all preview blocks)
    block_golds: dict[str, list[str]] = defaultdict(list)
    for r in clean:
        block_golds[r["weak_block_id"]].append(r["namespaced_gold_story_id"])

    survival_rows: list[dict[str, Any]] = []
    status_ctr = Counter()
    multi_block = 0
    posts_multi_blocks = 0
    for bid, bs in sorted(blocks_meta.items()):
        row = _block_survival(bid, bs, block_golds.get(bid, []))
        survival_rows.append(row)
        status_ctr[row["survival_status"]] += 1
        if row["has_any_multi_gold_story"]:
            multi_block += 1
            posts_multi_blocks += row["clean_rows_in_block"]

    n_blocks = len(blocks_meta)
    survival_summary = {
        "weak_blocks_total": n_blocks,
        "full_survived_count": status_ctr["full_survived"],
        "full_survived_rate": round(status_ctr["full_survived"] / n_blocks, 4),
        "partial_survived_count": status_ctr["partial_survived"],
        "partial_survived_rate": round(status_ctr["partial_survived"] / n_blocks, 4),
        "split_to_singletons_count": status_ctr["split_to_singletons"],
        "split_to_singletons_rate": round(status_ctr["split_to_singletons"] / n_blocks, 4),
        "dropped_or_invalid_count": status_ctr["dropped_or_invalid"],
        "dropped_or_invalid_rate": round(status_ctr["dropped_or_invalid"] / n_blocks, 4),
        "mixed_other_count": status_ctr["mixed_other"],
        "blocks_with_any_multi_gold_count": multi_block,
        "blocks_with_any_multi_gold_rate": round(multi_block / n_blocks, 4),
        "posts_in_blocks_with_any_multi_gold": posts_multi_blocks,
    }

    non_sing_rate = stats["non_singleton_story_count"] / max(stats["gold_story_count"], 1)

    gate_checks = {
        "clean_rows_ge_50": len(clean) >= GATE_THRESHOLDS["clean_rows_min"],
        "overlap_previous_eq_0": overlap_count == 0,
        "whitelist_only": whitelist_only,
        "forbidden_channels_present_false": not forbidden_present,
        "non_singleton_gold_story_rate_ge_0_30": non_sing_rate >= GATE_THRESHOLDS["non_singleton_gold_story_rate_min"],
        "blocks_with_any_multi_gold_rate_ge_0_40": survival_summary["blocks_with_any_multi_gold_rate"]
        >= GATE_THRESHOLDS["blocks_with_any_multi_gold_rate_min"],
        "split_to_singletons_rate_le_0_50": survival_summary["split_to_singletons_rate"]
        <= GATE_THRESHOLDS["split_to_singletons_rate_max"],
    }
    expand = all(gate_checks.values())
    failed = [k for k, v in gate_checks.items() if not v]

    validation_ok = (
        labeled_path.is_file()
        and not dup_pids
        and not empty_pid
        and not extra_pids
        and len(lab_rows) > 0
    )

    # Write CSVs
    clean_fields = list(clean[0].keys()) if clean else ["post_id"]
    with (audit_dir / "clean_gold_labels_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=clean_fields)
        w.writeheader()
        w.writerows(clean)

    story_summary: list[dict[str, Any]] = []
    for gid, pids in sorted(by_story.items(), key=lambda x: (-len(x[1]), x[0])):
        story_summary.append(
            {
                "namespaced_gold_story_id": gid,
                "post_count": len(pids),
                "story_size_bucket": _size_bucket(len(pids)),
                "post_ids": ";".join(str(p) for p in sorted(pids)),
            }
        )
    with (audit_dir / "gold_story_summary_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["namespaced_gold_story_id", "post_count", "story_size_bucket", "post_ids"],
        )
        w.writeheader()
        w.writerows(story_summary)

    surv_fields = list(survival_rows[0].keys()) if survival_rows else ["weak_block_id"]
    with (audit_dir / "weak_block_survival_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=surv_fields)
        w.writeheader()
        w.writerows(survival_rows)

    with (audit_dir / "channel_audit_labeled_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "channel_id",
                "channel_username",
                "channel_title",
                "post_count_clean",
                "in_strict_whitelist",
                "forbidden_flag",
            ],
        )
        w.writeheader()
        w.writerows(ch_audit)

    with (audit_dir / "overlap_audit_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "path", "overlap_count", "overlap_post_ids"])
        w.writeheader()
        w.writerows(overlap_rows)

    composition = {
        "labeled_rows": len(lab_rows),
        "dropped_rows_action_drop": n_drop,
        "invalid_empty_gold_rows": n_empty_gold,
        "removed_rows": len(removed),
        "clean_rows": len(clean),
        "preview_posts_total": len(preview_posts),
        "preview_posts_not_labeled": len(missing_pids),
        **stats,
        "singleton_story_rate": round(stats["singleton_story_count"] / max(stats["gold_story_count"], 1), 4),
        "non_singleton_story_rate": round(non_sing_rate, 4),
        "story_size_bucket_stories": dict(story_buckets),
        "story_size_bucket_posts": dict(post_by_bucket),
        **survival_summary,
    }

    gate = {
        "expand_with_same_method": expand,
        "gate_checks": gate_checks,
        "failed_checks": failed,
        "thresholds": GATE_THRESHOLDS,
        "composition": composition,
        "validation_ok": validation_ok,
        "validation_issues": validation_issues,
        "overlap_previous_total": overlap_count,
        "whitelist_only": whitelist_only,
        "forbidden_channels_present": forbidden_present,
        "diagnostic_preview_only": False,
    }

    diagnostic_done = False
    if expand and not args.skip_diagnostic and clean:
        exp_v3._bootstrap_pg_password()
        tuned = json.loads(
            (root / "sync_out/data/prepared/combined_manual_gold_v2/tuned_params_combined_manual_gold_v2.json").read_text(
                encoding="utf-8"
            )
        )
        pids = [int(r["post_id"]) for r in clean]
        pg_meta: dict[int, dict[str, Any]] = {}
        for r in clean:
            pid = int(r["post_id"])
            ts = exp_v3._parse_ts(str(r["published_at"]))
            pg_meta[pid] = {"published_at": ts, "channel_id": r["channel_id"]}
        emb = exp_v3._load_embeddings(pids)
        ent = _load_entity_sets(pids)
        feats = _build_pairwise_features(pids, emb, pg_meta, ent)
        order = sorted(
            pids,
            key=lambda p: pg_meta[p]["published_at"] or datetime.min.replace(tzinfo=timezone.utc),
        )
        gold = {int(r["post_id"]): r["namespaced_gold_story_id"] for r in clean}
        diag_rows: list[dict[str, Any]] = []
        for method in ("semantic_only", "hybrid_v4_1_soft_entity_penalty"):
            thr = float(tuned["methods"][method]["tuned_threshold"])
            m = _eval_method(method, order, gold, feats, thr)
            row = _metrics_row(method, "gadv1_preview_clean", m, thr, "combined_manual_gold_v2_audit_v2_frozen")
            row["diagnostic_preview_only"] = True
            diag_rows.append(row)
        with (audit_dir / "diagnostic_metrics_preview_only_v1.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            w.writeheader()
            w.writerows(diag_rows)
        reports.joinpath("graph_advantage_extension_preview_diagnostic_metrics_v1.md").write_text(
            "# Diagnostic metrics (preview only)\n\n"
            "**Not confirmatory.** Thresholds frozen from `combined_manual_gold_v2` / commission audit v2.\n\n"
            + "\n".join(
                f"- **{r['method']}**: F1={r['pairwise_f1']}, acc={r['post_assignment_accuracy']}, "
                f"thr={r['tuned_threshold']}"
                for r in diag_rows
            )
            + "\n",
            encoding="utf-8",
        )
        gate["diagnostic_preview_only"] = True
        diagnostic_done = True

    (audit_dir / "decision_gate_v1.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")

    reports.joinpath("graph_advantage_extension_preview_labeled_audit_v1.md").write_text(
        "# Graph advantage preview — labeled audit\n\n"
        f"- **validation_ok:** {validation_ok}\n"
        f"- **labeled rows:** {len(lab_rows)} / preview **{len(preview_posts)}** "
        f"(missing from labeled: **{len(missing_pids)}**)\n"
        f"- **clean rows:** {len(clean)} | removed: {len(removed)} | action=drop: {n_drop} | empty gold: {n_empty_gold}\n"
        f"- **overlap with previous datasets:** {overlap_count}\n"
        f"- **whitelist_only:** {whitelist_only}\n"
        f"- **forbidden channels in clean:** {forbidden_present}\n\n"
        "## Gold composition\n\n"
        + "\n".join(f"- **{k}:** {v}" for k, v in composition.items() if k not in survival_summary)
        + "\n\n## Validation issues\n\n"
        + ("\n".join(f"- {x}" for x in validation_issues) if validation_issues else "- none")
        + "\n",
        encoding="utf-8",
    )

    reports.joinpath("graph_advantage_extension_preview_weak_block_survival_v1.md").write_text(
        "# Weak block survival (preview labeled)\n\n"
        + "\n".join(f"- **{k}:** {v}" for k, v in survival_summary.items())
        + "\n\n## Per-block status counts\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in status_ctr.items())
        + "\n\n## Incomplete labeling blocks\n\n"
        + "\n".join(
            f"- {r['weak_block_id']}: clean={r['clean_rows_in_block']} / input={r['block_size_input']}"
            for r in survival_rows
            if r["incomplete_labeling"]
        )
        + "\n",
        encoding="utf-8",
    )

    reports.joinpath("graph_advantage_extension_preview_decision_gate_v1.md").write_text(
        "# Decision gate — expand with same graph-advantage method?\n\n"
        f"- **expand_with_same_method:** `{expand}`\n\n"
        "## Checks\n\n"
        + "\n".join(f"- `{k}`: **{v}**" for k, v in gate_checks.items())
        + "\n\n"
        + (f"## Failed\n\n" + "\n".join(f"- {x}" for x in failed) if failed else "## All checks passed\n")
        + f"\n\n**Diagnostic metrics run:** {diagnostic_done}\n",
        encoding="utf-8",
    )

    if not args.skip_rclone:
        r = subprocess.run(
            ["rclone", "sync", str(root / "sync_out"), "gdrive:diploma_story_tracking_experiment"],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print("rclone failed:", r.stderr[:500], file=sys.stderr)

    print(json.dumps({"validation_ok": validation_ok, "expand": expand, **composition}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
