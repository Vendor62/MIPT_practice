#!/usr/bin/env python3.11
"""
build_same_source_graph_advantage_extension_preview_v1

Graph-advantage preview extension (same source logic as combined v2 seed).
No metrics, no manual gold for selection, no fulltext in sync_out.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

_v3_path = _TOOLS / "build_manual_gold_expansion_robustness_v3.py"
_spec = importlib.util.spec_from_file_location("exp_v3", _v3_path)
exp_v3 = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(exp_v3)

_rerun = _TOOLS / "rerun_v3d_eval_with_neo4j_graph_features_v2.py"
_spec2 = importlib.util.spec_from_file_location("neo_v2", _rerun)
neo_v2 = importlib.util.module_from_spec(_spec2)
assert _spec2 and _spec2.loader
_spec2.loader.exec_module(neo_v2)

FORBIDDEN_USERS = frozenset({"nachemodanah", "trendwatching24", "rueventjob4at"})
ADS_RE = re.compile(r"(travel|aviasales|job|ваканс|eventjob|marketing)", re.I)
TRUMP_KW = exp_v3.TRUMP_KW
IRAN_KW = exp_v3.IRAN_KW

ARTIFACT_PATHS = [
    "sync_out/data/prepared/hybrid_weak_target_advantage_v1b/selected_posts_metadata_v1b.csv",
    "sync_out/data/prepared/hybrid_weak_target_advantage_v1b/selected_blocks_v1b.csv",
    "sync_out/data/prepared/hybrid_weak_target_advantage_v1b/weak_target_labels_v1b.csv",
    "sync_out/data/prepared/final_graph_advantage_gold_seed_200_v1/seed_manifest_v1.json",
    "sync_out/data/prepared/final_graph_advantage_gold_seed_200_v1/seed_posts_metadata_v1.csv",
    "sync_out/data/prepared/final_graph_advantage_gold_seed_200_v1/seed_blocks_metadata_v1.csv",
    "sync_out/data/prepared/combined_manual_gold_v2/combined_manual_gold_labels_v2.csv",
    "sync_out/data/prepared/combined_manual_gold_v2/combined_posts_metadata_v2.csv",
    "sync_out/data/prepared/combined_manual_gold_v2/combined_gold_story_summary_v2.csv",
]

LABEL_COLS = [
    "action",
    "gold_story_id",
    "reviewer_notes",
    "weak_block_id",
    "row_in_block",
    "block_size",
    "weak_storyline_id",
    "candidate_reason",
    "published_at",
    "channel_id",
    "channel_username",
    "channel_title",
    "post_id",
    "temporal_span_hours",
    "graph_entity_count",
    "graph_storyline_count",
    "semantic_summary_fields",
    "text",
]

TARGET_ROWS_MIN, TARGET_ROWS_MAX = 60, 80
TARGET_BLOCKS_MIN, TARGET_BLOCKS_MAX = 15, 30
TRUMP_IRAN_CAP = 0.30


def _exp_root() -> Path:
    return _TOOLS.parent


def _size_bucket(n: int) -> str:
    return exp_v3._size_bucket(n)


def _topic_tags(text: str) -> set[str]:
    low = (text or "").lower()
    tags: set[str] = set()
    if any(k in low for k in TRUMP_KW):
        tags.add("trump")
    if any(k in low for k in IRAN_KW):
        tags.add("iran")
    return tags


def _artifact_inventory(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    alts: list[tuple[str, str]] = [
        ("hybrid_weak_target_advantage_v1", "sync_out/data/prepared/hybrid_weak_target_advantage_v1"),
        ("hybrid_weak_target_advantage_v1", "sync_out/data/prepared/hybrid_weak_target_advantage_v1/selected_posts_metadata_v1.csv"),
    ]
    for rel in ARTIFACT_PATHS:
        p = root / rel
        rows.append({"requested_path": rel, "status": "found" if p.is_file() else "missing", "resolved_path": str(p) if p.is_file() else ""})
    for name, rel in alts:
        p = root / rel
        if p.exists():
            rows.append({"requested_path": f"analog:{name}", "status": "found_analog", "resolved_path": str(p)})
    return rows


def _strict_whitelist(root: Path) -> list[dict[str, str]]:
    meta_p = root / "sync_out/data/prepared/combined_manual_gold_v2/combined_posts_metadata_v2.csv"
    by: dict[str, dict[str, str]] = {}
    for r in csv.DictReader(meta_p.open(encoding="utf-8")):
        cid = r["channel_id"].strip()
        if cid not in by:
            ch = r.get("channel_name", "")
            by[cid] = {
                "channel_id": cid,
                "channel_name": ch,
                "channel_username": ch.lstrip("@"),
                "channel_title": ch,
                "whitelist_source": "combined_manual_gold_v2",
            }
    return [by[c] for c in sorted(by)]


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
        ("same_protocol", "sync_out/data/prepared/same_protocol_extension_v1/selected_posts_metadata_v1.csv"),
        ("same_protocol_labeled_60", "sync_in/same_protocol_extension_v1_labeling_labeled_60.csv"),
    ]
    for tag, rel in paths:
        p = root / rel
        n = 0
        if p.is_file():
            n = len({int(r["post_id"]) for r in csv.DictReader(p.open(encoding="utf-8"))})
            exclude |= {int(r["post_id"]) for r in csv.DictReader(p.open(encoding="utf-8"))}
        summary.append({"dataset": tag, "path": rel, "present": str(p.is_file()), "post_ids": n})
    return exclude, summary


def _audit_combined_v2(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Proxy audit when v1b/seed files missing — decompose OLD vs EXP2 gold."""
    labels_p = root / "sync_out/data/prepared/combined_manual_gold_v2/combined_manual_gold_labels_v2.csv"
    meta_p = root / "sync_out/data/prepared/combined_manual_gold_v2/combined_posts_metadata_v2.csv"
    summary_p = root / "sync_out/data/prepared/combined_manual_gold_v2/combined_gold_story_summary_v2.csv"

    gold = {int(r["post_id"]): r["gold_story_id"].strip() for r in csv.DictReader(labels_p.open(encoding="utf-8"))}
    meta = {int(r["post_id"]): r for r in csv.DictReader(meta_p.open(encoding="utf-8"))}
    story_sizes = {r["gold_story_id"]: int(r["post_count"]) for r in csv.DictReader(summary_p.open(encoding="utf-8"))}

    pool_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    by_story: dict[str, list[int]] = defaultdict(list)
    for p, g in gold.items():
        by_story[g].append(p)

    for g, ps in by_story.items():
        prefix = "OLD" if g.startswith("OLD_") else ("EXP2" if g.startswith("EXP2_") else "other")
        sz = len(ps)
        months = Counter((meta[p]["published_at"] or "")[:7] for p in ps if p in meta)
        chans = Counter(meta[p].get("channel_name", "") for p in ps if p in meta)
        block_rows.append(
            {
                "proxy_block_id": g,
                "source_prefix": prefix,
                "block_size": sz,
                "story_size_bucket": _size_bucket(sz),
                "post_ids": ";".join(str(x) for x in sorted(ps)),
                "month_top": months.most_common(1)[0][0] if months else "",
                "channel_top": chans.most_common(1)[0][0] if chans else "",
                "survival_non_singleton_gold": 1 if sz >= 2 else 0,
            }
        )

    for p, g in gold.items():
        m = meta.get(p, {})
        pool_rows.append(
            {
                "post_id": p,
                "gold_story_id": g,
                "source_prefix": "OLD" if g.startswith("OLD_") else ("EXP2" if g.startswith("EXP2_") else "other"),
                "published_at": m.get("published_at", ""),
                "channel_name": m.get("channel_name", ""),
                "proxy_block_id": g,
            }
        )

    old_ps = [p for p, g in gold.items() if g.startswith("OLD_")]
    exp_ps = [p for p, g in gold.items() if g.startswith("EXP2_")]
    old_stories = [g for g in by_story if g.startswith("OLD_")]
    exp_stories = [g for g in by_story if g.startswith("EXP2_")]

    def _stats(ps: list[int], stories: list[str]) -> dict[str, Any]:
        sizes = [len(by_story[s]) for s in stories]
        months = Counter((meta[p]["published_at"] or "")[:7] for p in ps if p in meta)
        return {
            "n_posts": len(ps),
            "n_stories": len(stories),
            "singleton_stories": sum(1 for s in sizes if s == 1),
            "multi_stories": sum(1 for s in sizes if s >= 2),
            "bucket_posts": dict(Counter(_size_bucket(len(by_story[s])) for s in stories for _ in [0])),
            "months": dict(months),
        }

    stats = {
        "v1b_artifacts_found": False,
        "combined_v2_posts": len(gold),
        "combined_v2_stories": len(by_story),
        "old_prefix": _stats(old_ps, old_stories),
        "exp2_prefix": _stats(exp_ps, exp_stories),
        "survival_note": "proxy: gold story size>=2 treated as manual non-singleton survival (weak blocks not on disk)",
        "singleton_story_rate": sum(1 for s in story_sizes.values() if s == 1) / max(len(story_sizes), 1),
        "multi_story_rate": sum(1 for s in story_sizes.values() if s >= 2) / max(len(story_sizes), 1),
    }
    return pool_rows, block_rows, stats


def _entity_jaccard(ga: dict[str, Any], gb: dict[str, Any]) -> float:
    ea, eb = ga["entities"], gb["entities"]
    if not ea and not eb:
        return 0.0
    return len(ea & eb) / len(ea | eb)


def _graph_cohesion(pids: list[int], graph: dict[int, dict[str, Any]]) -> float:
    if len(pids) < 2:
        return 0.0
    js = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            js.append(_entity_jaccard(graph[pids[i]], graph[pids[j]]))
    return float(np.mean(js)) if js else 0.0


def _near_dup_block(pids: list[int], meta: dict[int, dict[str, Any]], emb: dict[int, np.ndarray], bodies: dict[int, str]) -> bool:
    if len(pids) < 2:
        return False
    mean_s, min_s, max_s = exp_v3._pairwise_sem_stats(pids, emb)
    span_h = 0.0
    ts = [meta[p]["published_at"] for p in pids if p in meta]
    if len(ts) >= 2:
        span_h = (max(ts) - min(ts)).total_seconds() / 3600.0
    chans = {meta[p]["channel_id"] for p in pids if p in meta}
    if mean_s > 0.92 and span_h < 6.0:
        return True
    if len(chans) == 1 and min_s > 0.88:
        lens = [len(bodies.get(p, "")) for p in pids]
        if max(lens) > 0 and min(lens) / max(lens) > 0.85:
            return True
    return False


def _passes_graph_advantage(
    pids: list[int],
    wid: str,
    meta: dict[int, dict[str, Any]],
    emb: dict[int, np.ndarray],
    graph: dict[int, dict[str, Any]],
    bodies: dict[int, str],
    d0: datetime,
    d1: datetime,
) -> tuple[dict[str, Any] | None, str]:
    if len(pids) < 2 or len(pids) > 12:
        return None, "block_size"
    pids = [p for p in pids if p in meta]
    if len(pids) < 2:
        return None, "meta"
    ts_list = [meta[p]["published_at"] for p in pids]
    if min(ts_list) < d0 or max(ts_list) > d1:
        return None, "date_window"
    if any(meta[p]["text_len"] < 100 for p in pids):
        return None, "short_text"
    span_h = (max(ts_list) - min(ts_list)).total_seconds() / 3600.0
    if span_h < 1.0 and len(pids) > 2:
        return None, "span_lt_1h"
    if _near_dup_block(pids, meta, emb, bodies):
        return None, "near_duplicate"
    mean_s, min_s, max_s = exp_v3._pairwise_sem_stats(pids, emb)
    if mean_s > 0.88:
        return None, "semantic_too_high"
    coh = _graph_cohesion(pids, graph)
    ent_pp = sum(len(graph[p]["entities"]) for p in pids) / len(pids)
    em_pp = sum(len(graph[p]["event_mentions"]) for p in pids) / len(pids)
    if em_pp < 0.5 and ent_pp < 0.1:
        return None, "low_neo_graph_signal"
    em_cnt = sum(len(graph[p]["event_mentions"]) for p in pids)
    sl_cnt = len({sl for p in pids for sl in graph[p]["storylines"]})
    chans = {meta[p]["channel_id"] for p in pids}
    contrast = coh - mean_s + 0.05 * min(ent_pp, 3.0)
    return {
        "weak_storyline_id": wid,
        "post_ids": sorted(pids, key=lambda p: meta[p]["published_at"]),
        "post_count": len(pids),
        "time_span_hours": round(span_h, 2),
        "mean_pairwise_semantic": round(mean_s, 4),
        "min_pairwise_semantic": round(min_s, 4),
        "max_pairwise_semantic": round(max_s, 4),
        "graph_cohesion_score": round(coh, 4),
        "mean_neo_entities_per_post": round(ent_pp, 2),
        "eventmention_total": em_cnt,
        "storyline_count": sl_cnt,
        "distinct_channels": len(chans),
        "graph_semantic_contrast": round(contrast, 4),
        "semantic_regime_proxy": "semantic_low_40" if mean_s <= 0.40 else ("semantic_low_60" if mean_s <= 0.60 else "semantic_moderate"),
    }, ""


def _mine_window(
    exclude: set[int],
    wl: set[str],
    d0: datetime,
    d1: datetime,
    ch_names: dict[str, str],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    snippets_cache: dict[int, str] = {}
    for skip in range(0, 12000, 400):
        neo_blocks = exp_v3._neo_storyline_blocks(skip, 400)
        if not neo_blocks:
            break
        all_pids = list({p for _, ps in neo_blocks for p in ps})
        meta = exp_v3._fetch_posts_meta(all_pids, wl)
        need = [p for p in all_pids if p in meta and p not in exclude]
        if not need:
            continue
        emb = exp_v3._load_embeddings(need)

        for sid, pids in neo_blocks:
            filt = [p for p in pids if p in meta and p not in exclude]
            if len(filt) < 2 or len(filt) > 12:
                continue
            ts_list = [meta[p]["published_at"] for p in filt]
            if min(ts_list) < d0 or max(ts_list) > d1:
                continue
            bid = f"neo_{sid}_{len(filt)}"
            if bid in seen:
                continue
            sub_graph = neo_v2._extract_post_graph(filt)
            missing_sn = [p for p in filt if p not in snippets_cache]
            if missing_sn:
                arr = "ARRAY[" + ",".join(str(x) for x in missing_sn[:10]) + "]::bigint[]"
                for r in exp_v3._psql_json(
                    f"SELECT coalesce(json_agg(row_to_json(t)), '[]'::json)::text FROM ("
                    f"SELECT id AS post_id, left(content, 1200) AS snippet FROM public.posts WHERE id = ANY({arr})) t;"
                ):
                    snippets_cache[int(r["post_id"])] = r.get("snippet") or ""
            rec, _skip_reason = _passes_graph_advantage(filt, sid, meta, emb, sub_graph, snippets_cache, d0, d1)
            if rec is None:
                continue
            seen.add(bid)
            rec["candidate_block_id"] = bid
            rec["source"] = "neo4j_storyline"
            tags: set[str] = set()
            for p in rec["post_ids"][:3]:
                tags |= _topic_tags(snippets_cache.get(p, ""))
            rec["topic_tags_proxy"] = ",".join(sorted(tags)) if tags else "general"
            rec["month_bucket"] = min(meta[p]["published_at"] for p in rec["post_ids"]).strftime("%Y-%m")
            rec["story_size_bucket"] = _size_bucket(len(rec["post_ids"]))
            blocks.append(rec)
        if len(blocks) >= 400:
            break
    return blocks


def _select_preview(blocks: list[dict[str, Any]], target_rows: int = 70) -> list[dict[str, Any]]:
    blocks = [b for b in blocks if len(b.get("post_ids", [])) >= 2]
    blocks = sorted(
        blocks,
        key=lambda b: (
            -float(b.get("graph_semantic_contrast", 0)),
            -float(b.get("time_span_hours", 0)),
            -len(b.get("post_ids", [])),
        ),
    )
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    ti_posts = 0

    def row_stats() -> dict[str, int]:
        posts = [p for g in chosen for p in g["post_ids"]]
        if not posts:
            return {"n": 0, "sing": 0, "sz2": 0, "sz3p": 0}
        c = Counter(_size_bucket(len(g["post_ids"])) for g in chosen)
        n = len(posts)
        sing = sum(len(g["post_ids"]) for g in chosen if len(g["post_ids"]) == 1)
        sz2 = c.get("2", 0) * 2 if "2" in c else sum(len(g["post_ids"]) for g in chosen if len(g["post_ids"]) == 2)
        sz3p = sum(len(g["post_ids"]) for g in chosen if len(g["post_ids"]) >= 3)
        return {"n": n, "sing": sing, "sz2": sum(len(g["post_ids"]) for g in chosen if len(g["post_ids"]) == 2), "sz3p": sz3p}

    for b in blocks:
        pids = [p for p in b["post_ids"] if p not in used]
        if not pids:
            continue
        tags = set((b.get("topic_tags_proxy") or "").split(","))
        add_ti = len(pids) if ("trump" in tags or "iran" in tags) else 0
        st = row_stats()
        new_n = st["n"] + len(pids)
        if (ti_posts + add_ti) / max(new_n, 1) > TRUMP_IRAN_CAP:
            continue
        if new_n > TARGET_ROWS_MAX:
            continue
        if len(chosen) >= TARGET_BLOCKS_MAX:
            break
        bb = dict(b)
        bb["post_ids"] = pids
        bb["post_count"] = len(pids)
        chosen.append(bb)
        used.update(pids)
        if "trump" in tags or "iran" in tags:
            ti_posts += len(pids)
        rs = row_stats()
        if rs["n"] >= TARGET_ROWS_MIN and len(chosen) >= TARGET_BLOCKS_MIN:
            break

    if row_stats()["n"] < TARGET_ROWS_MIN:
        for b in blocks:
            pids = [p for p in b["post_ids"] if p not in used]
            if not pids:
                continue
            chosen.append({**b, "post_ids": pids, "post_count": len(pids)})
            used.update(pids)
            if len(used) >= TARGET_ROWS_MIN:
                break
    return chosen


def _write_labeling(rows: list[dict[str, str]], csv_p: Path, xlsx_p: Path) -> None:
    with csv_p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLS)
        w.writeheader()
        w.writerows(rows)

    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    ins = wb.active
    ins.title = "instructions"
    ins["A1"] = "Graph-advantage extension preview v1"
    for i, t in enumerate(
        [
            "gold_story_id — заполнить после ручной проверки (сейчас пусто).",
            "weak_block_id — целый Neo4j weak block, не менять состав блока без причины.",
            "candidate_reason — почему блок в preview (graph-advantage zone).",
            "action=drop если мусор/реклама/не новость.",
            "reviewer_notes — спорные split/merge.",
        ],
        start=3,
    ):
        ins[f"A{i}"] = t
    ws = wb.create_sheet("review")
    hf = PatternFill("solid", fgColor="D9E1F2")
    bf = PatternFill("solid", fgColor="FFF2CC")
    for j, col in enumerate(LABEL_COLS, 1):
        c = ws.cell(1, j, col)
        c.font = Font(bold=True)
        c.fill = hf
    prev = None
    for i, r in enumerate(rows, 2):
        first = r["weak_block_id"] != prev
        prev = r["weak_block_id"]
        for j, col in enumerate(LABEL_COLS, 1):
            cell = ws.cell(i, j, r.get(col, ""))
            if col == "text":
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            if first and col in ("weak_block_id", "block_size", "candidate_reason"):
                cell.fill = bf
        ws.row_dimensions[i].height = 70 if r.get("text") else 20
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(LABEL_COLS))}{len(rows)+1}"
    wb.save(xlsx_p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rows", type=int, default=70)
    args = ap.parse_args()

    exp_v3._bootstrap_pg_password()
    root = _exp_root()
    out = root / "sync_out/data/prepared/graph_advantage_extension_preview_v1"
    rep = root / "sync_out/reports"
    sync_in = root / "sync_in"
    for p in (out, rep, sync_in):
        p.mkdir(parents=True, exist_ok=True)

    inv = _artifact_inventory(root)
    with (out / "artifact_search_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["requested_path", "status", "resolved_path"])
        w.writeheader()
        w.writerows(inv)

    pool_rows, block_rows, v2_stats = _audit_combined_v2(root)
    with (out / "original_pool_audit_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pool_rows[0].keys()) if pool_rows else ["post_id"])
        w.writeheader()
        w.writerows(pool_rows)
    with (out / "original_pool_block_audit_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(block_rows[0].keys()) if block_rows else ["proxy_block_id"])
        w.writeheader()
        w.writerows(block_rows)

    wl_rows = _strict_whitelist(root)
    with (out / "channel_audit_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wl_rows[0].keys()))
        w.writeheader()
        w.writerows(wl_rows)
    wl_ids = {r["channel_id"] for r in wl_rows}
    ch_map = {r["channel_id"]: r for r in wl_rows}

    exclude, excl_sum = _load_exclude(root)
    with (out / "excluded_post_ids_summary_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "path", "present", "post_ids"])
        w.writeheader()
        w.writerows(excl_sum)

    windows = [
        ("adjacent_2026_q1", datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc)),
        ("fallback_2026_recent", datetime(2026, 4, 1, tzinfo=timezone.utc), datetime(2026, 5, 21, 23, 59, 59, tzinfo=timezone.utc)),
        ("primary_2025_q4", datetime(2025, 9, 1, tzinfo=timezone.utc), datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)),
        ("secondary_2025_q2q3", datetime(2025, 5, 1, tzinfo=timezone.utc), datetime(2025, 8, 31, 23, 59, 59, tzinfo=timezone.utc)),
    ]

    all_blocks: list[dict[str, Any]] = []
    window_used = ""
    chosen: list[dict[str, Any]] = []
    for wname, d0, d1 in windows:
        mined = _mine_window(exclude, wl_ids, d0, d1, ch_map)
        all_blocks.extend(mined)
        chosen = _select_preview(all_blocks, args.target_rows)
        n_sel = len({p for g in chosen for p in g["post_ids"]})
        window_used = wname
        if n_sel >= TARGET_ROWS_MIN:
            break
    if len({p for g in chosen for p in g["post_ids"]}) < TARGET_ROWS_MIN:
        print("WARN: fewer than 60 posts selected", file=sys.stderr)

    # metadata + labeling
    pids_all = sorted({p for g in chosen for p in g["post_ids"]})
    graph = neo_v2._extract_post_graph(pids_all)
    pg = exp_v3._psql_json(
        f"""
        SELECT coalesce(json_agg(row_to_json(t) ORDER BY t.post_id), '[]'::json)::text FROM (
          SELECT p.id AS post_id, p.timestamp AS published_at, p.channel_id::text AS channel_id,
                 coalesce(c.username,'') AS username, coalesce(c.name,'') AS channel_title,
                 coalesce(p.content,'') AS content
          FROM public.posts p
          LEFT JOIN communities c ON c.chat_id::text = p.channel_id
          WHERE p.id = ANY(ARRAY[{','.join(str(x) for x in pids_all)}]::bigint[])
        ) t;
        """
    )
    pg_map = {int(r["post_id"]): r for r in pg}

    post_meta_rows: list[dict[str, Any]] = []
    block_meta_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, str]] = []
    months: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    ti = 0

    for bi, g in enumerate(chosen, 1):
        wblk = f"GAXP1_BLOCK_{bi:04d}"
        pids = g["post_ids"]
        reason = (
            f"neo4j_storyline;contrast={g.get('graph_semantic_contrast')};"
            f"sem_mean={g.get('mean_pairwise_semantic')};span_h={g.get('time_span_hours')};"
            f"regime={g.get('semantic_regime_proxy')}"
        )
        block_meta_rows.append(
            {
                "weak_block_id": wblk,
                "weak_storyline_id": g.get("weak_storyline_id", ""),
                "post_count": len(pids),
                "story_size_bucket": g.get("story_size_bucket", ""),
                "time_span_hours": g.get("time_span_hours", ""),
                "mean_pairwise_semantic": g.get("mean_pairwise_semantic", ""),
                "graph_cohesion_score": g.get("graph_cohesion_score", ""),
                "graph_semantic_contrast": g.get("graph_semantic_contrast", ""),
                "semantic_regime_proxy": g.get("semantic_regime_proxy", ""),
                "topic_tags_proxy": g.get("topic_tags_proxy", ""),
                "month_bucket": g.get("month_bucket", ""),
                "selection_window": window_used,
            }
        )
        for ri, pid in enumerate(sorted(pids, key=lambda x: pg_map[x]["published_at"]), 1):
            pr = pg_map[pid]
            ts = exp_v3._parse_ts(str(pr.get("published_at") or ""))
            cid = str(pr.get("channel_id") or "")
            un = (pr.get("username") or "").strip()
            ch_user = f"@{un}" if un and not un.startswith("@") else un
            ch_title = str(pr.get("channel_title") or ch_map.get(cid, {}).get("channel_name", cid))
            gph = graph.get(pid, {"entities": set(), "storylines": set()})
            sem_f = f"mean={g.get('mean_pairwise_semantic')};min={g.get('min_pairwise_semantic')};max={g.get('max_pairwise_semantic')}"
            text = exp_v3._normalize_text(pr.get("content") or "")
            tags = _topic_tags(text)
            if "trump" in tags or "iran" in tags:
                ti += 1
            if ts:
                months[ts.strftime("%Y-%m")] += 1
            channels[ch_user or ch_title] += 1
            post_meta_rows.append(
                {
                    "post_id": pid,
                    "weak_block_id": wblk,
                    "weak_storyline_id": g.get("weak_storyline_id", ""),
                    "row_in_block": ri,
                    "block_size": len(pids),
                    "published_at": ts.isoformat() if ts else "",
                    "channel_id": cid,
                    "channel_name": ch_user or ch_title,
                    "month_bucket": ts.strftime("%Y-%m") if ts else "",
                    "graph_entity_count": len(gph["entities"]),
                    "graph_storyline_count": len(gph["storylines"]),
                    "mean_pairwise_semantic": g.get("mean_pairwise_semantic", ""),
                    "selection_window": window_used,
                }
            )
            label_rows.append(
                {
                    "action": "",
                    "gold_story_id": "",
                    "reviewer_notes": "",
                    "weak_block_id": wblk,
                    "row_in_block": str(ri),
                    "block_size": str(len(pids)),
                    "weak_storyline_id": str(g.get("weak_storyline_id", "")),
                    "candidate_reason": reason,
                    "published_at": ts.isoformat() if ts else "",
                    "channel_id": cid,
                    "channel_username": ch_user,
                    "channel_title": ch_title,
                    "post_id": str(pid),
                    "temporal_span_hours": str(g.get("time_span_hours", "")),
                    "graph_entity_count": str(len(gph["entities"])),
                    "graph_storyline_count": str(len(gph["storylines"])),
                    "semantic_summary_fields": sem_f,
                    "text": text,
                }
            )

    n_posts = len(label_rows)
    bucket_posts = Counter(_size_bucket(int(r["block_size"])) for r in label_rows)
    sing_posts = sum(1 for r in label_rows if int(r["block_size"]) == 1)
    multi_posts = n_posts - sing_posts
    ti_rate = ti / n_posts if n_posts else 0

    manifest = {
        "dataset": "graph_advantage_extension_preview_v1",
        "protocol": "combined_v2_graph_advantage_same_source",
        "v1b_artifacts_on_disk": False,
        "selection_window": window_used,
        "selected_posts": n_posts,
        "selected_blocks": len(chosen),
        "strict_whitelist_channels": len(wl_rows),
        "excluded_total_ids": len(exclude),
        "singleton_post_rate": round(sing_posts / n_posts, 4) if n_posts else 0,
        "multi_post_rate": round(multi_posts / n_posts, 4) if n_posts else 0,
        "story_size_bucket_posts": dict(bucket_posts),
        "trump_iran_proxy_rate": round(ti_rate, 4),
        "month_distribution": dict(months),
        "channel_distribution": dict(channels.most_common(20)),
        "v2_audit_proxy": v2_stats,
        "target_rows": [TARGET_ROWS_MIN, TARGET_ROWS_MAX],
        "target_blocks": [TARGET_BLOCKS_MIN, TARGET_BLOCKS_MAX],
    }

    with (out / "selected_posts_metadata_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(post_meta_rows[0].keys()) if post_meta_rows else ["post_id"])
        w.writeheader()
        w.writerows(post_meta_rows)
    with (out / "selected_blocks_metadata_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(block_meta_rows[0].keys()) if block_meta_rows else ["weak_block_id"])
        w.writeheader()
        w.writerows(block_meta_rows)
    (out / "selection_manifest_v1.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_label = sync_in / "graph_advantage_extension_preview_v1_labeling.csv"
    xlsx_label = sync_in / "graph_advantage_extension_preview_v1_labeling.xlsx"
    _write_labeling(label_rows, csv_label, xlsx_label)

    found = [x for x in inv if x["status"] == "found"]
    missing = [x for x in inv if x["status"] == "missing"]
    audit_md = [
        "# Graph-advantage original pool audit v1",
        "",
        "## Artifact search",
        "",
        f"- **Found:** {len(found)} / {len([x for x in ARTIFACT_PATHS])} requested paths",
        f"- **Missing v1b/seed files:** {len(missing)} (see `artifact_search_v1.csv`)",
        "",
        "Closest analog on disk: script `tools/build_hybrid_as_weak_target_advantage_diagnostic_v1.py` "
        "(outputs `hybrid_weak_target_advantage_v1/`, not v1b). No `selected_*_v1b.csv` or `final_graph_advantage_gold_seed_200_v1/` present.",
        "",
        "## Proxy audit from combined_manual_gold_v2",
        "",
        f"- Combined posts: **{v2_stats['combined_v2_posts']}**, stories: **{v2_stats['combined_v2_stories']}**",
        f"- OLD prefix: {v2_stats['old_prefix']}",
        f"- EXP2 prefix: {v2_stats['exp2_prefix']}",
        f"- Singleton story rate (gold): **{v2_stats['singleton_story_rate']:.1%}**",
        f"- Multi story rate: **{v2_stats['multi_story_rate']:.1%}**",
        "",
        "## Survival proxy (no weak blocks on disk)",
        "",
        v2_stats.get("survival_note", ""),
        "",
        "Interpretation: multi-post gold stories (size≥2) ≈ weak block survived as non-singleton manual story.",
        "",
    ]
    (rep / "graph_advantage_original_pool_audit_v1.md").write_text("\n".join(audit_md), encoding="utf-8")

    val_ok = (
        n_posts >= TARGET_ROWS_MIN
        and n_posts <= TARGET_ROWS_MAX
        and TARGET_BLOCKS_MIN <= len(chosen) <= TARGET_BLOCKS_MAX
        and ti_rate <= TRUMP_IRAN_CAP
    )
    val_lines = [
        f"validation_ok={val_ok}",
        f"rows={n_posts} blocks={len(chosen)}",
        f"window={window_used}",
        f"singleton_rate={sing_posts/max(n_posts,1):.3f}",
        f"trump_iran={ti_rate:.3f}",
    ]
    (rep / "graph_advantage_extension_preview_validation_v1.md").write_text(
        "\n".join(
            [
                "# Graph-advantage extension preview validation v1",
                "",
                f"**validation_ok:** `{val_ok}`",
                "",
                *[f"- {ln}" for ln in val_lines],
                "",
                f"**Bucket posts:** `{dict(bucket_posts)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (rep / "graph_advantage_extension_preview_v1.md").write_text(
        "\n".join(
            [
                "# Graph-advantage extension preview v1",
                "",
                f"- Preview: **{n_posts}** posts, **{len(chosen)}** blocks",
                f"- Window: **{window_used}**",
                f"- Whitelist: **{len(wl_rows)}** channels (combined v2)",
                f"- Labeling: `sync_in/graph_advantage_extension_preview_v1_labeling.csv`",
                "",
                "## Recommendation",
                "",
                "Proceed with **manual preview labeling** if blocks look like real multi-post news arcs "
                "(not repost duplicates). Then merge into extension eval with frozen audit v2 thresholds.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    for p in out.rglob("*.csv"):
        hdr = next(csv.reader(p.open(encoding="utf-8")))
        if {"text", "content", "fulltext"} & set(hdr):
            raise SystemExit(f"forbidden col in {p}")

    print("OK", n_posts, len(chosen), window_used, "validation_ok", val_ok)
    return 0 if val_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
