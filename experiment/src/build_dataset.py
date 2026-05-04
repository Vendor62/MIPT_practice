from __future__ import annotations

"""
Dataset build logic (public reference).

This module describes how the manual gold benchmark was assembled:
- whitelist channels
- Neo4j Storyline used only as a candidate source (read-only)
- Postgres used for embeddings/entities/text during private runs
- fulltext is never stored in public artifacts
- old manual gold + expansion manual gold are combined with strict namespace isolation
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ManualGoldSource:
    source_dataset: str  # "old" or "expansion"
    raw_gold_story_id: str
    gold_story_id: str  # namespaced: OLD_* or EXP2_*


def namespace_gold_id(*, source_dataset: str, raw_gold_story_id: str) -> ManualGoldSource:
    raw = (raw_gold_story_id or "").strip()
    if not raw:
        raise ValueError("empty raw_gold_story_id")
    if source_dataset == "old":
        return ManualGoldSource(source_dataset="old", raw_gold_story_id=raw, gold_story_id=f"OLD_{raw}")
    if source_dataset == "expansion":
        if raw.startswith("EXP2_"):
            return ManualGoldSource(source_dataset="expansion", raw_gold_story_id=raw.replace("EXP2_", "", 1), gold_story_id=raw)
        return ManualGoldSource(source_dataset="expansion", raw_gold_story_id=raw, gold_story_id=f"EXP2_{raw}")
    raise ValueError(f"unknown source_dataset: {source_dataset}")


def namespace_audit(*, old_labels: dict[int, str], expansion_labels: dict[int, str], combined_labels: dict[int, str]) -> dict[str, Any]:
    old_raw = set(old_labels.values())
    exp_raw = {v.replace("EXP2_", "", 1) if v.startswith("EXP2_") else v for v in expansion_labels.values()}
    raw_collisions = sorted(list(old_raw & exp_raw))

    old_pids = set(old_labels.keys())
    exp_pids = set(expansion_labels.keys())
    by_gid: dict[str, set[str]] = {}
    for pid, gid in combined_labels.items():
        srcs = by_gid.setdefault(gid, set())
        if pid in old_pids:
            srcs.add("old")
        if pid in exp_pids:
            srcs.add("expansion")

    cross_source_merged = sorted([gid for gid, srcs in by_gid.items() if len(srcs) >= 2])
    prefix_ok = all(g.startswith("OLD_") or g.startswith("EXP2_") for g in combined_labels.values())

    return {
        "namespace_ok": bool(prefix_ok and (len(cross_source_merged) == 0)),
        "prefix_ok": prefix_ok,
        "raw_id_collisions_old_vs_expansion": len(raw_collisions),
        "cross_source_merged_gold_ids": len(cross_source_merged),
    }

from __future__ import annotations

"""
Dataset build logic (public reference).

The original experiment used:
- whitelist channels (Telegram news sources),
- PostgreSQL (posts, embeddings, heuristic entities),
- Neo4j Storyline as a candidate source (read-only),
- manual gold labeling files (CSV/XLSX),
- strict hygiene: no fulltext in public artifacts.

This module documents the logic and provides a minimal, reusable interface.
It is not expected to run without private database access.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ManualGoldSource:
    source_dataset: str  # "old" or "expansion"
    raw_gold_story_id: str
    gold_story_id: str  # namespaced: OLD_* or EXP2_*


def namespace_gold_id(*, source_dataset: str, raw_gold_story_id: str) -> ManualGoldSource:
    raw = (raw_gold_story_id or "").strip()
    if not raw:
        raise ValueError("empty raw_gold_story_id")
    if source_dataset == "old":
        return ManualGoldSource(source_dataset="old", raw_gold_story_id=raw, gold_story_id=f"OLD_{raw}")
    if source_dataset == "expansion":
        if raw.startswith("EXP2_"):
            return ManualGoldSource(source_dataset="expansion", raw_gold_story_id=raw.replace("EXP2_", "", 1), gold_story_id=raw)
        return ManualGoldSource(source_dataset="expansion", raw_gold_story_id=raw, gold_story_id=f"EXP2_{raw}")
    raise ValueError(f"unknown source_dataset: {source_dataset}")


def namespace_audit(*, old_labels: dict[int, str], expansion_labels: dict[int, str], combined_labels: dict[int, str]) -> dict[str, Any]:
    """
    Audit accidental cross-source merges and raw-id collisions.
    """
    old_raw = set(old_labels.values())
    exp_raw = {v.replace("EXP2_", "", 1) if v.startswith("EXP2_") else v for v in expansion_labels.values()}
    raw_collisions = sorted(list(old_raw & exp_raw))

    old_pids = set(old_labels.keys())
    exp_pids = set(expansion_labels.keys())
    by_gid: dict[str, set[str]] = {}
    for pid, gid in combined_labels.items():
        srcs = by_gid.setdefault(gid, set())
        if pid in old_pids:
            srcs.add("old")
        if pid in exp_pids:
            srcs.add("expansion")

    cross_source_merged = sorted([gid for gid, srcs in by_gid.items() if len(srcs) >= 2])

    prefix_ok = all(g.startswith("OLD_") or g.startswith("EXP2_") for g in combined_labels.values())
    return {
        "namespace_ok": bool(prefix_ok and (len(cross_source_merged) == 0)),
        "prefix_ok": prefix_ok,
        "raw_id_collisions_old_vs_expansion": len(raw_collisions),
        "cross_source_merged_gold_ids": len(cross_source_merged),
    }

