#!/usr/bin/env python3.11
"""Проверка набора для ручной разметки manual_gold_expansion_v3d (строгий «развивающийся» режим)."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FORBIDDEN_COLS = frozenset({"fulltext", "content", "text", "raw_text"})
FORBIDDEN_USERS = frozenset({"nachemodanah", "trendwatching24", "rueventjob4at"})
MIN_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _norm_user(s: str) -> str:
    return (s or "").strip().lower().lstrip("@")


def _parse_ts(s: str) -> datetime | None:
    t = (s or "").strip().replace(" ", "T", 1)
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rclone", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = root / "sync_out/data/prepared/manual_gold_expansion_v3d"
    prep_v2 = root / "sync_out/data/prepared/combined_manual_gold_v2"
    wl_path = out_dir / "strict_news_whitelist_v3d.csv"
    label_csv = root / "sync_in/grouped_manual_gold_expansion_v3d_strict_developmental_labeling.csv"
    label_xlsx = root / "sync_in/grouped_manual_gold_expansion_v3d_strict_developmental_labeling.xlsx"
    manifest_path = out_dir / "selection_manifest_v3d.json"
    report_path = root / "sync_out/reports/manual_gold_expansion_validation_v3d.md"
    raw_path = root / "sync_out/runs/manual_gold_expansion_v3d/validation_raw.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    issues: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []

    def check(cond: bool, msg: str, *, hard: bool = True) -> None:
        lines.append(f"{'OK' if cond else ('FAIL' if hard else 'WARN')}: {msg}")
        if not cond:
            (issues if hard else warnings).append(msg)

    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(label_csv.open(encoding="utf-8-sig")))
    n = len(rows)
    check(n >= 100, f"rows {n} >= 100 (quality floor with warning path)")
    if n < 150:
        warnings.append(f"rows {n} below target 150")
    check(n <= 250, f"rows {n} <= 250")

    blocks = int(man.get("selected_blocks", 0))
    if blocks < 30:
        warnings.append(f"blocks {blocks} below target 30")
    check(blocks <= 60, f"blocks {blocks} <= 60", hard=False)

    pids = [int(r["post_id"]) for r in rows]
    check(len(pids) == len(set(pids)), "post_id unique")

    def load_pids(p: Path) -> set[int]:
        if not p.is_file():
            return set()
        return {int(r["post_id"]) for r in csv.DictReader(p.open(encoding="utf-8"))}

    old = load_pids(prep_v2 / "combined_manual_gold_labels_v2.csv")
    v3 = load_pids(root / "sync_out/data/prepared/manual_gold_expansion_v3/selected_posts_metadata_v3.csv")
    v3b = load_pids(root / "sync_out/data/prepared/manual_gold_expansion_v3b/selected_posts_metadata_v3b.csv")
    v3c = load_pids(root / "sync_out/data/prepared/manual_gold_expansion_v3c/selected_posts_metadata_v3c.csv")
    v3c_clean = load_pids(
        root / "sync_out/data/prepared/manual_gold_expansion_v3c_clean_audit/v3c_labeled_clean_posts_v1.csv"
    )
    ov = {
        "combined_v2": len(set(pids) & old),
        "v3": len(set(pids) & v3),
        "v3b": len(set(pids) & v3b),
        "v3c": len(set(pids) & v3c),
        "v3c_clean": len(set(pids) & v3c_clean),
    }
    check(sum(ov.values()) == 0, f"overlap with prior sets = 0 ({ov})")

    wl_ids = {r["channel_id"].strip() for r in csv.DictReader(wl_path.open(encoding="utf-8"))}
    meta = list(csv.DictReader((out_dir / "selected_posts_metadata_v3d.csv").open(encoding="utf-8")))
    check(wl_ids and all(r.get("channel_id") in wl_ids for r in meta), "strict_news_whitelist_only")

    bad_ch = [
        r["channel_name"]
        for r in rows
        if _norm_user(r.get("channel_name", "")) in FORBIDDEN_USERS
        or "nachemodanah" in (r.get("channel_name") or "").lower()
    ]
    check(not bad_ch, f"forbidden suspicious channels absent (found {bad_ch[:3]})")

    dev_r = float(man.get("developmental_post_rate", 0))
    sing_r = float(man.get("singleton_control_post_rate", 0))
    nd_r = float(man.get("near_duplicate_diagnostic_post_rate", 0))
    check(dev_r >= 0.60 - 1e-9, f"developmental_post_rate {dev_r:.3f} >= 0.60")
    check(sing_r <= 0.25 + 1e-9, f"singleton_control {sing_r:.3f} <= 0.25")
    check(nd_r <= 0.10 + 1e-9, f"near_duplicate_diagnostic {nd_r:.3f} <= 0.10")

    med_span = float(man.get("developmental_median_time_span_hours", 0))
    check(med_span >= 48 - 1e-9, f"developmental median span {med_span:.1f}h >= 48")
    if med_span < 72:
        warnings.append(f"developmental median span {med_span:.1f}h below target 72h")

    phase_share = float(man.get("developmental_blocks_phase_ge2_share", 0))
    check(phase_share >= 0.80 - 1e-9, f"developmental blocks phase>=2 share {phase_share:.3f} >= 0.80")

    check(all((r.get("gold_story_id") or "").strip() == "" for r in rows), "gold_story_id empty")
    check(all((r.get("suggested_story_id") or "").strip() != "" for r in rows), "suggested_story_id filled")
    check(all((r.get("text") or "").strip() != "" for r in rows), "text non-empty")

    old2023 = int(man.get("2023_or_older_count", -1))
    check(old2023 == 0, f"2023_or_older_count={old2023}")
    min_ts = _parse_ts(man.get("published_at_min", ""))
    if min_ts:
        check(min_ts >= MIN_DATE, f"min published_at {min_ts}")

    expanded = root / "sync_out/data/prepared/dataset_candidate_mining_whitelist_v1/channel_whitelist_v1.csv"
    if expanded.is_file():
        exp_ids = {r["channel_id"].strip() for r in csv.DictReader(expanded.open(encoding="utf-8"))}
        check(wl_ids <= exp_ids or wl_ids.issubset(exp_ids), "strict whitelist subset of known lists", hard=False)

    for p in out_dir.glob("*.csv"):
        hdr = next(csv.reader(p.open(encoding="utf-8")), [])
        bad = [c for c in hdr if c.lower() in FORBIDDEN_COLS]
        check(not bad, f"no forbidden cols in {p.name}")

    check(label_xlsx.is_file(), "xlsx exists")
    check(man.get("strict_whitelist_source") == "combined_manual_gold_v2", "strict whitelist source documented")

    hard_ok = len(issues) == 0
    ok = hard_ok

    if not args.skip_rclone:
        for cmd in (
            [
                "rclone",
                "copyto",
                str(label_csv),
                "gdrive:diploma_story_tracking_experiment/sync_in/grouped_manual_gold_expansion_v3d_strict_developmental_labeling.csv",
            ],
            [
                "rclone",
                "copyto",
                str(label_xlsx),
                "gdrive:diploma_story_tracking_experiment/sync_in/grouped_manual_gold_expansion_v3d_strict_developmental_labeling.xlsx",
            ],
            ["rclone", "sync", str(root / "sync_out"), "gdrive:diploma_story_tracking_experiment"],
        ):
            r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
            lines.append(f"rclone exit={r.returncode} ({cmd[1]})")
            if r.returncode != 0:
                issues.append(f"rclone failed: {r.stderr[:200]}")
                ok = False

    report = (
        "# manual_gold_expansion_v3d validation\n\n"
        f"- **validation_ok:** `{str(ok).lower()}`\n"
        f"- **hard_checks_ok:** `{str(hard_ok).lower()}`\n\n"
        "## Checks\n\n"
        + "\n".join(f"- {ln}" for ln in lines)
        + "\n\n## Hard issues\n\n"
        + ("\n".join(f"- {i}" for i in issues) if issues else "- none")
        + "\n\n## Warnings\n\n"
        + ("\n".join(f"- {w}" for w in warnings) if warnings else "- none")
        + "\n"
    )
    report_path.write_text(report, encoding="utf-8")
    raw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
