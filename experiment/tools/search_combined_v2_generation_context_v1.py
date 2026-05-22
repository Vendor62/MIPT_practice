#!/usr/bin/env python3.11
"""search_combined_v2_generation_context_v1 — producer provenance grep (read-only)."""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

KEYWORDS = [
    "predictions_combined_manual_gold_v2",
    "metrics_combined_manual_gold_v2",
    "tuned_params_combined_manual_gold_v2",
    "combined_pairwise_features_v2",
    "combined_manual_gold_v2",
    "evaluate_expansion_300",
    "expansion_300",
    "large_evaluation_v4",
    "large_evaluation_v4_guarded",
    "guard_grid",
    "hybrid_v4_1",
    "hybrid_v4_1_soft_entity_penalty",
    "soft_entity_penalty",
    "simulate_v4_1",
    "CatBoost",
    "catboost",
    "reranker",
    "pair_reranker",
    "storyline_resolver",
    "build_large_graph_features",
    "evaluate_expansion_300_and_combined_manual_gold_v2",
]

TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".log", ".sh", ".py"}
CSV_HEADER_ONLY = True
MAX_LINE = 400
MAX_MATCHES = 8000

EXTRA_ROOTS = [
    Path("/home/dima_p/.cursor/projects/srv-project/agent-transcripts"),
]


def _exp_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _mtime_iso(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()


def _should_scan(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.stat().st_size > 50_000_000:
        return False
    name = p.name.lower()
    if name.endswith(".csv") and CSV_HEADER_ONLY:
        return "combined" in str(p).lower() or "expansion_300" in str(p).lower() or "manual_gold" in str(p).lower()
    return p.suffix.lower() in TEXT_SUFFIXES or p.suffix.lower() == ".csv"


def _scan_file(p: Path, rel: str) -> list[dict]:
    rows: list[dict] = []
    try:
        if p.suffix.lower() == ".csv" and CSV_HEADER_ONLY:
            with p.open(encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i > 5:
                        break
                    low = line.lower()
                    for kw in KEYWORDS:
                        if kw.lower() in low:
                            rows.append(
                                {
                                    "path": rel,
                                    "line_number": i + 1,
                                    "keyword": kw,
                                    "line_text": line.strip()[:MAX_LINE],
                                    "mtime": _mtime_iso(p),
                                    "size": p.stat().st_size,
                                }
                            )
            return rows
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for i, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        for kw in KEYWORDS:
            if kw.lower() in low:
                rows.append(
                    {
                        "path": rel,
                        "line_number": i,
                        "keyword": kw,
                        "line_text": line.strip()[:MAX_LINE],
                        "mtime": _mtime_iso(p),
                        "size": p.stat().st_size,
                    }
                )
                if len(rows) >= 50:
                    return rows
    return rows


def _collect_scan_roots(exp: Path) -> list[Path]:
    roots = [
        exp,
        exp / "tools",
        exp / "sync_out",
        exp / "sync_out/reports",
        exp / "sync_out/logs",
        exp / "sync_out/runs",
        exp / "sync_out/data/prepared",
        exp / "instructions",
        exp / "sync_in/instructions",
        exp / "sync_out/instructions",
    ]
    roots.extend(EXTRA_ROOTS)
    return [r for r in roots if r.is_dir()]


def build_candidates(exp: Path) -> list[dict]:
    cands: list[dict] = []
    # Primary producer (transcript patch, May 4 2026) — script absent on disk
    cands.append(
        {
            "path": "tools/evaluate_expansion_300_and_combined_manual_gold_v2.py",
            "evidence_type": "producer_script_transcript",
            "confidence": "high",
            "reason": "Agent transcript f93466da: created+run 2026-05-04; wrote combined v2 preds/metrics/tuned/pairwise; subprocess build_large_graph_features_v3; seeds large_tuned_params v3/v4_1",
        }
    )
    cands.append(
        {
            "path": "tools/build_large_graph_features_v3.py",
            "evidence_type": "feature_builder_subprocess",
            "confidence": "high",
            "reason": "Called from evaluate_expansion via subprocess to build combined_pairwise_features_v2.csv",
        }
    )
    cands.append(
        {
            "path": "sync_out/data/prepared/large_evaluation_v4_1_guard_grid/large_tuned_params_v4_1.json",
            "evidence_type": "tuning_seed_input",
            "confidence": "medium",
            "reason": "Default --tuned-v41 seed for combined dev retune; directory may be absent now",
        }
    )
    cands.append(
        {
            "path": "sync_out/data/prepared/large_evaluation_v3/large_tuned_params_v3.json",
            "evidence_type": "tuning_seed_input",
            "confidence": "medium",
            "reason": "Default --tuned-v3 seed for semantic/hybrid_graph on combined",
        }
    )
    cands.append(
        {
            "path": "tools/build_hybrid_as_weak_target_advantage_diagnostic_v1.py",
            "evidence_type": "tuning_library_import",
            "confidence": "high",
            "reason": "Imported as _adv: _tune_structural_standard/_tune_structural_v41 during combined eval",
        }
    )
    cands.append(
        {
            "path": "tools/evaluate_large_hybrid_v4_1_guard_grid.py",
            "evidence_type": "formula_reference",
            "confidence": "medium",
            "reason": "simulate_v4_1 imported; guard grid tuned v4.1 on large_v3 not necessarily re-run for combined preds",
        }
    )
    cands.append(
        {
            "path": "sync_out/reports/combined_manual_gold_v2_validation.md",
            "evidence_type": "run_report",
            "confidence": "medium",
            "reason": "Listed in ARTIFACTS_MANIFEST 2026-05-04 from evaluate_expansion; file may be missing locally",
        }
    )
    cands.append(
        {
            "path": "sync_out/reports/combined_manual_gold_v2_evaluation.md",
            "evidence_type": "run_report",
            "confidence": "medium",
            "reason": "Written by evaluate_expansion main(); may be missing locally",
        }
    )
    cands.append(
        {
            "path": "tools/recompute_combined_manual_gold_v2_metrics_unified.py",
            "evidence_type": "metrics_consumer_not_producer",
            "confidence": "high",
            "reason": "2026-05-21: reloads saved hybrid predictions; does not regenerate May-04 preds",
        }
    )
    for p in [
        exp / "sync_out/data/prepared/combined_manual_gold_v2/predictions_combined_manual_gold_v2.csv",
        exp / "sync_out/data/prepared/combined_manual_gold_v2/metrics_combined_manual_gold_v2.csv",
        exp / "sync_out/data/prepared/combined_manual_gold_v2/tuned_params_combined_manual_gold_v2.json",
        exp / "sync_out/data/prepared/combined_manual_gold_v2/combined_pairwise_features_v2.csv",
    ]:
        if p.is_file():
            cands.append(
                {
                    "path": str(p.relative_to(exp)),
                    "evidence_type": "artifact_output",
                    "confidence": "high",
                    "reason": f"Output batch mtime {_mtime_iso(p)}; no sidecar manifest in directory",
                }
            )
    return cands


def main() -> int:
    exp = _exp_root()
    out = exp / "sync_out/data/prepared/combined_v2_generation_context_search_v1"
    out.mkdir(parents=True, exist_ok=True)
    files_scanned = 0
    matches: list[dict] = []
    seen_files: set[str] = set()

    for root in _collect_scan_roots(exp):
        for dirpath, _dn, filenames in os.walk(root):
            for fn in filenames:
                p = Path(dirpath) / fn
                try:
                    rel = str(p.relative_to(exp)) if p.is_relative_to(exp) else str(p)
                except ValueError:
                    rel = str(p)
                key = str(p.resolve())
                if key in seen_files:
                    continue
                if not _should_scan(p):
                    continue
                seen_files.add(key)
                files_scanned += 1
                matches.extend(_scan_file(p, rel))
                if len(matches) >= MAX_MATCHES:
                    break
            if len(matches) >= MAX_MATCHES:
                break

    match_fields = ["path", "line_number", "keyword", "line_text", "mtime", "size"]
    with (out / "matches_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=match_fields)
        w.writeheader()
        for r in matches:
            w.writerow(r)

    cands = build_candidates(exp)
    cand_fields = ["path", "evidence_type", "confidence", "reason"]
    with (out / "candidate_producer_files_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cand_fields)
        w.writeheader()
        for r in cands:
            w.writerow(r)

    report = f"""# Combined v2 generation context search v1

## Scan summary
- Files scanned: **{files_scanned}**
- Keyword matches: **{len(matches)}** (cap {MAX_MATCHES})
- Artifact batch mtime (UTC): predictions/metrics/tuned **2026-05-04 ~13:55**; pairwise **13:55:20**; labels **13:54:57**

## Producer script (primary)
| Item | Value |
|------|--------|
| Script | `tools/evaluate_expansion_300_and_combined_manual_gold_v2.py` |
| On disk now | **No** (removed after May-04 run; recovered from agent transcript [f93466da](f93466da-7c50-4816-ad63-18b0c74dc013)) |
| Command | `python3.11 tools/evaluate_expansion_300_and_combined_manual_gold_v2.py` |
| Task date | 2026-05-04 (user task `evaluate_expansion_300_and_combined_manual_gold_v2`) |

## Pipeline (from transcript patch)
1. Download `grouped_manual_gold_expansion_300_v2_labeling_labeled` from Drive → `sync_in/`
2. Clean expansion gold → `grouped_manual_gold_expansion_300_v2_manual_gold/`
3. Merge old `manual_gold_labels_v1` + expansion → `combined_manual_gold_v2/` labels, split, metadata
4. **Subprocess** `tools/build_large_graph_features_v3.py` → `combined_pairwise_features_v2.csv`
5. Evaluate 4 methods with dev-only structural tuning via `build_hybrid_as_weak_target_advantage_diagnostic_v1`:
   - `semantic_only`, `semantic_time`, `hybrid_graph_v1_entity`, `hybrid_v4_1_soft_entity_penalty`
6. Seed thresholds: `large_evaluation_v3/large_tuned_params_v3.json`, `large_evaluation_v4_1_guard_grid/large_tuned_params_v4_1.json`
7. Write `predictions_combined_manual_gold_v2.csv`, `metrics_combined_manual_gold_v2.csv`, `tuned_params_combined_manual_gold_v2.json`
8. Reports: `combined_manual_gold_v2_evaluation.md`, `combined_manual_gold_v2_validation.md` (manifest May-04; **not found** in current `sync_out/reports/`)

## CatBoost / reranker
- **Not referenced** in `evaluate_expansion_300` patch (no CatBoost in generation command).
- Prod `storyline_resolver` CatBoost mentioned only in later audit reports as **not** used for combined CSV path.

## large_evaluation_v4_guarded
- Referenced as **seed JSON** path `large_evaluation_v4_1_guard_grid/large_tuned_params_v4_1.json`, not as on-disk prediction bundle for combined.
- `large_evaluation_v4_guarded/` predictions directory **absent** in workspace.

## Later consumers (not producers)
- `tools/recompute_combined_manual_gold_v2_metrics_unified.py` (2026-05-21): hybrid = **load saved** predictions only.
- `tools/evaluate_manual_gold_expansion_v3d_strict_developmental_labeled.py`: hybrid old_test = saved preds.

## Conclusion
**partially documented** — strong transcript + mtime + tuned_params `labeled_file` + STATUS/Manifest May-04 entries; **undocumented** on disk: generator script, run log, command manifest beside artifacts, May-04 evaluation reports.

Outputs: `data/prepared/combined_v2_generation_context_search_v1/`
"""
    (exp / "sync_out/reports/combined_v2_generation_context_search_v1.md").write_text(report, encoding="utf-8")
    print(json.dumps({"validation_ok": True, "files_scanned": files_scanned, "matches": len(matches)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
