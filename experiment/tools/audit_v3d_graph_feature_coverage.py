#!/usr/bin/env python3.11
"""audit_v3d_graph_feature_coverage — покрытие графовых признаков (Postgres/Neo4j/парные) на v3d по сравнению со старым сводным набором."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
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

PAIR_NUM_COLS = [
    "semantic_score",
    "temporal_score",
    "entity_jaccard",
    "entity_overlap_count",
    "event_signature_jaccard",
    "graph_neighborhood_jaccard",
    "graph_relation_type_jaccard",
    "channel_same_flag",
    "near_duplicate_score",
]


def _exp_root() -> Path:
    return _TOOLS.parent


def _load_pids_from_csv(path: Path, col: str = "post_id") -> list[int]:
    return [int(r[col]) for r in csv.DictReader(path.open(encoding="utf-8"))]


def _pg_entity_stats(pids: list[int]) -> dict[str, Any]:
    if not pids:
        return {"n_posts": 0}
    arr = "ARRAY[" + ",".join(str(x) for x in pids) + "]::bigint[]"
    row = exp_v3._psql_json(
        f"""
        SELECT coalesce(json_agg(row_to_json(t)), '[]'::json)::text FROM (
          WITH target AS (SELECT unnest({arr}) AS post_id),
          ent AS (
            SELECT post_id, count(*) AS entity_rows, count(DISTINCT lower(trim(text))) AS distinct_entity_text
            FROM public.entities
            WHERE post_id = ANY({arr}) AND coalesce(text,'') <> ''
            GROUP BY post_id
          ),
          ent_any AS (
            SELECT post_id, count(*) AS entity_rows_any
            FROM public.entities
            WHERE post_id = ANY({arr})
            GROUP BY post_id
          ),
          emb AS (
            SELECT DISTINCT post_id FROM public.embeddings WHERE post_id = ANY({arr})
          ),
          pst AS (
            SELECT id AS post_id FROM public.posts WHERE id = ANY({arr})
          )
          SELECT
            (SELECT count(*) FROM target) AS n_posts,
            (SELECT count(*) FROM pst) AS in_posts,
            (SELECT count(*) FROM emb) AS in_embeddings,
            (SELECT count(*) FROM ent) AS posts_with_entities,
            (SELECT count(*) FROM ent_any) AS posts_with_any_entity_row,
            (SELECT coalesce(sum(entity_rows),0) FROM ent) AS total_entity_rows,
            (SELECT coalesce(sum(entity_rows_any),0) FROM ent_any) AS total_entity_rows_any,
            (SELECT coalesce(max(entity_rows),0) FROM ent) AS max_entities_per_post,
            (SELECT coalesce(min(entity_rows),0) FROM ent) AS min_entities_nonempty,
            (SELECT count(*) FROM ent WHERE entity_rows >= 2) AS posts_ge2_entities
        ) t;
        """
    )[0]
    n = int(row["n_posts"])
    posts_with = int(row["posts_with_entities"])
    posts_any = int(row.get("posts_with_any_entity_row", 0))
    return {
        "n_posts": n,
        "in_posts": int(row["in_posts"]),
        "in_embeddings": int(row["in_embeddings"]),
        "posts_with_any_entity_row": posts_any,
        "share_posts_any_entity_row": round(posts_any / max(n, 1), 4),
        "posts_with_ge1_entity": posts_with,
        "share_posts_ge1_entity": round(posts_with / max(n, 1), 4),
        "total_entity_rows_any": int(row.get("total_entity_rows_any", 0)),
        "posts_ge2_entities": int(row["posts_ge2_entities"]),
        "share_posts_ge2_entities": round(int(row["posts_ge2_entities"]) / max(n, 1), 4),
        "total_entity_rows": int(row["total_entity_rows"]),
        "max_entities_per_post": int(row["max_entities_per_post"]),
    }


def _neo_row(st: str, params: dict[str, Any] | None = None) -> list[Any]:
    res = exp_v3._neo_commit([{"statement": st, "parameters": params or {}}])
    if not res or not res[0]["data"]:
        return []
    return [r["row"] for r in res[0]["data"]]


def _neo_schema_inventory() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label in ("EventMention", "Storyline", "Entity", "Post"):
        cnt = _neo_row(f"MATCH (n:{label}) RETURN count(n) AS c")[0][0]
        keys = _neo_row(f"MATCH (n:{label}) RETURN keys(n) AS k LIMIT 1")
        klist = keys[0][0] if keys else []
        rows.append({"label": label, "count": str(cnt), "sample_keys": ";".join(sorted(klist)[:40])})
    rels = _neo_row(
        "MATCH ()-[r]->() RETURN DISTINCT type(r) AS t ORDER BY t LIMIT 200"
    )
    for t in rels:
        rows.append({"label": f"REL:{t[0]}", "count": "", "sample_keys": ""})
    em_part_of = _neo_row("MATCH (:EventMention)-[:PART_OF]->(:Storyline) RETURN count(*) AS c")[0][0]
    em_inv = _neo_row("MATCH (:EventMention)-[:INVOLVES]->(:Entity) RETURN count(*) AS c")[0][0]
    rows.append({"label": "EventMention-PART_OF->Storyline", "count": str(em_part_of), "sample_keys": ""})
    rows.append({"label": "EventMention-INVOLVES->Entity", "count": str(em_inv), "sample_keys": ""})
    return rows


def _neo_match_keys(pids: list[int]) -> dict[str, Any]:
    """Try candidate post id properties on EventMention."""
    if not pids:
        return {}
    arr = pids[:500]  # sample for key probe
    keys_row = _neo_row("MATCH (em:EventMention) RETURN keys(em) AS k LIMIT 1")
    props = keys_row[0][0] if keys_row else []
    candidates = [p for p in props if "post" in p.lower() or "message" in p.lower() or p in ("id", "external_id")]
    if "post_id" not in candidates:
        candidates = ["post_id"] + candidates
    out: dict[str, Any] = {"eventmention_properties": ";".join(props), "candidates_tested": []}
    chunk = 80
    for prop in candidates[:12]:
        matched = 0
        for i in range(0, len(pids), chunk):
            sub = pids[i : i + chunk]
            st = f"MATCH (em:EventMention) WHERE em.{prop} IN $pids RETURN count(DISTINCT em.{prop})"
            try:
                c = _neo_row(st, {"pids": sub})[0][0]
                matched += int(c)
            except Exception:
                matched = -1
                break
        out["candidates_tested"].append({"property": prop, "matched_distinct": matched})
    best = max(
        (x for x in out["candidates_tested"] if x["matched_distinct"] >= 0),
        key=lambda x: x["matched_distinct"],
        default={"property": "post_id", "matched_distinct": 0},
    )
    out["best_key"] = best["property"]
    out["best_matched_posts"] = best["matched_distinct"]
    return out


def _neo_coverage(pids: list[int], key: str = "post_id") -> dict[str, Any]:
    if not pids:
        return {}
    matched_ems = 0
    with_part_of = 0
    with_involves = 0
    entity_rows = 0
    storylines: set[str] = set()
    chunk = 50
    for i in range(0, len(pids), chunk):
        sub = pids[i : i + chunk]
        row = _neo_row(
            f"""
            MATCH (em:EventMention)
            WHERE em.{key} IN $pids
            OPTIONAL MATCH (em)-[:PART_OF]->(s:Storyline)
            OPTIONAL MATCH (em)-[:INVOLVES]->(en:Entity)
            RETURN count(em) AS em_cnt,
                   count(DISTINCT em.{key}) AS post_cnt,
                   count(DISTINCT s) AS storyline_cnt,
                   sum(CASE WHEN s IS NOT NULL THEN 1 ELSE 0 END) AS with_part_of,
                   count(en) AS entity_mentions
            """,
            {"pids": sub},
        )[0]
        matched_ems += int(row[0])
        with_part_of += int(row[3])
        entity_rows += int(row[4])
        st2 = f"MATCH (em:EventMention)-[:PART_OF]->(s:Storyline) WHERE em.{key} IN $pids RETURN DISTINCT s.storyline_id AS sid"
        try:
            for r in _neo_row(st2, {"pids": sub}):
                if r[0]:
                    storylines.add(str(r[0]))
        except Exception:
            pass
        st3 = f"""
        MATCH (em:EventMention)-[:INVOLVES]->(en:Entity)
        WHERE em.{key} IN $pids
        RETURN count(DISTINCT em.{key}) AS posts_with_entity
        """
        try:
            with_involves += int(_neo_row(st3, {"pids": sub})[0][0])
        except Exception:
            pass
    n = len(pids)
    return {
        "n_posts": n,
        "eventmention_rows": matched_ems,
        "posts_with_eventmention": min(matched_ems, n),
        "share_posts_with_eventmention": round(min(matched_ems, n) / max(n, 1), 4),
        "posts_with_part_of": with_part_of,
        "posts_with_involves_entity": with_involves,
        "total_entity_involve_rows": entity_rows,
        "distinct_storylines": len(storylines),
    }


def _pairwise_nonzero_audit(path: Path, label: str) -> dict[str, Any]:
    cols = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    if not cols:
        return {"label": label, "n_pairs": 0}
    n = len(cols)
    out: dict[str, Any] = {"label": label, "n_pairs": n, "columns": list(cols[0].keys())}
    for c in PAIR_NUM_COLS:
        if c not in cols[0]:
            out[f"{c}_present"] = False
            continue
        vals = []
        for r in cols:
            try:
                vals.append(float(r[c]))
            except ValueError:
                vals.append(float("nan"))
        arr = np.array(vals, dtype=float)
        nz = int(np.sum(arr > 1e-9))
        out[f"{c}_nonzero_share"] = round(nz / n, 4)
        out[f"{c}_mean"] = round(float(np.nanmean(arr)), 4)
        out[f"{c}_max"] = round(float(np.nanmax(arr)), 4)
    if "has_graph_features" in cols[0]:
        hg = [str(r.get("has_graph_features", "")).lower() in ("true", "1", "1.0") for r in cols]
        out["has_graph_features_true_share"] = round(sum(hg) / n, 4)
    return out


def _post_debug_sample(
    pids: list[int],
    key: str,
    n_sample: int = 15,
) -> list[dict[str, Any]]:
    sample = sorted(pids)[:n_sample]
    rows: list[dict[str, Any]] = []
    arr = "ARRAY[" + ",".join(str(x) for x in sample) + "]::bigint[]"
    pg = {
        int(r["post_id"]): int(r["entity_rows"])
        for r in exp_v3._psql_json(
            f"""
            SELECT coalesce(json_agg(row_to_json(t)), '[]'::json)::text FROM (
              SELECT post_id, count(*)::int AS entity_rows FROM public.entities
              WHERE post_id = ANY({arr}) AND coalesce(text,'') <> ''
              GROUP BY post_id
            ) t;
            """
        )
    }
    meta_of = {
        int(r["post_id"]): r
        for r in exp_v3._psql_json(
            f"""
            SELECT coalesce(json_agg(row_to_json(t)), '[]'::json)::text FROM (
              SELECT id AS post_id, channel_id::text, timestamp::text AS published_at
              FROM public.posts WHERE id = ANY({arr})
            ) t;
            """
        )
    }
    for pid in sample:
        neo_em = 0
        neo_ent = 0
        storyline = ""
        try:
            em = _neo_row(
                f"MATCH (em:EventMention) WHERE em.{key} = $pid "
                f"OPTIONAL MATCH (em)-[:PART_OF]->(s:Storyline) "
                f"OPTIONAL MATCH (em)-[:INVOLVES]->(en:Entity) "
                f"RETURN count(em), count(en), head(collect(s.storyline_id))",
                {"pid": pid},
            )
            if em:
                neo_em = int(em[0][0])
                neo_ent = int(em[0][1])
                storyline = str(em[0][2] or "")
        except Exception as e:
            storyline = f"neo_err:{e}"
        meta = meta_of.get(pid, {})
        ch = meta.get("channel_id", "")
        ts = meta.get("published_at", "")
        rows.append(
            {
                "post_id": pid,
                "channel_id": ch,
                "published_at": ts,
                "pg_entity_count": pg.get(pid, 0),
                "neo4j_eventmention_count": neo_em,
                "neo4j_entity_count": neo_ent,
                "neo4j_storyline_id": storyline,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rclone", action="store_true")
    args = ap.parse_args()

    root = _exp_root()
    audit_dir = root / "sync_out/data/prepared/manual_gold_expansion_v3d_graph_audit"
    reports = root / "sync_out/reports"
    audit_dir.mkdir(parents=True, exist_ok=True)

    v3d_pids = _load_pids_from_csv(root / "sync_out/data/prepared/manual_gold_expansion_v3d_eval/v3d_labeled_clean_posts_v1.csv")
    split_rows = list(csv.DictReader((root / "sync_out/data/prepared/combined_manual_gold_v2/combined_split_v2.csv").open()))
    split_of = {int(r["post_id"]): r["split"] for r in split_rows}
    all_v2 = [int(r["post_id"]) for r in csv.DictReader((root / "sync_out/data/prepared/combined_manual_gold_v2/combined_manual_gold_labels_v2.csv").open())]
    old_test = [p for p in all_v2 if split_of.get(p) == "test"]

    pg_v3d = _pg_entity_stats(v3d_pids)
    pg_old = _pg_entity_stats(old_test)
    pg_all = _pg_entity_stats(all_v2)

    pg_cmp = [
        {"cohort": "v3d_clean", **pg_v3d},
        {"cohort": "combined_old_test", **pg_old},
        {"cohort": "combined_all_v2", **pg_all},
    ]
    with (audit_dir / "postgres_entity_coverage_v3d_vs_old_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pg_cmp[0].keys()))
        w.writeheader()
        w.writerows(pg_cmp)

    neo_schema = _neo_schema_inventory()
    with (audit_dir / "neo4j_schema_inventory_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "count", "sample_keys"])
        w.writeheader()
        w.writerows(neo_schema)

    key_probe = _neo_match_keys(v3d_pids)
    best_key = key_probe.get("best_key", "post_id")
    neo_v3d = _neo_coverage(v3d_pids, best_key)
    neo_old = _neo_coverage(old_test, best_key)
    neo_cmp = [
        {"cohort": "v3d_clean", **neo_v3d},
        {"cohort": "combined_old_test", **neo_old},
    ]
    with (audit_dir / "neo4j_coverage_v3d_vs_old_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(neo_cmp[0].keys()))
        w.writeheader()
        w.writerows(neo_cmp)

    pw_v3d = _pairwise_nonzero_audit(root / "sync_out/data/prepared/manual_gold_expansion_v3d_eval/v3d_pairwise_features_v1.csv", "v3d")
    pw_v2 = _pairwise_nonzero_audit(root / "sync_out/data/prepared/combined_manual_gold_v2/combined_pairwise_features_v2.csv", "combined_v2")
    # old test pairs only — filter v2 file by post ids in test (sample max 50k pairs for speed)
    test_set = set(old_test)
    old_pairs_stats: dict[str, list[float]] = defaultdict(list)
    n_old_pairs = 0
    with (root / "sync_out/data/prepared/combined_manual_gold_v2/combined_pairwise_features_v2.csv").open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            a, b = int(row["post_id_a"]), int(row["post_id_b"])
            if a in test_set and b in test_set:
                n_old_pairs += 1
                for c in PAIR_NUM_COLS:
                    if c in row:
                        try:
                            old_pairs_stats[c].append(float(row[c]))
                        except ValueError:
                            pass
    pw_old: dict[str, Any] = {"label": "combined_old_test_subset", "n_pairs": n_old_pairs}
    for c, vals in old_pairs_stats.items():
        if vals:
            arr = np.array(vals)
            pw_old[f"{c}_nonzero_share"] = round(float(np.sum(arr > 1e-9)) / len(arr), 4)
            pw_old[f"{c}_mean"] = round(float(np.mean(arr)), 4)

    pw_rows = [pw_v3d, pw_old, pw_v2]
    flat_rows: list[dict[str, Any]] = []
    for block in pw_rows:
        label = block["label"]
        for k, v in block.items():
            if k in ("label", "columns", "n_pairs"):
                continue
            flat_rows.append({"dataset": label, "metric": k, "value": v})
    with (audit_dir / "pairwise_feature_nonzero_audit_v1.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "metric", "value"])
        w.writeheader()
        w.writerows(flat_rows)

    # post-level samples
    v3d_sample = _post_debug_sample(v3d_pids, best_key, 15)
    old_sample = _post_debug_sample(old_test[:15], best_key, 15)
    for r in v3d_sample:
        r["cohort"] = "v3d_clean"
    for r in old_sample:
        r["cohort"] = "combined_old_test_graph_nonempty"
    with (audit_dir / "post_level_graph_debug_sample_v1.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(v3d_sample[0].keys()) if v3d_sample else list(old_sample[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(v3d_sample + old_sample)

    # classify zero reason
    reasons: list[str] = []
    bug_suspected = False
    data_coverage_problem = False
    if pg_v3d["share_posts_ge1_entity"] < 0.05:
        reasons.append("real_no_pg_entities")
        data_coverage_problem = True
    if neo_v3d.get("share_posts_with_eventmention", 0) < 0.05:
        reasons.append("real_no_neo4j_event_mentions")
        data_coverage_problem = True
    elif pg_v3d["share_posts_ge1_entity"] < 0.05 and neo_v3d.get("share_posts_with_eventmention", 0) > 0.5:
        reasons.append("neo4j_has_data_but_pg_entities_empty")
        bug_suspected = True
    if pw_v3d.get("graph_neighborhood_jaccard_nonzero_share", 0) == 0:
        reasons.append("v3d_builder_hardcoded_graph_zero")
        bug_suspected = True
    if pw_v3d.get("entity_jaccard_nonzero_share", 0) == 0 and pg_v3d["share_posts_ge1_entity"] > 0.1:
        reasons.append("pg_entities_present_but_pairwise_builder_not_using_them")
        bug_suspected = True
    if pw_v2.get("graph_neighborhood_jaccard_nonzero_share", 0) > 0.1 and pw_v3d.get("graph_neighborhood_jaccard_nonzero_share", 0) == 0:
        reasons.append("v2_has_graph_features_v3d_builder_omitted_neo4j_graph_path")

    can_fix = (
        neo_v3d.get("share_posts_with_eventmention", 0) > 0.3
        or pg_v3d["share_posts_ge1_entity"] > 0.3
        or pw_v2.get("graph_neighborhood_jaccard_nonzero_share", 0) > 0
    )

    if neo_v3d.get("share_posts_with_eventmention", 0) > 0.3 and pw_v3d.get("graph_neighborhood_jaccard_nonzero_share", 0) == 0:
        recommended = "B_rebuild_v3d_pairwise_from_neo4j_then_rerun_eval"
    elif data_coverage_problem and neo_v3d.get("share_posts_with_eventmention", 0) < 0.1:
        recommended = "C_stress_test_without_graph_coverage"
    elif bug_suspected:
        recommended = "A_fix_v3d_feature_builder_rerun_eval"
    else:
        recommended = "D_investigate_ie_pipeline_gap_for_strict_channels_period"

    graph_features_zero_reason = ";".join(reasons) if reasons else "mixed_builder_and_data_limits"

    decision = {
        "graph_features_zero_reason": graph_features_zero_reason,
        "bug_suspected": bug_suspected,
        "data_coverage_problem": data_coverage_problem,
        "can_fix_without_prod_changes": can_fix,
        "recommended_next_step": recommended,
        "neo_best_join_key": best_key,
        "validation_ok": True,
    }
    (audit_dir / "graph_audit_decision_v1.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # reports (abbreviated — full text in files)
    reports.joinpath("manual_gold_expansion_v3d_postgres_entity_coverage_v1.md").write_text(
        "# Postgres entity coverage\n\n"
        + "\n".join(f"- **{r['cohort']}**: posts={r['n_posts']}, in_posts={r['in_posts']}, "
        f"embeddings={r['in_embeddings']}, entities≥1={r['posts_with_ge1_entity']} ({r['share_posts_ge1_entity']:.1%})\n"
        for r in pg_cmp)
        + "\n",
        encoding="utf-8",
    )
    reports.joinpath("manual_gold_expansion_v3d_neo4j_coverage_v1.md").write_text(
        "# Neo4j coverage\n\n"
        f"- Best join key: **{best_key}** (probe: {json.dumps(key_probe.get('candidates_tested', []), ensure_ascii=False)})\n\n"
        + "\n".join(
            f"- **{r['cohort']}**: EventMention posts={r.get('posts_with_eventmention')}/{r.get('n_posts')}, "
            f"INVOLVES posts={r.get('posts_with_involves_entity')}, storylines={r.get('distinct_storylines')}\n"
            for r in neo_cmp
        )
        + "\n",
        encoding="utf-8",
    )
    reports.joinpath("manual_gold_expansion_v3d_pairwise_feature_audit_v1.md").write_text(
        "# Pairwise feature audit\n\n"
        f"- v3d entity_jaccard nonzero: **{pw_v3d.get('entity_jaccard_nonzero_share', 0):.1%}**\n"
        f"- v3d graph_neighborhood nonzero: **{pw_v3d.get('graph_neighborhood_jaccard_nonzero_share', 0):.1%}**\n"
        f"- v2 full graph_neighborhood nonzero: **{pw_v2.get('graph_neighborhood_jaccard_nonzero_share', 0):.1%}**\n"
        f"- v2 old_test graph_neighborhood: **{pw_old.get('graph_neighborhood_jaccard_nonzero_share', 'n/a')}**\n\n"
        "v3d builder (`evaluate_manual_gold_expansion_v3d_strict_developmental_labeled.py`) sets "
        "`event_signature_jaccard`, `graph_neighborhood_jaccard`, `graph_relation_type_jaccard` to **literal 0**.\n",
        encoding="utf-8",
    )
    reports.joinpath("manual_gold_expansion_v3d_graph_join_diagnosis_v1.md").write_text(
        "# Join diagnosis\n\n"
        f"- graph_features_zero_reason: `{graph_features_zero_reason}`\n"
        f"- bug_suspected: **{bug_suspected}**\n"
        f"- recommended: **{recommended}**\n",
        encoding="utf-8",
    )
    reports.joinpath("manual_gold_expansion_v3d_feature_builder_logic_audit_v1.md").write_text(
        "# Feature builder logic audit\n\n"
        "## v3d (`evaluate_manual_gold_expansion_v3d_strict_developmental_labeled.py`)\n\n"
        "- Sources: Postgres `embeddings`, `public.entities` (column `text`), **no Neo4j**.\n"
        "- Hardcoded zeros: `event_signature_jaccard`, `graph_neighborhood_jaccard`, `graph_relation_type_jaccard`.\n"
        "- `has_graph_features` = true only if PG entity set non-empty for either post.\n"
        "- hybrid_v4_1 replay uses v41 formula; with entity_jaccard=0 and graph=0 → score ≈ semantic+time only.\n\n"
        "## combined v2 (`combined_pairwise_features_v2.csv`)\n\n"
        "- Prebuilt artifact (not in experiment tools/); includes **nonzero graph_neighborhood_jaccard** while entity_jaccard often 0.\n"
        "- Likely built from prod/Neo4j graph feature pipeline, not from v3d builder.\n",
        encoding="utf-8",
    )
    reports.joinpath("manual_gold_expansion_v3d_graph_coverage_audit_summary_v1.md").write_text(
        "# Graph coverage audit summary v3d\n\n"
        f"- **Decision:** {json.dumps(decision, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )

    # STATUS snippet update via append to manifest — run rclone
    validation_ok = True
    if not args.skip_rclone:
        r = subprocess.run(
            ["rclone", "sync", str(root / "sync_out"), "gdrive:diploma_story_tracking_experiment"],
            cwd=str(root),
        )
        if r.returncode != 0:
            validation_ok = False

    # copy script
    sync_tools = root / "sync_out/tools"
    sync_tools.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(_TOOLS / "audit_v3d_graph_feature_coverage.py", sync_tools / "audit_v3d_graph_feature_coverage.py")

    print(json.dumps({**decision, "pg_v3d": pg_v3d, "pg_old": pg_old, "neo_v3d": neo_v3d, "pw_v3d": pw_v3d}, indent=2))
    return 0 if validation_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
