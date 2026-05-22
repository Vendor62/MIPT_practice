#!/usr/bin/env python3
"""Train a lightweight collaborative-filtering user clustering model.

The script is intentionally offline-only: it reads reactions/clicks from a
Postgres snapshot, builds a sparse user-post matrix, trains TruncatedSVD +
MiniBatchKMeans, and writes CSV/JSON artifacts that can later be loaded into
runtime tables or used for shadow analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


REACTION_WEIGHTS = {
    "fire": 3.0,
    "up": 2.0,
    "down": -2.5,
}
CLICK_WEIGHT = 1.0


@dataclass(frozen=True, slots=True)
class Interaction:
    user_id: int
    post_id: int
    score: float
    positive: bool
    negative: bool
    source: str


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _connect(args: argparse.Namespace):
    return psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )


def _fetch_interactions(args: argparse.Namespace) -> list[Interaction]:
    interactions: dict[tuple[int, int], dict[str, float | bool]] = defaultdict(
        lambda: {"score": 0.0, "positive": False, "negative": False, "click": False}
    )
    with _connect(args) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, post_id, lower(coalesce(reaction, ''))
                FROM reactions
                WHERE user_id IS NOT NULL
                  AND post_id IS NOT NULL
                  AND reaction IS NOT NULL
                """
            )
            for user_id, post_id, reaction in cur.fetchall():
                reaction_key = str(reaction or "").strip().lower()
                weight = REACTION_WEIGHTS.get(reaction_key)
                if weight is None:
                    continue
                key = (int(user_id), int(post_id))
                interactions[key]["score"] = float(interactions[key]["score"]) + weight
                if weight > 0:
                    interactions[key]["positive"] = True
                elif weight < 0:
                    interactions[key]["negative"] = True

            cur.execute(
                """
                SELECT user_id, post_id, coalesce(click_count, 1)
                FROM clicks
                WHERE user_id IS NOT NULL
                  AND post_id IS NOT NULL
                """
            )
            for user_id, post_id, click_count in cur.fetchall():
                key = (int(user_id), int(post_id))
                clicks = max(1.0, min(3.0, float(click_count or 1)))
                interactions[key]["score"] = float(interactions[key]["score"]) + (CLICK_WEIGHT * clicks)
                interactions[key]["positive"] = True
                interactions[key]["click"] = True

    rows: list[Interaction] = []
    for (user_id, post_id), payload in interactions.items():
        score = max(-3.0, min(4.0, float(payload["score"])))
        rows.append(
            Interaction(
                user_id=user_id,
                post_id=post_id,
                score=score,
                positive=bool(payload["positive"]) and score > 0.0,
                negative=bool(payload["negative"]) and score < 0.0,
                source="mixed" if payload["click"] else "reaction",
            )
        )
    return rows


def _filter_interactions(
    interactions: list[Interaction],
    *,
    min_user_events: int,
    min_post_events: int,
) -> list[Interaction]:
    user_counts = Counter(row.user_id for row in interactions)
    post_counts = Counter(row.post_id for row in interactions)
    return [
        row
        for row in interactions
        if user_counts[row.user_id] >= min_user_events and post_counts[row.post_id] >= min_post_events
    ]


def _auto_clusters(active_users: int) -> int:
    if active_users <= 2:
        return max(1, active_users)
    return max(2, min(32, int(round(math.sqrt(active_users) * 1.5))))


def _build_matrix(interactions: list[Interaction]) -> tuple[csr_matrix, list[int], list[int]]:
    user_ids = sorted({row.user_id for row in interactions})
    post_ids = sorted({row.post_id for row in interactions})
    user_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    post_idx = {post_id: idx for idx, post_id in enumerate(post_ids)}

    rows = [user_idx[row.user_id] for row in interactions]
    cols = [post_idx[row.post_id] for row in interactions]
    data = [row.score for row in interactions]
    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(post_ids)), dtype=np.float32)
    return matrix, user_ids, post_ids


def _train_latents(matrix: csr_matrix, requested_components: int, random_state: int) -> tuple[np.ndarray, dict]:
    max_components = max(1, min(matrix.shape[0] - 1, matrix.shape[1] - 1, requested_components))
    if max_components < 2:
        dense = matrix.toarray().astype(np.float32)
        return normalize(dense), {"method": "raw_normalized", "components": int(dense.shape[1])}

    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    latent = svd.fit_transform(matrix)
    latent = normalize(latent)
    explained = float(np.sum(svd.explained_variance_ratio_))
    return latent, {
        "method": "truncated_svd",
        "components": int(max_components),
        "explained_variance_ratio_sum": round(explained, 6),
    }


def _cluster_users(latent: np.ndarray, requested_clusters: int | None, random_state: int) -> np.ndarray:
    n_users = latent.shape[0]
    n_clusters = requested_clusters or _auto_clusters(n_users)
    n_clusters = max(1, min(int(n_clusters), n_users))
    if n_clusters == 1:
        return np.zeros(n_users, dtype=np.int32)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
        batch_size=max(32, min(256, n_users)),
    )
    return model.fit_predict(latent).astype(np.int32)


def _cluster_affinities(
    interactions: list[Interaction],
    user_clusters: dict[int, int],
    cluster_sizes: dict[int, int],
    *,
    prior_strength: float,
) -> list[dict[str, float | int]]:
    global_positive_users_by_post: dict[int, set[int]] = defaultdict(set)
    for row in interactions:
        if row.positive:
            global_positive_users_by_post[row.post_id].add(row.user_id)

    active_users_total = max(1, len(set(user_clusters)))
    post_cluster: dict[tuple[int, int], dict[str, set[int]]] = defaultdict(
        lambda: {"positive": set(), "negative": set(), "interacted": set()}
    )
    for row in interactions:
        cluster_id = user_clusters.get(row.user_id)
        if cluster_id is None:
            continue
        key = (cluster_id, row.post_id)
        post_cluster[key]["interacted"].add(row.user_id)
        if row.positive:
            post_cluster[key]["positive"].add(row.user_id)
        if row.negative:
            post_cluster[key]["negative"].add(row.user_id)

    rows: list[dict[str, float | int]] = []
    for (cluster_id, post_id), payload in sorted(post_cluster.items()):
        cluster_size = max(1, int(cluster_sizes.get(cluster_id, 0)))
        positive_count = len(payload["positive"])
        negative_count = len(payload["negative"])
        interacted_count = len(payload["interacted"])
        global_prior = len(global_positive_users_by_post.get(post_id, set())) / active_users_total
        cluster_positive_user_ratio = positive_count / cluster_size
        interacted_positive_ratio = positive_count / interacted_count if interacted_count else 0.0
        affinity = (positive_count + prior_strength * global_prior) / (cluster_size + prior_strength)
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "post_id": int(post_id),
                "cluster_size": int(cluster_size),
                "positive_count": int(positive_count),
                "negative_count": int(negative_count),
                "interacted_count": int(interacted_count),
                "cluster_positive_user_ratio": round(float(cluster_positive_user_ratio), 6),
                "interacted_positive_ratio": round(float(interacted_positive_ratio), 6),
                "affinity": round(float(affinity), 6),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> dict:
    raw_interactions = _fetch_interactions(args)
    interactions = _filter_interactions(
        raw_interactions,
        min_user_events=args.min_user_events,
        min_post_events=args.min_post_events,
    )
    if len(interactions) < 2:
        raise RuntimeError("Not enough interactions after filtering.")

    matrix, user_ids, post_ids = _build_matrix(interactions)
    latent, latent_meta = _train_latents(matrix, args.components, args.random_state)
    labels = _cluster_users(latent, args.clusters, args.random_state)

    user_clusters = {user_id: int(label) for user_id, label in zip(user_ids, labels, strict=True)}
    cluster_sizes = Counter(user_clusters.values())
    affinity_rows = _cluster_affinities(
        interactions,
        user_clusters,
        dict(cluster_sizes),
        prior_strength=args.prior_strength,
    )

    run_dir = Path(args.out_dir) / f"user_cf_clusters_{_utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    user_rows = [
        {
            "user_id": int(user_id),
            "cluster_id": int(user_clusters[user_id]),
            "events_count": int(sum(1 for row in interactions if row.user_id == user_id)),
        }
        for user_id in user_ids
    ]
    cluster_rows = [
        {
            "cluster_id": int(cluster_id),
            "users_count": int(size),
            "events_count": int(sum(1 for row in interactions if user_clusters.get(row.user_id) == cluster_id)),
            "positive_events": int(
                sum(1 for row in interactions if user_clusters.get(row.user_id) == cluster_id and row.positive)
            ),
            "negative_events": int(
                sum(1 for row in interactions if user_clusters.get(row.user_id) == cluster_id and row.negative)
            ),
        }
        for cluster_id, size in sorted(cluster_sizes.items())
    ]

    _write_csv(run_dir / "user_clusters.csv", user_rows)
    _write_csv(run_dir / "cluster_post_affinities.csv", affinity_rows)
    _write_csv(run_dir / "cluster_summary.csv", cluster_rows)
    np.save(run_dir / "user_latents.npy", latent)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "db_name": args.db_name,
        "raw_interactions": len(raw_interactions),
        "interactions": len(interactions),
        "active_users": len(user_ids),
        "active_posts": len(post_ids),
        "matrix_density": round(float(matrix.nnz / max(1, matrix.shape[0] * matrix.shape[1])), 8),
        "reaction_weights": REACTION_WEIGHTS,
        "click_weight": CLICK_WEIGHT,
        "min_user_events": args.min_user_events,
        "min_post_events": args.min_post_events,
        "clusters": len(cluster_sizes),
        "cluster_sizes": {str(k): int(v) for k, v in sorted(cluster_sizes.items())},
        "latent": latent_meta,
        "affinity_rows": len(affinity_rows),
        "affinity_prior_strength": args.prior_strength,
        "outputs": {
            "user_clusters": str(run_dir / "user_clusters.csv"),
            "cluster_post_affinities": str(run_dir / "cluster_post_affinities.csv"),
            "cluster_summary": str(run_dir / "cluster_summary.csv"),
            "user_latents": str(run_dir / "user_latents.npy"),
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **metadata}, ensure_ascii=False, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", "db"))
    parser.add_argument("--db-port", default=os.getenv("DB_PORT", "5432"))
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "newshub_prod_snapshot"))
    parser.add_argument("--db-user", default=os.getenv("DB_USER", "newshub"))
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD", "newshub"))
    parser.add_argument("--out-dir", default="artifacts/model_runs")
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--min-user-events", type=int, default=2)
    parser.add_argument("--min-post-events", type=int, default=1)
    parser.add_argument("--prior-strength", type=float, default=3.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
