import os
import time
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph_gateway import graph_gateway
from app.models import Embedding


log = structlog.get_logger()

RECO_EMBEDDING_SOURCE = os.getenv("RECO_EMBEDDING_SOURCE", "sql").strip().lower()


def _to_vector(raw: Any) -> list[float]:
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for item in raw:
        try:
            out.append(float(item))
        except Exception:
            return []
    return out


async def _get_sql_post_embedding(session: AsyncSession, post_id: int) -> list[float]:
    res = await session.execute(select(Embedding.embedding).where(Embedding.post_id == post_id))
    return _to_vector(res.scalar_one_or_none())


async def _get_neo4j_post_embedding(post_id: int) -> list[float]:
    query = """
    MATCH (p:Post {post_id: $post_id})-[:DESCRIBES]->(e:Event)
    WHERE e.embedding IS NOT NULL
    RETURN e.embedding AS embedding
    ORDER BY coalesce(e.recorded_at, e.published_at, '') DESC
    LIMIT 1
    """
    rows = await graph_gateway.run_read_query(query, post_id=int(post_id))
    if not rows:
        return []
    return _to_vector(rows[0].get("embedding"))


async def get_post_embeddings_dual_probe(
    session: AsyncSession,
    post_id: int,
) -> tuple[list[float], list[float]]:
    sql_emb = await _get_sql_post_embedding(session, post_id)
    neo4j_emb = await _get_neo4j_post_embedding(post_id)
    return sql_emb, neo4j_emb


async def get_post_embedding(
    session: AsyncSession,
    post_id: int,
    *,
    source: str | None = None,
) -> list[float]:
    mode = (source or RECO_EMBEDDING_SOURCE or "sql").strip().lower()

    if mode == "neo4j":
        started = time.perf_counter()
        emb = await _get_neo4j_post_embedding(post_id)
        neo4j_ms = round((time.perf_counter() - started) * 1000.0, 2)
        log.info(
            "reco_embedding_fetch",
            mode=mode,
            post_id=post_id,
            selected_source="neo4j" if emb else "sql_fallback",
            selected_dim=len(emb),
            neo4j_ms=neo4j_ms,
        )
        if emb:
            return emb
        started = time.perf_counter()
        sql_emb = await _get_sql_post_embedding(session, post_id)
        sql_ms = round((time.perf_counter() - started) * 1000.0, 2)
        log.info(
            "reco_embedding_fetch",
            mode=mode,
            post_id=post_id,
            selected_source="sql_fallback",
            selected_dim=len(sql_emb),
            sql_ms=sql_ms,
            neo4j_ms=neo4j_ms,
        )
        return sql_emb

    if mode == "dual":
        sql_started = time.perf_counter()
        sql_emb = await _get_sql_post_embedding(session, post_id)
        sql_ms = round((time.perf_counter() - sql_started) * 1000.0, 2)
        neo4j_started = time.perf_counter()
        neo4j_emb = await _get_neo4j_post_embedding(post_id)
        neo4j_ms = round((time.perf_counter() - neo4j_started) * 1000.0, 2)
        selected_source = "sql" if sql_emb else ("neo4j" if neo4j_emb else "none")
        selected_emb = sql_emb if sql_emb else neo4j_emb
        log.info(
            "reco_embedding_dual_probe",
            post_id=post_id,
            sql_dim=len(sql_emb),
            neo4j_dim=len(neo4j_emb),
            sql_present=bool(sql_emb),
            neo4j_present=bool(neo4j_emb),
            selected_source=selected_source,
            selected_dim=len(selected_emb),
            sql_ms=sql_ms,
            neo4j_ms=neo4j_ms,
        )
        return selected_emb

    started = time.perf_counter()
    sql_emb = await _get_sql_post_embedding(session, post_id)
    sql_ms = round((time.perf_counter() - started) * 1000.0, 2)
    log.info(
        "reco_embedding_fetch",
        mode=mode,
        post_id=post_id,
        selected_source="sql",
        selected_dim=len(sql_emb),
        sql_ms=sql_ms,
    )
    return sql_emb
