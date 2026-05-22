from __future__ import annotations

import re
from typing import Any

import structlog

from app.graph_gateway import graph_gateway


log = structlog.get_logger()


_RECOVERY_STOPWORDS = {
    "что", "это", "как", "или", "для", "при", "над", "под", "после", "перед", "если", "только",
    "когда", "пока", "очень", "снова", "также", "между", "через", "этого", "этой", "этом", "этот",
    "который", "которая", "которые", "уже", "были", "было", "будет", "сегодня", "завтра", "вчера",
    "there", "their", "about", "after", "before", "while", "from", "with", "into", "been", "have",
    "has", "had", "that", "this", "those", "these", "over", "under", "than", "then", "they",
    "them", "were", "will", "would", "could", "should",
    "россия", "россии", "россию", "сша", "израиль", "иран", "дагестан", "санкт", "петербурге", "питере",
}


def _normalize_storyline_text(value: str | None) -> str:
    text = str(value or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^\w\sа-яё-]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _storyline_tokens(value: str | None) -> list[str]:
    tokens = re.findall(r"[a-zа-яё0-9-]{4,}", _normalize_storyline_text(value), flags=re.IGNORECASE)
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        tkn = token.strip("-")
        if len(tkn) < 4 or tkn in _RECOVERY_STOPWORDS or tkn in seen:
            continue
        seen.add(tkn)
        out.append(tkn)
    return out


def _render_storyline_context_query(match_clause: str) -> str:
    return f"""
    {match_clause}
    OPTIONAL MATCH (s)-[:BELONGS_TO]->(sf:StoryFamily)
    OPTIONAL MATCH (s)-[:BELONGS_TO_TOPIC]->(mt:MacroTopic)
    OPTIONAL MATCH (s)-[:BELONGS_TO_EPISODE]->(ep:Episode)
    CALL {{
        WITH s, sf
        OPTIONAL MATCH (sf)<-[:BELONGS_TO]-(sib:Storyline)
        RETURN collect(DISTINCT sib) AS family_siblings
    }}
    CALL {{
        WITH s, mt
        OPTIONAL MATCH (mt)<-[:BELONGS_TO_TOPIC]-(topic_sib:Storyline)
        RETURN collect(DISTINCT topic_sib) AS topic_siblings
    }}
    WITH s, sf, mt, ep, family_siblings + topic_siblings AS siblings
    RETURN
        s.id AS storyline_id,
        coalesce(s.title, '') AS storyline_title,
        coalesce(s.seed_preview, '') AS storyline_seed_preview,
        coalesce(sf.id, '') AS story_family_id,
        coalesce(sf.root_storyline_id, s.id) AS family_root_storyline_id,
        coalesce(mt.id, coalesce(s.macro_topic_id, '')) AS macro_topic_id,
        coalesce(mt.canonical_title, coalesce(s.macro_topic_title, '')) AS macro_topic_title,
        coalesce(ep.id, coalesce(s.story_episode_id, '')) AS story_episode_id,
        coalesce(ep.canonical_title, coalesce(s.story_episode_title, '')) AS story_episode_title,
        coalesce(mt.topic_signatures, []) AS macro_topic_signatures,
        coalesce(s.entity_keys, []) AS entity_keys,
        coalesce(s.family_signatures, []) AS family_signatures,
        coalesce(s.topic_signatures, []) AS topic_signatures,
        coalesce(sf.topic_signatures, []) AS family_topic_signatures,
        coalesce(s.posts_count, 0) AS posts_count,
        [sib IN siblings
         WHERE sib IS NOT NULL AND sib.id <> s.id |
            {{
                storyline_id: sib.id,
                title: coalesce(sib.title, ''),
                seed_preview: coalesce(sib.seed_preview, ''),
                posts_count: coalesce(sib.posts_count, 0)
            }}
        ][0..5] AS sibling_candidates
    LIMIT 1
    """


async def get_storyline_context_for_post(post_id: int) -> dict[str, Any] | None:
    query = _render_storyline_context_query("MATCH (p:Post {post_id: $post_id})-[:SOURCE_OF]->(s:Storyline)")
    try:
        rows = await graph_gateway.run_read_query(query, post_id=int(post_id))
    except Exception:
        log.exception("storyline_context_lookup_failed", post_id=post_id)
        return None
    return rows[0] if rows else None


async def get_storyline_context_for_storyline_id(storyline_id: str) -> dict[str, Any] | None:
    storyline_id = str(storyline_id or "").strip()
    if not storyline_id:
        return None
    query = _render_storyline_context_query("MATCH (s:Storyline {id: $storyline_id})")
    try:
        rows = await graph_gateway.run_read_query(query, storyline_id=storyline_id)
    except Exception:
        log.exception("storyline_context_by_id_lookup_failed", storyline_id=storyline_id)
        return None
    return rows[0] if rows else None


async def get_update_step_for_post(post_id: int) -> dict[str, Any] | None:
    query = """
    MATCH (u:UpdateStep)
    WHERE $post_id IN coalesce(u.source_post_ids, [])
    OPTIONAL MATCH (u)-[:PART_OF_EPISODE]->(e:Episode)
    OPTIONAL MATCH (e)-[:PART_OF_TOPIC]->(episode_topic:MacroTopic)
    OPTIONAL MATCH (u)-[:PART_OF_TOPIC]->(direct_topic:MacroTopic)
    WITH u, e, coalesce(episode_topic, direct_topic) AS mt
    RETURN
        u.id AS step_id,
        coalesce(u.canonical_summary, '') AS canonical_summary,
        coalesce(u.event_signature, '') AS event_signature,
        coalesce(u.anchor_entities, []) AS anchor_entities,
        coalesce(u.source_post_ids, []) AS source_post_ids,
        coalesce(u.storyline_ids, []) AS storyline_ids,
        coalesce(u.first_seen_at, '') AS first_seen_at,
        coalesce(u.last_seen_at, '') AS last_seen_at,
        coalesce(e.id, '') AS episode_id,
        coalesce(e.canonical_title, '') AS episode_title,
        coalesce(mt.id, '') AS macro_topic_id,
        coalesce(mt.canonical_title, '') AS macro_topic_title
    ORDER BY size(coalesce(u.source_post_ids, [])) DESC, coalesce(u.last_seen_at, '') DESC
    LIMIT 1
    """
    try:
        rows = await graph_gateway.run_read_query(query, post_id=int(post_id))
    except Exception:
        log.exception("update_step_for_post_lookup_failed", post_id=post_id)
        return None
    return rows[0] if rows else None


async def get_macro_topic_timeline_steps(macro_topic_id: str, *, limit: int = 120) -> list[dict[str, Any]]:
    macro_topic_id = str(macro_topic_id or "").strip()
    if not macro_topic_id:
        return []
    query = """
    CALL {
        MATCH (t:MacroTopic {id: $macro_topic_id})
        MATCH (t)<-[:PART_OF_TOPIC]-(e:Episode)<-[:PART_OF_EPISODE]-(u:UpdateStep)
        RETURN
            coalesce(e.id, '') AS episode_id,
            coalesce(e.canonical_title, '') AS episode_title,
            coalesce(e.phase_type, '') AS episode_phase_type,
            coalesce(e.first_seen_at, '') AS episode_first_seen_at,
            coalesce(e.last_seen_at, '') AS episode_last_seen_at,
            u.id AS step_id,
            coalesce(u.canonical_summary, '') AS canonical_summary,
            coalesce(u.first_seen_at, '') AS first_seen_at,
            coalesce(u.last_seen_at, '') AS last_seen_at,
            coalesce(u.event_signature, '') AS event_signature,
            coalesce(u.anchor_entities, []) AS anchor_entities,
            coalesce(u.source_post_ids, []) AS source_post_ids,
            coalesce(u.storyline_ids, []) AS storyline_ids
        UNION
        MATCH (t:MacroTopic {id: $macro_topic_id})<-[:PART_OF_TOPIC]-(u:UpdateStep)
        WHERE NOT EXISTS { (u)-[:PART_OF_EPISODE]->(:Episode) }
        RETURN
            '' AS episode_id,
            '' AS episode_title,
            '' AS episode_phase_type,
            '' AS episode_first_seen_at,
            '' AS episode_last_seen_at,
            u.id AS step_id,
            coalesce(u.canonical_summary, '') AS canonical_summary,
            coalesce(u.first_seen_at, '') AS first_seen_at,
            coalesce(u.last_seen_at, '') AS last_seen_at,
            coalesce(u.event_signature, '') AS event_signature,
            coalesce(u.anchor_entities, []) AS anchor_entities,
            coalesce(u.source_post_ids, []) AS source_post_ids,
            coalesce(u.storyline_ids, []) AS storyline_ids
    }
    RETURN *
    ORDER BY coalesce(episode_first_seen_at, ''), coalesce(first_seen_at, ''), step_id
    LIMIT $limit
    """
    try:
        return await graph_gateway.run_read_query(
            query,
            macro_topic_id=macro_topic_id,
            limit=max(1, int(limit)),
        )
    except Exception:
        log.exception("macro_topic_timeline_steps_lookup_failed", macro_topic_id=macro_topic_id)
        return []


async def get_storyline_cluster_timeline_steps(
    storyline_ids: list[str],
    *,
    limit: int = 240,
) -> list[dict[str, Any]]:
    normalized_storyline_ids = [
        str(storyline_id or "").strip()
        for storyline_id in (storyline_ids or [])
        if str(storyline_id or "").strip()
    ][:48]
    if not normalized_storyline_ids:
        return []

    query = """
    MATCH (u:UpdateStep)
    WHERE any(storyline_id IN coalesce(u.storyline_ids, []) WHERE storyline_id IN $storyline_ids)
    OPTIONAL MATCH (u)-[:PART_OF_EPISODE]->(e:Episode)
    OPTIONAL MATCH (e)-[:PART_OF_TOPIC]->(episode_topic:MacroTopic)
    OPTIONAL MATCH (u)-[:PART_OF_TOPIC]->(direct_topic:MacroTopic)
    WITH u, e, coalesce(episode_topic, direct_topic) AS mt
    RETURN
        coalesce(e.id, '') AS episode_id,
        coalesce(e.canonical_title, '') AS episode_title,
        coalesce(e.phase_type, '') AS episode_phase_type,
        coalesce(e.first_seen_at, '') AS episode_first_seen_at,
        coalesce(e.last_seen_at, '') AS episode_last_seen_at,
        u.id AS step_id,
        coalesce(u.canonical_summary, '') AS canonical_summary,
        coalesce(u.first_seen_at, '') AS first_seen_at,
        coalesce(u.last_seen_at, '') AS last_seen_at,
        coalesce(u.event_signature, '') AS event_signature,
        coalesce(u.anchor_entities, []) AS anchor_entities,
        coalesce(u.source_post_ids, []) AS source_post_ids,
        coalesce(u.storyline_ids, []) AS storyline_ids,
        coalesce(mt.id, '') AS macro_topic_id,
        coalesce(mt.canonical_title, '') AS macro_topic_title
    ORDER BY coalesce(e.first_seen_at, ''), coalesce(u.first_seen_at, ''), step_id
    LIMIT $limit
    """
    try:
        return await graph_gateway.run_read_query(
            query,
            storyline_ids=normalized_storyline_ids,
            limit=max(1, int(limit)),
        )
    except Exception:
        log.exception(
            "storyline_cluster_timeline_steps_lookup_failed",
            storyline_count=len(normalized_storyline_ids),
            limit=limit,
        )
        return []


async def search_storyline_candidates(query_text: str, limit: int = 8) -> list[dict[str, Any]]:
    q = " ".join(str(query_text or "").strip().lower().split())
    if not q:
        return []
    query = """
    MATCH (s:Storyline)
    OPTIONAL MATCH (s)-[:BELONGS_TO]->(f:StoryFamily)
    OPTIONAL MATCH (s)-[:BELONGS_TO_TOPIC]->(t:MacroTopic)
    WITH s, f, t, toLower(
        coalesce(s.title, '') + ' '
        + coalesce(s.seed_preview, '') + ' '
        + reduce(acc = '', item IN coalesce(t.topic_aliases, []) | acc + ' ' + toLower(item))
    ) AS hay
    WHERE hay CONTAINS $q
       OR any(k IN coalesce(s.entity_keys, []) WHERE toLower(k) CONTAINS $q)
       OR any(sig IN coalesce(s.family_signatures, []) WHERE toLower(sig) CONTAINS $q)
       OR any(sig IN coalesce(s.topic_signatures, []) WHERE toLower(sig) CONTAINS $q)
    RETURN
        s.id AS storyline_id,
        coalesce(f.id, '') AS story_family_id,
        coalesce(f.root_storyline_id, s.id) AS family_root_storyline_id,
        coalesce(t.id, coalesce(s.macro_topic_id, '')) AS macro_topic_id,
        coalesce(t.canonical_title, coalesce(s.macro_topic_title, '')) AS macro_topic_title,
        coalesce(s.title, '') AS title,
        coalesce(s.seed_preview, '') AS seed_preview,
        coalesce(s.posts_count, 0) AS posts_count,
        coalesce(s.updated_at, '') AS updated_at
    ORDER BY coalesce(s.posts_count, 0) DESC, coalesce(s.updated_at, '') DESC
    LIMIT $limit
    """
    try:
        return await graph_gateway.run_read_query(query, q=q, limit=max(1, int(limit)))
    except Exception:
        log.exception("storyline_search_failed", query_text=q, limit=limit)
        return []


async def search_storyline_branch_candidates(
    *,
    query_tokens: list[str],
    limit: int = 12,
    min_match_count: int = 2,
    exclude_storyline_id: str | None = None,
) -> list[dict[str, Any]]:
    normalized_query_tokens = [
        str(token or "").strip().lower()
        for token in (query_tokens or [])
        if len(str(token or "").strip()) >= 4
    ][:12]
    if not normalized_query_tokens:
        return []

    query = """
    MATCH (cand:Storyline)
    OPTIONAL MATCH (cand)-[:BELONGS_TO]->(cf:StoryFamily)
    OPTIONAL MATCH (cand)-[:BELONGS_TO_TOPIC]->(ct:MacroTopic)
    WITH cand, cf, ct, $query_tokens AS query_tokens,
         toLower(
            coalesce(cand.title, '') + ' '
            + coalesce(cand.seed_preview, '') + ' '
            + reduce(acc = '', item IN coalesce(cand.entity_keys, []) | acc + ' ' + toLower(item))
            + ' '
            + reduce(acc = '', item IN coalesce(cand.family_signatures, []) | acc + ' ' + toLower(item))
            + ' '
            + reduce(acc = '', item IN coalesce(cand.topic_signatures, []) | acc + ' ' + toLower(item))
            + ' '
            + reduce(acc = '', item IN coalesce(ct.topic_aliases, []) | acc + ' ' + toLower(item))
         ) AS hay
    WITH cand, cf, ct, [tok IN query_tokens WHERE hay CONTAINS tok] AS matched_tokens
    WITH cand, cf, ct, matched_tokens, size(matched_tokens) AS match_count
    WHERE match_count >= $min_match_count
      AND ($exclude_storyline_id = '' OR cand.id <> $exclude_storyline_id)
    RETURN
        cand.id AS storyline_id,
        coalesce(cf.id, '') AS story_family_id,
        coalesce(cf.root_storyline_id, cand.id) AS family_root_storyline_id,
        coalesce(ct.id, coalesce(cand.macro_topic_id, '')) AS macro_topic_id,
        coalesce(ct.canonical_title, coalesce(cand.macro_topic_title, '')) AS macro_topic_title,
        coalesce(cand.title, '') AS title,
        coalesce(cand.seed_preview, '') AS seed_preview,
        coalesce(cand.posts_count, 0) AS posts_count,
        matched_tokens,
        match_count
    ORDER BY match_count DESC, coalesce(cand.posts_count, 0) DESC, coalesce(cand.updated_at, '') DESC
    LIMIT $limit
    """
    try:
        return await graph_gateway.run_read_query(
            query,
            query_tokens=normalized_query_tokens,
            limit=max(1, int(limit)),
            min_match_count=max(1, int(min_match_count)),
            exclude_storyline_id=str(exclude_storyline_id or "").strip(),
        )
    except Exception:
        log.exception("storyline_branch_search_failed", limit=limit, exclude_storyline_id=exclude_storyline_id)
        return []


async def search_storyline_structural_candidates(
    *,
    anchor_storyline_id: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    storyline_id = str(anchor_storyline_id or "").strip()
    if not storyline_id:
        return []

    query = """
    MATCH (anchor:Storyline {id: $anchor_storyline_id})
    OPTIONAL MATCH (anchor)-[:BELONGS_TO]->(af:StoryFamily)
    OPTIONAL MATCH (anchor)-[:BELONGS_TO_TOPIC]->(at:MacroTopic)
    WITH anchor, af, at,
         coalesce(anchor.entity_keys, []) AS anchor_entity_keys,
         coalesce(anchor.family_signatures, []) AS anchor_family_signatures,
         (coalesce(anchor.topic_signatures, []) + coalesce(af.topic_signatures, []) + coalesce(anchor.family_topic_signatures, [])) AS anchor_topic_signatures
    MATCH (cand:Storyline)
    OPTIONAL MATCH (cand)-[:BELONGS_TO]->(cf:StoryFamily)
    OPTIONAL MATCH (cand)-[:BELONGS_TO_TOPIC]->(ct:MacroTopic)
    WITH anchor, af, at, cand, cf, ct,
         size([k IN coalesce(cand.entity_keys, []) WHERE k IN anchor_entity_keys]) AS shared_entity_count,
         size([sig IN coalesce(cand.family_signatures, []) WHERE sig IN anchor_family_signatures]) AS shared_signature_count,
         size([sig IN coalesce(cand.topic_signatures, []) WHERE sig IN anchor_topic_signatures]) AS shared_topic_count,
         CASE
            WHEN at IS NOT NULL AND ct IS NOT NULL AND ct.id = at.id THEN 1
            ELSE 0
         END AS same_macro_topic,
         CASE
            WHEN af IS NOT NULL AND coalesce(cf.root_storyline_id, cand.id) = coalesce(af.root_storyline_id, anchor.id) THEN 1
            ELSE 0
         END AS family_match
    WITH cand,
         head(collect(DISTINCT cf)) AS cf,
         head(collect(DISTINCT ct)) AS ct,
         max(shared_entity_count) AS shared_entity_count,
         max(shared_signature_count) AS shared_signature_count,
         max(shared_topic_count) AS shared_topic_count,
         max(same_macro_topic) AS same_macro_topic,
         max(family_match) AS family_match
    WITH cand, cf, ct, shared_entity_count, shared_signature_count, shared_topic_count, same_macro_topic, family_match,
         (
            same_macro_topic * 10
            + family_match * 6
            + shared_entity_count * 4
            + shared_topic_count * 3
            + shared_signature_count
         ) AS structural_score
    WHERE cand.id <> $anchor_storyline_id
      AND (
            same_macro_topic > 0
            OR family_match > 0
            OR shared_entity_count >= 1
            OR shared_topic_count >= 1
            OR shared_signature_count >= 3
      )
      AND structural_score >= 3
    RETURN
        cand.id AS storyline_id,
        coalesce(cf.id, '') AS story_family_id,
        coalesce(cf.root_storyline_id, cand.id) AS family_root_storyline_id,
        coalesce(ct.id, coalesce(cand.macro_topic_id, '')) AS macro_topic_id,
        coalesce(ct.canonical_title, coalesce(cand.macro_topic_title, '')) AS macro_topic_title,
        coalesce(cand.title, '') AS title,
        coalesce(cand.seed_preview, '') AS seed_preview,
        coalesce(cand.posts_count, 0) AS posts_count,
        shared_entity_count,
        shared_signature_count,
        shared_topic_count,
        same_macro_topic,
        family_match,
        structural_score
    ORDER BY structural_score DESC, coalesce(cand.posts_count, 0) DESC, coalesce(cand.updated_at, '') DESC
    LIMIT $limit
    """
    try:
        return await graph_gateway.run_read_query(
            query,
            anchor_storyline_id=storyline_id,
            limit=max(1, int(limit)),
        )
    except Exception:
        log.exception("storyline_structural_search_failed", anchor_storyline_id=storyline_id, limit=limit)
        return []


async def search_storyline_recovery_candidates(
    post_text: str,
    *,
    query_tokens: list[str] | None = None,
    limit: int = 8,
    exclude_storyline_id: str | None = None,
) -> list[dict[str, Any]]:
    normalized_query_tokens = [
        str(token or "").strip().lower()
        for token in (query_tokens or _storyline_tokens(post_text))
        if len(str(token or "").strip()) >= 4
    ][:12]
    if not normalized_query_tokens:
        return []

    query = """
    MATCH (cand:Storyline)
    OPTIONAL MATCH (cand)-[:BELONGS_TO]->(cf:StoryFamily)
    OPTIONAL MATCH (cand)-[:BELONGS_TO_TOPIC]->(ct:MacroTopic)
    WITH cand, cf, ct, $query_tokens AS query_tokens,
         toLower(
            coalesce(cand.title, '') + ' '
            + coalesce(cand.seed_preview, '') + ' '
            + reduce(acc = '', item IN coalesce(cand.entity_keys, []) | acc + ' ' + toLower(item))
            + ' '
            + reduce(acc = '', item IN coalesce(cand.family_signatures, []) | acc + ' ' + toLower(item))
            + ' '
            + reduce(acc = '', item IN coalesce(cand.topic_signatures, []) | acc + ' ' + toLower(item))
            + ' '
            + reduce(acc = '', item IN coalesce(ct.topic_aliases, []) | acc + ' ' + toLower(item))
         ) AS hay
    WITH cand, cf, ct, hay,
         size([tok IN query_tokens WHERE hay CONTAINS tok]) AS token_overlap,
         size([tok IN query_tokens WHERE any(k IN coalesce(cand.entity_keys, []) WHERE toLower(k) CONTAINS tok OR tok CONTAINS toLower(k))]) AS entity_token_overlap,
         size([tok IN query_tokens WHERE any(sig IN coalesce(cand.family_signatures, []) WHERE toLower(sig) CONTAINS tok OR tok CONTAINS toLower(sig))]) AS signature_token_overlap,
         size([tok IN query_tokens WHERE any(sig IN coalesce(cand.topic_signatures, []) WHERE toLower(sig) CONTAINS tok OR tok CONTAINS toLower(sig))]) AS topic_token_overlap
    WITH cand, cf, ct, token_overlap, entity_token_overlap, signature_token_overlap, topic_token_overlap,
         (
            token_overlap * 4
            + entity_token_overlap * 3
            + signature_token_overlap
            + topic_token_overlap * 2
            + CASE WHEN coalesce(cand.posts_count, 0) >= 2 THEN 2 ELSE 0 END
            + CASE WHEN coalesce(cand.posts_count, 0) >= 5 THEN 1 ELSE 0 END
         ) AS recovery_score
    WHERE recovery_score >= 6
      AND (token_overlap >= 2 OR topic_token_overlap >= 2 OR (entity_token_overlap >= 1 AND token_overlap >= 1))
      AND ($exclude_storyline_id = '' OR cand.id <> $exclude_storyline_id)
    RETURN
        cand.id AS storyline_id,
        coalesce(cand.title, '') AS title,
        coalesce(cand.seed_preview, '') AS seed_preview,
        coalesce(cand.posts_count, 0) AS posts_count,
        coalesce(cf.id, '') AS story_family_id,
        coalesce(cf.root_storyline_id, cand.id) AS family_root_storyline_id,
        coalesce(cand.entity_keys, []) AS entity_keys,
        coalesce(cand.family_signatures, []) AS family_signatures,
        coalesce(cand.topic_signatures, []) AS topic_signatures,
        coalesce(ct.id, coalesce(cand.macro_topic_id, '')) AS macro_topic_id,
        coalesce(ct.canonical_title, coalesce(cand.macro_topic_title, '')) AS macro_topic_title,
        token_overlap,
        entity_token_overlap,
        signature_token_overlap,
        topic_token_overlap,
        recovery_score
    ORDER BY recovery_score DESC, topic_token_overlap DESC, token_overlap DESC, entity_token_overlap DESC, coalesce(cand.posts_count, 0) DESC
    LIMIT $limit
    """
    try:
        return await graph_gateway.run_read_query(
            query,
            query_tokens=normalized_query_tokens,
            limit=max(1, int(limit)),
            exclude_storyline_id=str(exclude_storyline_id or "").strip(),
        )
    except Exception:
        log.exception("storyline_recovery_search_failed", limit=limit, exclude_storyline_id=exclude_storyline_id)
        return []


async def search_storyline_neighborhood_candidates(
    *,
    anchor_storyline_id: str,
    limit: int = 24,
) -> list[dict[str, Any]]:
    storyline_id = str(anchor_storyline_id or "").strip()
    if not storyline_id:
        return []

    query = """
    MATCH (anchor:Storyline {id: $anchor_storyline_id})
    CALL {
        WITH anchor
        MATCH (anchor)-[:BELONGS_TO]->(:StoryFamily)<-[:BELONGS_TO]-(cand:Storyline)
        WHERE cand.id <> anchor.id
        RETURN cand, 1 AS family_match, 0 AS same_macro_topic, 0 AS same_episode, 8 AS path_score
        LIMIT $per_path_limit
        UNION
        WITH anchor
        MATCH (anchor)-[:BELONGS_TO_TOPIC]->(:MacroTopic)<-[:BELONGS_TO_TOPIC]-(cand:Storyline)
        WHERE cand.id <> anchor.id
        RETURN cand, 0 AS family_match, 1 AS same_macro_topic, 0 AS same_episode, 10 AS path_score
        LIMIT $per_path_limit
        UNION
        WITH anchor
        MATCH (anchor)-[:BELONGS_TO_EPISODE]->(:Episode)<-[:BELONGS_TO_EPISODE]-(cand:Storyline)
        WHERE cand.id <> anchor.id
        RETURN cand, 0 AS family_match, 0 AS same_macro_topic, 1 AS same_episode, 16 AS path_score
        LIMIT $per_path_limit
    }
    WITH anchor, cand,
         max(family_match) AS family_match,
         max(same_macro_topic) AS same_macro_topic,
         max(same_episode) AS same_episode,
         max(path_score) AS path_score
    OPTIONAL MATCH (anchor)-[:BELONGS_TO]->(af:StoryFamily)
    OPTIONAL MATCH (cand)-[:BELONGS_TO]->(cf:StoryFamily)
    OPTIONAL MATCH (cand)-[:BELONGS_TO_TOPIC]->(ct:MacroTopic)
    WITH anchor, cand, cf, ct, family_match, same_macro_topic, same_episode, path_score,
         size([k IN coalesce(cand.entity_keys, []) WHERE k IN coalesce(anchor.entity_keys, [])]) AS shared_entity_count,
         size([sig IN coalesce(cand.family_signatures, []) WHERE sig IN coalesce(anchor.family_signatures, [])]) AS shared_signature_count,
         size([sig IN coalesce(cand.topic_signatures, []) WHERE sig IN (coalesce(anchor.topic_signatures, []) + coalesce(af.topic_signatures, []))]) AS shared_topic_count
    WITH cand,
         head(collect(DISTINCT cf)) AS cf,
         head(collect(DISTINCT ct)) AS ct,
         max(family_match) AS family_match,
         max(same_macro_topic) AS same_macro_topic,
         max(same_episode) AS same_episode,
         max(shared_entity_count) AS shared_entity_count,
         max(shared_signature_count) AS shared_signature_count,
         max(shared_topic_count) AS shared_topic_count,
         max(path_score) AS path_score
    WITH cand, cf, ct, family_match, same_macro_topic, same_episode, shared_entity_count, shared_signature_count, shared_topic_count,
         (
            path_score
            + CASE WHEN coalesce(cand.posts_count, 0) >= 2 THEN 2 ELSE 0 END
            + same_episode * 16
            + same_macro_topic * 10
            + family_match * 8
            + shared_entity_count * 4
            + shared_topic_count * 3
            + shared_signature_count
         ) AS neighborhood_score
    WHERE neighborhood_score >= 5
      AND (
        same_episode > 0
        OR same_macro_topic > 0
        OR family_match > 0
        OR shared_entity_count >= 1
        OR shared_topic_count >= 1
        OR shared_signature_count >= 2
      )
    CALL {
        WITH cand
        MATCH (cand)<-[:SOURCE_OF]-(p:Post)
        WITH p
        ORDER BY p.post_id DESC
        RETURN collect(p.post_id)[0..4] AS source_post_ids,
               collect(coalesce(p.summary, ''))[0..3] AS source_summaries
    }
    RETURN
        cand.id AS storyline_id,
        coalesce(cf.id, '') AS story_family_id,
        coalesce(cf.root_storyline_id, cand.id) AS family_root_storyline_id,
        coalesce(ct.id, coalesce(cand.macro_topic_id, '')) AS macro_topic_id,
        coalesce(ct.canonical_title, coalesce(cand.macro_topic_title, '')) AS macro_topic_title,
        coalesce(cand.title, '') AS title,
        coalesce(cand.seed_preview, '') AS seed_preview,
        coalesce(cand.posts_count, 0) AS posts_count,
        family_match,
        same_macro_topic,
        same_episode,
        shared_entity_count,
        shared_signature_count,
        shared_topic_count,
        neighborhood_score,
        source_post_ids,
        source_summaries
    ORDER BY neighborhood_score DESC, same_episode DESC, same_macro_topic DESC, family_match DESC, coalesce(cand.posts_count, 0) DESC, coalesce(cand.updated_at, '') DESC
    LIMIT $limit
    """
    try:
        return await graph_gateway.run_read_query(
            query,
            anchor_storyline_id=storyline_id,
            limit=max(1, int(limit)),
            per_path_limit=max(8, int(limit)),
        )
    except Exception:
        log.exception("storyline_neighborhood_search_failed", anchor_storyline_id=storyline_id, limit=limit)
        return []


def _related_storyline_query_tokens(anchor: dict[str, Any]) -> list[str]:
    raw_parts: list[str] = [
        str(anchor.get("storyline_title") or "").strip(),
        str(anchor.get("storyline_seed_preview") or "").strip(),
        str(anchor.get("macro_topic_title") or "").strip(),
        str(anchor.get("story_episode_title") or "").strip(),
    ]
    for key in ("entity_keys", "family_signatures", "topic_signatures", "family_topic_signatures", "macro_topic_signatures"):
        for item in anchor.get(key) or []:
            value = str(item or "").strip()
            if not value:
                continue
            if ":" in value:
                value = value.split(":", 1)[1]
            raw_parts.append(value.replace("::", " "))

    tokens: list[str] = []
    seen: set[str] = set()
    for token in _storyline_tokens(" ".join(raw_parts)):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= 12:
            break
    return tokens


def _related_storyline_candidate_key(item: dict[str, Any]) -> str:
    return str(item.get("storyline_id") or "").strip()


def _related_storyline_family_key(item: dict[str, Any]) -> str:
    return str(item.get("family_root_storyline_id") or item.get("storyline_id") or "").strip()


def _related_storyline_score(item: dict[str, Any]) -> float:
    same_macro_topic = int(item.get("same_macro_topic") or 0)
    family_match = int(item.get("family_match") or 0)
    shared_entity_count = int(item.get("shared_entity_count") or 0)
    shared_signature_count = int(item.get("shared_signature_count") or 0)
    shared_topic_count = int(item.get("shared_topic_count") or 0)
    token_overlap = int(item.get("token_overlap") or item.get("match_count") or 0)
    focus_token_hits = int(item.get("focus_token_hits") or 0)
    context_token_hits = int(item.get("context_token_hits") or 0)
    topic_token_overlap = int(item.get("topic_token_overlap") or 0)
    structural_score = float(item.get("structural_score") or 0.0)
    posts_count = int(item.get("posts_count") or 0)
    sibling_seed = 4.0 if item.get("_source") == "context_sibling" else 0.0
    neighborhood_score = float(item.get("neighborhood_score") or 0.0)
    same_episode = int(item.get("same_episode") or 0)
    score = (
        sibling_seed
        + min(neighborhood_score, 40.0) * 0.45
        + same_episode * 12.0
        + same_macro_topic * 10.0
        + family_match * 8.0
        + min(shared_entity_count, 4) * 3.5
        + min(shared_topic_count, 4) * 3.0
        + min(shared_signature_count, 5) * 1.2
        + min(token_overlap, 5) * 1.5
        + min(focus_token_hits, 6) * 7.0
        + min(context_token_hits, 5) * 2.5
        + min(topic_token_overlap, 4) * 2.0
        + min(structural_score, 30.0) * 0.35
        + min(posts_count, 8) * 0.35
    )
    return round(score, 4)


def _related_storyline_token_hits(item: dict[str, Any], query_tokens: list[str]) -> int:
    if not query_tokens:
        return 0
    hay = _related_storyline_text(item)
    if not hay:
        return 0
    return sum(1 for token in query_tokens if token and token in hay)


_RELATED_SAME_CASE_TERMS = (
    "адвокат",
    "арест",
    "обыск",
    "залог",
    "суд",
    "прокур",
    "набу",
    "сап",
    "легализац",
    "отмыв",
    "имуществ",
    "подозрева",
    "антикоррупц",
    "следствен",
)

_RELATED_CONTEXT_TERMS = (
    "зеленск",
    "украин",
    "киев",
    "верховн",
    "рада",
    "правительств",
    "президент",
    "офис",
    "вашингтон",
    "запад",
    "полит",
    "институт",
    "переговор",
    "миндич",
    "умеров",
)

_RELATED_PRINCIPAL_STOP_TOKENS = {
    "general",
    "бывшего",
    "главы",
    "главу",
    "глава",
    "офиса",
    "офис",
    "президента",
    "президент",
    "страны",
    "украины",
    "украина",
}


def _related_storyline_text(item: dict[str, Any]) -> str:
    return _normalize_storyline_text(
        " ".join(
            [
                str(item.get("title") or item.get("storyline_title") or ""),
                str(item.get("seed_preview") or item.get("storyline_seed_preview") or ""),
                str(item.get("macro_topic_title") or ""),
                " ".join(str(x or "") for x in (item.get("matched_tokens") or [])),
                " ".join(str(x or "") for x in (item.get("source_summaries") or [])),
                " ".join(str(x or "") for x in (item.get("example_posts") or [])),
            ]
        )
    )


def _related_storyline_primary_text(item: dict[str, Any]) -> str:
    return _normalize_storyline_text(
        " ".join(
            [
                str(item.get("title") or item.get("storyline_title") or ""),
                str(item.get("seed_preview") or item.get("storyline_seed_preview") or ""),
                str(item.get("macro_topic_title") or ""),
            ]
        )
    )


def _related_storyline_primary_token_hits(item: dict[str, Any], query_tokens: list[str]) -> int:
    hay = _related_storyline_primary_text(item)
    if not hay:
        return 0
    return sum(1 for token in query_tokens if token and token in hay)


def _related_storyline_context_hits(item: dict[str, Any]) -> int:
    hay = _related_storyline_text(item)
    if not hay:
        return 0
    return sum(1 for token in _RELATED_CONTEXT_TERMS if token in hay)


def _related_storyline_primary_context_hits(item: dict[str, Any]) -> int:
    hay = _related_storyline_primary_text(item)
    if not hay:
        return 0
    return sum(1 for token in _RELATED_CONTEXT_TERMS if token in hay)


def _related_storyline_anchor_principal_tokens(anchor: dict[str, Any]) -> list[str]:
    title_text = str(anchor.get("storyline_title") or anchor.get("title") or "").strip()
    raw_text = title_text or str(anchor.get("storyline_seed_preview") or anchor.get("seed_preview") or "")[:220]
    tokens = [
        token
        for token in _storyline_tokens(raw_text)
        if len(token) >= 5 and token not in _RELATED_PRINCIPAL_STOP_TOKENS
    ]
    expanded = _related_storyline_expand_query_tokens(tokens[:8])
    result: list[str] = []
    seen: set[str] = set()
    for token in expanded:
        if token in seen or token in _RELATED_PRINCIPAL_STOP_TOKENS:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= 10:
            break
    return result


def _related_storyline_same_case_duplicate(anchor: dict[str, Any], item: dict[str, Any]) -> bool:
    hay = _related_storyline_primary_text(item)
    if not hay:
        return False
    anchor_hay = _related_storyline_text(
        {
            "storyline_title": anchor.get("storyline_title") or anchor.get("title"),
            "storyline_seed_preview": anchor.get("storyline_seed_preview") or anchor.get("seed_preview"),
            "macro_topic_title": anchor.get("macro_topic_title"),
        }
    )
    if not any(term in anchor_hay for term in _RELATED_SAME_CASE_TERMS):
        return False
    principal_tokens = _related_storyline_anchor_principal_tokens(anchor)
    if not principal_tokens or not any(token in hay for token in principal_tokens):
        return False
    return any(term in hay for term in _RELATED_SAME_CASE_TERMS)


def _related_storyline_expand_query_tokens(query_tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    suffixes = (
        "ского",
        "скому",
        "скими",
        "ских",
        "ская",
        "ское",
        "ские",
        "ский",
        "ской",
        "ыми",
        "ими",
        "ого",
        "ому",
        "ами",
        "ями",
        "ая",
        "ое",
        "ые",
        "ий",
        "ый",
        "ой",
        "ых",
        "их",
        "ов",
        "ев",
        "ью",
    )
    for raw_token in query_tokens:
        token = str(raw_token or "").strip().lower()
        if len(token) < 4:
            continue
        variants = [token]
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) - len(suffix) >= 4:
                variants.append(token[: -len(suffix)])
                break
        if token.startswith("иран"):
            variants.append("иран")
        if token.startswith("китай"):
            variants.append("китай")
        if token.startswith("санкц"):
            variants.append("санкц")
        if token.startswith("ормуз"):
            variants.append("ормуз")
        if token.startswith("нефт"):
            variants.append("нефт")
        for variant in variants:
            if variant in seen or len(variant) < 4:
                continue
            seen.add(variant)
            expanded.append(variant)
            if len(expanded) >= 18:
                return expanded
    return expanded


async def collect_related_storyline_branches(
    post_id: int,
    *,
    anchor_storyline_id: str | None = None,
    query_tokens: list[str] | None = None,
    enable_expensive_search: bool = False,
    limit: int = 12,
) -> dict[str, Any] | None:
    """Collect graph-neighbor storyline branches for a user-facing related-stories card."""
    anchor = (
        await get_storyline_context_for_storyline_id(anchor_storyline_id)
        if str(anchor_storyline_id or "").strip()
        else await get_storyline_context_for_post(post_id)
    )
    if not anchor:
        return None

    anchor_storyline_id_value = str(anchor.get("storyline_id") or anchor_storyline_id or "").strip()
    anchor_family_root_id = str(
        anchor.get("family_root_storyline_id") or anchor.get("storyline_id") or ""
    ).strip()
    normalized_query_tokens = [
        str(token or "").strip().lower()
        for token in (query_tokens or [])
        if len(str(token or "").strip()) >= 4
    ][:12]
    if not normalized_query_tokens:
        normalized_query_tokens = _related_storyline_query_tokens(anchor)
    normalized_query_tokens = _related_storyline_expand_query_tokens(normalized_query_tokens)

    raw_candidates: list[dict[str, Any]] = []
    for sibling in anchor.get("sibling_candidates") or []:
        if not isinstance(sibling, dict):
            continue
        item = dict(sibling)
        item.setdefault("family_root_storyline_id", anchor_family_root_id)
        item.setdefault("story_family_id", str(anchor.get("story_family_id") or ""))
        item.setdefault("macro_topic_id", str(anchor.get("macro_topic_id") or ""))
        item.setdefault("macro_topic_title", str(anchor.get("macro_topic_title") or ""))
        item["_source"] = "context_sibling"
        if normalized_query_tokens and _related_storyline_token_hits(item, normalized_query_tokens) < 1:
            continue
        raw_candidates.append(item)

    if anchor_storyline_id_value:
        for item in await search_storyline_neighborhood_candidates(
            anchor_storyline_id=anchor_storyline_id_value,
            limit=max(limit * 3, 24),
        ):
            local = dict(item)
            local["_source"] = "neighborhood"
            raw_candidates.append(local)

        if enable_expensive_search and len(raw_candidates) < min(max(6, int(limit) // 2), int(limit)):
            for item in await search_storyline_structural_candidates(
                anchor_storyline_id=anchor_storyline_id_value,
                limit=max(limit, 12),
            ):
                local = dict(item)
                local["_source"] = "structural"
                raw_candidates.append(local)

    if enable_expensive_search and normalized_query_tokens and len(raw_candidates) < min(max(6, int(limit) // 2), int(limit)):
        for item in await search_storyline_branch_candidates(
            query_tokens=normalized_query_tokens,
            limit=max(limit, 12),
            min_match_count=2,
            exclude_storyline_id=anchor_storyline_id_value,
        ):
            local = dict(item)
            local["_source"] = "branch_tokens"
            raw_candidates.append(local)
        if len(raw_candidates) < min(max(6, int(limit) // 2), int(limit)):
            for item in await search_storyline_branch_candidates(
                query_tokens=normalized_query_tokens,
                limit=max(limit, 12),
                min_match_count=1,
                exclude_storyline_id=anchor_storyline_id_value,
            ):
                local = dict(item)
                local["_source"] = "branch_tokens_wide"
                raw_candidates.append(local)

    by_storyline: dict[str, dict[str, Any]] = {}
    for item in raw_candidates:
        storyline_id = _related_storyline_candidate_key(item)
        if not storyline_id or storyline_id == anchor_storyline_id_value:
            continue
        if anchor_family_root_id and _related_storyline_family_key(item) == anchor_family_root_id:
            continue
        token_hits = _related_storyline_token_hits(item, normalized_query_tokens)
        primary_token_hits = _related_storyline_primary_token_hits(item, normalized_query_tokens)
        context_hits = _related_storyline_context_hits(item)
        primary_context_hits = _related_storyline_primary_context_hits(item)
        same_case_duplicate = _related_storyline_same_case_duplicate(anchor, item)
        item["focus_token_hits"] = token_hits
        item["primary_focus_token_hits"] = primary_token_hits
        item["context_token_hits"] = context_hits
        item["primary_context_token_hits"] = primary_context_hits
        item["same_case_duplicate"] = same_case_duplicate
        if same_case_duplicate:
            continue
        if str(item.get("_source") or "") == "neighborhood" and normalized_query_tokens:
            same_episode = int(item.get("same_episode") or 0)
            same_macro_topic = int(item.get("same_macro_topic") or 0)
            family_match = int(item.get("family_match") or 0)
            principal_tokens = _related_storyline_anchor_principal_tokens(anchor)
            primary_hay = _related_storyline_primary_text(item)
            primary_principal_hit = any(token in primary_hay for token in principal_tokens)
            if token_hits <= 0 and same_episode <= 0:
                continue
            if family_match > 0 and same_macro_topic <= 0 and same_episode <= 0 and primary_context_hits <= 0 and not primary_principal_hit:
                continue
            if family_match > 0 and same_macro_topic <= 0 and same_episode <= 0 and token_hits < 2 and context_hits < 2:
                continue
        if _related_storyline_family_key(item) == anchor_family_root_id and str(item.get("_source") or "") == "context_sibling":
            # Same-family siblings are useful, but exact duplicate roots should be collapsed below.
            pass
        item["related_score"] = max(float(item.get("related_score") or 0.0), _related_storyline_score(item))
        existing = by_storyline.get(storyline_id)
        if not existing or float(item.get("related_score") or 0.0) > float(existing.get("related_score") or 0.0):
            by_storyline[storyline_id] = item

    candidates = sorted(
        by_storyline.values(),
        key=lambda row: (
            -int(row.get("focus_token_hits") or 0),
            -float(row.get("related_score") or 0.0),
            -int(row.get("context_token_hits") or 0),
            -int(row.get("same_episode") or 0),
            -int(row.get("same_macro_topic") or 0),
            -int(row.get("family_match") or 0),
            -int(row.get("shared_topic_count") or 0),
            -int(row.get("posts_count") or 0),
            str(row.get("title") or ""),
        ),
    )[: max(1, int(limit))]

    if candidates:
        source_post_ids_by_storyline: dict[str, list[int]] = {}
        for item in candidates:
            storyline_id = str(item.get("storyline_id") or "").strip()
            source_post_ids = [
                int(post_id)
                for post_id in (item.get("source_post_ids") or [])
                if int(post_id or 0) > 0
            ]
            if storyline_id and source_post_ids:
                source_post_ids_by_storyline[storyline_id] = source_post_ids[:4]

        missing_storyline_ids = [
            str(item.get("storyline_id") or "").strip()
            for item in candidates
            if str(item.get("storyline_id") or "").strip()
            and str(item.get("storyline_id") or "").strip() not in source_post_ids_by_storyline
        ]
        if missing_storyline_ids:
            posts_query = """
            UNWIND $storyline_ids AS storyline_id
            MATCH (s:Storyline {id: storyline_id})<-[:SOURCE_OF]-(p:Post)
            WITH storyline_id, p
            ORDER BY storyline_id, p.post_id DESC
            RETURN storyline_id, collect(p.post_id)[0..4] AS source_post_ids
            """
            try:
                post_rows = await graph_gateway.run_read_query(posts_query, storyline_ids=missing_storyline_ids)
                for row in post_rows:
                    sid = str(row.get("storyline_id") or "").strip()
                    ids = [int(pid) for pid in (row.get("source_post_ids") or []) if int(pid or 0) > 0]
                    if sid and ids:
                        source_post_ids_by_storyline[sid] = ids[:4]
            except Exception:
                log.exception("storyline_related_post_refs_lookup_failed", storyline_count=len(missing_storyline_ids))

        for item in candidates:
            storyline_id = str(item.get("storyline_id") or "").strip()
            if storyline_id and source_post_ids_by_storyline.get(storyline_id):
                item["source_post_ids"] = source_post_ids_by_storyline[storyline_id]

    return {
        "anchor": anchor,
        "candidate_branches": [dict(item) for item in candidates],
        "query_tokens": normalized_query_tokens,
    }


async def collect_storyline_follow_candidates(
    post_id: int,
    *,
    anchor_storyline_id: str | None = None,
    max_storylines: int | None = None,
    query_tokens: list[str] | None = None,
) -> dict[str, Any] | None:
    anchor = (
        await get_storyline_context_for_storyline_id(anchor_storyline_id)
        if str(anchor_storyline_id or "").strip()
        else await get_storyline_context_for_post(post_id)
    )
    if not anchor:
        return None

    normalized_query_tokens = [
        str(token or "").strip().lower()
        for token in (query_tokens or [])
        if len(str(token or "").strip()) >= 4
    ][:12]
    if not normalized_query_tokens:
        raw_query = " ".join(
            part
            for part in [
                str(anchor.get("storyline_title") or "").strip(),
                str(anchor.get("storyline_seed_preview") or "").strip(),
            ]
            if part
        ).lower()
        normalized_query_tokens = [
            token
            for token in {
                piece.strip(".,:;!?()[]{}\"'`").lower()
                for piece in raw_query.split()
            }
            if len(token) >= 4
        ][:10]

    limit_clause = "\n    LIMIT $max_storylines" if max_storylines else ""

    query = """
    __ANCHOR_MATCH__
    OPTIONAL MATCH (anchor)-[:BELONGS_TO]->(af:StoryFamily)
    OPTIONAL MATCH (anchor)-[:BELONGS_TO_TOPIC]->(at:MacroTopic)
    WITH anchor, af AS anchor_family, at AS anchor_topic, $query_tokens AS query_tokens,
         (coalesce(anchor.topic_signatures, []) + coalesce(af.topic_signatures, [])) AS anchor_topic_signatures
    MATCH (cand:Storyline)
    OPTIONAL MATCH (cand)-[:BELONGS_TO]->(cf:StoryFamily)
    OPTIONAL MATCH (cand)-[:BELONGS_TO_TOPIC]->(ct:MacroTopic)
    WITH anchor, anchor_family, anchor_topic, cand, cf, ct, query_tokens, anchor_topic_signatures,
         CASE
            WHEN cand.id = anchor.id THEN 2
            WHEN anchor_family IS NOT NULL AND coalesce(cf.root_storyline_id, cand.id) = coalesce(anchor_family.root_storyline_id, anchor.id) THEN 1
            ELSE 0
         END AS family_match,
         CASE
            WHEN anchor_topic IS NOT NULL AND ct IS NOT NULL AND ct.id = anchor_topic.id THEN 1
            ELSE 0
         END AS same_macro_topic,
         size([k IN coalesce(cand.entity_keys, []) WHERE k IN coalesce(anchor.entity_keys, [])]) AS shared_entity_count,
         size([sig IN coalesce(cand.family_signatures, []) WHERE sig IN coalesce(anchor.family_signatures, [])]) AS shared_signature_count,
         size([sig IN coalesce(cand.topic_signatures, []) WHERE sig IN anchor_topic_signatures]) AS shared_topic_count,
         size([tok IN query_tokens WHERE toLower(coalesce(cand.title, '') + ' ' + coalesce(cand.seed_preview, '')) CONTAINS tok]) AS token_overlap,
         size([tok IN query_tokens WHERE any(sig IN coalesce(cand.topic_signatures, []) WHERE toLower(sig) CONTAINS tok)]) AS topic_token_overlap
    WITH anchor, anchor_family, anchor_topic, cand, cf, ct, family_match, same_macro_topic, shared_entity_count, shared_signature_count, shared_topic_count, token_overlap, topic_token_overlap,
         (
            CASE WHEN cand.id = anchor.id THEN 100 ELSE 0 END +
            CASE WHEN family_match > 0 THEN 3 ELSE 0 END +
            CASE WHEN same_macro_topic > 0 THEN 8 ELSE 0 END +
            shared_entity_count * 3 +
            shared_topic_count * 4 +
            CASE WHEN shared_entity_count > 0 OR family_match > 0 THEN shared_signature_count ELSE 0 END +
            token_overlap * 2 +
            topic_token_overlap * 2
         ) AS retrieval_score
    WHERE cand.id = anchor.id
        OR (
            retrieval_score >= 6
            AND (
                same_macro_topic > 0
                OR shared_entity_count >= 2
                OR shared_topic_count >= 2
                OR token_overlap >= 3
                OR topic_token_overlap >= 2
                OR (shared_entity_count >= 1 AND token_overlap >= 1 AND coalesce(cand.posts_count, 0) >= 2)
                OR (shared_topic_count >= 1 AND token_overlap >= 1 AND coalesce(cand.posts_count, 0) >= 2)
                OR (family_match > 0 AND (shared_entity_count >= 1 OR token_overlap >= 1))
                OR (same_macro_topic > 0 AND (topic_token_overlap >= 1 OR shared_topic_count >= 1 OR token_overlap >= 1))
            )
       )
    RETURN
        cand.id AS storyline_id,
        coalesce(cand.title, '') AS title,
        coalesce(cand.seed_preview, '') AS seed_preview,
        coalesce(cand.posts_count, 0) AS posts_count,
        coalesce(cf.id, '') AS story_family_id,
        coalesce(cf.root_storyline_id, cand.id) AS family_root_storyline_id,
        coalesce(ct.id, coalesce(cand.macro_topic_id, '')) AS macro_topic_id,
        coalesce(ct.canonical_title, coalesce(cand.macro_topic_title, '')) AS macro_topic_title,
        coalesce(cand.topic_signatures, []) AS topic_signatures,
        family_match,
        same_macro_topic,
        shared_entity_count,
        shared_signature_count,
        shared_topic_count,
        token_overlap,
        topic_token_overlap,
        retrieval_score
    ORDER BY retrieval_score DESC, same_macro_topic DESC, shared_topic_count DESC, family_match DESC, shared_signature_count DESC, shared_entity_count DESC, token_overlap DESC, coalesce(cand.posts_count, 0) DESC
    __LIMIT_CLAUSE__
    """
    anchor_match = (
        "MATCH (anchor:Storyline {id: $anchor_storyline_id})"
        if str(anchor_storyline_id or "").strip()
        else "MATCH (p:Post {post_id: $post_id})-[:SOURCE_OF]->(anchor:Storyline)"
    )
    query = query.replace("__ANCHOR_MATCH__", anchor_match)
    query = query.replace("__LIMIT_CLAUSE__", limit_clause)
    try:
        candidates = await graph_gateway.run_read_query(
            query,
            post_id=int(post_id),
            anchor_storyline_id=str(anchor_storyline_id or "").strip(),
            query_tokens=normalized_query_tokens,
            **({"max_storylines": max(1, int(max_storylines))} if max_storylines else {}),
        )
    except Exception:
        log.exception("storyline_follow_candidate_lookup_failed", post_id=post_id)
        return {"anchor": anchor, "candidate_storylines": []}

    deduped_candidates: list[dict[str, Any]] = []
    seen_storyline_ids: set[str] = set()
    for item in candidates:
        storyline_id = str(item.get("storyline_id") or "").strip()
        if not storyline_id or storyline_id in seen_storyline_ids:
            continue
        seen_storyline_ids.add(storyline_id)
        deduped_candidates.append(dict(item))

    storyline_ids = [
        str(item.get("storyline_id") or "").strip()
        for item in deduped_candidates
        if str(item.get("storyline_id") or "").strip()
    ]
    post_refs: list[dict[str, Any]] = []
    if storyline_ids:
        posts_query = """
        UNWIND $storyline_ids AS storyline_id
        MATCH (s:Storyline {id: storyline_id})<-[:SOURCE_OF]-(p:Post)
        RETURN
            storyline_id,
            p.post_id AS post_id
        ORDER BY storyline_id, p.post_id ASC
        """
        try:
            post_refs = await graph_gateway.run_read_query(posts_query, storyline_ids=storyline_ids)
        except Exception:
            log.exception("storyline_follow_post_refs_lookup_failed", post_id=post_id, storyline_count=len(storyline_ids))
            post_refs = []

    return {
        "anchor": anchor,
        "candidate_storylines": deduped_candidates,
        "post_refs": [dict(item) for item in post_refs],
        "query_tokens": normalized_query_tokens,
    }
