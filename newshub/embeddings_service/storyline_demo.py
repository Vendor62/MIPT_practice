from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from embeddings_service.entity_normalizer import normalize_entity_text, partition_entities
from embeddings_service.ie_pipeline import embed_passage, extract_entities, extract_relations
from embeddings_service.relation_validator import filter_relations


DEFAULT_RELATION_MODEL_DIR = (
    "./artifacts/model_runs/relation_scorer_catboost_20260321T140313Z"
)
DEFAULT_RELATION_THRESHOLD = 0.52
DEFAULT_STORYLINE_ASSIGN_THRESHOLD = 0.57

ORG_KEYWORDS = {
    "банк", "компания", "корпорац", "холдинг", "госкорпорац",
    "министер", "ведомств", "правительств", "администрац",
    "парламент", "дума", "совет", "университет", "институт",
    "центр", "служба", "комитет", "агентств", "фонд", "армия",
    "командование", "партия", "суд", "прокуратур",
}

LOCATION_KEYWORDS = {
    "город", "область", "край", "район", "республик", "страна",
    "пролив", "регион", "округ", "остров", "посел", "провинц",
    "столица", "улица", "море", "океан", "континент",
}

RELATION_DIRECTION_PRIORS = {
    "employer": ("person", "org"),
    "position_held": ("person", "org"),
    "member_of": ("person", "org"),
    "member_of_political_party": ("person", "org"),
    "founded_by": ("org", "person"),
    "chairperson": ("person", "org"),
    "director": ("person", "org"),
    "born_in": ("person", "location"),
    "place_of_death": ("person", "location"),
    "citizen_of": ("person", "location"),
    "located_in": ("org", "location"),
    "country_of_origin": ("org", "location"),
}


@dataclass
class StorylineState:
    storyline_id: str
    title: str
    created_at: datetime | None
    last_timestamp: datetime | None
    centroid_embedding: list[float]
    entity_keys: set[str]
    relation_signatures: set[str]
    post_ids: list[int]


def _tokenize(text_value: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text_value or "")


def _to_num(value: Any) -> float:
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    if s.lower() == "true":
        return 1.0
    if s.lower() == "false":
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _has(pattern: str, text_value: str) -> int:
    return 1 if re.search(pattern, text_value or "") else 0


def _guess_entity_type(surface: str) -> str:
    s = (surface or "").strip()
    if not s:
        return "other"
    s_low = s.lower()

    person_like_tokens = re.findall(r"\b[А-ЯЁA-Z][а-яёa-z-]{2,}\b", s)
    if len(person_like_tokens) >= 2 and not any(k in s_low for k in ORG_KEYWORDS):
        return "person"

    if any(k in s_low for k in ORG_KEYWORDS):
        return "org"
    if any(k in s_low for k in LOCATION_KEYWORDS):
        return "location"
    return "other"


def _guess_label_from_surface(surface: str) -> str:
    t = _guess_entity_type(surface)
    if t == "org":
        return "organization"
    return t


def _build_relation_features(row: dict[str, Any]) -> dict[str, Any]:
    head = str(row.get("head", "") or "")
    tail = str(row.get("tail", "") or "")
    relation = str(row.get("relation", "") or "").strip().lower()

    head_tokens = _tokenize(head)
    tail_tokens = _tokenize(tail)
    head_set = {t.lower() for t in head_tokens}
    tail_set = {t.lower() for t in tail_tokens}
    union = head_set | tail_set
    inter = head_set & tail_set
    jaccard = (len(inter) / len(union)) if union else 0.0

    head_type = _guess_entity_type(head)
    tail_type = _guess_entity_type(tail)
    expected = RELATION_DIRECTION_PRIORS.get(relation)

    return {
        "head_token_count": len(head_tokens),
        "tail_token_count": len(tail_tokens),
        "head_char_len": len(head),
        "tail_char_len": len(tail),
        "token_count_diff_abs": abs(len(head_tokens) - len(tail_tokens)),
        "char_len_diff_abs": abs(len(head) - len(tail)),
        "head_tail_token_jaccard": round(jaccard, 6),
        "head_has_digit": _has(r"\d", head),
        "tail_has_digit": _has(r"\d", tail),
        "head_has_latin": _has(r"[A-Za-z]", head),
        "tail_has_latin": _has(r"[A-Za-z]", tail),
        "head_has_cyrillic": _has(r"[А-Яа-яЁё]", head),
        "tail_has_cyrillic": _has(r"[А-Яа-яЁё]", tail),
        "head_type_guess": head_type,
        "tail_type_guess": tail_type,
        "type_pair_guess": f"{head_type}->{tail_type}",
        "same_type_guess": 1 if head_type == tail_type else 0,
        "relation_has_direction_prior": 1 if expected else 0,
        "direction_match_expected": 1 if expected and (head_type, tail_type) == expected else 0,
        "direction_reverse_expected": 1 if expected and (tail_type, head_type) == expected else 0,
        "is_person_head_guess": 1 if head_type == "person" else 0,
        "is_person_tail_guess": 1 if tail_type == "person" else 0,
        "is_org_head_guess": 1 if head_type == "org" else 0,
        "is_org_tail_guess": 1 if tail_type == "org" else 0,
        "is_location_head_guess": 1 if head_type == "location" else 0,
        "is_location_tail_guess": 1 if tail_type == "location" else 0,
    }


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _safe_iso(ts: datetime | None) -> str | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return _safe_iso(value)
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _normalize_dt(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _temporal_proximity(current_ts: datetime | None, last_ts: datetime | None) -> float:
    if current_ts is None or last_ts is None:
        return 0.5
    delta_hours = abs((current_ts - last_ts).total_seconds()) / 3600.0
    return max(0.0, min(1.0, math.exp(-delta_hours / 72.0)))


def _relation_signature(rel: dict[str, Any]) -> str:
    head = normalize_entity_text(str(rel.get("head_key") or rel.get("head") or ""))
    tail = normalize_entity_text(str(rel.get("tail_key") or rel.get("tail") or ""))
    relation = str(rel.get("relation") or "")
    return f"{relation}:{head}->{tail}"


def _event_type(relations: list[dict[str, Any]]) -> str:
    rels = {str(r.get("relation") or "") for r in relations}
    if "diplomatic_relation" in rels:
        return "diplomatic"
    if {"position_held", "chairperson", "member_of"} & rels:
        return "personnel"
    if {"located_in", "country", "headquarters_in", "citizen_of"} & rels:
        return "location_update"
    if {"participant", "part_of", "significant_event"} & rels:
        return "event_update"
    return "general_news"


def _storyline_title(entities: list[dict[str, Any]], relations: list[dict[str, Any]], storyline_id: str) -> str:
    names = [str(e.get("text") or "").strip() for e in entities if str(e.get("text") or "").strip()]
    names = names[:2]
    rel = str(relations[0].get("relation")) if relations else "general"
    if names:
        return f"{', '.join(names)} [{rel}]"
    return f"{storyline_id} [{rel}]"


def _vector_average(current: list[float], incoming: list[float], n_current: int) -> list[float]:
    if not current:
        return list(incoming)
    if not incoming or len(current) != len(incoming):
        return current
    n_next = n_current + 1
    return [((x * n_current) + y) / n_next for x, y in zip(current, incoming)]


class RelationScorer:
    def __init__(self, model_dir: str):
        from catboost import CatBoostClassifier

        meta_path = Path(model_dir) / "relation_scorer_feature_cols.json"
        model_path = Path(model_dir) / "relation_scorer_catboost.cbm"

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_cols = list(meta["feature_cols"])
        self.cat_feature_names = set(meta.get("cat_features", []))
        self.cat_idx = [i for i, c in enumerate(self.feature_cols) if c in self.cat_feature_names]

        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))

    def score(self, relation_row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        from catboost import Pool

        full = dict(relation_row)
        derived = _build_relation_features(full)
        full.update(derived)

        features: list[Any] = []
        for col in self.feature_cols:
            if col in self.cat_feature_names:
                features.append(str(full.get(col, "") or ""))
            else:
                features.append(_to_num(full.get(col)))

        proba = float(self.model.predict_proba(Pool([features], cat_features=self.cat_idx))[0][1])
        explanation = {
            "direction_match_expected": int(derived["direction_match_expected"]),
            "direction_reverse_expected": int(derived["direction_reverse_expected"]),
            "head_tail_token_jaccard": float(derived["head_tail_token_jaccard"]),
            "head_type_guess": derived["head_type_guess"],
            "tail_type_guess": derived["tail_type_guess"],
            "rule_confidence": round(float(_to_num(relation_row.get("confidence"))), 4),
        }
        return proba, explanation


def _load_relation_rows_by_post(
    csv_path: str,
    relation_threshold: float,
) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        try:
            post_id = int(str(row.get("post_id") or "").strip())
        except Exception:
            continue

        proba = _to_num(row.get("model_proba_accept"))
        model_decision = str(row.get("model_decision") or "").strip().lower()
        if model_decision not in {"accept", "reject"}:
            model_decision = "accept" if proba >= relation_threshold else "reject"

        item = {
            "head": normalize_entity_text(str(row.get("head") or "")),
            "tail": normalize_entity_text(str(row.get("tail") or "")),
            "relation": str(row.get("relation") or "").strip().lower(),
            "head_key": str(row.get("head_key") or "").strip(),
            "tail_key": str(row.get("tail_key") or "").strip(),
            "reason": str(row.get("reason") or "").strip(),
            "reason_group": str(row.get("reason_group") or "").strip() or "accepted",
            "model_proba_accept": round(float(proba), 6),
            "model_decision": model_decision,
            "confidence": round(float(proba), 4),
            "model_explanation": {
                "source": "relation_candidates_csv",
                "rule_decision": str(row.get("rule_decision") or "").strip(),
            },
        }
        out.setdefault(post_id, []).append(item)

    return out


def _db_dsn() -> str:
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "db")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME") or os.getenv("POSTGRES_DB")
    if not all([db_user, db_password, db_name]):
        raise RuntimeError("DB env vars are not set (DB_USER/DB_PASSWORD/DB_NAME or POSTGRES_DB)")
    return f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


async def _fetch_posts_batch(
    session: AsyncSession,
    *,
    limit: int,
    min_post_id: int,
    max_post_id: int | None,
) -> list[dict[str, Any]]:
    query = """
        SELECT p.id, p.processed_content, p.content, p.timestamp, e.embedding
        FROM posts p
        LEFT JOIN embeddings e ON e.post_id = p.id
        WHERE p.content IS NOT NULL
          AND p.id >= :min_post_id
        ORDER BY p.id DESC
        LIMIT :limit
    """
    params: dict[str, Any] = {
        "limit": int(limit),
        "min_post_id": int(min_post_id),
    }
    if max_post_id is not None:
        query = query.replace("ORDER BY", "AND p.id <= :max_post_id\n        ORDER BY")
        params["max_post_id"] = int(max_post_id)

    rows = (
        await session.execute(
            text(query),
            params,
        )
    ).all()

    out: list[dict[str, Any]] = []
    for post_id, processed_content, content, timestamp, embedding in rows:
        text_value = processed_content or content
        if not text_value:
            continue
        out.append(
            {
                "post_id": int(post_id),
                "text": str(text_value),
                "raw_text": str(content or text_value),
                "timestamp": _normalize_dt(timestamp),
                "embedding": embedding if isinstance(embedding, list) else [],
            }
        )

    # model stream behavior: process older -> newer
    out.sort(key=lambda item: item["post_id"])
    return out


def _event_confidence(promoted_entities: list[dict[str, Any]], kept_relations: list[dict[str, Any]]) -> float:
    relation_mean = 0.0
    if kept_relations:
        relation_mean = sum(float(r.get("model_proba_accept", 0.0)) for r in kept_relations) / len(kept_relations)
    entity_signal = min(1.0, len(promoted_entities) / 6.0)
    relation_signal = min(1.0, len(kept_relations) / 4.0)
    value = 0.5 * relation_mean + 0.3 * entity_signal + 0.2 * relation_signal
    return round(max(0.0, min(1.0, value)), 4)


def _score_storyline_match(
    event_embedding: list[float],
    event_entity_keys: set[str],
    event_relation_signatures: set[str],
    event_ts: datetime | None,
    storyline: StorylineState,
) -> dict[str, Any]:
    emb = _cosine_similarity(event_embedding, storyline.centroid_embedding)
    ent = _jaccard(event_entity_keys, storyline.entity_keys)
    rel = _jaccard(event_relation_signatures, storyline.relation_signatures)
    tmp = _temporal_proximity(event_ts, storyline.last_timestamp)

    weights = {
        "embedding_similarity": 0.45,
        "entity_overlap": 0.3,
        "relation_overlap": 0.15,
        "temporal_proximity": 0.1,
    }
    contributions = {
        key: round(weights[key] * value, 4)
        for key, value in {
            "embedding_similarity": emb,
            "entity_overlap": ent,
            "relation_overlap": rel,
            "temporal_proximity": tmp,
        }.items()
    }
    final_score = round(sum(contributions.values()), 4)
    return {
        "storyline_id": storyline.storyline_id,
        "score": final_score,
        "components": {
            "embedding_similarity": round(emb, 4),
            "entity_overlap": round(ent, 4),
            "relation_overlap": round(rel, 4),
            "temporal_proximity": round(tmp, 4),
        },
        "contributions": contributions,
    }


def assign_to_storyline_or_create_new(
    *,
    event: dict[str, Any],
    storylines: list[StorylineState],
    assignment_threshold: float,
) -> tuple[str, float, dict[str, Any]]:
    best: dict[str, Any] | None = None
    for storyline in storylines:
        result = _score_storyline_match(
            event_embedding=event["embedding"],
            event_entity_keys=event["entity_keys"],
            event_relation_signatures=event["relation_signatures"],
            event_ts=event.get("timestamp"),
            storyline=storyline,
        )
        if best is None or result["score"] > best["score"]:
            best = result

    if best and best["score"] >= assignment_threshold:
        return best["storyline_id"], best["score"], best

    return "", (best["score"] if best else 0.0), {
        "score": best["score"] if best else 0.0,
        "components": best["components"] if best else {},
        "contributions": best["contributions"] if best else {},
        "reason": "below_threshold_or_no_candidates",
    }


async def run_storyline_demo(
    *,
    limit: int,
    min_post_id: int,
    max_post_id: int | None,
    relation_model_dir: str,
    relation_threshold: float,
    assignment_threshold: float,
    out_dir: str,
    relation_candidates_csv: str | None = None,
) -> dict[str, Any]:
    dsn = _db_dsn()
    engine = create_async_engine(dsn, pool_pre_ping=True)
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    scorer = RelationScorer(relation_model_dir) if not relation_candidates_csv else None
    relation_rows_by_post = (
        _load_relation_rows_by_post(relation_candidates_csv, relation_threshold)
        if relation_candidates_csv
        else {}
    )

    async with async_session() as session:
        posts = await _fetch_posts_batch(
            session,
            limit=limit,
            min_post_id=min_post_id,
            max_post_id=max_post_id,
        )

    storylines: list[StorylineState] = []
    events_out: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    storyline_seq = 0

    for post in posts:
        post_id = post["post_id"]
        post_text = post["text"]
        raw_text = post["raw_text"]

        if relation_rows_by_post:
            post_rows = relation_rows_by_post.get(post_id, [])
            entity_map: dict[str, dict[str, Any]] = {}
            for rel in post_rows:
                for side in ("head", "tail"):
                    surface = str(rel.get(side) or "").strip()
                    if not surface:
                        continue
                    key_field = "head_key" if side == "head" else "tail_key"
                    canonical_key = str(rel.get(key_field) or "").strip() or normalize_entity_text(surface)
                    entity_key = canonical_key or normalize_entity_text(surface)
                    if entity_key in entity_map:
                        continue
                    label = _guess_label_from_surface(surface)
                    entity_map[entity_key] = {
                        "text": surface,
                        "raw_text": surface,
                        "label": label,
                        "canonical_key": canonical_key,
                        "decision": "promote",
                        "reason": "from_relation_candidates_csv",
                    }
            promoted = list(entity_map.values())
            mentions = []
            rejected = []
            scored_relations = post_rows
            rule_relations = post_rows
            kept_relations = [r for r in scored_relations if r["model_decision"] == "accept"]
        else:
            raw_entities = extract_entities(post_text)
            promoted, mentions, rejected = partition_entities(raw_entities, full_text=raw_text, post_id=post_id)

            raw_relations = extract_relations(raw_text, promoted)
            rule_relations, _rule_rejected = filter_relations(raw_relations, promoted)

            scored_relations = []
            assert scorer is not None
            for rel in rule_relations:
                proba, rel_expl = scorer.score(rel)
                rel_item = dict(rel)
                rel_item["model_proba_accept"] = round(proba, 6)
                rel_item["model_decision"] = "accept" if proba >= relation_threshold else "reject"
                rel_item["model_explanation"] = rel_expl
                scored_relations.append(rel_item)

            kept_relations = [r for r in scored_relations if r["model_decision"] == "accept"]

        embedding = post["embedding"]
        if not embedding:
            embedding = embed_passage(post_text)

        entity_keys = {
            normalize_entity_text(str(e.get("canonical_key") or e.get("text") or ""))
            for e in promoted
            if str(e.get("canonical_key") or e.get("text") or "").strip()
        }
        relation_signatures = {_relation_signature(r) for r in kept_relations}

        event_payload = {
            "event_mention_id": f"event_mention_lite_{post_id}",
            "post_id": post_id,
            "timestamp": post.get("timestamp"),
            "event_type": _event_type(kept_relations),
            "entities": [
                {
                    "text": e.get("text"),
                    "label": e.get("label"),
                    "canonical_key": e.get("canonical_key"),
                    "decision": e.get("decision"),
                    "reason": e.get("reason"),
                }
                for e in promoted
            ],
            "actors": [
                e.get("text")
                for e in promoted
                if str(e.get("label") or "").lower() in {"person", "organization"}
            ],
            "places": [
                e.get("text")
                for e in promoted
                if str(e.get("label") or "").lower() == "location"
            ],
            "time": _safe_iso(post.get("timestamp")),
            "relations_rule_accepted": len(rule_relations),
            "relations_v10_accepted": len(kept_relations),
            "relations": kept_relations,
            "relation_gate": {
                "model_dir": relation_model_dir,
                "threshold": relation_threshold,
                "source_mode": "relation_candidates_csv" if relation_rows_by_post else "live_ie_inference",
                "kept": len(kept_relations),
                "dropped": len(scored_relations) - len(kept_relations),
                "top_kept": [
                    {
                        "relation": r.get("relation"),
                        "head": r.get("head"),
                        "tail": r.get("tail"),
                        "model_proba_accept": r.get("model_proba_accept"),
                        "reason_group": r.get("reason_group"),
                        "feature_hints": r.get("model_explanation"),
                    }
                    for r in sorted(
                        kept_relations,
                        key=lambda item: float(item.get("model_proba_accept", 0.0)),
                        reverse=True,
                    )[:5]
                ],
            },
            "confidence": _event_confidence(promoted, kept_relations),
            "embedding": embedding,
            "entity_keys": entity_keys,
            "relation_signatures": relation_signatures,
            "mentions_count": len(mentions),
            "rejected_entities_count": len(rejected),
        }

        storyline_id, assign_score, match = assign_to_storyline_or_create_new(
            event=event_payload,
            storylines=storylines,
            assignment_threshold=assignment_threshold,
        )

        if not storyline_id:
            storyline_seq += 1
            storyline_id = f"storyline_{storyline_seq:04d}"
            new_state = StorylineState(
                storyline_id=storyline_id,
                title=_storyline_title(promoted, kept_relations, storyline_id),
                created_at=event_payload["timestamp"],
                last_timestamp=event_payload["timestamp"],
                centroid_embedding=list(embedding),
                entity_keys=set(entity_keys),
                relation_signatures=set(relation_signatures),
                post_ids=[post_id],
            )
            storylines.append(new_state)
            assignment_reason = "new_storyline_created"
        else:
            st = next(s for s in storylines if s.storyline_id == storyline_id)
            st.centroid_embedding = _vector_average(st.centroid_embedding, embedding, len(st.post_ids))
            st.entity_keys |= entity_keys
            st.relation_signatures |= relation_signatures
            st.last_timestamp = event_payload["timestamp"]
            st.post_ids.append(post_id)
            assignment_reason = "assigned_to_existing_storyline"

        shared_entities = []
        shared_relations = []
        if assignment_reason == "assigned_to_existing_storyline":
            st = next(s for s in storylines if s.storyline_id == storyline_id)
            shared_entities = sorted(entity_keys & st.entity_keys)[:8]
            shared_relations = sorted(relation_signatures & st.relation_signatures)[:8]

        event_payload["storyline_assignment"] = {
            "storyline_id": storyline_id,
            "assignment_confidence": round(assign_score, 4),
            "assignment_threshold": assignment_threshold,
            "reason": assignment_reason,
            "match_details": match,
            "shared_entities": shared_entities,
            "shared_relations": shared_relations,
            "explanation": (
                f"{assignment_reason}; score={round(assign_score, 4)}; "
                f"components={match.get('components', {})}; "
                f"shared_entities={shared_entities[:4]}"
            ),
        }

        events_out.append(event_payload)

        flat_rows.append(
            {
                "post_id": post_id,
                "timestamp": _safe_iso(post.get("timestamp")) or "",
                "storyline_id": storyline_id,
                "assignment_confidence": round(assign_score, 4),
                "event_type": event_payload["event_type"],
                "event_confidence": event_payload["confidence"],
                "relations_rule_accepted": len(rule_relations),
                "relations_v10_accepted": len(kept_relations),
                "embedding_similarity": match.get("components", {}).get("embedding_similarity", 0.0),
                "entity_overlap": match.get("components", {}).get("entity_overlap", 0.0),
                "relation_overlap": match.get("components", {}).get("relation_overlap", 0.0),
                "temporal_proximity": match.get("components", {}).get("temporal_proximity", 0.0),
                "explanation": event_payload["storyline_assignment"]["explanation"],
            }
        )

    storyline_cards = []
    for st in storylines:
        timeline = [
            {
                "post_id": ev["post_id"],
                "timestamp": ev["time"],
                "event_type": ev["event_type"],
                "confidence": ev["confidence"],
                "explanation": ev["storyline_assignment"]["explanation"],
            }
            for ev in events_out
            if ev["storyline_assignment"]["storyline_id"] == st.storyline_id
        ]
        storyline_cards.append(
            {
                "storyline_id": st.storyline_id,
                "title": st.title,
                "posts_count": len(st.post_ids),
                "post_ids": st.post_ids,
                "entity_keys_top": sorted(st.entity_keys)[:12],
                "relation_signatures_top": sorted(st.relation_signatures)[:12],
                "timeline": timeline,
            }
        )

    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = out_base / f"storyline_demo_{ts}.json"
    out_csv = out_base / f"storyline_demo_{ts}.csv"

    report = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_posts": len(posts),
            "relation_model_dir": relation_model_dir,
            "relation_threshold": relation_threshold,
            "relation_candidates_csv": relation_candidates_csv,
            "storyline_assignment_threshold": assignment_threshold,
            "storylines_count": len(storylines),
            "event_type_distribution": Counter(ev["event_type"] for ev in events_out),
        },
        "storyline_cards": storyline_cards,
        "event_mentions": events_out,
    }

    out_json.write_text(
        json.dumps(_to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if flat_rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
    else:
        out_csv.write_text("", encoding="utf-8")

    await engine.dispose()

    return {
        "json_path": str(out_json),
        "csv_path": str(out_csv),
        "posts_processed": len(posts),
        "storylines_count": len(storylines),
    }
