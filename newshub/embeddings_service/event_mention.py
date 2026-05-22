from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


EVENT_SUBTYPE_PRIORITY = [
    "diplomatic_relation",
    "significant_event",
    "participant",
    "part_of",
    "position_held",
    "chairperson",
    "member_of_political_party",
    "member_of",
    "employer",
    "founded_by",
    "owned_by",
    "applies_to_jurisdiction",
    "located_in",
    "headquarters_in",
    "country",
    "citizen_of",
    "country_of_origin",
    "born_in",
    "place_of_death",
]

EVENT_SUBTYPE_TO_COARSE = {
    "diplomatic_relation": "diplomatic",
    "participant": "event_update",
    "part_of": "event_update",
    "significant_event": "event_update",
    "position_held": "personnel",
    "chairperson": "personnel",
    "member_of": "personnel",
    "member_of_political_party": "personnel",
    "employer": "personnel",
    "founded_by": "organizational_change",
    "owned_by": "organizational_change",
    "applies_to_jurisdiction": "jurisdiction_update",
    "located_in": "location_update",
    "headquarters_in": "location_update",
    "country": "location_update",
    "citizen_of": "background_fact",
    "country_of_origin": "background_fact",
    "born_in": "background_fact",
    "place_of_death": "background_fact",
}

RELATION_ROLE_MAP = {
    "diplomatic_relation": ("counterpart", "counterpart"),
    "participant": ("participant", "event_anchor"),
    "part_of": ("subevent", "parent_event"),
    "significant_event": ("context_subject", "event_anchor"),
    "position_held": ("person", "office_or_org"),
    "chairperson": ("organization", "chairperson"),
    "member_of": ("member", "organization"),
    "member_of_political_party": ("member", "political_party"),
    "employer": ("employee", "employer"),
    "founded_by": ("organization", "founder"),
    "owned_by": ("asset", "owner"),
    "applies_to_jurisdiction": ("authority_or_law", "jurisdiction"),
    "located_in": ("subject", "location"),
    "headquarters_in": ("organization", "location"),
    "country": ("subject", "country"),
    "citizen_of": ("person", "country"),
    "country_of_origin": ("subject", "country_of_origin"),
    "born_in": ("person", "birth_place"),
    "place_of_death": ("person", "death_place"),
}

UPDATE_KIND_CUES = {
    "denial": ("опроверг", "опровергла", "отрица", "не подтверд", "не призна"),
    "confirmation": ("подтверд", "подтверди", "согласовал", "одобрил", "одобрила"),
    "followup": ("позже", "затем", "после этого", "далее", "вслед за этим", "обновил"),
}


def _safe_iso(ts: datetime | None) -> str | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().replace("ё", "е").split())


def _relation_value(rel: dict[str, Any], key: str) -> str:
    for field in (f"{key}_key", f"{key}_canonical", key):
        value = str(rel.get(field) or "").strip()
        if value:
            return value
    return ""


def _relation_utility_counts(relations: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    core = 0
    support = 0
    background = 0
    shadow = 0
    for rel in relations:
        bucket = str(rel.get("utility_bucket") or "").strip().lower()
        if bucket == "storyline_core":
            core += 1
        elif bucket == "storyline_support":
            support += 1
        elif bucket == "background_static":
            background += 1
        else:
            shadow += 1
    return core, support, background, shadow


def _event_subtype(relations: list[dict[str, Any]]) -> str:
    relation_set = {str(r.get("relation") or "") for r in relations}
    for relation in EVENT_SUBTYPE_PRIORITY:
        if relation in relation_set:
            return relation
    return "general_news"


def _event_type_coarse(relations: list[dict[str, Any]]) -> str:
    subtype = _event_subtype(relations)
    return EVENT_SUBTYPE_TO_COARSE.get(subtype, "general_news")


def _update_kind(text: str, subtype: str) -> str:
    norm_text = _normalize_text(text)
    for kind, cues in UPDATE_KIND_CUES.items():
        if any(cue in norm_text for cue in cues):
            return kind
    if subtype in {"position_held", "chairperson", "member_of", "member_of_political_party", "employer"}:
        return "personnel_update"
    if subtype in {"participant", "part_of", "significant_event", "diplomatic_relation"}:
        return "new_fact"
    if subtype == "general_news":
        return "general_update"
    return "attribute_update"


def _status(update_kind: str) -> str:
    if update_kind == "denial":
        return "denied"
    if update_kind == "confirmation":
        return "confirmed"
    return "reported"


def _event_signatures(
    *,
    event_type_coarse: str,
    subtype: str,
    update_kind: str,
    participant_role_signatures: list[str],
    location_keys: list[str],
) -> list[str]:
    signatures: list[str] = [
        f"event_frame:{event_type_coarse}:{subtype}",
        f"event_subtype:{subtype}",
        f"update_kind:{update_kind}",
    ]
    for sig in participant_role_signatures[:8]:
        clean = str(sig or "").strip()
        if clean:
            signatures.append(f"participant_role:{clean}")
    for key in location_keys[:4]:
        clean = str(key or "").strip()
        if clean:
            signatures.append(f"event_place:{clean}")
    deduped: list[str] = []
    seen = set()
    for sig in signatures:
        if sig not in seen:
            seen.add(sig)
            deduped.append(sig)
    return deduped


def _build_participants(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    participants: dict[str, dict[str, Any]] = {}
    location_keys: set[str] = set()

    for rel in relations:
        relation = str(rel.get("relation") or "")
        roles = RELATION_ROLE_MAP.get(relation)
        for side in ("head", "tail"):
            entity_key = _relation_value(rel, side)
            if not entity_key:
                continue
            role = "participant"
            if roles:
                role = roles[0] if side == "head" else roles[1]
            item = participants.setdefault(
                entity_key,
                {
                    "entity_key": entity_key,
                    "name": str(rel.get(f"{side}_canonical") or rel.get(side) or entity_key),
                    "roles": set(),
                },
            )
            item["roles"].add(role)
            if role in {"location", "country", "jurisdiction", "birth_place", "death_place"}:
                location_keys.add(entity_key)

    for entity in entities:
        entity_key = str(entity.get("canonical_key") or entity.get("text") or "").strip()
        if not entity_key:
            continue
        label = str(entity.get("label") or "").strip().lower()
        role = None
        if label in {"person", "organization"}:
            role = "actor"
        elif label == "location":
            role = "place"
            location_keys.add(entity_key)
        if not role:
            continue
        item = participants.setdefault(
            entity_key,
            {
                "entity_key": entity_key,
                "name": str(entity.get("text") or entity_key),
                "roles": set(),
            },
        )
        item["roles"].add(role)

    participant_rows: list[dict[str, str]] = []
    participant_role_signatures: set[str] = set()
    for entity_key, item in participants.items():
        roles = sorted(item["roles"])
        primary_role = roles[0] if roles else "participant"
        participant_rows.append(
            {
                "entity_key": entity_key,
                "name": str(item.get("name") or entity_key),
                "role": primary_role,
            }
        )
        for role in roles:
            participant_role_signatures.add(f"{role}:{entity_key}")

    participant_rows.sort(key=lambda row: (row["role"], row["entity_key"]))
    return participant_rows[:20], sorted(location_keys)[:20], sorted(participant_role_signatures)[:40]


def _confidence_breakdown(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> dict[str, float]:
    relation_mean = 0.0
    if relations:
        relation_mean = sum(float(r.get("confidence") or 0.0) for r in relations) / len(relations)
    entity_signal = min(1.0, len(entities) / 6.0)
    relation_signal = min(1.0, len(relations) / 4.0)
    storyline_core_count, storyline_support_count, background_count, _shadow_count = _relation_utility_counts(relations)
    core_relation_signal = min(1.0, storyline_core_count / 3.0)
    support_relation_signal = min(1.0, storyline_support_count / 3.0)
    total = (
        0.35 * relation_mean
        + 0.25 * entity_signal
        + 0.15 * relation_signal
        + 0.15 * core_relation_signal
        + 0.10 * support_relation_signal
    )
    return {
        "relation_mean": round(relation_mean, 4),
        "entity_signal": round(entity_signal, 4),
        "relation_signal": round(relation_signal, 4),
        "core_relation_signal": round(core_relation_signal, 4),
        "support_relation_signal": round(support_relation_signal, 4),
        "storyline_core_relation_count": float(storyline_core_count),
        "storyline_support_relation_count": float(storyline_support_count),
        "background_relation_count": float(background_count),
        "total": round(max(0.0, min(1.0, total)), 4),
    }


def build_event_mention_lite(
    *,
    post_id: int,
    text: str,
    timestamp: datetime | None,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    entity_usage_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    participants, location_keys, participant_role_signatures = _build_participants(entities, relations)
    subtype = _event_subtype(relations)
    event_type_coarse = _event_type_coarse(relations)
    update_kind = _update_kind(text, subtype)
    status = _status(update_kind)
    confidence_breakdown = _confidence_breakdown(entities, relations)
    storyline_core_count, storyline_support_count, background_count, shadow_count = _relation_utility_counts(relations)
    actors = [row["name"] for row in participants if row["role"] not in {"place", "location"}]
    places = [
        str(entity.get("text") or "")
        for entity in entities
        if str(entity.get("label") or "").lower() == "location"
    ]
    event_signature = f"{event_type_coarse}:{subtype}"
    event_signatures = _event_signatures(
        event_type_coarse=event_type_coarse,
        subtype=subtype,
        update_kind=update_kind,
        participant_role_signatures=participant_role_signatures,
        location_keys=location_keys,
    )

    payload = {
        "id": f"event_mention_{post_id}",
        "post_id": post_id,
        "event_type": event_type_coarse,
        "event_type_coarse": event_type_coarse,
        "event_subtype": subtype,
        "event_signature": event_signature,
        "event_signatures": event_signatures,
        "summary": (text or "")[:500],
        "time": _safe_iso(timestamp),
        "event_time_text": None,
        "event_time_resolved": _safe_iso(timestamp),
        "update_kind": update_kind,
        "status": status,
        "actors": [a for a in actors if a][:20],
        "places": [p for p in places if p][:20],
        "participants": participants,
        "participant_keys": [row["entity_key"] for row in participants],
        "participant_role_signatures": participant_role_signatures,
        "location_keys": location_keys,
        "entities_count": len(entities),
        "relations_count": len(relations),
        "storyline_core_relation_count": storyline_core_count,
        "storyline_support_relation_count": storyline_support_count,
        "background_relation_count": background_count,
        "shadow_relation_count": shadow_count,
        "confidence_breakdown": confidence_breakdown,
        "confidence": confidence_breakdown["total"],
    }

    if entity_usage_stats:
        payload.update(
            {
                "core_entities_count": int(entity_usage_stats.get("core_entities_count", 0) or 0),
                "support_entities_count": int(entity_usage_stats.get("support_entities_count", 0) or 0),
                "event_argument_entities_count": int(
                    entity_usage_stats.get("event_argument_entities_count", 0) or 0
                ),
                "event_argument_entities_from_mentions": int(
                    entity_usage_stats.get("event_argument_entities_from_mentions", 0) or 0
                ),
            }
        )

    return payload
