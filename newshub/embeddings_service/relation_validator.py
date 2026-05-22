import json
import os
import re
from pathlib import Path

try:
    from embeddings_service.entity_normalizer import normalize_entity_text
except Exception:
    try:
        from entity_normalizer import normalize_entity_text
    except Exception:
        def normalize_entity_text(text: str) -> str:
            t = (text or "").strip().lower()
            t = t.replace("ё", "е")
            t = re.sub(r"\s+", " ", t)
            return t.strip(".,;:!?\"'()[]{}")


GENERIC_HEADTAIL = {
    "как сообщает",
    "по данным",
    "новый лидер",
    "в 2024",
    "в 2025",
    "в 2026",
}

BLOCKED_ENTITY_REASONS = {
    "generic_person_role",
    "generic_org",
    "generic_location",
    "generic_event",
    "single_token_event",
    "single_token_product",
    "single_token_law",
    "descriptor_only_org",
    "weak_org_phrase",
    "descriptor_location",
    "suspicious_location_form",
    "suspicious_person_phrase",
    "descriptor_role_person",
    "role_person",
}

DEFAULT_RELATION_SCHEMA = {
    "version": "v1",
    "aliases": {
        "located in": "located_in",
        "location": "located_in",
        "located in the administrative territorial entity": "located_in",
        "located in or next to body of water": "located_in",
        "place of birth": "born_in",
        "country of citizenship": "citizen_of",
        "founded by": "founded_by",
        "owned by": "owned_by",
        "instance of": "instance_of",
        "subclass of": "subclass_of",
        "point in time": "point_in_time",
        "headquarters location": "headquarters_in",
        "position held": "position_held",
        "member of": "member_of",
        "diplomatic relation": "diplomatic_relation",
        "language used": "language_used",
        "military rank": "military_rank",
        "significant event": "significant_event",
        "applies to jurisdiction": "applies_to_jurisdiction",
        "member of political party": "member_of_political_party",
        "country of origin": "country_of_origin",
        "place of death": "place_of_death",
        "spouse": "spouse_of",
    },
    "blocked_relations": [
        "related_to",
        "relation",
        "is",
        "are",
        "point_in_time",
    ],
    "weak_relations": [
        "creator",
        "author",
        "publisher",
        "winner",
        "sport",
        "manufacturer",
        "occupation",
    ],
    "symmetric_relations": [
        "ally_of",
        "partner_of",
        "competitor_of",
        "associated_with",
        "sibling_of",
        "spouse_of",
        "diplomatic_relation",
        "associated_with",
    ],
    "swappable_relations": [
        "located_in",
        "member_of",
        "part_of",
        "participant",
        "position_held",
        "country",
        "subclass_of",
        "instance_of",
        "chairperson",
    ],
    "allowed_relations": [
        "country",
        "subclass_of",
        "instance_of",
        "located_in",
        "headquarters_in",
        "participant",
        "part_of",
        "member_of",
        "chairperson",
        "position_held",
        "military_rank",
        "language_used",
        "applies_to_jurisdiction",
        "founded_by",
        "owned_by",
        "significant_event",
        "diplomatic_relation",
        "citizen_of",
        "developer",
        "subsidiary",
        "facet_of",
        "legislative_body",
        "employer",
        "author",
        "manufacturer",
        "occupation",
        "country_of_origin",
        "member_of_political_party",
        "born_in",
        "place_of_death",
        "spouse_of",
        "associated_with",
    ],
    "storyline_core_relations": [
        "participant",
        "part_of",
        "position_held",
        "chairperson",
        "member_of",
        "member_of_political_party",
        "founded_by",
        "owned_by",
        "significant_event",
        "diplomatic_relation",
        "employer",
        "applies_to_jurisdiction",
    ],
    "background_static_relations": [
        "country",
        "subclass_of",
        "instance_of",
        "located_in",
        "headquarters_in",
        "military_rank",
        "language_used",
        "citizen_of",
        "developer",
        "subsidiary",
        "facet_of",
        "legislative_body",
        "author",
        "manufacturer",
        "occupation",
        "country_of_origin",
        "born_in",
        "place_of_death",
        "spouse_of",
    ],
    "storyline_support_relations": [
        "located_in",
        "headquarters_in",
        "country",
        "citizen_of",
        "country_of_origin",
        "born_in",
        "place_of_death",
        "developer",
        "manufacturer",
        "author",
        "legislative_body",
        "associated_with",
    ],
    "type_constraints": {
        "country": [["organization", "location"], ["person", "location"], ["event", "location"]],
        "located_in": [["*", "location"]],
        "headquarters_in": [["organization", "location"]],
        "member_of": [["person", "organization"], ["organization", "organization"]],
        "participant": [["person", "event"], ["organization", "event"], ["event", "event"]],
        "part_of": [["event", "event"], ["organization", "organization"], ["location", "location"]],
        "position_held": [["person", "organization"], ["person", "person"]],
        "military_rank": [["person", "organization"], ["person", "product"]],
        "language_used": [["organization", "product"], ["location", "product"]],
        "chairperson": [["organization", "person"]],
        "subclass_of": [["*", "*"]],
        "instance_of": [["*", "*"]],
        "significant_event": [["location", "event"], ["organization", "event"], ["person", "event"]],
        "applies_to_jurisdiction": [["organization", "location"], ["law", "location"]],
        "citizen_of": [["person", "location"]],
        "founded_by": [["organization", "person"], ["organization", "organization"]],
        "owned_by": [["organization", "organization"]],
        "diplomatic_relation": [["location", "location"], ["organization", "organization"]],
        "developer": [["product", "organization"], ["product", "person"]],
        "subsidiary": [["organization", "organization"]],
        "facet_of": [["event", "event"], ["product", "product"]],
        "legislative_body": [["location", "organization"], ["organization", "organization"]],
        "employer": [["person", "organization"], ["person", "product"]],
        "author": [["person", "product"], ["person", "event"], ["organization", "product"]],
        "manufacturer": [["product", "organization"], ["product", "location"]],
        "occupation": [["person", "organization"], ["person", "product"], ["person", "event"]],
        "country_of_origin": [["organization", "location"], ["product", "location"], ["person", "location"]],
        "member_of_political_party": [["person", "organization"]],
        "born_in": [["person", "location"]],
        "place_of_death": [["person", "location"]],
        "spouse_of": [["person", "person"]],
        "associated_with": [["*", "*"]],
    },
}


def _load_relation_schema() -> dict:
    schema_path = os.getenv(
        "RELATION_SCHEMA_PATH",
        str(Path(__file__).with_name("relation_schema_v1.json")),
    )
    try:
        with Path(schema_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return DEFAULT_RELATION_SCHEMA


RELATION_SCHEMA = _load_relation_schema()
RELATION_SCHEMA_VERSION = RELATION_SCHEMA.get("version", "v1")
RELATION_ALIASES = RELATION_SCHEMA.get("aliases", {})
BAD_RELATIONS = set(RELATION_SCHEMA.get("blocked_relations", []))
WEAK_RELATIONS = set(RELATION_SCHEMA.get("weak_relations", []))
SYMMETRIC_RELATIONS = set(RELATION_SCHEMA.get("symmetric_relations", []))
SWAPPABLE_RELATIONS = set(RELATION_SCHEMA.get("swappable_relations", []))
ALLOWED_RELATIONS = set(RELATION_SCHEMA.get("allowed_relations", []))
STORYLINE_CORE_RELATIONS = set(RELATION_SCHEMA.get("storyline_core_relations", []))
BACKGROUND_STATIC_RELATIONS = set(RELATION_SCHEMA.get("background_static_relations", []))
STORYLINE_SUPPORT_RELATIONS = set(RELATION_SCHEMA.get("storyline_support_relations", []))
TYPE_CONSTRAINTS = RELATION_SCHEMA.get("type_constraints", {})
STRICT_MATCH_THRESHOLD = float(os.getenv("RELATION_STRICT_MATCH_THRESHOLD", "0.8"))
SOFT_MATCH_THRESHOLD = float(os.getenv("RELATION_SOFT_MATCH_THRESHOLD", "0.25"))
GRAPH_WRITE_MIN_CONFIDENCE = float(os.getenv("RELATION_GRAPH_WRITE_MIN_CONFIDENCE", "0.72"))
ENABLE_PROVISIONAL_RELATIONS = os.getenv("ENABLE_PROVISIONAL_RELATIONS", "1") != "0"
LINK_REPAIR_MARGIN = float(os.getenv("RELATION_LINK_REPAIR_MARGIN", "0.85"))
RELATION_ALLOW_MENTION_LINKING = os.getenv("RELATION_ALLOW_MENTION_LINKING", "1") != "0"
RELATION_MENTION_MIN_MODEL_PROBA = float(os.getenv("RELATION_MENTION_MIN_MODEL_PROBA", "0.75"))
RELATION_MENTION_MIN_GLINER_SCORE = float(os.getenv("RELATION_MENTION_MIN_GLINER_SCORE", "0.85"))
RELATION_STORYLINE_ANCHOR_MENTION_MIN_MODEL_PROBA = float(
    os.getenv("RELATION_STORYLINE_ANCHOR_MENTION_MIN_MODEL_PROBA", "0.35")
)
RELATION_STORYLINE_ANCHOR_MENTION_MIN_GLINER_SCORE = float(
    os.getenv("RELATION_STORYLINE_ANCHOR_MENTION_MIN_GLINER_SCORE", "0.55")
)
RELATION_MENTION_ALLOWED_LABELS = set(
    x.strip().lower()
    for x in os.getenv("RELATION_MENTION_ALLOWED_LABELS", "person,organization,location,product,law,event").split(",")
    if x.strip()
)
RELATION_MENTION_ALLOWED_REASONS = set(
    x.strip().lower()
    for x in os.getenv(
        "RELATION_MENTION_ALLOWED_REASONS",
        "single_token_person,single_token_org,named_person,named_org,known_location,named_location,single_token_location,valid_short_person,valid_short_org,source_mention,product_mention_only,named_product_topic,single_token_product,law_mention_only,event_mention_only,single_token_event",
    ).split(",")
    if x.strip()
)

RELATION_ARG_NOISE_TOKENS = {
    "карточка",
    "карточки",
    "главреда",
    "главред",
    "поста",
    "пост",
    "сообщил",
    "сообщает",
    "сообщению",
    "заявил",
    "заявила",
}


def _looks_like_relation_phrase(text: str) -> bool:
    tokens = re.findall(r"[a-zа-я0-9]+", text or "")
    if len(tokens) < 2 or len(tokens) > 6:
        return False
    if any(tok.isdigit() for tok in tokens):
        return False
    return True


def _heuristic_relation_alias(rel: str) -> str:
    t = (rel or "").strip().lower()
    if not t:
        return ""
    checks = [
        (("administrative territorial entity", "body of water", "located in", "location"), "located_in"),
        (("headquarters",), "headquarters_in"),
        (("country of citizenship", "citizenship"), "citizen_of"),
        (("country of origin",), "country_of_origin"),
        (("place of birth", "birth place"), "born_in"),
        (("place of death", "death place"), "place_of_death"),
        (("member of political party", "political party"), "member_of_political_party"),
        (("member of",), "member_of"),
        (("position held", "office held"), "position_held"),
        (("significant event",), "significant_event"),
        (("applies to jurisdiction", "jurisdiction"), "applies_to_jurisdiction"),
        (("founded by", "founder"), "founded_by"),
        (("owned by", "owner"), "owned_by"),
        (("diplomatic relation", "relations with"), "diplomatic_relation"),
        (("developer",), "developer"),
        (("manufacturer",), "manufacturer"),
        (("legislative body",), "legislative_body"),
        (("language used", "official language"), "language_used"),
        (("military rank",), "military_rank"),
        (("participant",), "participant"),
        (("part of", "component of"), "part_of"),
        (("instance of",), "instance_of"),
        (("subclass of",), "subclass_of"),
        (("employer",), "employer"),
        (("author",), "author"),
        (("subsidiary",), "subsidiary"),
        (("chairperson",), "chairperson"),
    ]
    for needles, target in checks:
        if any(needle in t for needle in needles):
            return target
    if any(needle in t for needle in ("shares border with", "military branch", "allied with", "connected to")):
        return "associated_with"
    if _looks_like_relation_phrase(t):
        return "associated_with"
    return ""


def normalize_relation(rel: str) -> str:
    t = (rel or "").strip().lower()
    t = t.replace("ё", "е")
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t)
    t = t.strip(".,;:!?\"'()[]{}")
    direct = t.replace(" ", "_")
    if direct in BAD_RELATIONS:
        return direct
    aliased = RELATION_ALIASES.get(t, "")
    if aliased:
        return aliased.replace(" ", "_")
    heuristic = _heuristic_relation_alias(t)
    if heuristic:
        return heuristic.replace(" ", "_")
    return t.replace(" ", "_")


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zа-я0-9\-]+", normalize_entity_text(text)))


def _clean_relation_argument(text: str, relation: str) -> str:
    base = normalize_entity_text(text)
    tokens = re.findall(r"[a-zа-я0-9\-]+", base)
    if not tokens:
        return base
    if relation in {"position_held", "citizen_of", "located_in"}:
        cleaned = [t for t in tokens if t not in RELATION_ARG_NOISE_TOKENS]
        if cleaned:
            return " ".join(cleaned)
    return base


def _entity_type(entity: dict | None) -> str:
    if not entity:
        return ""
    return (entity.get("label") or entity.get("original_label") or "").strip().lower()


def _entity_is_relation_eligible(entity: dict | None) -> bool:
    if not entity:
        return False

    decision = (entity.get("decision") or "").strip().lower()
    reason = (entity.get("reason") or "").strip().lower()
    label = (entity.get("label") or entity.get("original_label") or "").strip().lower()
    model_proba = entity.get("decision_model_proba")
    gliner_score = entity.get("gliner_score")
    storyline_anchor_eligible = bool(entity.get("storyline_anchor_eligible"))
    rag_anchor_eligible = bool(entity.get("rag_anchor_eligible"))
    support_anchor_eligible = storyline_anchor_eligible or rag_anchor_eligible

    mention_override_allowed = bool(
        decision == "mention"
        and RELATION_ALLOW_MENTION_LINKING
        and label
        and (not RELATION_MENTION_ALLOWED_LABELS or label in RELATION_MENTION_ALLOWED_LABELS)
        and reason
        and (not RELATION_MENTION_ALLOWED_REASONS or reason in RELATION_MENTION_ALLOWED_REASONS)
    )

    if reason in BLOCKED_ENTITY_REASONS and not support_anchor_eligible and not mention_override_allowed:
        return False

    if not decision or decision == "promote":
        return True

    if decision == "mention" and RELATION_ALLOW_MENTION_LINKING:
        try:
            mp = float(model_proba) if model_proba is not None else 0.0
        except Exception:
            mp = 0.0
        try:
            gs = float(gliner_score) if gliner_score is not None else 0.0
        except Exception:
            gs = 0.0

        if support_anchor_eligible:
            if label and label not in {"person", "organization", "location", "product", "law", "event"}:
                return False
            return (
                mp >= RELATION_STORYLINE_ANCHOR_MENTION_MIN_MODEL_PROBA
                or gs >= RELATION_STORYLINE_ANCHOR_MENTION_MIN_GLINER_SCORE
            )

        if label and RELATION_MENTION_ALLOWED_LABELS and label not in RELATION_MENTION_ALLOWED_LABELS:
            return False
        if reason and RELATION_MENTION_ALLOWED_REASONS and reason not in RELATION_MENTION_ALLOWED_REASONS:
            return False

        return mp >= RELATION_MENTION_MIN_MODEL_PROBA or gs >= RELATION_MENTION_MIN_GLINER_SCORE

    return False


def _entity_priority_bonus(entity: dict | None) -> float:
    if not entity:
        return 0.0
    bonus = 0.0
    if bool(entity.get("storyline_anchor_eligible")):
        bonus += 0.08
    elif bool(entity.get("rag_anchor_eligible")):
        bonus += 0.06
    decision = (entity.get("decision") or "").strip().lower()
    if decision == "promote":
        bonus += 0.04
    elif decision == "mention":
        bonus += 0.01
    reason = (entity.get("reason") or "").strip().lower()
    if reason in {"named_product_topic", "law_mention_only", "event_mention_only", "source_mention"}:
        bonus += 0.03
    return bonus


def _best_entity_candidate(
    value: str,
    entities: list[dict],
    *,
    eligible_only: bool,
) -> tuple[dict | None, float]:
    value_norm = normalize_entity_text(value)
    value_tokens = _token_set(value_norm)
    if not value_tokens:
        return None, 0.0

    candidates = [
        e for e in entities
        if (not eligible_only or _entity_is_relation_eligible(e))
    ]
    if not candidates:
        return None, 0.0

    best = None
    best_score = 0.0

    for e in candidates:
        variants = {
            normalize_entity_text(e.get("text", "")),
            normalize_entity_text(e.get("raw_text", "")),
        }
        canonical_key = normalize_entity_text(e.get("canonical_key", ""))
        if ":" in canonical_key:
            variants.add(normalize_entity_text(canonical_key.split(":", 1)[1].replace("_", " ")))
        variants = {v for v in variants if v}
        if value_norm in variants:
            return e, 1.0
        for variant in variants:
            if len(variant) >= 4 and variant in value_norm:
                variant_tokens = _token_set(variant)
                coverage = len(variant_tokens) / len(value_tokens) if value_tokens else 0.0
                score = min(1.0, 0.84 + 0.12 * coverage + _entity_priority_bonus(e))
                if score > best_score:
                    best_score = score
                    best = e

        entity_tokens = set()
        for v in variants:
            entity_tokens |= _token_set(v)

        if not entity_tokens:
            continue

        inter = len(value_tokens & entity_tokens)
        union = len(value_tokens | entity_tokens)
        jaccard = inter / union if union else 0.0
        recall_like = inter / len(value_tokens) if value_tokens else 0.0
        precision_like = inter / len(entity_tokens) if entity_tokens else 0.0
        subset_bonus = 0.0
        if entity_tokens and entity_tokens <= value_tokens:
            subset_bonus += 0.22
        if value_tokens and value_tokens <= entity_tokens:
            subset_bonus += 0.1
        score = min(
            1.0,
            0.45 * jaccard + 0.25 * recall_like + 0.15 * precision_like + subset_bonus + _entity_priority_bonus(e),
        )
        if score > best_score:
            best_score = score
            best = e

    return best, best_score


def _relation_type_compatible(relation: str, head_type: str, tail_type: str) -> bool:
    constraints = TYPE_CONSTRAINTS.get(relation) or []
    if not constraints:
        return True

    for pair in constraints:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        allowed_head, allowed_tail = pair
        head_ok = allowed_head == "*" or allowed_head == head_type
        tail_ok = allowed_tail == "*" or allowed_tail == tail_type
        if head_ok and tail_ok:
            return True
    return False


def _proximity_signal(head_match: dict | None, tail_match: dict | None) -> float:
    if not head_match or not tail_match:
        return 0.0

    head_start = head_match.get("span_start")
    tail_start = tail_match.get("span_start")
    if not isinstance(head_start, int) or not isinstance(tail_start, int):
        return 0.0

    distance = abs(head_start - tail_start)
    return max(0.0, 1.0 - min(distance, 200) / 200.0)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _relation_confidence(
    relation: str,
    head_score: float,
    tail_score: float,
    proximity_score: float,
    *,
    swapped: bool,
    provisional: bool,
) -> float:
    score = 0.45 + 0.2 * head_score + 0.2 * tail_score + 0.15 * proximity_score
    if relation in WEAK_RELATIONS:
        score -= 0.08
    if swapped:
        score -= 0.07
    if provisional:
        score -= 0.12
    return round(_clamp01(score), 4)


def _reason_group(reason: str) -> str:
    if reason in {
        "head_not_in_entities",
        "tail_not_in_entities",
        "head_entity_blocked",
        "tail_entity_blocked",
    }:
        return "linking_error"
    if reason in {"type_mismatch", "direction_mismatch", "unknown_relation", "bad_relation"}:
        return "schema_error"
    if reason in {"duplicate", "symmetric_duplicate"}:
        return "dedup"
    if reason in {"ok", "direction_swapped_auto", "provisional_link_match"}:
        return "accepted"
    return "content_error"


def relation_storyline_utility(relation: str) -> str:
    rel = normalize_relation(relation)
    if rel in STORYLINE_CORE_RELATIONS:
        return "storyline_core"
    if rel in STORYLINE_SUPPORT_RELATIONS:
        return "storyline_support"
    if rel in BACKGROUND_STATIC_RELATIONS:
        return "background_static"
    return "drop_or_shadow"


def _candidate_payload(entity: dict | None, score: float) -> tuple[str, str, float]:
    if not entity:
        return "", "", round(score, 4)
    return (
        normalize_entity_text(entity.get("text", "")),
        (entity.get("canonical_key") or "").strip(),
        round(score, 4),
    )


def validate_relation(
    triplet: dict,
    entities: list[dict],
) -> tuple[bool, str, dict | None, dict | None, dict]:
    rel = normalize_relation(triplet.get("relation", ""))
    head_raw = normalize_entity_text(triplet.get("head", ""))
    tail_raw = normalize_entity_text(triplet.get("tail", ""))
    head = _clean_relation_argument(head_raw, rel)
    tail = _clean_relation_argument(tail_raw, rel)

    head_candidate, head_candidate_score = _best_entity_candidate(head, entities, eligible_only=False)
    tail_candidate, tail_candidate_score = _best_entity_candidate(tail, entities, eligible_only=False)
    head_cand_text, head_cand_key, head_cand_score = _candidate_payload(head_candidate, head_candidate_score)
    tail_cand_text, tail_cand_key, tail_cand_score = _candidate_payload(tail_candidate, tail_candidate_score)

    meta = {
        "schema_version": RELATION_SCHEMA_VERSION,
        "head_match_candidate": head_cand_text,
        "tail_match_candidate": tail_cand_text,
        "head_match_candidate_key": head_cand_key,
        "tail_match_candidate_key": tail_cand_key,
        "head_match_score": head_cand_score,
        "tail_match_score": tail_cand_score,
        "swapped": False,
        "provisional": False,
        "write_to_graph": False,
        "proximity_score": 0.0,
        "confidence": 0.0,
        "head_cleaned": head,
        "tail_cleaned": tail,
        "link_repair_head": False,
        "link_repair_tail": False,
    }

    if not head or not tail or not rel:
        return False, "empty_part", None, None, meta
    if head == tail:
        return False, "self_loop", None, None, meta
    if rel == head or rel == tail:
        return False, "relation_equals_arg", None, None, meta
    if rel in BAD_RELATIONS:
        return False, "bad_relation", None, None, meta
    if rel not in ALLOWED_RELATIONS:
        return False, "unknown_relation", None, None, meta
    if len(rel) < 3:
        return False, "too_short_relation", None, None, meta
    if len(rel.split("_")) > 6:
        return False, "too_long_relation", None, None, meta
    if any(ch.isdigit() for ch in rel):
        return False, "digit_in_relation", None, None, meta
    if head in GENERIC_HEADTAIL or tail in GENERIC_HEADTAIL:
        return False, "generic_arg", None, None, meta
    if head.isdigit() or tail.isdigit():
        return False, "digit_arg", None, None, meta
    if len(head.split()) > 10 or len(tail.split()) > 10:
        return False, "too_long_arg", None, None, meta

    head_match, head_score = _best_entity_candidate(head, entities, eligible_only=True)
    tail_match, tail_score = _best_entity_candidate(tail, entities, eligible_only=True)

    strict_head = head_score >= STRICT_MATCH_THRESHOLD
    strict_tail = tail_score >= STRICT_MATCH_THRESHOLD

    if head_score < SOFT_MATCH_THRESHOLD:
        head_match = None
    if tail_score < SOFT_MATCH_THRESHOLD:
        tail_match = None

    repair_min = SOFT_MATCH_THRESHOLD * LINK_REPAIR_MARGIN
    if not head_match and head_candidate and head_candidate_score >= repair_min and _entity_is_relation_eligible(head_candidate):
        head_match = head_candidate
        head_score = head_candidate_score
        meta["link_repair_head"] = True
    if not tail_match and tail_candidate and tail_candidate_score >= repair_min and _entity_is_relation_eligible(tail_candidate):
        tail_match = tail_candidate
        tail_score = tail_candidate_score
        meta["link_repair_tail"] = True

    if not head_match:
        return False, "head_not_in_entities", None, tail_match, meta
    if not tail_match:
        return False, "tail_not_in_entities", head_match, None, meta

    if not _entity_is_relation_eligible(head_match):
        return False, "head_entity_blocked", head_match, tail_match, meta
    if not _entity_is_relation_eligible(tail_match):
        return False, "tail_entity_blocked", head_match, tail_match, meta

    if rel in WEAK_RELATIONS and len(head.split()) == 1 and len(tail.split()) == 1:
        return False, "underspecified_relation", head_match, tail_match, meta

    if normalize_entity_text(head_match.get("text", "")) == normalize_entity_text(tail_match.get("text", "")):
        return False, "same_canonical_entity", head_match, tail_match, meta

    head_type = _entity_type(head_match)
    tail_type = _entity_type(tail_match)
    meta["head_type"] = head_type
    meta["tail_type"] = tail_type

    if not _relation_type_compatible(rel, head_type, tail_type):
        if rel in SWAPPABLE_RELATIONS and _relation_type_compatible(rel, tail_type, head_type):
            head_match, tail_match = tail_match, head_match
            head_score, tail_score = tail_score, head_score
            meta["swapped"] = True
        else:
            return False, "type_mismatch", head_match, tail_match, meta

    proximity_score = _proximity_signal(head_match, tail_match)
    meta["proximity_score"] = round(proximity_score, 4)
    meta["head_match_score"] = round(head_score, 4)
    meta["tail_match_score"] = round(tail_score, 4)
    meta["confidence"] = _relation_confidence(
        rel,
        head_score,
        tail_score,
        proximity_score,
        swapped=meta["swapped"],
        provisional=not (strict_head and strict_tail),
    )
    if strict_head and strict_tail:
        meta["write_to_graph"] = meta["confidence"] >= GRAPH_WRITE_MIN_CONFIDENCE
    elif ENABLE_PROVISIONAL_RELATIONS:
        meta["provisional"] = True
        meta["write_to_graph"] = False
    else:
        if not strict_head:
            return False, "head_not_in_entities", None, tail_match, meta
        if not strict_tail:
            return False, "tail_not_in_entities", head_match, None, meta

    if rel in SWAPPABLE_RELATIONS and meta["swapped"]:
        return True, "direction_swapped_auto", head_match, tail_match, meta
    if meta["provisional"]:
        return True, "provisional_link_match", head_match, tail_match, meta
    return True, "ok", head_match, tail_match, meta


def filter_relations(relations: list[dict], entities: list[dict]) -> tuple[list[dict], list[dict]]:
    accepted = []
    rejected = []
    seen = set()

    for rel in relations:
        ok, reason, head_match, tail_match, meta = validate_relation(rel, entities)

        item = {
            "head": normalize_entity_text(rel.get("head", "")),
            "tail": normalize_entity_text(rel.get("tail", "")),
            "relation": normalize_relation(rel.get("relation", "")),
            "reason": reason,
            "reason_group": _reason_group(reason),
            "schema_version": meta.get("schema_version"),
            "head_match_candidate": meta.get("head_match_candidate"),
            "tail_match_candidate": meta.get("tail_match_candidate"),
            "head_match_candidate_key": meta.get("head_match_candidate_key"),
            "tail_match_candidate_key": meta.get("tail_match_candidate_key"),
            "head_match_score": meta.get("head_match_score"),
            "tail_match_score": meta.get("tail_match_score"),
            "proximity_score": meta.get("proximity_score"),
            "confidence": meta.get("confidence"),
            "swapped": bool(meta.get("swapped")),
            "provisional": bool(meta.get("provisional")),
            "write_to_graph": bool(meta.get("write_to_graph")),
            "utility_bucket": relation_storyline_utility(rel.get("relation", "")),
        }
        item["is_storyline_core"] = item["utility_bucket"] == "storyline_core"

        if head_match:
            item["head_canonical"] = normalize_entity_text(head_match.get("text", ""))
            item["head_key"] = head_match.get("canonical_key")
            item["head_type"] = _entity_type(head_match)
            item["head_decision"] = (head_match.get("decision") or "").strip().lower()
            item["head_reason"] = (head_match.get("reason") or "").strip().lower()
        if tail_match:
            item["tail_canonical"] = normalize_entity_text(tail_match.get("text", ""))
            item["tail_key"] = tail_match.get("canonical_key")
            item["tail_type"] = _entity_type(tail_match)
            item["tail_decision"] = (tail_match.get("decision") or "").strip().lower()
            item["tail_reason"] = (tail_match.get("reason") or "").strip().lower()

        if not ok:
            rejected.append(item)
            continue

        canon_head = normalize_entity_text(head_match["text"])
        canon_tail = normalize_entity_text(tail_match["text"])
        head_key = head_match.get("canonical_key")
        tail_key = tail_match.get("canonical_key")

        left = head_key or canon_head
        right = tail_key or canon_tail
        relation_type = item["relation"]

        key = (left, relation_type, right)
        symmetric_key = None
        if relation_type in SYMMETRIC_RELATIONS:
            a, b = sorted([left, right])
            symmetric_key = (a, relation_type, b)

        if key in seen:
            item["reason"] = "duplicate"
            item["reason_group"] = _reason_group("duplicate")
            rejected.append(item)
            continue
        if symmetric_key and symmetric_key in seen:
            item["reason"] = "symmetric_duplicate"
            item["reason_group"] = _reason_group("symmetric_duplicate")
            rejected.append(item)
            continue

        seen.add(key)
        if symmetric_key:
            seen.add(symmetric_key)
        accepted.append(
            {
                "head": canon_head,
                "tail": canon_tail,
                "relation": item["relation"],
                "head_key": head_key,
                "tail_key": tail_key,
                "head_canonical": item.get("head_canonical"),
                "tail_canonical": item.get("tail_canonical"),
                "head_type": item.get("head_type"),
                "tail_type": item.get("tail_type"),
                "head_decision": item.get("head_decision"),
                "tail_decision": item.get("tail_decision"),
                "head_reason": item.get("head_reason"),
                "tail_reason": item.get("tail_reason"),
                "reason": item["reason"],
                "reason_group": item["reason_group"],
                "confidence": item["confidence"],
                "proximity_score": item["proximity_score"],
                "head_match_score": item["head_match_score"],
                "tail_match_score": item["tail_match_score"],
                "swapped": item["swapped"],
                "provisional": item["provisional"],
                "write_to_graph": item["write_to_graph"],
                "schema_version": item["schema_version"],
                "utility_bucket": item["utility_bucket"],
                "is_storyline_core": item["is_storyline_core"],
            }
        )

    return accepted, rejected
