from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from embeddings_service.storyline_llm_arbiter import arbitrate_storyline_candidate
from embeddings_service.storyline_pair_scorer import score_candidates, scorer_status
from embeddings_service.text_filters import classify_storyline_content

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional runtime dependency
    CatBoostClassifier = None


_DEFAULT_FAMILY_ALIAS_MAP_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "storyline_family_aliases.json"
)
_DEFAULT_FAMILY_MATCHER_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "model_runs"
    / "storyline_family_matcher_catboost_20260328T150319Z"
)
_DEFAULT_FACET_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "entity_facet_registry_v1.json"
)
log = structlog.get_logger()

STORYLINE_ENABLED = os.getenv("STORYLINE_RESOLVER_ENABLED", "1") != "0"
STORYLINE_ASSIGN_THRESHOLD = float(os.getenv("STORYLINE_ASSIGN_THRESHOLD", "0.57"))
STORYLINE_MAX_CANDIDATES = int(os.getenv("STORYLINE_MAX_CANDIDATES", "30"))
STORYLINE_NEW_MIN_EVENT_CONFIDENCE = float(
    os.getenv("STORYLINE_NEW_MIN_EVENT_CONFIDENCE", "0.35")
)
STORYLINE_FORCE_ASSIGN_MIN_SCORE = float(
    os.getenv("STORYLINE_FORCE_ASSIGN_MIN_SCORE", "0.40")
)
STORYLINE_FORCE_ASSIGN_MIN_ENTITY_OVERLAP = float(
    os.getenv("STORYLINE_FORCE_ASSIGN_MIN_ENTITY_OVERLAP", "0.05")
)
STORYLINE_FORCE_ASSIGN_MIN_RELATION_OVERLAP = float(
    os.getenv("STORYLINE_FORCE_ASSIGN_MIN_RELATION_OVERLAP", "0.05")
)
STORYLINE_FORCE_ASSIGN_MIN_STRUCTURAL_OVERLAP = float(
    os.getenv("STORYLINE_FORCE_ASSIGN_MIN_STRUCTURAL_OVERLAP", "0.05")
)
STORYLINE_FORCE_ASSIGN_MIN_CONTINUITY = float(
    os.getenv("STORYLINE_FORCE_ASSIGN_MIN_CONTINUITY", "0.24")
)
STORYLINE_FORCE_ASSIGN_MIN_FAMILY_SIGNAL = float(
    os.getenv("STORYLINE_FORCE_ASSIGN_MIN_FAMILY_SIGNAL", "0.18")
)
STORYLINE_FORCE_ASSIGN_MIN_TFIDF = float(
    os.getenv("STORYLINE_FORCE_ASSIGN_MIN_TFIDF", "0.20")
)
STORYLINE_FORCE_ASSIGN_MIN_USEFUL_FACET = float(
    os.getenv("STORYLINE_FORCE_ASSIGN_MIN_USEFUL_FACET", "0.10")
)
STORYLINE_PAIR_ASSIGN_THRESHOLD = float(
    os.getenv("STORYLINE_PAIR_ASSIGN_THRESHOLD", "0.50")
)
STORYLINE_PAIR_FORCE_ASSIGN_MIN_SCORE = float(
    os.getenv("STORYLINE_PAIR_FORCE_ASSIGN_MIN_SCORE", "0.45")
)
STORYLINE_PAIR_ASSIGN_MIN_SCORE_NO_STRUCTURAL_LOW_CONF = float(
    os.getenv("STORYLINE_PAIR_ASSIGN_MIN_SCORE_NO_STRUCTURAL_LOW_CONF", "0.56")
)
STORYLINE_HYBRID_ASSIGN_MIN_SCORE_NO_STRUCTURAL_LOW_CONF = float(
    os.getenv("STORYLINE_HYBRID_ASSIGN_MIN_SCORE_NO_STRUCTURAL_LOW_CONF", "0.65")
)
STORYLINE_EVENT_ONLY_LOW_CONF_MAX = float(
    os.getenv("STORYLINE_EVENT_ONLY_LOW_CONF_MAX", "0.20")
)
STORYLINE_EVENT_ONLY_MIN_OVERLAP = float(
    os.getenv("STORYLINE_EVENT_ONLY_MIN_OVERLAP", "0.50")
)
STORYLINE_DIGEST_SEED_PENALTY = float(
    os.getenv("STORYLINE_DIGEST_SEED_PENALTY", "0.12")
)
STORYLINE_AD_SEED_PENALTY = float(
    os.getenv("STORYLINE_AD_SEED_PENALTY", "0.16")
)
STORYLINE_PAIR_HYBRID_BLEND = float(
    os.getenv("STORYLINE_PAIR_HYBRID_BLEND", "0.35")
)
STORYLINE_CANDIDATE_FETCH_MULTIPLIER = int(
    os.getenv("STORYLINE_CANDIDATE_FETCH_MULTIPLIER", "3")
)
STORYLINE_BROAD_ENTITY_WEIGHT = float(
    os.getenv("STORYLINE_BROAD_ENTITY_WEIGHT", "0.35")
)
STORYLINE_DUPLICATE_TEXT_JACCARD_MIN = float(
    os.getenv("STORYLINE_DUPLICATE_TEXT_JACCARD_MIN", "0.72")
)
STORYLINE_DUPLICATE_EMBEDDING_MIN = float(
    os.getenv("STORYLINE_DUPLICATE_EMBEDDING_MIN", "0.97")
)
STORYLINE_DUPLICATE_EVENT_OVERLAP_MIN = float(
    os.getenv("STORYLINE_DUPLICATE_EVENT_OVERLAP_MIN", "0.80")
)
STORYLINE_DUPLICATE_TEMPORAL_MIN = float(
    os.getenv("STORYLINE_DUPLICATE_TEMPORAL_MIN", "0.75")
)
STORYLINE_RECENT_EXEMPLAR_LIMIT = int(
    os.getenv("STORYLINE_RECENT_EXEMPLAR_LIMIT", "3")
)
STORYLINE_FAMILY_NEAR_BEST_DELTA = float(
    os.getenv("STORYLINE_FAMILY_NEAR_BEST_DELTA", "0.09")
)
STORYLINE_FAMILY_MIN_EMBEDDING = float(
    os.getenv("STORYLINE_FAMILY_MIN_EMBEDDING", "0.84")
)
STORYLINE_FAMILY_MIN_SIGNAL = float(
    os.getenv("STORYLINE_FAMILY_MIN_SIGNAL", "0.16")
)
STORYLINE_FAMILY_MAX_BOOST = float(
    os.getenv("STORYLINE_FAMILY_MAX_BOOST", "0.08")
)
STORYLINE_FAMILY_SIGNATURE_MIN_OVERLAP = float(
    os.getenv("STORYLINE_FAMILY_SIGNATURE_MIN_OVERLAP", "0.25")
)
STORYLINE_FAMILY_ROOT_REROUTE_DELTA = float(
    os.getenv("STORYLINE_FAMILY_ROOT_REROUTE_DELTA", "0.12")
)
STORYLINE_FAMILY_ALIAS_MAP_PATH = os.getenv(
    "STORYLINE_FAMILY_ALIAS_MAP_PATH",
    str(_DEFAULT_FAMILY_ALIAS_MAP_PATH),
)
STORYLINE_MIN_RETRIEVABLE_SEED_QUALITY = float(
    os.getenv("STORYLINE_MIN_RETRIEVABLE_SEED_QUALITY", "0.45")
)
STORYLINE_DUPLICATE_FASTPATH_TEXT_MIN = float(
    os.getenv("STORYLINE_DUPLICATE_FASTPATH_TEXT_MIN", "0.90")
)
STORYLINE_TFIDF_SEED_WEIGHT = float(
    os.getenv("STORYLINE_TFIDF_SEED_WEIGHT", "0.10")
)
STORYLINE_TFIDF_PEER_WEIGHT = float(
    os.getenv("STORYLINE_TFIDF_PEER_WEIGHT", "0.14")
)
STORYLINE_FAMILY_MATCHER_ENABLED = os.getenv("STORYLINE_FAMILY_MATCHER_ENABLED", "1") != "0"
STORYLINE_FAMILY_MATCHER_THRESHOLD = float(
    os.getenv("STORYLINE_FAMILY_MATCHER_THRESHOLD", "0.59")
)
STORYLINE_FAMILY_MATCHER_FALLBACK_MIN = float(
    os.getenv("STORYLINE_FAMILY_MATCHER_FALLBACK_MIN", "0.42")
)
STORYLINE_WITHIN_FAMILY_ATTACH_ENABLED = os.getenv(
    "STORYLINE_WITHIN_FAMILY_ATTACH_ENABLED", "1"
) != "0"
STORYLINE_WITHIN_FAMILY_ATTACH_MIN_ROOT_SCORE = float(
    os.getenv("STORYLINE_WITHIN_FAMILY_ATTACH_MIN_ROOT_SCORE", "0.59")
)
STORYLINE_WITHIN_FAMILY_ATTACH_MIN_PAIR_SCORE = float(
    os.getenv("STORYLINE_WITHIN_FAMILY_ATTACH_MIN_PAIR_SCORE", "0.43")
)
STORYLINE_WITHIN_FAMILY_ATTACH_MIN_CONTINUITY = float(
    os.getenv("STORYLINE_WITHIN_FAMILY_ATTACH_MIN_CONTINUITY", "0.12")
)
STORYLINE_WITHIN_FAMILY_ATTACH_MIN_ENTITY = float(
    os.getenv("STORYLINE_WITHIN_FAMILY_ATTACH_MIN_ENTITY", "0.08")
)
STORYLINE_WITHIN_FAMILY_ATTACH_MIN_EVENT = float(
    os.getenv("STORYLINE_WITHIN_FAMILY_ATTACH_MIN_EVENT", "0.12")
)
STORYLINE_LLM_ARBITER_ENABLED = os.getenv("STORYLINE_LLM_ARBITER_ENABLED", "1") != "0"
STORYLINE_LLM_ARBITER_MIN_SCORE = float(
    os.getenv("STORYLINE_LLM_ARBITER_MIN_SCORE", "0.34")
)
STORYLINE_LLM_ARBITER_MAX_MARGIN = float(
    os.getenv("STORYLINE_LLM_ARBITER_MAX_MARGIN", "0.16")
)
STORYLINE_LLM_ARBITER_MIN_EMBEDDING = float(
    os.getenv("STORYLINE_LLM_ARBITER_MIN_EMBEDDING", "0.82")
)
STORYLINE_LLM_ARBITER_MIN_EVENT_OVERLAP = float(
    os.getenv("STORYLINE_LLM_ARBITER_MIN_EVENT_OVERLAP", "0.18")
)
STORYLINE_LLM_ARBITER_MIN_FAMILY_SIGNAL = float(
    os.getenv("STORYLINE_LLM_ARBITER_MIN_FAMILY_SIGNAL", "0.12")
)
STORYLINE_LLM_ARBITER_MIN_CONTINUITY = float(
    os.getenv("STORYLINE_LLM_ARBITER_MIN_CONTINUITY", "0.10")
)
STORYLINE_LLM_ARBITER_MIN_CONFIDENCE = float(
    os.getenv("STORYLINE_LLM_ARBITER_MIN_CONFIDENCE", "0.55")
)
STORYLINE_FAMILY_MATCHER_MODEL_DIR = os.getenv(
    "STORYLINE_FAMILY_MATCHER_MODEL_DIR",
    str(_DEFAULT_FAMILY_MATCHER_DIR),
)
STORYLINE_FACET_REGISTRY_PATH = os.getenv(
    "STORYLINE_FACET_REGISTRY_PATH",
    str(_DEFAULT_FACET_REGISTRY_PATH),
)
STORYLINE_FACET_USEFUL_MATCH_BONUS = float(
    os.getenv("STORYLINE_FACET_USEFUL_MATCH_BONUS", "0.06")
)
STORYLINE_FACET_TYPE_MATCH_BONUS = float(
    os.getenv("STORYLINE_FACET_TYPE_MATCH_BONUS", "0.025")
)
STORYLINE_FACET_BROAD_NO_MATCH_PENALTY = float(
    os.getenv("STORYLINE_FACET_BROAD_NO_MATCH_PENALTY", "0.07")
)
STORYLINE_FACET_MIN_SIMILARITY = float(
    os.getenv("STORYLINE_FACET_MIN_SIMILARITY", "0.03")
)

_DIGEST_TITLE_RE = re.compile(
    r"(главн(ые|ое)\s+новост|дайджест|итоги\s+(дня|недели)|"
    r"что\s+известно|главное\s+к\s+\d|сводка|коротко\s+о\s+главном)",
    re.IGNORECASE,
)
_AD_TITLE_RE = re.compile(
    r"(реклам|партнерск|спонсор|подписк|скидк|промокод|"
    r"виртуальн\w+\s+карт|крипто-?карт|chatgpt|netflix|"
    r"выпуск\s+за\s+\d+\s+минут|пополнен\w+\s+рубл|работает\s+там,\s+где\s+нужно)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,}")
_BROAD_ENTITY_RE = re.compile(
    r"(росси|russia|москв|moscow|сша|usa|иран|iran|израил|israel|"
    r"украин|ukraine|трамп|trump|путин|putin|телеграм|telegram|"
    r"кремл|kremlin|европ|europe|ес\b|eu\b|"
    r"сбер|sber|матч\s*тв|match\s*tv|госдум|duma)",
    re.IGNORECASE,
)
_SALIENT_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ]{5,}")
_FAMILY_STOPWORDS = {
    "который", "которая", "которые", "которое", "последние", "заявил", "заявила",
    "заявили", "заявление", "сообщил", "сообщила", "сообщили", "сообщение",
    "сказал", "сказала", "данным", "стороны", "продолжит", "завершить", "завершения",
    "ситуация", "ситуации", "условиях", "условия", "своих", "сегодня", "после",
    "будет", "стало", "среди", "также", "через", "снова", "теперь", "затем",
}
_ABSTRACT_LOCATION_ANCHOR_TERMS = (
    "пространств",
    "корабл",
    "борту",
    "орбит",
    "космос",
    "отсеке",
    "модуле",
    "кабине",
)
_WEAK_TOPIC_TERM_STEMS = {
    "местн",
    "власт",
    "служб",
    "коммуналь",
    "вице",
    "министр",
    "президент",
    "официальн",
    "заявлен",
    "сообщ",
    "работник",
    "чиновник",
}
_EPISODE_TITLE_SUFFIX_RE = re.compile(r"\s*\[[^\]]+\]\s*$")
_EPISODE_LAUNCH_RE = re.compile(
    r"(старт|запуск|launch|liftoff|takeoff|crewed mission begins|отправил|отправилась)",
    re.IGNORECASE,
)
_EPISODE_FLYBY_RE = re.compile(
    r"(облет|обл[её]т|flyby|approach(ed)? the moon|пролетел[аи]? мимо луны|орбит)",
    re.IGNORECASE,
)
_EPISODE_RETURN_RE = re.compile(
    r"(вернул|возвраща|приводнен|splashdown|returned to earth|возвратил(?:ась)? на землю)",
    re.IGNORECASE,
)
_EPISODE_INCIDENT_RE = re.compile(
    r"(пожар|взрыв|газ|метан|водород|risk|опасност|проблем|авар|incident|утечк)",
    re.IGNORECASE,
)
_EPISODE_POLICY_RE = re.compile(
    r"(обсуд|объясн|заявил|заявила|рассказал|рассказала|подчеркнул|политик|дискусс|зачем)",
    re.IGNORECASE,
)
_US_IRAN_RE = re.compile(r"(сша|usa|трамп|trump).*(иран|iran)|(иран|iran).*(сша|usa|трамп|trump)", re.IGNORECASE | re.DOTALL)
_NEGOTIATION_RE = re.compile(r"(переговор|сделк|соглашен|перемири|план мир|урегулир)", re.IGNORECASE)
_CONFLICT_RE = re.compile(r"(войн|удар|атак|операци|вторжен|бомбард|обстрел|конфликт)", re.IGNORECASE)
_ENERGY_RE = re.compile(r"(ормуз|нефт|пролив|санкц)", re.IGNORECASE)
_CHELYABINSK_SCHOOL_RE = re.compile(
    r"(челябин).*(школ|девятикласс|одноклассниц|арбалет|ракетниц|напад)|(школ|девятикласс|одноклассниц|арбалет|ракетниц|напад).*(челябин)",
    re.IGNORECASE | re.DOTALL,
)
_FAMILY_ALIAS_CACHE: tuple[str, dict[str, str]] | None = None
_FAMILY_ALIAS_WARNED_PATHS: set[str] = set()
_FAMILY_MATCHER_CACHE: tuple[str, Any] | None = None
_FAMILY_MATCHER_META_CACHE: tuple[str, dict[str, Any]] | None = None
_FAMILY_MATCHER_WARNED_PATHS: set[str] = set()
_FACET_REGISTRY_CACHE: tuple[str, dict[str, list[dict[str, Any]]]] | None = None
_FACET_REGISTRY_WARNED_PATHS: set[str] = set()


def _safe_iso(ts: datetime | None) -> str | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return _clamp01(dot / (na * nb))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _normalized_tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(str(text or ""))}


def _normalize_entity_text(text: str) -> str:
    t = str(text or "").strip().lower().replace("ё", "е")
    while t and not t[0].isalnum():
        t = t[1:]
    while t and not t[-1].isalnum():
        t = t[:-1]
    return " ".join(t.split())


def _text_jaccard(a: str, b: str) -> float:
    return _jaccard(_normalized_tokens(a), _normalized_tokens(b))


def _is_broad_entity_key(key: str) -> bool:
    return bool(_BROAD_ENTITY_RE.search(str(key or "")))


def _entity_key_parts(key: str) -> tuple[str, str]:
    label, _, raw = str(key or "").partition(":")
    return label.strip().lower(), raw.strip()


def _entity_key_display_text(key: str) -> str:
    _label, raw = _entity_key_parts(key)
    return raw.replace("_", " ").strip()


def _is_abstract_location_key(key: str) -> bool:
    label, _raw = _entity_key_parts(key)
    if label != "location":
        return False
    text = _normalize_entity_text(_entity_key_display_text(key))
    return any(term in text for term in _ABSTRACT_LOCATION_ANCHOR_TERMS)


def _is_weak_single_token_event_key(key: str) -> bool:
    label, _raw = _entity_key_parts(key)
    if label != "event":
        return False
    text = _normalize_entity_text(_entity_key_display_text(key))
    tokens = list(_normalized_tokens(text))
    if len(tokens) != 1:
        return False
    if re.search(r"[A-Za-z0-9]", text):
        return False
    return True


def _anchor_key_score(key: str) -> float:
    label, _raw = _entity_key_parts(key)
    text = _entity_key_display_text(key)
    score = 0.0
    if label == "product":
        score += 4.8
    elif label == "organization":
        score += 4.4
    elif label in {"event", "law"}:
        score += 4.0
    elif label == "person":
        score += 3.2
    elif label == "location":
        score += 2.3
    if not _is_broad_entity_key(key):
        score += 1.5
    if _is_abstract_location_key(key):
        score -= 4.0
    if _is_weak_single_token_event_key(key):
        score -= 2.2
    if len(text) >= 6:
        score += 0.25
    return score


def _topic_term_value(sig: str) -> str:
    return str(sig or "").split(":", 1)[1].strip() if ":" in str(sig or "") else ""


_CYR_TO_LAT = str.maketrans(
    {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ё": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "i",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
        "ц": "c",
        "ч": "ch",
        "ш": "sh",
        "щ": "sch",
        "ъ": "",
        "ы": "y",
        "ь": "",
        "э": "e",
        "ю": "yu",
        "я": "ya",
    }
)


def _topic_semantic_token_key(token: str) -> str:
    normalized = _normalize_entity_text(token)
    if not normalized:
        return ""
    translit = normalized.translate(_CYR_TO_LAT)
    translit = translit.replace("ii", "2").replace("iii", "3").replace("iv", "4")
    translit = re.sub(r"[^a-z0-9]+", "", translit)
    if not translit:
        return ""
    if translit.isdigit():
        return translit
    if len(translit) <= 4:
        return translit
    return translit[:6]


def _macro_topic_semantic_keys(
    *,
    topic_signatures: set[str],
    anchor_entities: set[str],
    canonical_title: str = "",
    topic_aliases: set[str] | None = None,
) -> set[str]:
    raw_terms: set[str] = set()
    for sig in topic_signatures:
        value = str(sig or "").strip()
        if value.startswith("topic_term:"):
            term = _topic_term_value(value)
            if term and not _is_weak_topic_term(term):
                raw_terms.add(term)
        elif value.startswith("topic_pair:"):
            pair = _topic_term_value(value)
            if "::" not in pair:
                continue
            left, right = pair.split("::", 1)
            for term in (left, right):
                if term and not _is_weak_topic_term(term):
                    raw_terms.add(term)
    for key in anchor_entities:
        text = _entity_key_display_text(key)
        if not text:
            continue
        for token in _normalized_tokens(text):
            if token and not _is_weak_topic_term(token):
                raw_terms.add(token)
    for text in {str(canonical_title or "").strip(), *(topic_aliases or set())}:
        if not text:
            continue
        for token in _normalized_tokens(text):
            if token and not _is_weak_topic_term(token):
                raw_terms.add(token)

    out: set[str] = set()
    for term in raw_terms:
        key = _topic_semantic_token_key(term)
        if not key:
            continue
        if len(key) >= 4 or key.isdigit():
            out.add(key)
    return out


def _is_weak_topic_term(term: str) -> bool:
    normalized = _normalize_entity_text(term)
    if not normalized:
        return True
    if normalized in _FAMILY_STOPWORDS:
        return True
    return any(stem in normalized for stem in _WEAK_TOPIC_TERM_STEMS)


def _topic_term_rank(term: str) -> tuple[int, int, int, str]:
    normalized = _normalize_entity_text(term)
    return (
        0 if _is_weak_topic_term(normalized) else 1,
        0 if _BROAD_ENTITY_RE.search(normalized) else 1,
        len(normalized),
        normalized,
    )


def _topic_pair_rank(sig: str) -> tuple[int, int, int, str]:
    pair = _topic_term_value(sig)
    if "::" not in pair:
        return (0, 0, 0, "")
    left, right = pair.split("::", 1)
    left_rank = _topic_term_rank(left)
    right_rank = _topic_term_rank(right)
    useful = int(left_rank[0] > 0 and right_rank[0] > 0)
    non_broad = int(left_rank[1] > 0 or right_rank[1] > 0)
    total_len = len(left) + len(right)
    return (useful, non_broad, total_len, pair)


def _macro_topic_anchor_keys(entity_keys: set[str]) -> list[str]:
    label_priority = {
        "product": 5,
        "organization": 4,
        "event": 3,
        "law": 3,
        "location": 2,
        "person": 1,
    }
    return sorted(
        (
            key
            for key in entity_keys
            if key
            and not _is_broad_entity_key(key)
            and _anchor_key_score(key) >= 4.0
        ),
        key=lambda key: (
            -label_priority.get(_entity_key_parts(key)[0], 0),
            -_anchor_key_score(key),
            key,
        ),
    )


def _title_entity_names(title: str) -> list[str]:
    base = re.sub(r"\s*\[[^\]]+\]\s*$", "", str(title or "").strip())
    return [part.strip() for part in base.split(",") if part.strip()]


def _title_quality_score(title: str, entity_keys: set[str]) -> float:
    title = str(title or "").strip()
    if not title:
        return -10.0
    if title.startswith("Storyline "):
        return -6.0

    names = _title_entity_names(title)
    if not names:
        return -5.0

    score = 0.0
    for name in names[:2]:
        norm_name = _normalize_entity_text(name)
        if not norm_name:
            continue
        matched_scores: list[float] = []
        for key in entity_keys:
            key_text = _normalize_entity_text(_entity_key_display_text(key))
            if not key_text:
                continue
            if norm_name == key_text or norm_name in key_text or key_text in norm_name:
                matched_scores.append(_anchor_key_score(key))
        if matched_scores:
            score += max(matched_scores)
        elif any(term in norm_name for term in _ABSTRACT_LOCATION_ANCHOR_TERMS):
            score -= 2.5
        elif re.search(r"[a-z0-9]", norm_name):
            score += 0.5
        else:
            score += 0.2
    return score


def _merge_storyline_entity_keys(prev_keys: set[str], incoming_keys: set[str]) -> list[str]:
    merged = sorted({str(key or "").strip() for key in (prev_keys | incoming_keys) if str(key or "").strip()})
    has_strong_non_location = any(
        _anchor_key_score(key) >= 4.0 and _entity_key_parts(key)[0] != "location"
        for key in merged
    )
    if has_strong_non_location:
        merged = [key for key in merged if not _is_abstract_location_key(key)]
    return merged


def _refresh_storyline_title(
    *,
    prev_title: str,
    incoming_title: str,
    merged_entity_keys: set[str],
) -> tuple[str, dict[str, float | bool]]:
    prev_score = _title_quality_score(prev_title, merged_entity_keys)
    incoming_score = _title_quality_score(incoming_title, merged_entity_keys)

    use_incoming = bool(incoming_title) and (
        not prev_title
        or incoming_score >= prev_score + 1.0
        or (prev_score < 1.5 and incoming_score > prev_score)
    )
    selected = incoming_title if use_incoming else (prev_title or incoming_title)
    return selected, {
        "title_refreshed": use_incoming,
        "prev_title_score": round(prev_score, 4),
        "incoming_title_score": round(incoming_score, 4),
    }


def _refresh_storyline_payload(
    *,
    prev: dict[str, Any],
    incoming_title: str,
    incoming_source_text: str,
    incoming_entity_keys: set[str],
    incoming_relation_signatures: set[str],
    incoming_all_relation_signatures: set[str],
    incoming_event_signatures: set[str],
    incoming_topic_signatures: set[str],
) -> dict[str, Any]:
    merged_entities = _merge_storyline_entity_keys(
        {str(x or "").strip() for x in (prev.get("entity_keys") or []) if str(x or "").strip()},
        incoming_entity_keys,
    )
    merged_relations = sorted(
        {str(x or "").strip() for x in (prev.get("relation_signatures") or []) if str(x or "").strip()}
        | incoming_relation_signatures
    )
    merged_all_relations = sorted(
        {str(x or "").strip() for x in (prev.get("all_relation_signatures") or []) if str(x or "").strip()}
        | incoming_all_relation_signatures
    )
    merged_events = sorted(
        {str(x or "").strip() for x in (prev.get("event_signatures") or []) if str(x or "").strip()}
        | incoming_event_signatures
    )

    refreshed_title, title_meta = _refresh_storyline_title(
        prev_title=str(prev.get("title") or "").strip(),
        incoming_title=str(incoming_title or "").strip(),
        merged_entity_keys=set(merged_entities),
    )
    refreshed_seed_preview = (
        str(prev.get("seed_preview") or "").strip()
        or str(incoming_source_text or incoming_title or "").strip()
    )[:500]
    carry_family = {
        sig
        for sig in (prev.get("family_signatures") or [])
        if str(sig or "").strip()
        and not str(sig or "").startswith("term:")
        and not str(sig or "").startswith("entity:")
    }
    refreshed_family_signatures = sorted(
        carry_family
        | _family_signatures(
            source_text=f"{refreshed_title}\n{refreshed_seed_preview}".strip(),
            entity_keys=set(merged_entities),
            event_signatures=set(merged_events),
        )
    )
    refreshed_topic_signatures = sorted(
        {
            str(sig or "").strip()
            for sig in (prev.get("topic_signatures") or [])
            if str(sig or "").strip()
        }
        | incoming_topic_signatures
        | _topic_signatures(
            source_text=f"{refreshed_title}\n{refreshed_seed_preview}".strip(),
            entity_keys=set(merged_entities),
            family_signatures=set(refreshed_family_signatures),
        )
    )

    return {
        "title": refreshed_title,
        "seed_preview": refreshed_seed_preview,
        "entity_keys": merged_entities,
        "relation_signatures": merged_relations,
        "all_relation_signatures": merged_all_relations,
        "event_signatures": merged_events,
        "family_signatures": refreshed_family_signatures,
        "topic_signatures": refreshed_topic_signatures,
        **title_meta,
    }


def _entity_weight(key: str) -> float:
    return STORYLINE_BROAD_ENTITY_WEIGHT if _is_broad_entity_key(key) else 1.0


def _weighted_entity_overlap(a: set[str], b: set[str]) -> tuple[float, float]:
    if not a or not b:
        return 0.0, 0.0
    inter = a & b
    if not inter:
        return 0.0, 0.0
    inter_w = sum(_entity_weight(x) for x in inter)
    union_w = float(len(a | b))
    broad_inter = sum(1 for x in inter if _is_broad_entity_key(x))
    broad_ratio = broad_inter / len(inter) if inter else 0.0
    return (inter_w / union_w if union_w else 0.0), broad_ratio


def _load_family_alias_map() -> dict[str, str]:
    global _FAMILY_ALIAS_CACHE
    path = str(STORYLINE_FAMILY_ALIAS_MAP_PATH or "").strip()
    if not path:
        return {}
    if _FAMILY_ALIAS_CACHE and _FAMILY_ALIAS_CACHE[0] == path:
        return _FAMILY_ALIAS_CACHE[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        if path not in _FAMILY_ALIAS_WARNED_PATHS:
            _FAMILY_ALIAS_WARNED_PATHS.add(path)
            log.warning("storyline.family_alias_map_unavailable", path=path)
        _FAMILY_ALIAS_CACHE = (path, {})
        return {}

    aliases = raw.get("aliases", raw) if isinstance(raw, dict) else {}
    out: dict[str, str] = {}
    if isinstance(aliases, dict):
        for k, v in aliases.items():
            ks = str(k or "").strip()
            vs = str(v or "").strip()
            if ks and vs and ks != vs:
                out[ks] = vs
    _FAMILY_ALIAS_CACHE = (path, out)
    return out


def _load_facet_registry() -> dict[str, list[dict[str, Any]]]:
    global _FACET_REGISTRY_CACHE
    path = str(STORYLINE_FACET_REGISTRY_PATH or "").strip()
    if not path:
        return {}
    if _FACET_REGISTRY_CACHE and _FACET_REGISTRY_CACHE[0] == path:
        return _FACET_REGISTRY_CACHE[1]
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        if path not in _FACET_REGISTRY_WARNED_PATHS:
            _FACET_REGISTRY_WARNED_PATHS.add(path)
            log.warning("storyline.facet_registry_unavailable", path=path)
        _FACET_REGISTRY_CACHE = (path, {})
        return {}

    rows = raw if isinstance(raw, list) else []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        entity_cluster_id = str(row.get("entity_cluster_id") or "").strip()
        if not entity_cluster_id:
            continue
        prototype = " ".join(
            [
                str(row.get("resolved_facet_label") or "").strip(),
                str(row.get("resolved_facet_type") or "").strip(),
                str(row.get("top_context_terms") or "").replace("|||", " "),
                str(row.get("top_co_entities") or "").replace("|||", " "),
                str(row.get("top_raw_variants") or "").replace("|||", " "),
            ]
        ).strip()
        grouped.setdefault(entity_cluster_id, []).append(
            {
                "facet_group_id": str(row.get("facet_group_id") or "").strip(),
                "resolved_facet_label": str(row.get("resolved_facet_label") or "").strip(),
                "resolved_facet_type": str(row.get("resolved_facet_type") or "").strip(),
                "resolved_is_useful_for_storyline": str(row.get("resolved_is_useful_for_storyline") or "").strip(),
                "prototype_text": prototype,
            }
        )
    _FACET_REGISTRY_CACHE = (path, grouped)
    return grouped


def _text_contains_entity_cluster(text: str, entity_cluster_id: str) -> bool:
    norm_text = _normalize_entity_text(text)
    if not norm_text:
        return False
    _, _, canonical_text = str(entity_cluster_id).partition(":")
    canonical_text = _normalize_entity_text(canonical_text)
    if not canonical_text:
        return False
    if " " in canonical_text:
        return canonical_text in norm_text
    return canonical_text in set(norm_text.split())


def _facet_entity_hits(text: str) -> set[str]:
    registry = _load_facet_registry()
    if not registry:
        return set()
    return {entity_cluster_id for entity_cluster_id in registry if _text_contains_entity_cluster(text, entity_cluster_id)}


def _best_facet_group_for_entity(text: str, entity_cluster_id: str) -> dict[str, Any] | None:
    registry = _load_facet_registry()
    groups = registry.get(entity_cluster_id) or []
    if not groups:
        return None
    local_idf = _local_tfidf_idf([text] + [str(g.get("prototype_text") or "") for g in groups])
    best: dict[str, Any] | None = None
    best_score = -1.0
    for group in groups:
        proto = str(group.get("prototype_text") or "").strip()
        if not proto:
            continue
        score = _tfidf_cosine(text, proto, local_idf)
        if score > best_score:
            best_score = score
            best = {**group, "similarity": round(score, 4)}
    if best is None or best_score < STORYLINE_FACET_MIN_SIMILARITY:
        return None
    return best


def _facet_pair_features(source_text: str, candidate_text: str) -> dict[str, float | int | str]:
    if not source_text or not candidate_text:
        return {
            "shared_registry_entity_count": 0.0,
            "facet_label_match_count": 0.0,
            "facet_type_match_count": 0.0,
            "useful_facet_match_count": 0.0,
            "facet_label_overlap_ratio": 0.0,
            "facet_type_overlap_ratio": 0.0,
            "useful_facet_overlap_ratio": 0.0,
            "broad_entity_no_useful_facet_ratio": 0.0,
            "facet_post_max_similarity": 0.0,
            "facet_seed_max_similarity": 0.0,
            "facet_shared_entities": "",
        }
    shared_entities = sorted(_facet_entity_hits(source_text) & _facet_entity_hits(candidate_text))
    if not shared_entities:
        return {
            "shared_registry_entity_count": 0.0,
            "facet_label_match_count": 0.0,
            "facet_type_match_count": 0.0,
            "useful_facet_match_count": 0.0,
            "facet_label_overlap_ratio": 0.0,
            "facet_type_overlap_ratio": 0.0,
            "useful_facet_overlap_ratio": 0.0,
            "broad_entity_no_useful_facet_ratio": 0.0,
            "facet_post_max_similarity": 0.0,
            "facet_seed_max_similarity": 0.0,
            "facet_shared_entities": "",
        }

    label_match = 0
    type_match = 0
    useful_match = 0
    post_max = 0.0
    seed_max = 0.0
    for entity_cluster_id in shared_entities:
        src = _best_facet_group_for_entity(source_text, entity_cluster_id)
        cand = _best_facet_group_for_entity(candidate_text, entity_cluster_id)
        if src:
            post_max = max(post_max, float(src.get("similarity", 0.0) or 0.0))
        if cand:
            seed_max = max(seed_max, float(cand.get("similarity", 0.0) or 0.0))
        if not src or not cand:
            continue
        if str(src.get("resolved_facet_label") or "") and src.get("resolved_facet_label") == cand.get("resolved_facet_label"):
            label_match += 1
        if str(src.get("resolved_facet_type") or "") and src.get("resolved_facet_type") == cand.get("resolved_facet_type"):
            type_match += 1
        if (
            str(src.get("resolved_is_useful_for_storyline") or "") == "1"
            and str(cand.get("resolved_is_useful_for_storyline") or "") == "1"
            and str(src.get("resolved_facet_label") or "")
            and src.get("resolved_facet_label") == cand.get("resolved_facet_label")
        ):
            useful_match += 1
    total = max(1, len(shared_entities))
    return {
        "shared_registry_entity_count": float(len(shared_entities)),
        "facet_label_match_count": float(label_match),
        "facet_type_match_count": float(type_match),
        "useful_facet_match_count": float(useful_match),
        "facet_label_overlap_ratio": round(label_match / total, 4),
        "facet_type_overlap_ratio": round(type_match / total, 4),
        "useful_facet_overlap_ratio": round(useful_match / total, 4),
        "broad_entity_no_useful_facet_ratio": round((len(shared_entities) - useful_match) / total, 4),
        "facet_post_max_similarity": round(post_max, 4),
        "facet_seed_max_similarity": round(seed_max, 4),
        "facet_shared_entities": " ||| ".join(shared_entities[:8]),
    }


def _family_root_alias(storyline_id: str) -> str:
    sid = str(storyline_id or "").strip()
    if not sid:
        return ""
    aliases = _load_family_alias_map()
    seen = set()
    cur = sid
    while cur and cur not in seen:
        seen.add(cur)
        nxt = aliases.get(cur)
        if not nxt:
            break
        cur = nxt
    return cur or sid


def _load_family_matcher_meta() -> dict[str, Any] | None:
    global _FAMILY_MATCHER_META_CACHE
    model_dir = str(STORYLINE_FAMILY_MATCHER_MODEL_DIR or "").strip()
    if not model_dir:
        return None
    if _FAMILY_MATCHER_META_CACHE and _FAMILY_MATCHER_META_CACHE[0] == model_dir:
        return _FAMILY_MATCHER_META_CACHE[1]
    path = Path(model_dir) / "storyline_family_matcher_feature_cols.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        if model_dir not in _FAMILY_MATCHER_WARNED_PATHS:
            _FAMILY_MATCHER_WARNED_PATHS.add(model_dir)
            log.warning("storyline.family_matcher_meta_unavailable", path=str(path))
        return None
    _FAMILY_MATCHER_META_CACHE = (model_dir, meta)
    return meta


def _load_family_matcher() -> Any | None:
    global _FAMILY_MATCHER_CACHE
    model_dir = str(STORYLINE_FAMILY_MATCHER_MODEL_DIR or "").strip()
    if not STORYLINE_FAMILY_MATCHER_ENABLED or not model_dir or CatBoostClassifier is None:
        return None
    if _FAMILY_MATCHER_CACHE and _FAMILY_MATCHER_CACHE[0] == model_dir:
        return _FAMILY_MATCHER_CACHE[1]
    model_path = Path(model_dir) / "storyline_family_matcher_catboost.cbm"
    try:
        model = CatBoostClassifier()
        model.load_model(str(model_path))
    except Exception:
        if model_dir not in _FAMILY_MATCHER_WARNED_PATHS:
            _FAMILY_MATCHER_WARNED_PATHS.add(model_dir)
            log.warning("storyline.family_matcher_unavailable", path=str(model_path))
        return None
    _FAMILY_MATCHER_CACHE = (model_dir, model)
    return model


def _family_matcher_feature_row(
    feature_cols: list[str],
    cat_feature_names: set[str],
    *,
    candidate: dict[str, Any],
    event_confidence: float | None,
    content_meta: dict[str, Any],
) -> list[Any]:
    comps = candidate.get("components", {}) or {}
    continuity_signal = max(
        float(comps.get("duplicate_text_overlap", 0.0) or 0.0),
        float(comps.get("exemplar_role_overlap", 0.0) or 0.0),
        float(comps.get("exemplar_event_frame_overlap", 0.0) or 0.0),
    )
    values = {
        "score": float(candidate.get("score", 0.0) or 0.0),
        "best_candidate_score": float(candidate.get("score", 0.0) or 0.0),
        "best_candidate_hybrid_score": float(candidate.get("score", 0.0) or 0.0),
        "best_candidate_pair_score": float(candidate.get("pair_score", 0.0) or 0.0),
        "pair_blended_score": float(candidate.get("pair_score", candidate.get("score", 0.0)) or 0.0),
        "pair_hybrid_blend": float(STORYLINE_PAIR_HYBRID_BLEND),
        "continuity_signal": continuity_signal,
        "duplicate_text_overlap": float(comps.get("duplicate_text_overlap", 0.0) or 0.0),
        "exemplar_role_overlap": float(comps.get("exemplar_role_overlap", 0.0) or 0.0),
        "exemplar_event_frame_overlap": float(comps.get("exemplar_event_frame_overlap", 0.0) or 0.0),
        "family_signature_overlap": float(comps.get("family_signature_overlap", 0.0) or 0.0),
        "entity_overlap": float(comps.get("entity_overlap", 0.0) or 0.0),
        "relation_overlap": float(comps.get("relation_overlap", 0.0) or 0.0),
        "event_overlap": float(comps.get("event_overlap", 0.0) or 0.0),
        "broad_entity_overlap_ratio": float(comps.get("broad_entity_overlap_ratio", 0.0) or 0.0),
        "event_mention_confidence": float(event_confidence or 0.0),
        "tfidf_post_seed_cosine": float(comps.get("tfidf_post_seed_cosine", 0.0) or 0.0),
        "tfidf_post_peer_max_cosine": float(comps.get("tfidf_post_peer_max_cosine", 0.0) or 0.0),
        "shared_registry_entity_count": float(comps.get("shared_registry_entity_count", 0.0) or 0.0),
        "facet_label_match_count": float(comps.get("facet_label_match_count", 0.0) or 0.0),
        "facet_type_match_count": float(comps.get("facet_type_match_count", 0.0) or 0.0),
        "useful_facet_match_count": float(comps.get("useful_facet_match_count", 0.0) or 0.0),
        "facet_label_overlap_ratio": float(comps.get("facet_label_overlap_ratio", 0.0) or 0.0),
        "facet_type_overlap_ratio": float(comps.get("facet_type_overlap_ratio", 0.0) or 0.0),
        "useful_facet_overlap_ratio": float(comps.get("useful_facet_overlap_ratio", 0.0) or 0.0),
        "broad_entity_no_useful_facet_ratio": float(comps.get("broad_entity_no_useful_facet_ratio", 0.0) or 0.0),
        "facet_post_max_similarity": float(comps.get("facet_post_max_similarity", 0.0) or 0.0),
        "facet_seed_max_similarity": float(comps.get("facet_seed_max_similarity", 0.0) or 0.0),
        "decision": "",
        "content_type": str(content_meta.get("content_type") or ""),
        "content_gate_reason": str(content_meta.get("content_gate_reason") or ""),
        "content_state": str(content_meta.get("content_state") or ""),
    }
    out = []
    for col in feature_cols:
        val = values.get(col, "")
        if col in cat_feature_names:
            out.append(str(val or ""))
        else:
            out.append(float(val or 0.0))
    return out


def _apply_family_matcher_prefilter(
    scored_candidates: list[dict[str, Any]],
    *,
    event_confidence: float | None,
    content_meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not scored_candidates:
        return scored_candidates, {}
    meta = _load_family_matcher_meta()
    model = _load_family_matcher()
    if not meta or model is None:
        return scored_candidates, {}

    feature_cols = list(meta.get("feature_cols") or [])
    cat_feature_names = set(meta.get("cat_features") or [])
    if not feature_cols:
        return scored_candidates, {}

    rows = [
        _family_matcher_feature_row(
            feature_cols,
            cat_feature_names,
            candidate=cand,
            event_confidence=event_confidence,
            content_meta=content_meta,
        )
        for cand in scored_candidates
    ]
    try:
        probs = [float(p[1]) for p in model.predict_proba(rows)]
    except Exception:
        log.warning("storyline.family_matcher_predict_failed")
        return scored_candidates, {}

    enriched: list[dict[str, Any]] = []
    root_scores: dict[str, float] = {}
    for cand, prob in zip(scored_candidates, probs):
        local = dict(cand)
        local["family_matcher_score"] = round(float(prob), 4)
        root_id = _family_root_alias(
            str(local.get("family_root_storyline_id") or local.get("storyline_id") or "")
        ) or str(local.get("storyline_id") or "")
        local["family_matcher_root_id"] = root_id
        enriched.append(local)
        root_scores[root_id] = max(root_scores.get(root_id, 0.0), float(local["family_matcher_score"]))

    selected_root = max(root_scores.items(), key=lambda kv: kv[1])[0]
    selected_root_score = float(root_scores.get(selected_root, 0.0) or 0.0)
    if selected_root_score >= STORYLINE_FAMILY_MATCHER_THRESHOLD:
        filtered = [c for c in enriched if str(c.get("family_matcher_root_id") or "") == selected_root]
        mode = "threshold_pass"
    elif selected_root_score >= STORYLINE_FAMILY_MATCHER_FALLBACK_MIN:
        filtered = [c for c in enriched if str(c.get("family_matcher_root_id") or "") == selected_root]
        mode = "fallback_root_only"
    else:
        filtered = enriched
        mode = "disabled_low_confidence"

    return filtered, {
        "family_matcher_ready": True,
        "family_matcher_mode": mode,
        "family_matcher_threshold": STORYLINE_FAMILY_MATCHER_THRESHOLD,
        "family_matcher_fallback_min": STORYLINE_FAMILY_MATCHER_FALLBACK_MIN,
        "family_matcher_selected_root_id": selected_root,
        "family_matcher_selected_root_score": round(selected_root_score, 4),
        "family_matcher_best_score": round(max(root_scores.values()), 4),
        "family_matcher_candidate_count_before": len(scored_candidates),
        "family_matcher_candidate_count_after": len(filtered),
    }


def _family_signatures(
    *,
    source_text: str,
    entity_keys: set[str],
    event_signatures: set[str],
) -> set[str]:
    text = str(source_text or "")
    lowered = text.lower()
    out: set[str] = set()

    salient = [
        tok.lower()
        for tok in _SALIENT_TOKEN_RE.findall(text)
        if tok and tok.lower() not in _FAMILY_STOPWORDS
    ]
    for tok in salient[:4]:
        out.add(f"term:{tok}")

    specific_entities = [
        key for key in sorted(entity_keys)
        if key and not _is_broad_entity_key(key)
    ]
    for key in specific_entities[:4]:
        out.add(f"entity:{key}")

    if _US_IRAN_RE.search(lowered):
        out.add("macro:us_iran")
        if _NEGOTIATION_RE.search(lowered):
            out.add("macro:us_iran_negotiation")
        if _CONFLICT_RE.search(lowered):
            out.add("macro:us_iran_conflict")
        if _ENERGY_RE.search(lowered):
            out.add("macro:us_iran_energy")

    if _CHELYABINSK_SCHOOL_RE.search(lowered):
        out.add("macro:chelyabinsk_school_attack")

    for sig in event_signatures:
        s = str(sig or "").strip()
        if not s:
            continue
        if s.startswith("event_frame:") or s.startswith("event_subtype:") or s.startswith("update_kind:"):
            out.add(s)

    return out


def _topic_signatures(
    *,
    source_text: str,
    entity_keys: set[str],
    family_signatures: set[str],
) -> set[str]:
    text = str(source_text or "")
    out: set[str] = set()

    prioritized_tokens: list[str] = []
    for key in sorted(entity_keys):
        if not key or _is_broad_entity_key(key):
            continue
        if _anchor_key_score(key) < 4.0:
            continue
        out.add(f"topic_entity:{key}")
        for token in _normalized_tokens(_entity_key_display_text(key)):
            if len(token) < 4 or token in _FAMILY_STOPWORDS:
                continue
            prioritized_tokens.append(token)

    for token in _normalized_tokens(text):
        if len(token) < 4 or token in _FAMILY_STOPWORDS:
            continue
        prioritized_tokens.append(token)

    seen_terms: set[str] = set()
    strong_terms: list[str] = []
    for token in prioritized_tokens:
        lowered = token.lower()
        if lowered in seen_terms or _BROAD_ENTITY_RE.search(lowered):
            continue
        seen_terms.add(lowered)
        strong_terms.append(lowered)

    for token in strong_terms[:8]:
        out.add(f"topic_term:{token}")

    if len(strong_terms) >= 2:
        out.add(f"topic_pair:{strong_terms[0]}::{strong_terms[1]}")
    if len(strong_terms) >= 3:
        out.add(f"topic_pair:{strong_terms[0]}::{strong_terms[2]}")

    for sig in family_signatures:
        value = str(sig or "").strip()
        if value.startswith("macro:"):
            out.add(f"topic_{value}")

    return out


def _macro_topic_basis_components(
    *,
    entity_keys: set[str],
    topic_signatures: set[str],
    family_signatures: set[str],
) -> list[str]:
    components: list[str] = []

    strong_entity_keys = _macro_topic_anchor_keys(entity_keys)
    strong_non_person = [key for key in strong_entity_keys if _entity_key_parts(key)[0] != "person"]
    strong_person = [key for key in strong_entity_keys if _entity_key_parts(key)[0] == "person"]

    macro_sigs = sorted(
        sig for sig in topic_signatures
        if sig.startswith("topic_macro:")
    )
    for sig in macro_sigs[:2]:
        if sig not in components:
            components.append(sig)

    topic_pairs = sorted(
        (sig for sig in topic_signatures if sig.startswith("topic_pair:")),
        key=_topic_pair_rank,
        reverse=True,
    )
    topic_terms = sorted(
        (sig for sig in topic_signatures if sig.startswith("topic_term:")),
        key=lambda sig: _topic_term_rank(_topic_term_value(sig)),
        reverse=True,
    )

    best_pair = topic_pairs[0] if topic_pairs else ""
    if best_pair:
        for key in strong_non_person[:2]:
            components.append(f"entity:{key}")
        if strong_person and not strong_non_person:
            components.append(f"entity:{strong_person[0]}")
        if best_pair not in components:
            components.append(best_pair)
        for sig in topic_terms[:1]:
            if sig not in components:
                components.append(sig)
    else:
        primary_entities = strong_non_person[:2] if strong_non_person else strong_person[:1]
        for key in primary_entities:
            components.append(f"entity:{key}")
        for sig in topic_pairs[:2]:
            if sig not in components:
                components.append(sig)
        for sig in topic_terms[:2]:
            if sig not in components:
                components.append(sig)

    family_macros = sorted(
        sig for sig in family_signatures
        if sig.startswith("macro:")
    )
    for sig in family_macros[:2]:
        topic_sig = f"topic_{sig}"
        if topic_sig not in components:
            components.append(topic_sig)

    return components[:6]


def _macro_topic_title(
    *,
    entity_keys: set[str],
    topic_signatures: set[str],
    fallback_title: str,
) -> str:
    strong_entity_keys = _macro_topic_anchor_keys(entity_keys)
    title_label_priority = {
        "product": 5,
        "event": 4,
        "law": 4,
        "location": 3,
        "organization": 2,
    }
    strong_non_person = sorted(
        [key for key in strong_entity_keys if _entity_key_parts(key)[0] != "person"],
        key=lambda key: (
            -title_label_priority.get(_entity_key_parts(key)[0], 0),
            -_anchor_key_score(key),
            key,
        ),
    )
    strong_person = [key for key in strong_entity_keys if _entity_key_parts(key)[0] == "person"]

    displays = [
        _entity_key_display_text(key)
        for key in strong_non_person[:2]
        if _entity_key_display_text(key)
    ]
    strong_displays = [
        text for text in displays
        if text and not _is_weak_topic_term(text) and _title_quality_score(text, entity_keys) >= 2.0
    ]
    if strong_displays:
        return ", ".join(strong_displays[:2])
    if displays:
        return ", ".join(displays[:2])

    best_pair = next(
        (
            _topic_term_value(sig)
            for sig in sorted(
                (sig for sig in topic_signatures if sig.startswith("topic_pair:")),
                key=_topic_pair_rank,
                reverse=True,
            )
            if "::" in _topic_term_value(sig)
        ),
        "",
    )
    if strong_person:
        person_display = _entity_key_display_text(strong_person[0])
        if best_pair:
            left, right = best_pair.split("::", 1)
            person_tokens = set(_normalized_tokens(person_display))
            pair_terms = [term for term in (left, right) if term and term not in person_tokens and not _is_weak_topic_term(term)]
            if pair_terms:
                return f"{person_display}, {pair_terms[0]}"
        best_term = next(
            (
                _topic_term_value(sig)
                for sig in sorted(
                    (sig for sig in topic_signatures if sig.startswith("topic_term:")),
                    key=lambda sig: _topic_term_rank(_topic_term_value(sig)),
                    reverse=True,
                )
                if _topic_term_value(sig)
                and _topic_term_value(sig) not in person_tokens
                and not _is_weak_topic_term(_topic_term_value(sig))
            ),
            "",
        )
        if best_term:
            return f"{person_display}, {best_term}"
        return person_display

    if best_pair:
        left, right = best_pair.split("::", 1)
        if left and right:
            return f"{left}, {right}"

    terms = [
        _topic_term_value(sig)
        for sig in sorted(
            (sig for sig in topic_signatures if sig.startswith("topic_term:")),
            key=lambda sig: _topic_term_rank(_topic_term_value(sig)),
            reverse=True,
        )
        if _topic_term_value(sig) and not _is_weak_topic_term(_topic_term_value(sig))
    ]
    if terms:
        return ", ".join(terms[:2])

    fallback_terms = [
        _topic_term_value(sig)
        for sig in sorted(
            (sig for sig in topic_signatures if sig.startswith("topic_term:")),
            key=lambda sig: _topic_term_rank(_topic_term_value(sig)),
            reverse=True,
        )
        if sig.startswith("topic_term:")
    ]
    if fallback_terms:
        return ", ".join(fallback_terms[:2])

    return str(fallback_title or "").strip() or "macro topic"


def _macro_topic_payload(
    *,
    title: str,
    seed_preview: str,
    entity_keys: set[str],
    family_signatures: set[str],
    topic_signatures: set[str],
    existing_macro_topic_id: str = "",
) -> dict[str, Any]:
    basis_components = _macro_topic_basis_components(
        entity_keys=entity_keys,
        topic_signatures=topic_signatures,
        family_signatures=family_signatures,
    )
    if not basis_components:
        basis_components = sorted(topic_signatures)[:4] or [str(title or "").strip()[:120] or "macro_topic"]
    basis_blob = "|".join(basis_components)
    macro_topic_id = str(existing_macro_topic_id or "").strip()
    if not macro_topic_id:
        macro_topic_id = f"macro_topic_{hashlib.sha1(basis_blob.encode('utf-8')).hexdigest()[:16]}"

    strong_entity_keys = sorted(
        (
            key
            for key in entity_keys
            if key
            and not _is_broad_entity_key(key)
            and _anchor_key_score(key) >= 4.0
        ),
        key=lambda key: (-_anchor_key_score(key), key),
    )
    anchor_entities = strong_entity_keys[:6]
    canonical_title = _macro_topic_title(
        entity_keys=entity_keys,
        topic_signatures=topic_signatures,
        fallback_title=title,
    )
    title_score = _title_quality_score(
        canonical_title,
        set(anchor_entities) if anchor_entities else entity_keys,
    )
    topic_aliases = sorted(
        {
            canonical_title,
            str(title or "").strip(),
        }
        | {
            _entity_key_display_text(key)
            for key in anchor_entities
            if _entity_key_display_text(key)
        }
    )

    return {
        "macro_topic_id": macro_topic_id,
        "canonical_title": canonical_title,
        "title_score": round(float(title_score or 0.0), 4),
        "anchor_entities": anchor_entities,
        "topic_aliases": [alias for alias in topic_aliases if alias][:8],
        "topic_signatures": sorted(topic_signatures),
        "basis_components": basis_components,
        "seed_preview": str(seed_preview or "").strip()[:500],
    }


def _macro_topic_sig_values(topic_signatures: set[str], prefix: str) -> set[str]:
    out: set[str] = set()
    for sig in topic_signatures:
        value = str(sig or "").strip()
        if value.startswith(prefix):
            out.add(value.split(":", 1)[1].strip())
    return out


def _episode_phase_type(*, title: str, seed_preview: str, topic_signatures: set[str]) -> str:
    text = "\n".join(
        part for part in [str(title or "").strip(), str(seed_preview or "").strip()] if part
    )
    if not text and topic_signatures:
        text = " ".join(sorted(topic_signatures))
    if not text:
        return ""
    if _EPISODE_RETURN_RE.search(text):
        return "return"
    if _EPISODE_LAUNCH_RE.search(text):
        return "launch"
    if _EPISODE_FLYBY_RE.search(text):
        return "flyby"
    if _EPISODE_INCIDENT_RE.search(text):
        return "incident"
    if _EPISODE_POLICY_RE.search(text):
        return "commentary"
    return ""


def _episode_phase_label(phase_type: str) -> str:
    labels = {
        "launch": "launch phase",
        "flyby": "flyby phase",
        "return": "return phase",
        "incident": "incident",
        "commentary": "commentary",
    }
    return labels.get(str(phase_type or "").strip(), "")


def _episode_title(
    *,
    storyline_title: str,
    phase_type: str,
    anchor_entities: list[str],
    topic_signatures: set[str],
) -> str:
    cleaned_title = _EPISODE_TITLE_SUFFIX_RE.sub("", str(storyline_title or "").strip()).strip(" ,")
    display_entities = [
        _entity_key_display_text(key)
        for key in anchor_entities
        if _entity_key_display_text(key)
    ]
    phase_label = _episode_phase_label(phase_type)
    if display_entities and phase_label:
        return f"{', '.join(display_entities[:2])}: {phase_label}"
    if cleaned_title and not _is_weak_topic_term(cleaned_title):
        return cleaned_title
    if display_entities:
        return ", ".join(display_entities[:2])
    fallback_terms = [
        term
        for term in sorted(_macro_topic_sig_values(topic_signatures, "topic_term:"))
        if not _is_weak_topic_term(term)
    ]
    if fallback_terms and phase_label:
        return f"{', '.join(fallback_terms[:2])}: {phase_label}"
    if fallback_terms:
        return ", ".join(fallback_terms[:2])
    return phase_label or cleaned_title or "episode"


def _episode_signatures(
    *,
    title: str,
    seed_preview: str,
    entity_keys: set[str],
    family_signatures: set[str],
    topic_signatures: set[str],
) -> set[str]:
    phase_type = _episode_phase_type(
        title=title,
        seed_preview=seed_preview,
        topic_signatures=topic_signatures,
    )
    signatures: set[str] = set()
    if phase_type:
        signatures.add(f"episode_phase:{phase_type}")
    for key in _macro_topic_anchor_keys(entity_keys)[:4]:
        signatures.add(f"episode_anchor:{key}")
    for sig in topic_signatures:
        value = str(sig or "").strip()
        if not value:
            continue
        if value.startswith("topic_pair:"):
            signatures.add(value.replace("topic_pair:", "episode_pair:", 1))
            continue
        if value.startswith("topic_term:"):
            term = value.split(":", 1)[1].strip()
            if term and not _is_weak_topic_term(term):
                signatures.add(f"episode_term:{term}")
    for sig in family_signatures:
        value = str(sig or "").strip()
        if not value:
            continue
        if value.startswith("family_anchor:"):
            signatures.add(value.replace("family_anchor:", "episode_family:", 1))
    return signatures


def _episode_payload(
    *,
    macro_topic_id: str,
    title: str,
    seed_preview: str,
    entity_keys: set[str],
    family_signatures: set[str],
    topic_signatures: set[str],
    existing_episode_id: str = "",
) -> dict[str, Any]:
    episode_signatures = _episode_signatures(
        title=title,
        seed_preview=seed_preview,
        entity_keys=entity_keys,
        family_signatures=family_signatures,
        topic_signatures=topic_signatures,
    )
    phase_type = _episode_phase_type(
        title=title,
        seed_preview=seed_preview,
        topic_signatures=topic_signatures,
    )
    strong_entity_keys = sorted(
        (
            key
            for key in entity_keys
            if key
            and not _is_broad_entity_key(key)
            and _anchor_key_score(key) >= 4.0
        ),
        key=lambda key: (-_anchor_key_score(key), key),
    )
    anchor_entities = strong_entity_keys[:4]
    basis_components = [
        *sorted(sig for sig in episode_signatures if sig.startswith("episode_phase:"))[:1],
        *sorted(sig for sig in episode_signatures if sig.startswith("episode_pair:"))[:2],
        *sorted(sig for sig in episode_signatures if sig.startswith("episode_anchor:"))[:3],
        *sorted(sig for sig in episode_signatures if sig.startswith("episode_term:"))[:3],
        *sorted(sig for sig in episode_signatures if sig.startswith("episode_family:"))[:1],
    ]
    if not basis_components:
        basis_components = sorted(episode_signatures)[:4] or [str(title or "").strip()[:120] or "episode"]
    basis_blob = f"topic:{macro_topic_id}|{'|'.join(basis_components)}"
    episode_id = str(existing_episode_id or "").strip()
    if not episode_id:
        episode_id = f"episode_{hashlib.sha1(basis_blob.encode('utf-8')).hexdigest()[:16]}"
    canonical_title = _episode_title(
        storyline_title=title,
        phase_type=phase_type,
        anchor_entities=anchor_entities,
        topic_signatures=topic_signatures,
    )
    title_score = _title_quality_score(
        canonical_title,
        set(anchor_entities) if anchor_entities else entity_keys,
    )
    topic_aliases = sorted(
        {
            canonical_title,
            str(title or "").strip(),
        }
        | {
            _entity_key_display_text(key)
            for key in anchor_entities
            if _entity_key_display_text(key)
        }
    )
    return {
        "episode_id": episode_id,
        "canonical_title": canonical_title,
        "phase_type": phase_type,
        "title_score": round(float(title_score or 0.0), 4),
        "anchor_entities": anchor_entities,
        "topic_aliases": [alias for alias in topic_aliases if alias][:8],
        "episode_signatures": sorted(episode_signatures),
        "basis_components": basis_components,
        "seed_preview": str(seed_preview or "").strip()[:500],
        "macro_topic_id": str(macro_topic_id or "").strip(),
    }


def _episode_candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": str(candidate.get("episode_id") or candidate.get("id") or "").strip(),
        "canonical_title": str(candidate.get("canonical_title") or candidate.get("episode_title") or "").strip(),
        "phase_type": str(candidate.get("phase_type") or "").strip(),
        "episode_signatures": {
            str(x or "").strip()
            for x in (candidate.get("episode_signatures") or [])
            if str(x or "").strip()
        },
        "anchor_entities": {
            str(x or "").strip()
            for x in (candidate.get("anchor_entities") or [])
            if str(x or "").strip()
        },
        "topic_aliases": {
            str(x or "").strip()
            for x in (candidate.get("topic_aliases") or [])
            if str(x or "").strip()
        },
    }


def _episode_alignment_score(payload: dict[str, Any], candidate: dict[str, Any]) -> float:
    cur_signatures = {
        str(x or "").strip()
        for x in (payload.get("episode_signatures") or [])
        if str(x or "").strip()
    }
    cur_anchor_entities = {
        str(x or "").strip()
        for x in (payload.get("anchor_entities") or [])
        if str(x or "").strip()
    }
    cur_title = str(payload.get("canonical_title") or "").strip()
    cur_phase = str(payload.get("phase_type") or "").strip()
    cur_pairs = _macro_topic_sig_values(cur_signatures, "episode_pair:")
    cur_terms = _macro_topic_sig_values(cur_signatures, "episode_term:")
    cur_family = _macro_topic_sig_values(cur_signatures, "episode_family:")
    cur_semantic_keys = _macro_topic_semantic_keys(
        topic_signatures={sig.replace("episode_pair:", "topic_pair:", 1).replace("episode_term:", "topic_term:", 1) for sig in cur_signatures},
        anchor_entities=cur_anchor_entities,
        canonical_title=cur_title,
    )

    cand = _episode_candidate_payload(candidate)
    cand_signatures = cand["episode_signatures"]
    cand_anchor_entities = cand["anchor_entities"]
    cand_title = str(cand.get("canonical_title") or "").strip()
    cand_phase = str(cand.get("phase_type") or "").strip()
    cand_pairs = _macro_topic_sig_values(cand_signatures, "episode_pair:")
    cand_terms = _macro_topic_sig_values(cand_signatures, "episode_term:")
    cand_family = _macro_topic_sig_values(cand_signatures, "episode_family:")
    cand_semantic_keys = _macro_topic_semantic_keys(
        topic_signatures={sig.replace("episode_pair:", "topic_pair:", 1).replace("episode_term:", "topic_term:", 1) for sig in cand_signatures},
        anchor_entities=cand_anchor_entities,
        canonical_title=cand_title,
        topic_aliases=set(cand.get("topic_aliases") or []),
    )

    pair_overlap = _jaccard(cur_pairs, cand_pairs)
    term_overlap = _jaccard(cur_terms, cand_terms)
    family_overlap = _jaccard(cur_family, cand_family)
    anchor_overlap = _weighted_entity_overlap(cur_anchor_entities, cand_anchor_entities)[0]
    semantic_overlap = _jaccard(cur_semantic_keys, cand_semantic_keys)
    title_overlap = _text_jaccard(cur_title, cand_title)

    score = (
        0.22 * pair_overlap
        + 0.16 * term_overlap
        + 0.10 * family_overlap
        + 0.26 * anchor_overlap
        + 0.18 * semantic_overlap
        + 0.08 * title_overlap
    )
    if cur_phase and cand_phase and cur_phase == cand_phase:
        score += 0.18
    if pair_overlap >= 0.99 and cur_pairs and cand_pairs:
        score += 0.12
    if anchor_overlap >= 0.49 and cur_anchor_entities and cand_anchor_entities:
        score += 0.08
    if semantic_overlap >= 0.34:
        score += 0.08
    return round(_clamp01(score), 4)


def _select_episode_alignment(
    payload: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    preferred_episode_id: str = "",
    min_score: float = 0.56,
) -> dict[str, Any] | None:
    preferred_episode_id = str(preferred_episode_id or "").strip()
    best: dict[str, Any] | None = None
    best_score = 0.0
    for candidate in candidates:
        candidate_id = str(candidate.get("episode_id") or candidate.get("id") or "").strip()
        if not candidate_id:
            continue
        score = _episode_alignment_score(payload, candidate)
        if preferred_episode_id and candidate_id == preferred_episode_id:
            score += 0.06
        candidate_with_score = dict(candidate)
        candidate_with_score["alignment_score"] = round(_clamp01(score), 4)
        if score > best_score:
            best = candidate_with_score
            best_score = score
    if not best or best_score < float(min_score):
        return None
    return best


def _macro_topic_candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "macro_topic_id": str(candidate.get("macro_topic_id") or candidate.get("id") or "").strip(),
        "canonical_title": str(candidate.get("canonical_title") or candidate.get("macro_topic_title") or "").strip(),
        "topic_signatures": {
            str(x or "").strip()
            for x in (candidate.get("topic_signatures") or [])
            if str(x or "").strip()
        },
        "anchor_entities": {
            str(x or "").strip()
            for x in (candidate.get("anchor_entities") or [])
            if str(x or "").strip()
        },
        "basis_components": {
            str(x or "").strip()
            for x in (candidate.get("basis_components") or [])
            if str(x or "").strip()
        },
        "topic_aliases": {
            str(x or "").strip()
            for x in (candidate.get("topic_aliases") or [])
            if str(x or "").strip()
        },
    }


def _macro_topic_branch_anchor_keys(anchor_entities: set[str]) -> set[str]:
    return {
        key
        for key in anchor_entities
        if key and _entity_key_parts(key)[0] in {"organization", "product", "event", "law"}
    }


def _macro_topic_alignment_features(payload: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    cur_signatures = {str(x or "").strip() for x in (payload.get("topic_signatures") or []) if str(x or "").strip()}
    cur_anchor_entities = {str(x or "").strip() for x in (payload.get("anchor_entities") or []) if str(x or "").strip()}
    cur_title = str(payload.get("canonical_title") or "").strip()
    cur_pairs = _macro_topic_sig_values(cur_signatures, "topic_pair:")
    cur_terms = {
        term for term in _macro_topic_sig_values(cur_signatures, "topic_term:")
        if not _is_weak_topic_term(term)
    }
    cur_non_person_entities = {
        key for key in cur_anchor_entities
        if _entity_key_parts(key)[0] != "person"
    }
    cur_branch_entities = _macro_topic_branch_anchor_keys(cur_anchor_entities)
    cur_semantic_keys = _macro_topic_semantic_keys(
        topic_signatures=cur_signatures,
        anchor_entities=cur_anchor_entities,
        canonical_title=cur_title,
    )

    cand = _macro_topic_candidate_payload(candidate)
    cand_signatures = cand["topic_signatures"]
    cand_anchor_entities = cand["anchor_entities"]
    cand_title = str(cand.get("canonical_title") or "").strip()
    cand_pairs = _macro_topic_sig_values(cand_signatures, "topic_pair:")
    cand_terms = {
        term for term in _macro_topic_sig_values(cand_signatures, "topic_term:")
        if not _is_weak_topic_term(term)
    }
    cand_non_person_entities = {
        key for key in cand_anchor_entities
        if _entity_key_parts(key)[0] != "person"
    }
    cand_branch_entities = _macro_topic_branch_anchor_keys(cand_anchor_entities)
    cand_semantic_keys = _macro_topic_semantic_keys(
        topic_signatures=cand_signatures,
        anchor_entities=cand_anchor_entities,
        canonical_title=cand_title,
        topic_aliases=set(cand.get("topic_aliases") or []),
    )
    pair_overlap = _jaccard(cur_pairs, cand_pairs)
    term_overlap = _jaccard(cur_terms, cand_terms)
    anchor_overlap = _weighted_entity_overlap(cur_anchor_entities, cand_anchor_entities)[0]
    non_person_anchor_overlap = _weighted_entity_overlap(cur_non_person_entities, cand_non_person_entities)[0]
    branch_anchor_overlap = _weighted_entity_overlap(cur_branch_entities, cand_branch_entities)[0]
    semantic_overlap = _jaccard(cur_semantic_keys, cand_semantic_keys)
    title_overlap = _text_jaccard(cur_title, cand_title)
    alias_overlap = max(
        [
            _text_jaccard(cur_title, alias)
            for alias in cand.get("topic_aliases") or []
            if str(alias or "").strip()
        ]
        or [0.0]
    )
    cand_person_only = bool(cand_anchor_entities) and not cand_non_person_entities
    cand_location_person_only = bool(cand_anchor_entities) and not cand_branch_entities
    cur_has_branch_entities = bool(cur_branch_entities)
    cand_title_weak = _is_weak_topic_term(cand_title)
    cand_title_low_quality = _title_quality_score(cand_title, cand_anchor_entities or cand_non_person_entities) < 2.0
    return {
        "pair_overlap": pair_overlap,
        "term_overlap": term_overlap,
        "anchor_overlap": anchor_overlap,
        "non_person_anchor_overlap": non_person_anchor_overlap,
        "branch_anchor_overlap": branch_anchor_overlap,
        "semantic_overlap": semantic_overlap,
        "title_overlap": title_overlap,
        "alias_overlap": alias_overlap,
        "cand_person_only": cand_person_only,
        "cand_location_person_only": cand_location_person_only,
        "cur_has_branch_entities": cur_has_branch_entities,
        "cand_title_weak": cand_title_weak,
        "cand_title_low_quality": cand_title_low_quality,
        "shared_semantic": cur_semantic_keys & cand_semantic_keys,
    }


def _macro_topic_alignment_merge_allowed(features: dict[str, Any]) -> bool:
    pair_overlap = float(features.get("pair_overlap") or 0.0)
    non_person_anchor_overlap = float(features.get("non_person_anchor_overlap") or 0.0)
    branch_anchor_overlap = float(features.get("branch_anchor_overlap") or 0.0)
    semantic_overlap = float(features.get("semantic_overlap") or 0.0)
    title_overlap = float(features.get("title_overlap") or 0.0)
    alias_overlap = float(features.get("alias_overlap") or 0.0)
    anchor_overlap = float(features.get("anchor_overlap") or 0.0)

    if pair_overlap >= 0.34 or branch_anchor_overlap >= 0.34 or non_person_anchor_overlap >= 0.45:
        return True
    if semantic_overlap >= 0.45 and (pair_overlap >= 0.12 or branch_anchor_overlap >= 0.18):
        return True
    if semantic_overlap >= 0.55 and title_overlap >= 0.34:
        return True

    if (
        bool(features.get("cur_has_branch_entities"))
        and bool(features.get("cand_location_person_only"))
        and pair_overlap < 0.18
        and branch_anchor_overlap < 0.18
    ):
        return False

    if (
        bool(features.get("cand_title_weak")) or bool(features.get("cand_title_low_quality"))
    ) and pair_overlap < 0.18 and branch_anchor_overlap < 0.18 and semantic_overlap < 0.34:
        return False

    if (
        pair_overlap == 0.0
        and branch_anchor_overlap == 0.0
        and non_person_anchor_overlap < 0.18
        and semantic_overlap < 0.25
        and title_overlap < 0.25
        and alias_overlap < 0.25
        and anchor_overlap < 0.25
    ):
        return False

    return semantic_overlap >= 0.30 and (title_overlap >= 0.20 or alias_overlap >= 0.20)


def _macro_topic_alignment_score(payload: dict[str, Any], candidate: dict[str, Any]) -> float:
    features = _macro_topic_alignment_features(payload, candidate)
    pair_overlap = float(features["pair_overlap"])
    term_overlap = float(features["term_overlap"])
    anchor_overlap = float(features["anchor_overlap"])
    non_person_anchor_overlap = float(features["non_person_anchor_overlap"])
    branch_anchor_overlap = float(features["branch_anchor_overlap"])
    semantic_overlap = float(features["semantic_overlap"])
    title_overlap = float(features["title_overlap"])
    alias_overlap = float(features["alias_overlap"])
    shared_semantic = set(features["shared_semantic"] or set())

    score = (
        0.34 * pair_overlap
        + 0.08 * term_overlap
        + 0.14 * non_person_anchor_overlap
        + 0.18 * branch_anchor_overlap
        + 0.08 * anchor_overlap
        + 0.18 * semantic_overlap
        + 0.06 * title_overlap
        + 0.01 * alias_overlap
    )

    if pair_overlap >= 0.99:
        score += 0.18
    if non_person_anchor_overlap >= 0.49:
        score += 0.08
    if shared_semantic:
        strong_shared_semantic = {
            key for key in shared_semantic
            if key and not key.isdigit() and len(key) >= 5
        }
        if len(shared_semantic) >= 2:
            score += 0.10
        elif len(shared_semantic) == 1 and pair_overlap > 0.0:
            score += 0.05
        if len(strong_shared_semantic) >= 2:
            score += 0.30
        elif len(strong_shared_semantic) >= 1 and branch_anchor_overlap >= 0.20:
            score += 0.12

    if bool(features["cand_person_only"]) and bool(features["cur_has_branch_entities"]):
        score -= 0.20
    if bool(features["cand_location_person_only"]) and bool(features["cur_has_branch_entities"]):
        score -= 0.18
    if bool(features["cand_title_weak"]):
        score -= 0.08
    if bool(features["cand_title_low_quality"]) and pair_overlap < 0.20 and branch_anchor_overlap < 0.20:
        score -= 0.10
    if not _macro_topic_alignment_merge_allowed(features):
        score -= 0.35

    return round(_clamp01(score), 4)


def _select_macro_topic_alignment(
    payload: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    preferred_macro_topic_id: str = "",
    min_score: float = 0.50,
) -> dict[str, Any] | None:
    preferred_macro_topic_id = str(preferred_macro_topic_id or "").strip()
    best: dict[str, Any] | None = None
    best_score = 0.0
    for candidate in candidates:
        candidate_id = str(candidate.get("macro_topic_id") or candidate.get("id") or "").strip()
        if not candidate_id:
            continue
        features = _macro_topic_alignment_features(payload, candidate)
        if not _macro_topic_alignment_merge_allowed(features):
            continue
        score = _macro_topic_alignment_score(payload, candidate)
        if preferred_macro_topic_id and candidate_id == preferred_macro_topic_id:
            score += 0.06
        candidate_with_score = dict(candidate)
        candidate_with_score["alignment_score"] = round(_clamp01(score), 4)
        if score > best_score:
            best = candidate_with_score
            best_score = score
    if not best or best_score < float(min_score):
        return None
    return best


def _update_step_anchor_entities(entity_keys: set[str]) -> list[str]:
    strong = sorted(
        (
            key
            for key in entity_keys
            if key
            and not _is_broad_entity_key(key)
            and _anchor_key_score(key) >= 4.0
        ),
        key=lambda key: (-_anchor_key_score(key), key),
    )
    return strong[:3]


def _update_step_payload(
    *,
    macro_topic_id: str,
    storyline_id: str,
    event_mention_id: str,
    post_id: int,
    event_time: datetime | None,
    summary: str,
    event_signature: str,
    entity_keys: set[str],
    participant_role_signatures: set[str],
) -> dict[str, Any]:
    anchor_entities = _update_step_anchor_entities(entity_keys)
    role_tokens = sorted(
        {
            str(sig or "").strip()
            for sig in participant_role_signatures
            if str(sig or "").strip()
        }
    )[:2]
    time_bucket = event_time.strftime("%Y-%m-%d") if event_time else "undated"
    signature_parts = [f"topic:{macro_topic_id}", f"date:{time_bucket}"]
    if event_signature:
        signature_parts.append(f"event:{event_signature}")
    for key in anchor_entities[:2]:
        signature_parts.append(f"entity:{key}")
    for sig in role_tokens[:1]:
        signature_parts.append(f"role:{sig}")
    signature_blob = "|".join(signature_parts)
    step_id = f"update_step_{hashlib.sha1(signature_blob.encode('utf-8')).hexdigest()[:16]}"
    canonical_summary = str(summary or "").strip()[:500]
    if not canonical_summary:
        canonical_summary = str(event_signature or storyline_id or macro_topic_id)
    return {
        "step_id": step_id,
        "step_signature": signature_blob,
        "canonical_summary": canonical_summary,
        "anchor_entities": anchor_entities,
        "event_signature": str(event_signature or "").strip(),
        "role_signatures": role_tokens,
        "storyline_ids": [storyline_id] if storyline_id else [],
        "source_post_ids": [int(post_id)] if post_id else [],
        "event_mention_ids": [event_mention_id] if event_mention_id else [],
        "first_seen_at": _safe_iso(event_time),
        "last_seen_at": _safe_iso(event_time),
    }


def _candidate_family_signatures(candidate: dict[str, Any]) -> set[str]:
    existing = {
        str(sig or "").strip()
        for sig in (candidate.get("family_signatures") or [])
        if str(sig or "").strip()
    }
    if existing:
        return existing
    seed_preview = str(candidate.get("seed_preview") or "")
    title = str(candidate.get("title") or "")
    entity_keys = {str(x or "").strip() for x in (candidate.get("entity_keys") or []) if str(x or "").strip()}
    event_signatures = {str(x or "").strip() for x in (candidate.get("event_signatures") or []) if str(x or "").strip()}
    derived = _family_signatures(
        source_text=f"{title}\n{seed_preview}".strip(),
        entity_keys=entity_keys,
        event_signatures=event_signatures,
    )
    candidate["family_signatures"] = sorted(derived)
    return derived


def _candidate_topic_signatures(candidate: dict[str, Any]) -> set[str]:
    existing = {
        str(sig or "").strip()
        for sig in (candidate.get("topic_signatures") or [])
        if str(sig or "").strip()
    }
    if existing:
        return existing
    seed_preview = str(candidate.get("seed_preview") or "")
    title = str(candidate.get("title") or "")
    entity_keys = {str(x or "").strip() for x in (candidate.get("entity_keys") or []) if str(x or "").strip()}
    family_signatures = _candidate_family_signatures(candidate)
    derived = _topic_signatures(
        source_text=f"{title}\n{seed_preview}".strip(),
        entity_keys=entity_keys,
        family_signatures=family_signatures,
    )
    candidate["topic_signatures"] = sorted(derived)
    return derived


def _temporal(current_ts: datetime | None, ref_ts: datetime | None) -> float:
    if current_ts is None or ref_ts is None:
        return 0.5
    delta_h = abs((current_ts - ref_ts).total_seconds()) / 3600.0
    return _clamp01(math.exp(-delta_h / 72.0))


def _score_candidate(
    *,
    event_embedding: list[float],
    event_entity_keys: set[str],
    event_relation_signatures: set[str],
    event_signatures: set[str],
    family_signatures: set[str],
    topic_signatures: set[str],
    participant_role_signatures: set[str],
    location_keys: set[str],
    event_ts: datetime | None,
    title_hint: str,
    source_text: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    centroid = candidate.get("centroid_embedding") or []
    cand_entities = set(candidate.get("entity_keys") or [])
    cand_relations = set(candidate.get("relation_signatures") or candidate.get("all_relation_signatures") or [])
    cand_events = set(candidate.get("event_signatures") or [])
    cand_family = _candidate_family_signatures(candidate)
    cand_topic = _candidate_topic_signatures(candidate)
    cand_ts = _parse_dt(candidate.get("last_event_time"))

    emb = _cosine(event_embedding, centroid)
    ent, broad_entity_ratio = _weighted_entity_overlap(event_entity_keys, cand_entities)
    rel = _jaccard(event_relation_signatures, cand_relations)
    evt = _jaccard(event_signatures, cand_events)
    tmp = _temporal(event_ts, cand_ts)
    duplicate = _candidate_duplicate_text_overlap(candidate, source_text, title_hint)
    tfidf_seed_cosine, tfidf_peer_max_cosine = _candidate_tfidf_features(candidate, source_text, title_hint)
    candidate_text = f"{candidate.get('title') or ''}\n{candidate.get('seed_preview') or ''}".strip()
    facet_features = _facet_pair_features(source_text, candidate_text)
    family_overlap = _jaccard(family_signatures, cand_family)
    topic_overlap = _jaccard(topic_signatures, cand_topic)
    exemplar_role_overlap, exemplar_event_frame_overlap = _candidate_recent_exemplar_features(
        candidate,
        event_signatures=event_signatures,
        participant_role_signatures=participant_role_signatures,
        location_keys=location_keys,
    )
    digest_penalty = _candidate_digest_penalty(candidate, ent=ent, rel=rel, evt=evt)
    ad_penalty = _candidate_ad_penalty(
        candidate,
        ent=ent,
        rel=rel,
        evt=evt,
        duplicate=duplicate,
    )
    broad_entity_penalty = 0.0
    if ent > 0.0 and rel <= 1e-9 and broad_entity_ratio >= 0.75:
        broad_entity_penalty = round(0.06 + (0.06 * broad_entity_ratio), 4)
    useful_facet_overlap = float(facet_features.get("useful_facet_overlap_ratio", 0.0) or 0.0)
    facet_type_overlap = float(facet_features.get("facet_type_overlap_ratio", 0.0) or 0.0)
    facet_broad_penalty = 0.0
    facet_useful_bonus = 0.0
    facet_type_bonus = 0.0
    if useful_facet_overlap > 0.0:
        facet_useful_bonus = round(STORYLINE_FACET_USEFUL_MATCH_BONUS * useful_facet_overlap, 4)
    elif facet_type_overlap > 0.0:
        facet_type_bonus = round(STORYLINE_FACET_TYPE_MATCH_BONUS * facet_type_overlap, 4)
    if (
        float(facet_features.get("shared_registry_entity_count", 0.0) or 0.0) > 0.0
        and broad_entity_ratio >= 0.50
        and useful_facet_overlap <= 1e-9
    ):
        facet_broad_penalty = round(
            STORYLINE_FACET_BROAD_NO_MATCH_PENALTY
            * broad_entity_ratio
            * max(0.4, float(facet_features.get("broad_entity_no_useful_facet_ratio", 0.0) or 0.0)),
            4,
        )

    contributions = {
        "embedding_similarity": round(0.24 * emb, 4),
        "entity_overlap": round(0.18 * ent, 4),
        "relation_overlap": round(0.08 * rel, 4),
        "event_overlap": round(0.06 * evt, 4),
        "temporal_proximity": round(0.08 * tmp, 4),
        "duplicate_text_overlap": round(0.10 * duplicate, 4),
        "tfidf_post_seed_cosine": round(STORYLINE_TFIDF_SEED_WEIGHT * tfidf_seed_cosine, 4),
        "tfidf_post_peer_max_cosine": round(STORYLINE_TFIDF_PEER_WEIGHT * tfidf_peer_max_cosine, 4),
        "family_signature_overlap": round(0.12 * family_overlap, 4),
        "topic_signature_overlap": round(0.08 * topic_overlap, 4),
        "exemplar_role_overlap": round(0.10 * exemplar_role_overlap, 4),
        "exemplar_event_frame_overlap": round(0.16 * exemplar_event_frame_overlap, 4),
        "facet_useful_match_bonus": round(facet_useful_bonus, 4),
        "facet_type_match_bonus": round(facet_type_bonus, 4),
    }
    score = round(
        sum(contributions.values()) - digest_penalty - ad_penalty - broad_entity_penalty - facet_broad_penalty,
        4,
    )
    return {
        "storyline_id": str(candidate.get("id") or ""),
        "candidate_posts_count": int(candidate.get("posts_count") or 0),
        "score": score,
        "seed_preview": str(candidate.get("seed_preview") or ""),
        "recent_mentions": list(candidate.get("recent_mentions") or []),
        "seed_quality_score": round(_candidate_seed_quality(candidate), 4),
        "is_retrievable_seed": _candidate_is_retrievable_seed(candidate),
        "is_retrievable_storyline": _candidate_is_retrievable_storyline(candidate),
        "components": {
            "embedding_similarity": round(emb, 4),
            "entity_overlap": round(ent, 4),
            "relation_overlap": round(rel, 4),
            "event_overlap": round(evt, 4),
            "temporal_proximity": round(tmp, 4),
            "duplicate_text_overlap": round(duplicate, 4),
            "tfidf_post_seed_cosine": round(tfidf_seed_cosine, 4),
            "tfidf_post_peer_max_cosine": round(tfidf_peer_max_cosine, 4),
            "family_signature_overlap": round(family_overlap, 4),
            "topic_signature_overlap": round(topic_overlap, 4),
            "exemplar_role_overlap": round(exemplar_role_overlap, 4),
            "exemplar_event_frame_overlap": round(exemplar_event_frame_overlap, 4),
            "broad_entity_overlap_ratio": round(broad_entity_ratio, 4),
            "shared_registry_entity_count": round(float(facet_features.get("shared_registry_entity_count", 0.0) or 0.0), 4),
            "facet_label_match_count": round(float(facet_features.get("facet_label_match_count", 0.0) or 0.0), 4),
            "facet_type_match_count": round(float(facet_features.get("facet_type_match_count", 0.0) or 0.0), 4),
            "useful_facet_match_count": round(float(facet_features.get("useful_facet_match_count", 0.0) or 0.0), 4),
            "facet_label_overlap_ratio": round(float(facet_features.get("facet_label_overlap_ratio", 0.0) or 0.0), 4),
            "facet_type_overlap_ratio": round(float(facet_features.get("facet_type_overlap_ratio", 0.0) or 0.0), 4),
            "useful_facet_overlap_ratio": round(useful_facet_overlap, 4),
            "broad_entity_no_useful_facet_ratio": round(float(facet_features.get("broad_entity_no_useful_facet_ratio", 0.0) or 0.0), 4),
            "facet_post_max_similarity": round(float(facet_features.get("facet_post_max_similarity", 0.0) or 0.0), 4),
            "facet_seed_max_similarity": round(float(facet_features.get("facet_seed_max_similarity", 0.0) or 0.0), 4),
        },
        "contributions": {
            **contributions,
            "digest_seed_penalty": round(-digest_penalty, 4),
            "ad_seed_penalty": round(-ad_penalty, 4),
            "broad_entity_penalty": round(-broad_entity_penalty, 4),
            "facet_broad_no_match_penalty": round(-facet_broad_penalty, 4),
        },
        "seed_is_digest": bool(candidate.get("seed_is_digest")),
        "seed_digest_score": round(float(candidate.get("seed_digest_score", 0.0) or 0.0), 4),
        "seed_is_advertising": _candidate_ad_like(candidate),
        "seed_ad_score": round(float(candidate.get("seed_ad_score", 0.0) or 0.0), 4),
        "candidate_event_signature_count": _candidate_event_signature_count(candidate),
        "family_signatures": sorted(cand_family),
        "topic_signatures": sorted(cand_topic),
        "facet_shared_entities": str(facet_features.get("facet_shared_entities", "") or ""),
    }


def _merge_embedding(prev: list[float], incoming: list[float], n_prev: int) -> list[float]:
    if not prev:
        return list(incoming)
    if not incoming or len(prev) != len(incoming):
        return prev
    n_next = n_prev + 1
    return [((x * n_prev) + y) / n_next for x, y in zip(prev, incoming)]


def _storyline_seed_post_id(storyline_id: str) -> int | None:
    if not storyline_id.startswith("storyline_"):
        return None
    suffix = storyline_id.split("storyline_", 1)[1]
    if suffix.isdigit():
        return int(suffix)
    return None


def _family_id_for_root(root_storyline_id: str) -> str:
    root = str(root_storyline_id or "").strip()
    if not root:
        return ""
    return f"family_{root}"


def _digest_score(text: str) -> float:
    raw = str(text or "").strip()
    if not raw:
        return 0.0
    lowered = raw.lower()
    score = 0.0
    if _DIGEST_TITLE_RE.search(lowered):
        score += 0.45
    score += min(0.30, lowered.count("➤") * 0.08)
    score += min(0.20, lowered.count("\n") * 0.03)
    score += min(0.15, lowered.count(" • ") * 0.05)
    score += min(0.15, lowered.count(" - ") * 0.04)
    return _clamp01(score)


def _is_digest_like_text(text: str) -> bool:
    return _digest_score(text) >= 0.45


def _candidate_event_signature_count(candidate: dict[str, Any]) -> int:
    return len(set(candidate.get("event_signatures") or []))


def _ad_score(text: str) -> float:
    raw = str(text or "").strip()
    if not raw:
        return 0.0
    lowered = raw.lower()
    score = 0.0
    if _AD_TITLE_RE.search(lowered):
        score += 0.60
    score += min(0.20, lowered.count("https://") * 0.10)
    score += min(0.10, lowered.count("telegram") * 0.05)
    return _clamp01(score)


def _is_ad_like_text(text: str) -> bool:
    return _ad_score(text) >= 0.55


def _candidate_ad_like(candidate: dict[str, Any]) -> bool:
    if bool(candidate.get("seed_is_advertising")):
        return True
    if float(candidate.get("seed_ad_score", 0.0) or 0.0) >= 0.55:
        return True
    seed_preview = str(candidate.get("seed_preview") or "")
    title = str(candidate.get("title") or "")
    return _is_ad_like_text(f"{title}\n{seed_preview}")


def _normalized_text_signature(text: str) -> str:
    return " ".join(sorted(_normalized_tokens(text)))


def _tfidf_vector(text: str, local_idf: dict[str, float]) -> dict[str, float]:
    toks = list(_normalized_tokens(text))
    if not toks:
        return {}
    counts: dict[str, int] = {}
    for tok in toks:
        counts[tok] = counts.get(tok, 0) + 1
    total = float(len(toks))
    return {
        tok: (cnt / total) * float(local_idf.get(tok, 1.0))
        for tok, cnt in counts.items()
    }


def _local_tfidf_idf(texts: list[str]) -> dict[str, float]:
    docs = [set(_normalized_tokens(t)) for t in texts if str(t or "").strip()]
    if not docs:
        return {}
    n_docs = len(docs)
    df: dict[str, int] = {}
    for doc in docs:
        for tok in doc:
            df[tok] = df.get(tok, 0) + 1
    return {
        tok: math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0
        for tok, freq in df.items()
    }


def _tfidf_cosine(a: str, b: str, local_idf: dict[str, float]) -> float:
    va = _tfidf_vector(a, local_idf)
    vb = _tfidf_vector(b, local_idf)
    if not va or not vb:
        return 0.0
    inter = set(va) & set(vb)
    dot = sum(va[k] * vb[k] for k in inter)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return _clamp01(dot / (na * nb))


def _candidate_tfidf_features(candidate: dict[str, Any], source_text: str, title_hint: str) -> tuple[float, float]:
    src = str(source_text or "").strip()
    src_with_title = f"{title_hint}\n{src}".strip()
    seed_text = f"{candidate.get('title') or ''}\n{candidate.get('seed_preview') or ''}".strip()
    peer_texts = [
        str(mention.get("summary") or "").strip()
        for mention in (candidate.get("recent_mentions") or [])
        if str(mention.get("summary") or "").strip()
    ]
    corpus = [src_with_title, src, seed_text, *peer_texts]
    local_idf = _local_tfidf_idf(corpus)
    seed_cos = max(
        _tfidf_cosine(src, seed_text, local_idf),
        _tfidf_cosine(src_with_title, seed_text, local_idf),
    ) if seed_text else 0.0
    peer_cos = max(
        [_tfidf_cosine(src, peer, local_idf) for peer in peer_texts]
        + [_tfidf_cosine(src_with_title, peer, local_idf) for peer in peer_texts]
        + [0.0]
    )
    return seed_cos, peer_cos


def _candidate_seed_quality(candidate: dict[str, Any]) -> float:
    stored = candidate.get("seed_quality_score")
    try:
        if stored is not None and str(stored).strip() != "":
            return float(stored)
    except Exception:
        pass
    meta = classify_storyline_content(str(candidate.get("seed_preview") or ""))
    return float(meta.get("seed_quality_score", 0.0) or 0.0)


def _candidate_hygiene_suppressed(candidate: dict[str, Any]) -> bool:
    stored = candidate.get("retrieval_suppressed")
    if stored is not None and str(stored).strip() != "":
        return str(stored).strip().lower() in {"1", "true", "yes"}
    hygiene_reason = str(candidate.get("hygiene_reason") or "").strip().lower()
    if "suppressed" in hygiene_reason:
        return True
    try:
        hygiene_score = float(candidate.get("hygiene_score") or 0.0)
    except Exception:
        hygiene_score = 0.0
    return hygiene_score >= 1.0


def _candidate_is_retrievable_seed(candidate: dict[str, Any]) -> bool:
    if _candidate_hygiene_suppressed(candidate):
        return False
    stored = candidate.get("is_retrievable_seed")
    if stored is not None and str(stored).strip() != "":
        return str(stored).strip().lower() in {"1", "true", "yes"}
    return _candidate_seed_quality(candidate) >= STORYLINE_MIN_RETRIEVABLE_SEED_QUALITY


def _candidate_is_retrievable_storyline(candidate: dict[str, Any]) -> bool:
    if _candidate_hygiene_suppressed(candidate):
        return False
    if _candidate_is_retrievable_seed(candidate):
        return True

    posts_count = int(candidate.get("posts_count") or candidate.get("candidate_posts_count") or 0)
    recent_mentions = candidate.get("recent_mentions") or []
    rich_recent_mentions = sum(
        1 for mention in recent_mentions if len(str(mention.get("summary") or "").strip()) >= 40
    )
    return posts_count >= 3 and rich_recent_mentions >= 2


def _candidate_duplicate_text_overlap(candidate: dict[str, Any], source_text: str, title_hint: str) -> float:
    ref_source = str(source_text or "").strip()
    ref_a = f"{title_hint}\n{ref_source}".strip()
    ref_sig = _normalized_text_signature(ref_source)
    ref_b = f"{candidate.get('title') or ''}\n{candidate.get('seed_preview') or ''}".strip()
    seed_preview = str(candidate.get("seed_preview") or "").strip()
    overlaps = [_text_jaccard(ref_a, ref_b), _text_jaccard(ref_source, seed_preview)]
    if ref_sig and ref_sig == _normalized_text_signature(seed_preview):
        overlaps.append(1.0)
    for mention in candidate.get("recent_mentions") or []:
        mention_text = str(mention.get("summary") or "").strip()
        if mention_text:
            overlaps.append(_text_jaccard(ref_a, mention_text))
            overlaps.append(_text_jaccard(ref_source, mention_text))
            if ref_sig and ref_sig == _normalized_text_signature(mention_text):
                overlaps.append(1.0)
    return max(overlaps) if overlaps else 0.0


def _candidate_recent_exemplar_features(
    candidate: dict[str, Any],
    *,
    event_signatures: set[str],
    participant_role_signatures: set[str],
    location_keys: set[str],
) -> tuple[float, float]:
    current_roles = {str(sig or "").strip() for sig in participant_role_signatures if str(sig or "").strip()}
    current_event_frame = {str(sig or "").strip() for sig in event_signatures if str(sig or "").strip()}
    current_event_frame |= {f"event_place:{key}" for key in location_keys if str(key or "").strip()}
    best_role = 0.0
    best_event_frame = 0.0

    for mention in candidate.get("recent_mentions") or []:
        mention_roles = {
            str(sig or "").strip()
            for sig in (mention.get("participant_role_signatures") or [])
            if str(sig or "").strip()
        }
        mention_event_frame = set()
        event_signature = str(mention.get("event_signature") or "").strip()
        update_kind = str(mention.get("update_kind") or "").strip()
        if event_signature:
            mention_event_frame.add(event_signature)
        if update_kind:
            mention_event_frame.add(f"update_kind:{update_kind}")
        mention_event_frame |= {
            f"participant_role:{sig}"
            for sig in mention_roles
        }
        mention_event_frame |= {
            f"event_place:{key}"
            for key in (mention.get("location_keys") or [])
            if str(key or "").strip()
        }

        best_role = max(best_role, _jaccard(current_roles, mention_roles))
        best_event_frame = max(best_event_frame, _jaccard(current_event_frame, mention_event_frame))

    return best_role, best_event_frame


def _candidate_digest_like(candidate: dict[str, Any]) -> bool:
    seed_is_digest = bool(candidate.get("seed_is_digest"))
    seed_digest_score = float(candidate.get("seed_digest_score", 0.0) or 0.0)
    posts_count = int(candidate.get("candidate_posts_count") or candidate.get("posts_count") or 0)
    event_signature_count = _candidate_event_signature_count(candidate)
    return (
        seed_is_digest
        or seed_digest_score >= 0.45
        or (posts_count <= 2 and event_signature_count >= 3)
    )


def _candidate_digest_penalty(candidate: dict[str, Any], *, ent: float, rel: float, evt: float) -> float:
    if not _candidate_digest_like(candidate) or evt <= 0.0:
        return 0.0
    if (ent + rel) > 0.0:
        return round(STORYLINE_DIGEST_SEED_PENALTY * 0.35, 4)
    penalty = STORYLINE_DIGEST_SEED_PENALTY
    if _candidate_event_signature_count(candidate) >= 4:
        penalty += 0.04
    return round(penalty, 4)


def _candidate_ad_penalty(candidate: dict[str, Any], *, ent: float, rel: float, evt: float, duplicate: float) -> float:
    if not _candidate_ad_like(candidate):
        return 0.0
    if duplicate >= 0.85:
        return 0.0
    if rel > 0.0:
        return round(STORYLINE_AD_SEED_PENALTY * 0.35, 4)
    if ent > 0.20 and evt > 0.0:
        return round(STORYLINE_AD_SEED_PENALTY * 0.50, 4)
    return round(STORYLINE_AD_SEED_PENALTY, 4)


def _is_event_only_low_conf_match(best: dict[str, Any] | None, event_confidence: float | None) -> bool:
    if not best:
        return False
    comps = best.get("components", {}) or {}
    ent = float(comps.get("entity_overlap", 0.0) or 0.0)
    rel = float(comps.get("relation_overlap", 0.0) or 0.0)
    evt = float(comps.get("event_overlap", 0.0) or 0.0)
    conf = float(event_confidence or 0.0)
    return (
        evt >= STORYLINE_EVENT_ONLY_MIN_OVERLAP
        and ent <= 1e-9
        and rel <= 1e-9
        and conf < STORYLINE_EVENT_ONLY_LOW_CONF_MAX
    )


def _is_duplicate_like_match(best: dict[str, Any] | None) -> bool:
    if not best:
        return False
    comps = best.get("components", {}) or {}
    return (
        float(comps.get("duplicate_text_overlap", 0.0) or 0.0) >= STORYLINE_DUPLICATE_TEXT_JACCARD_MIN
        or (
            float(comps.get("embedding_similarity", 0.0) or 0.0) >= STORYLINE_DUPLICATE_EMBEDDING_MIN
            and float(comps.get("event_overlap", 0.0) or 0.0) >= STORYLINE_DUPLICATE_EVENT_OVERLAP_MIN
            and float(comps.get("temporal_proximity", 0.0) or 0.0) >= STORYLINE_DUPLICATE_TEMPORAL_MIN
        )
    )


def _candidate_family_signal(candidate: dict[str, Any]) -> float:
    comps = candidate.get("components", {}) or {}
    ent = float(comps.get("entity_overlap", 0.0) or 0.0)
    broad_ratio = float(comps.get("broad_entity_overlap_ratio", 0.0) or 0.0)
    specific_entity_signal = ent * max(0.0, 1.0 - broad_ratio)
    return max(
        float(comps.get("family_signature_overlap", 0.0) or 0.0),
        float(comps.get("duplicate_text_overlap", 0.0) or 0.0),
        float(comps.get("exemplar_role_overlap", 0.0) or 0.0),
        float(comps.get("exemplar_event_frame_overlap", 0.0) or 0.0),
        float(comps.get("useful_facet_overlap_ratio", 0.0) or 0.0),
        float(comps.get("event_overlap", 0.0) or 0.0) * 0.7,
        specific_entity_signal,
    )


def _reroute_to_family_root(
    *,
    best: dict[str, Any] | None,
    scored_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not best:
        return best
    best_comps = best.get("components", {}) or {}
    best_overlap = float(best_comps.get("family_signature_overlap", 0.0) or 0.0)
    if best_overlap < STORYLINE_FAMILY_SIGNATURE_MIN_OVERLAP:
        return best

    best_root_alias = _family_root_alias(str(best.get("storyline_id") or ""))
    same_family = [
        cand for cand in scored_candidates
        if float((cand.get("components", {}) or {}).get("family_signature_overlap", 0.0) or 0.0)
        >= max(STORYLINE_FAMILY_SIGNATURE_MIN_OVERLAP, best_overlap - 0.10)
    ]
    if best_root_alias:
        alias_candidates = [
            cand for cand in scored_candidates
            if _family_root_alias(str(cand.get("storyline_id") or "")) == best_root_alias
        ]
        if alias_candidates:
            same_family = list({id(c): c for c in (same_family + alias_candidates)}.values())
    if len(same_family) < 2:
        best["family_root_storyline_id"] = best_root_alias or best.get("storyline_id") or ""
        best["family_rerouted_to_root"] = False
        return best

    def _root_rank(c: dict[str, Any]) -> tuple[int, float, int]:
        overlap = float((c.get("components", {}) or {}).get("family_signature_overlap", 0.0) or 0.0)
        posts = int(c.get("candidate_posts_count") or 0)
        seed = _storyline_seed_post_id(str(c.get("storyline_id") or "")) or 10**12
        return (posts, overlap, -seed)

    root = max(same_family, key=_root_rank)
    alias_root_id = _family_root_alias(str(root.get("storyline_id") or ""))
    if alias_root_id:
        alias_root_candidate = next(
            (cand for cand in same_family if str(cand.get("storyline_id") or "") == alias_root_id),
            None,
        )
        if alias_root_candidate is not None:
            root = alias_root_candidate
    if root is best:
        best["family_root_storyline_id"] = alias_root_id or best_root_alias or best.get("storyline_id") or ""
        best["family_rerouted_to_root"] = False
        return best
    if float(root.get("score", 0.0) or 0.0) + STORYLINE_FAMILY_ROOT_REROUTE_DELTA < float(best.get("score", 0.0) or 0.0):
        best["family_root_storyline_id"] = alias_root_id or root.get("storyline_id") or ""
        best["family_rerouted_to_root"] = False
        return best

    root = dict(root)
    root["family_root_storyline_id"] = alias_root_id or root.get("storyline_id") or ""
    root["family_rerouted_to_root"] = True
    root["family_original_best_storyline_id"] = best.get("storyline_id") or ""
    return root


def _apply_family_consolidation_bonus(
    scored_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(scored_candidates) < 2:
        return scored_candidates

    best_score = max(float(c.get("score", 0.0)) for c in scored_candidates)
    family_candidates = [
        c
        for c in scored_candidates
        if (best_score - float(c.get("score", 0.0))) <= STORYLINE_FAMILY_NEAR_BEST_DELTA
        and float((c.get("components", {}) or {}).get("embedding_similarity", 0.0) or 0.0)
        >= STORYLINE_FAMILY_MIN_EMBEDDING
        and _candidate_family_signal(c) >= STORYLINE_FAMILY_MIN_SIGNAL
    ]
    if len(family_candidates) < 2:
        return scored_candidates

    max_posts = max(int(c.get("candidate_posts_count") or 0) for c in family_candidates)
    seed_ids = [sid for sid in (_storyline_seed_post_id(str(c.get("storyline_id") or "")) for c in family_candidates) if sid]
    oldest_seed_id = min(seed_ids) if seed_ids else None
    max_posts_norm = math.log1p(max(1, max_posts))

    for cand in family_candidates:
        comps = cand.get("components", {}) or {}
        continuity_signal = max(
            float(comps.get("duplicate_text_overlap", 0.0) or 0.0),
            float(comps.get("exemplar_role_overlap", 0.0) or 0.0),
            float(comps.get("exemplar_event_frame_overlap", 0.0) or 0.0),
        )
        posts_count = int(cand.get("candidate_posts_count") or 0)
        posts_bonus = 0.0
        if max_posts_norm > 0.0:
            posts_bonus = 0.035 * (math.log1p(max(1, posts_count)) / max_posts_norm)
        seed_id = _storyline_seed_post_id(str(cand.get("storyline_id") or ""))
        root_bonus = 0.025 if oldest_seed_id is not None and seed_id == oldest_seed_id else 0.0
        continuity_bonus = 0.02 * continuity_signal
        family_boost = min(STORYLINE_FAMILY_MAX_BOOST, posts_bonus + root_bonus + continuity_bonus)

        cand["score"] = round(float(cand.get("score", 0.0)) + family_boost, 4)
        cand["contributions"] = {
            **(cand.get("contributions", {}) or {}),
            "family_consolidation_bonus": round(family_boost, 4),
        }
        cand["family_cluster_size"] = len(family_candidates)
        cand["family_signal"] = round(_candidate_family_signal(cand), 4)
        cand["family_boost"] = round(family_boost, 4)
        cand["family_oldest_seed_id"] = oldest_seed_id or ""
        cand["family_preferred_root"] = bool(oldest_seed_id is not None and seed_id == oldest_seed_id)

    return scored_candidates


async def _fetch_storyline_candidates(
    *,
    session,
    entity_keys: set[str],
    relation_signatures: set[str],
    event_signatures: set[str],
    family_signatures: set[str],
    topic_signatures: set[str],
    participant_role_signatures: set[str],
    location_keys: set[str],
) -> list[dict[str, Any]]:
    fetch_limit = max(
        STORYLINE_MAX_CANDIDATES,
        STORYLINE_MAX_CANDIDATES * STORYLINE_CANDIDATE_FETCH_MULTIPLIER,
    )
    shared_return = """
        WITH s
        ORDER BY s.updated_at DESC
        LIMIT $limit
        CALL {
            WITH s
            MATCH (s)<-[:PART_OF]-(em:EventMention)
            WHERE em.summary IS NOT NULL
            WITH em
            ORDER BY coalesce(em.event_time, '') DESC, coalesce(em.updated_at, '') DESC
            RETURN collect({
                event_mention_id: em.id,
                post_id: em.post_id,
                summary: em.summary,
                event_signature: coalesce(em.event_signature, ''),
                update_kind: coalesce(em.update_kind, ''),
                participant_role_signatures: coalesce(em.participant_role_signatures, []),
                location_keys: coalesce(em.location_keys, [])
            })[0..""" + str(max(1, STORYLINE_RECENT_EXEMPLAR_LIMIT)) + """] AS recent_mentions
        }
        RETURN s.id AS id,
               s.title AS title,
               s.centroid_embedding AS centroid_embedding,
               s.entity_keys AS entity_keys,
               s.relation_signatures AS relation_signatures,
               s.all_relation_signatures AS all_relation_signatures,
               s.event_signatures AS event_signatures,
               s.last_event_time AS last_event_time,
               s.posts_count AS posts_count,
               s.updated_at AS updated_at,
               s.seed_is_digest AS seed_is_digest,
               s.seed_digest_score AS seed_digest_score,
               s.seed_is_advertising AS seed_is_advertising,
               s.seed_ad_score AS seed_ad_score,
               s.seed_quality_score AS seed_quality_score,
               s.is_retrievable_seed AS is_retrievable_seed,
               s.retrieval_suppressed AS retrieval_suppressed,
               s.hygiene_score AS hygiene_score,
               s.hygiene_reason AS hygiene_reason,
               s.family_signatures AS family_signatures,
               s.topic_signatures AS topic_signatures,
               s.macro_topic_id AS macro_topic_id,
               s.macro_topic_title AS macro_topic_title,
               s.story_episode_id AS story_episode_id,
               s.story_episode_title AS story_episode_title,
               s.family_root_id AS family_root_id,
               s.seed_preview AS seed_preview,
               recent_mentions AS recent_mentions
        ORDER BY s.updated_at DESC
        LIMIT $limit
    """
    queries: list[tuple[str, str, dict[str, Any]]] = [
        ("recent", f"MATCH (s:Storyline)\n{shared_return}", {"limit": fetch_limit}),
    ]

    signal_conditions: list[str] = []
    signal_params: dict[str, Any] = {}
    signal_groups = 0
    if entity_keys:
        signal_conditions.append(
            "any(k IN coalesce(s.entity_keys, []) WHERE k IN $entity_keys)"
        )
        signal_params["entity_keys"] = sorted(entity_keys)
        signal_groups += 1
    if relation_signatures:
        signal_conditions.append(
            "(any(sig IN coalesce(s.relation_signatures, []) WHERE sig IN $relation_signatures) "
            "OR any(sig IN coalesce(s.all_relation_signatures, []) WHERE sig IN $relation_signatures))"
        )
        signal_params["relation_signatures"] = sorted(relation_signatures)
        signal_groups += 1
    if event_signatures:
        signal_conditions.append(
            "any(sig IN coalesce(s.event_signatures, []) WHERE sig IN $event_signatures)"
        )
        signal_params["event_signatures"] = sorted(event_signatures)
        signal_groups += 1
    if family_signatures:
        signal_conditions.append(
            "any(sig IN coalesce(s.family_signatures, []) WHERE sig IN $family_signatures)"
        )
        signal_params["family_signatures"] = sorted(family_signatures)
        signal_groups += 1
    if topic_signatures:
        signal_conditions.append(
            "any(sig IN coalesce(s.topic_signatures, []) WHERE sig IN $topic_signatures)"
        )
        signal_params["topic_signatures"] = sorted(topic_signatures)
        signal_groups += 1

    if signal_conditions:
        # One scan over Storyline is materially cheaper than one scan per signal family.
        signal_fetch_limit = min(fetch_limit * max(2, signal_groups), fetch_limit * 4)
        queries.append(
            (
                "storyline_signals",
                f"""
                MATCH (s:Storyline)
                WHERE {' OR '.join(signal_conditions)}
                {shared_return}
                """,
                {**signal_params, "limit": signal_fetch_limit},
            )
        )

    participant_query: tuple[str, str, dict[str, Any]] | None = None
    if participant_role_signatures or location_keys:
        participant_query = (
            "participant_location_fallback",
            f"""
            MATCH (s:Storyline)<-[:PART_OF]-(em:EventMention)
            WHERE any(sig IN coalesce(em.participant_role_signatures, []) WHERE sig IN $participant_role_signatures)
               OR any(loc IN coalesce(em.location_keys, []) WHERE loc IN $location_keys)
            WITH DISTINCT s
            {shared_return}
            """,
            {
                "participant_role_signatures": sorted(participant_role_signatures),
                "location_keys": sorted(location_keys),
                "limit": fetch_limit,
            },
        )

    merged: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    raw_rows_count = 0
    executed_queries_count = 0

    async def _run_candidate_query(
        *,
        query_index: int,
        query_name: str,
        query: str,
        params: dict[str, Any],
    ) -> None:
        nonlocal raw_rows_count, executed_queries_count
        executed_queries_count += 1
        query_started = time.monotonic()
        rows = await session.run(query, **params)
        row_data = await rows.data()
        raw_rows_count += len(row_data)
        query_elapsed_ms = int((time.monotonic() - query_started) * 1000)
        if query_elapsed_ms >= 1000:
            log.info(
                "storyline_candidate_query_slow",
                query_index=query_index,
                query_name=query_name,
                rows_count=len(row_data),
                elapsed_ms=query_elapsed_ms,
                fetch_limit=int(params.get("limit") or fetch_limit),
            )
        for row in row_data:
            storyline_id = str(row.get("id") or "")
            if storyline_id:
                merged.setdefault(storyline_id, row)

    for query_index, (query_name, query, params) in enumerate(queries):
        await _run_candidate_query(
            query_index=query_index,
            query_name=query_name,
            query=query,
            params=params,
        )

    participant_fallback_run = bool(
        participant_query and (not signal_conditions or len(merged) < fetch_limit)
    )
    if participant_fallback_run and participant_query:
        query_name, query, params = participant_query
        await _run_candidate_query(
            query_index=len(queries),
            query_name=query_name,
            query=query,
            params=params,
        )

    elapsed_ms = int((time.monotonic() - started) * 1000)
    if elapsed_ms >= 1000:
        log.info(
            "storyline_candidate_fetch_done",
            queries_count=executed_queries_count,
            raw_rows_count=raw_rows_count,
            merged_count=len(merged),
            elapsed_ms=elapsed_ms,
            fetch_limit=fetch_limit,
            signal_groups=signal_groups,
            participant_fallback_run=participant_fallback_run,
        )
    return list(merged.values())


async def _fetch_macro_topic_candidates(
    *,
    session,
    topic_signatures: set[str],
    anchor_entities: set[str],
    preferred_macro_topic_id: str = "",
    limit: int = 80,
) -> list[dict[str, Any]]:
    if not topic_signatures and not anchor_entities and not preferred_macro_topic_id:
        return []

    query = """
    MATCH (t:MacroTopic)
    WHERE ($preferred_macro_topic_id <> '' AND t.id = $preferred_macro_topic_id)
       OR any(sig IN coalesce(t.topic_signatures, []) WHERE sig IN $topic_signatures)
       OR any(ent IN coalesce(t.anchor_entities, []) WHERE ent IN $anchor_entities)
    RETURN
        t.id AS macro_topic_id,
        coalesce(t.canonical_title, '') AS canonical_title,
        coalesce(t.topic_signatures, []) AS topic_signatures,
        coalesce(t.anchor_entities, []) AS anchor_entities,
        coalesce(t.topic_aliases, []) AS topic_aliases,
        coalesce(t.basis_components, []) AS basis_components,
        coalesce(t.title_score, 0.0) AS title_score,
        coalesce(t.updated_at, '') AS updated_at
    ORDER BY coalesce(t.updated_at, '') DESC
    LIMIT $limit
    """
    rows = await session.run(
        query,
        topic_signatures=sorted(topic_signatures),
        anchor_entities=sorted(anchor_entities),
        preferred_macro_topic_id=str(preferred_macro_topic_id or "").strip(),
        limit=max(1, int(limit)),
    )
    return [dict(row or {}) for row in await rows.data()]


async def _fetch_episode_candidates(
    *,
    session,
    macro_topic_id: str,
    episode_signatures: set[str],
    anchor_entities: set[str],
    preferred_episode_id: str = "",
    limit: int = 60,
) -> list[dict[str, Any]]:
    macro_topic_id = str(macro_topic_id or "").strip()
    if not macro_topic_id:
        return []
    if not episode_signatures and not anchor_entities and not preferred_episode_id:
        return []

    query = """
    MATCH (e:Episode)-[:PART_OF_TOPIC]->(:MacroTopic {id: $macro_topic_id})
    WHERE ($preferred_episode_id <> '' AND e.id = $preferred_episode_id)
       OR any(sig IN coalesce(e.episode_signatures, []) WHERE sig IN $episode_signatures)
       OR any(ent IN coalesce(e.anchor_entities, []) WHERE ent IN $anchor_entities)
    RETURN
        e.id AS episode_id,
        coalesce(e.canonical_title, '') AS canonical_title,
        coalesce(e.phase_type, '') AS phase_type,
        coalesce(e.episode_signatures, []) AS episode_signatures,
        coalesce(e.anchor_entities, []) AS anchor_entities,
        coalesce(e.topic_aliases, []) AS topic_aliases,
        coalesce(e.title_score, 0.0) AS title_score,
        coalesce(e.updated_at, '') AS updated_at
    ORDER BY coalesce(e.updated_at, '') DESC
    LIMIT $limit
    """
    rows = await session.run(
        query,
        macro_topic_id=macro_topic_id,
        episode_signatures=sorted(episode_signatures),
        anchor_entities=sorted(anchor_entities),
        preferred_episode_id=str(preferred_episode_id or "").strip(),
        limit=max(1, int(limit)),
    )
    return [dict(row or {}) for row in await rows.data()]


def _has_force_assign_structural_signal(best: dict[str, Any] | None) -> bool:
    if not best:
        return False
    comps = best.get("components", {}) or {}
    ent = float(comps.get("entity_overlap", 0.0) or 0.0)
    rel = float(comps.get("relation_overlap", 0.0) or 0.0)
    evt = float(comps.get("event_overlap", 0.0) or 0.0)
    duplicate = float(comps.get("duplicate_text_overlap", 0.0) or 0.0)
    broad_ratio = float(comps.get("broad_entity_overlap_ratio", 0.0) or 0.0)
    if duplicate >= STORYLINE_DUPLICATE_TEXT_JACCARD_MIN:
        return True
    generic_only_entity = ent > 0.0 and rel <= 1e-9 and broad_ratio >= 0.75
    return (
        (ent >= STORYLINE_FORCE_ASSIGN_MIN_ENTITY_OVERLAP and not generic_only_entity)
        or rel >= STORYLINE_FORCE_ASSIGN_MIN_RELATION_OVERLAP
        or (ent + rel + evt) >= STORYLINE_FORCE_ASSIGN_MIN_STRUCTURAL_OVERLAP
    )


def _force_assign_ok(
    best: dict[str, Any] | None,
    *,
    decision_score: float,
    force_assign_min_score: float,
    structural_signal_ok: bool,
    duplicate_like_match: bool,
    duplicate_fastpath_ok: bool,
    continuity_signal: float,
    assign_blocked_by_event_only_low_conf: bool,
    assign_blocked_by_digest_seed: bool,
    assign_blocked_by_ad_seed: bool,
) -> bool:
    if not best or not best.get("storyline_id"):
        return False
    if decision_score < force_assign_min_score:
        return False
    if not structural_signal_ok:
        return False
    if assign_blocked_by_digest_seed or assign_blocked_by_ad_seed:
        return False
    if assign_blocked_by_event_only_low_conf and not duplicate_like_match and not duplicate_fastpath_ok:
        return False
    if not bool(best.get("is_retrievable_storyline")):
        return False

    comps = best.get("components", {}) or {}
    tfidf_seed = float(comps.get("tfidf_post_seed_cosine", 0.0) or 0.0)
    tfidf_peer = float(comps.get("tfidf_post_peer_max_cosine", 0.0) or 0.0)
    family_signal = max(
        float(comps.get("family_signature_overlap", 0.0) or 0.0),
        float(comps.get("exemplar_event_frame_overlap", 0.0) or 0.0),
        float(comps.get("exemplar_role_overlap", 0.0) or 0.0),
        float(comps.get("useful_facet_overlap_ratio", 0.0) or 0.0),
    )
    useful_facet = float(comps.get("useful_facet_overlap_ratio", 0.0) or 0.0)
    broad_no_useful = float(comps.get("broad_entity_no_useful_facet_ratio", 0.0) or 0.0)
    broad_ratio = float(comps.get("broad_entity_overlap_ratio", 0.0) or 0.0)
    duplicate = float(comps.get("duplicate_text_overlap", 0.0) or 0.0)

    if (
        broad_ratio >= 0.75
        and broad_no_useful >= 0.75
        and useful_facet <= 1e-9
        and duplicate < STORYLINE_DUPLICATE_FASTPATH_TEXT_MIN
        and tfidf_peer < STORYLINE_FORCE_ASSIGN_MIN_TFIDF
    ):
        return False

    if duplicate_like_match or duplicate_fastpath_ok:
        return True

    strong_lexical_support = max(tfidf_seed, tfidf_peer) >= STORYLINE_FORCE_ASSIGN_MIN_TFIDF
    strong_family_support = family_signal >= STORYLINE_FORCE_ASSIGN_MIN_FAMILY_SIGNAL
    useful_facet_support = useful_facet >= STORYLINE_FORCE_ASSIGN_MIN_USEFUL_FACET
    return bool(
        continuity_signal >= STORYLINE_FORCE_ASSIGN_MIN_CONTINUITY
        and (strong_lexical_support or strong_family_support or useful_facet_support)
    )


def _within_family_attach_ok(
    best: dict[str, Any] | None,
    *,
    family_matcher_meta: dict[str, Any],
    continuity_signal: float,
    pair_blended_score: float,
    duplicate_like_match: bool,
    assign_blocked_by_event_only_low_conf: bool,
    assign_blocked_by_digest_seed: bool,
    assign_blocked_by_ad_seed: bool,
) -> bool:
    if not STORYLINE_WITHIN_FAMILY_ATTACH_ENABLED or not best:
        return False
    if assign_blocked_by_digest_seed or assign_blocked_by_ad_seed:
        return False
    if assign_blocked_by_event_only_low_conf and not duplicate_like_match:
        return False

    selected_root = str(family_matcher_meta.get("family_matcher_selected_root_id") or "").strip()
    if not selected_root:
        return False
    best_root = _family_root_alias(
        str(best.get("family_root_storyline_id") or best.get("storyline_id") or "")
    )
    if best_root != selected_root:
        return False

    root_score = float(family_matcher_meta.get("family_matcher_selected_root_score", 0.0) or 0.0)
    if root_score < STORYLINE_WITHIN_FAMILY_ATTACH_MIN_ROOT_SCORE:
        return False

    comps = best.get("components", {}) or {}
    ent = float(comps.get("entity_overlap", 0.0) or 0.0)
    evt = float(comps.get("event_overlap", 0.0) or 0.0)
    dup = float(comps.get("duplicate_text_overlap", 0.0) or 0.0)
    tfidf_seed = float(comps.get("tfidf_post_seed_cosine", 0.0) or 0.0)
    tfidf_peer = float(comps.get("tfidf_post_peer_max_cosine", 0.0) or 0.0)
    exemplar_evt = float(comps.get("exemplar_event_frame_overlap", 0.0) or 0.0)
    exemplar_role = float(comps.get("exemplar_role_overlap", 0.0) or 0.0)
    useful_facet = float(comps.get("useful_facet_overlap_ratio", 0.0) or 0.0)
    facet_type = float(comps.get("facet_type_overlap_ratio", 0.0) or 0.0)
    broad_no_useful = float(comps.get("broad_entity_no_useful_facet_ratio", 0.0) or 0.0)

    if broad_no_useful >= 0.75 and useful_facet <= 1e-9 and tfidf_seed < STORYLINE_WITHIN_FAMILY_ATTACH_MIN_CONTINUITY:
        return False

    return bool(
        duplicate_like_match
        or dup >= STORYLINE_DUPLICATE_TEXT_JACCARD_MIN
        or tfidf_seed >= STORYLINE_WITHIN_FAMILY_ATTACH_MIN_CONTINUITY
        or tfidf_peer >= STORYLINE_WITHIN_FAMILY_ATTACH_MIN_CONTINUITY
        or useful_facet >= STORYLINE_WITHIN_FAMILY_ATTACH_MIN_CONTINUITY
        or (
            pair_blended_score >= STORYLINE_WITHIN_FAMILY_ATTACH_MIN_PAIR_SCORE
            and (
                continuity_signal >= STORYLINE_WITHIN_FAMILY_ATTACH_MIN_CONTINUITY
                or exemplar_evt >= STORYLINE_WITHIN_FAMILY_ATTACH_MIN_CONTINUITY
                or exemplar_role >= STORYLINE_WITHIN_FAMILY_ATTACH_MIN_CONTINUITY
                or facet_type >= STORYLINE_WITHIN_FAMILY_ATTACH_MIN_CONTINUITY
                or ent >= STORYLINE_WITHIN_FAMILY_ATTACH_MIN_ENTITY
                or evt >= STORYLINE_WITHIN_FAMILY_ATTACH_MIN_EVENT
            )
        )
    )


def _prefer_existing_storyline_over_unassigned(
    *,
    low_confidence_event: bool,
    allow_new_storyline_creation: bool,
    can_force_assign: bool,
    concise_official_update: bool = False,
) -> bool:
    if concise_official_update and allow_new_storyline_creation and low_confidence_event:
        return False
    return bool(can_force_assign and (low_confidence_event or not allow_new_storyline_creation))


def _should_try_llm_storyline_arbiter(
    best: dict[str, Any] | None,
    *,
    decision_score: float,
    assign_threshold: float,
    continuity_signal: float,
    allow_new_storyline_creation: bool,
) -> bool:
    if not STORYLINE_LLM_ARBITER_ENABLED or not allow_new_storyline_creation or not best:
        return False
    if not str(best.get("storyline_id") or "").strip():
        return False
    if decision_score >= assign_threshold or decision_score < STORYLINE_LLM_ARBITER_MIN_SCORE:
        return False
    if (assign_threshold - decision_score) > STORYLINE_LLM_ARBITER_MAX_MARGIN:
        return False
    if not bool(best.get("is_retrievable_storyline", _candidate_is_retrievable_storyline(best))):
        return False
    if _candidate_digest_like(best) or _candidate_ad_like(best):
        return False

    comps = best.get("components", {}) or {}
    embedding = float(comps.get("embedding_similarity", 0.0) or 0.0)
    event_overlap = float(comps.get("event_overlap", 0.0) or 0.0)
    family_signal = max(
        float(best.get("family_signal", 0.0) or 0.0),
        float(comps.get("family_signature_overlap", 0.0) or 0.0),
    )
    tfidf_seed = float(comps.get("tfidf_post_seed_cosine", 0.0) or 0.0)
    tfidf_peer = float(comps.get("tfidf_post_peer_max_cosine", 0.0) or 0.0)
    duplicate = float(comps.get("duplicate_text_overlap", 0.0) or 0.0)

    if embedding < STORYLINE_LLM_ARBITER_MIN_EMBEDDING:
        return False
    if max(
        event_overlap,
        family_signal,
        continuity_signal,
        tfidf_seed,
        tfidf_peer,
        duplicate,
    ) < min(
        STORYLINE_LLM_ARBITER_MIN_EVENT_OVERLAP,
        STORYLINE_LLM_ARBITER_MIN_FAMILY_SIGNAL,
        STORYLINE_LLM_ARBITER_MIN_CONTINUITY,
    ):
        return False

    return bool(
        event_overlap >= STORYLINE_LLM_ARBITER_MIN_EVENT_OVERLAP
        or family_signal >= STORYLINE_LLM_ARBITER_MIN_FAMILY_SIGNAL
        or continuity_signal >= STORYLINE_LLM_ARBITER_MIN_CONTINUITY
        or duplicate >= STORYLINE_LLM_ARBITER_MIN_CONTINUITY
        or tfidf_seed >= STORYLINE_LLM_ARBITER_MIN_CONTINUITY
        or tfidf_peer >= STORYLINE_LLM_ARBITER_MIN_CONTINUITY
    )


async def resolve_and_write_storyline(
    *,
    session,
    post_id: int,
    event_mention_id: str,
    event_time: datetime | None,
    event_embedding: list[float],
    entity_keys: set[str],
    relation_signatures: set[str],
    all_relation_signatures: set[str],
    event_signatures: set[str],
    participant_role_signatures: set[str],
    location_keys: set[str],
    title_hint: str,
    source_text: str = "",
    source_is_advertising: bool = False,
    event_confidence: float | None = None,
    allow_new_storyline_creation: bool = True,
) -> dict[str, Any]:
    if not STORYLINE_ENABLED:
        return {
            "enabled": False,
            "storyline_id": "",
            "decision": "disabled",
            "score": 0.0,
            "explanation": "{}",
        }

    current_family_signatures = _family_signatures(
        source_text=f"{title_hint}\n{source_text}".strip(),
        entity_keys=entity_keys,
        event_signatures=event_signatures,
    )
    current_topic_signatures = _topic_signatures(
        source_text=f"{title_hint}\n{source_text}".strip(),
        entity_keys=entity_keys,
        family_signatures=current_family_signatures,
    )
    content_meta = classify_storyline_content(source_text)
    concise_official_update = bool(content_meta.get("concise_official_update")) or (
        str(content_meta.get("reason") or "") == "concise_official_update"
    )

    # Reruns must not let the current post match itself through stale graph links.
    # The final writes below recreate the current assignment after scoring succeeds.
    await session.run(
        """
        MATCH (em:EventMention {id: $event_mention_id})
        OPTIONAL MATCH (em)-[old_part_rel:PART_OF]->(:Storyline)
        DELETE old_part_rel
        WITH em
        OPTIONAL MATCH (em)-[old_step_rel:PART_OF_STEP]->(:UpdateStep)
        DELETE old_step_rel
        """,
        event_mention_id=event_mention_id,
    )
    await session.run(
        """
        MATCH (p:Post {post_id: $post_id})
        OPTIONAL MATCH (p)-[old_source_rel:SOURCE_OF]->(:Storyline)
        DELETE old_source_rel
        WITH p
        OPTIONAL MATCH (p)-[old_step_rel:PART_OF_STEP]->(:UpdateStep)
        DELETE old_step_rel
        """,
        post_id=post_id,
    )

    candidates = await _fetch_storyline_candidates(
        session=session,
        entity_keys=entity_keys,
        relation_signatures=relation_signatures,
        event_signatures=event_signatures,
        family_signatures=current_family_signatures,
        topic_signatures=current_topic_signatures,
        participant_role_signatures=participant_role_signatures,
        location_keys=location_keys,
    )
    candidates = [cand for cand in candidates if _candidate_is_retrievable_storyline(cand)]

    best: dict[str, Any] | None = None
    resolved_storyline_title = title_hint or f"Storyline {post_id}"
    final_storyline_payload: dict[str, Any] | None = None
    scored_candidates: list[dict[str, Any]] = []
    for cand in candidates:
        scored = _score_candidate(
            event_embedding=event_embedding,
            event_entity_keys=entity_keys,
            event_relation_signatures=relation_signatures,
            event_signatures=event_signatures,
            family_signatures=current_family_signatures,
            topic_signatures=current_topic_signatures,
            participant_role_signatures=participant_role_signatures,
            location_keys=location_keys,
            event_ts=event_time,
            title_hint=title_hint,
            source_text=source_text,
            candidate=cand,
        )
        scored_candidates.append(scored)
        if best is None or scored["score"] > best["score"]:
            best = scored

    family_matcher_meta: dict[str, Any] = {}
    scored_candidates, family_matcher_meta = _apply_family_matcher_prefilter(
        scored_candidates,
        event_confidence=event_confidence,
        content_meta=content_meta,
    )
    if scored_candidates:
        best = max(scored_candidates, key=lambda x: float(x.get("score", 0.0)))
    else:
        best = None

    scored_candidates = _apply_family_consolidation_bonus(scored_candidates)
    if scored_candidates:
        best = max(scored_candidates, key=lambda x: float(x.get("score", 0.0)))
        best = _reroute_to_family_root(best=best, scored_candidates=scored_candidates)

    score_source = "hybrid"
    pair_status = scorer_status()
    if pair_status.get("ready"):
        pair_rows = []
        for cand in scored_candidates:
            comps = cand.get("components", {}) or {}
            base_score = float(cand.get("score", 0.0))
            pair_rows.append(
                {
                    "best_candidate_score": base_score,
                    "score": base_score,
                    "threshold": STORYLINE_ASSIGN_THRESHOLD,
                    "abs_margin_to_threshold": abs(base_score - STORYLINE_ASSIGN_THRESHOLD),
                    "event_mention_confidence": float(event_confidence or 0.0),
                    "embedding_similarity": float(comps.get("embedding_similarity", 0.0) or 0.0),
                    "entity_overlap": float(comps.get("entity_overlap", 0.0) or 0.0),
                    "relation_overlap": float(comps.get("relation_overlap", 0.0) or 0.0),
                    "event_overlap": float(comps.get("event_overlap", 0.0) or 0.0),
                    "temporal_proximity": float(comps.get("temporal_proximity", 0.5) or 0.5),
                    "candidate_posts_count": int(cand.get("candidate_posts_count") or 0),
                    "reason": "runtime_pair_rerank",
                }
            )
        probs = score_candidates(pair_rows)
        if probs and len(probs) == len(scored_candidates):
            for cand, pr in zip(scored_candidates, probs):
                cand["pair_score"] = round(float(pr), 4)
            best = max(scored_candidates, key=lambda x: float(x.get("pair_score", x.get("score", 0.0))))
            score_source = "pair_catboost"

    now_iso = datetime.now(timezone.utc).isoformat()
    event_time_iso = _safe_iso(event_time)
    best_candidate_storyline_id = str((best or {}).get("storyline_id") or "")
    best_candidate_hybrid_score = round(float((best or {}).get("score", 0.0)), 4) if best else 0.0
    best_candidate_pair_score = round(float((best or {}).get("pair_score", 0.0)), 4) if best else 0.0
    continuity_signal = 0.0
    if best:
        comps = best.get("components", {}) or {}
        continuity_signal = max(
            float(comps.get("duplicate_text_overlap", 0.0) or 0.0),
            float(comps.get("tfidf_post_seed_cosine", 0.0) or 0.0),
            float(comps.get("tfidf_post_peer_max_cosine", 0.0) or 0.0),
            float(comps.get("exemplar_role_overlap", 0.0) or 0.0),
            float(comps.get("exemplar_event_frame_overlap", 0.0) or 0.0),
            float(comps.get("useful_facet_overlap_ratio", 0.0) or 0.0),
        )
    pair_hybrid_blend = min(0.7, STORYLINE_PAIR_HYBRID_BLEND + (0.25 * continuity_signal))
    pair_blended_score = round(
        ((1.0 - pair_hybrid_blend) * best_candidate_pair_score)
        + (pair_hybrid_blend * best_candidate_hybrid_score),
        4,
    ) if best else 0.0
    decision_score = (
        pair_blended_score
        if score_source == "pair_catboost"
        else best_candidate_hybrid_score
    )
    assign_threshold = STORYLINE_PAIR_ASSIGN_THRESHOLD if score_source == "pair_catboost" else STORYLINE_ASSIGN_THRESHOLD
    force_assign_min_score = (
        STORYLINE_PAIR_FORCE_ASSIGN_MIN_SCORE
        if score_source == "pair_catboost"
        else STORYLINE_FORCE_ASSIGN_MIN_SCORE
    )
    low_confidence_event = (
        event_confidence is not None
        and float(event_confidence) < STORYLINE_NEW_MIN_EVENT_CONFIDENCE
    )
    structural_signal_ok = _has_force_assign_structural_signal(best)
    no_structural_low_conf_min_score = (
        STORYLINE_PAIR_ASSIGN_MIN_SCORE_NO_STRUCTURAL_LOW_CONF
        if score_source == "pair_catboost"
        else STORYLINE_HYBRID_ASSIGN_MIN_SCORE_NO_STRUCTURAL_LOW_CONF
    )
    assign_blocked_by_low_conf_no_structural = bool(
        best
        and low_confidence_event
        and not structural_signal_ok
        and decision_score < no_structural_low_conf_min_score
    )
    assign_blocked_by_event_only_low_conf = _is_event_only_low_conf_match(best, event_confidence)
    duplicate_like_match = _is_duplicate_like_match(best)
    duplicate_fastpath_ok = bool(
        best
        and bool(best.get("is_retrievable_storyline"))
        and not _candidate_digest_like(best)
        and not _candidate_ad_like(best)
        and float((best.get("components", {}) or {}).get("duplicate_text_overlap", 0.0) or 0.0)
        >= STORYLINE_DUPLICATE_FASTPATH_TEXT_MIN
    )
    assign_blocked_by_concise_official_low_conf = bool(
        best
        and concise_official_update
        and low_confidence_event
        and allow_new_storyline_creation
        and not duplicate_like_match
        and not duplicate_fastpath_ok
    )
    assign_blocked_by_digest_seed = bool(
        best
        and _candidate_digest_like(best)
        and _is_event_only_low_conf_match(best, event_confidence)
    )
    assign_blocked_by_ad_seed = bool(
        best
        and _candidate_ad_like(best)
        and not duplicate_like_match
        and float((best.get("components", {}) or {}).get("relation_overlap", 0.0) or 0.0) <= 1e-9
    )

    if (
        best
        and (duplicate_fastpath_ok or decision_score >= assign_threshold)
        and not assign_blocked_by_low_conf_no_structural
        and not (assign_blocked_by_event_only_low_conf and not duplicate_like_match)
        and not assign_blocked_by_concise_official_low_conf
        and not assign_blocked_by_digest_seed
        and not assign_blocked_by_ad_seed
    ):
        storyline_id = best["storyline_id"]
        prev = next((c for c in candidates if str(c.get("id")) == storyline_id), {}) or {}
        prev_count = int(prev.get("posts_count") or 0)
        merged_embedding = _merge_embedding(
            prev=prev.get("centroid_embedding") or [],
            incoming=event_embedding,
            n_prev=prev_count,
        )
        refresh_payload = _refresh_storyline_payload(
            prev=prev,
            incoming_title=title_hint,
            incoming_source_text=source_text,
            incoming_entity_keys=entity_keys,
            incoming_relation_signatures=relation_signatures,
            incoming_all_relation_signatures=all_relation_signatures,
            incoming_event_signatures=event_signatures,
            incoming_topic_signatures=current_topic_signatures,
        )
        resolved_storyline_title = str(refresh_payload.get("title") or resolved_storyline_title)
        final_storyline_payload = dict(refresh_payload)

        await session.run(
            """
            MATCH (s:Storyline {id: $storyline_id})
            SET s.centroid_embedding = $centroid_embedding,
                s.title = $title,
                s.seed_preview = $seed_preview,
                s.entity_keys = $entity_keys,
                s.relation_signatures = $relation_signatures,
                s.all_relation_signatures = $all_relation_signatures,
                s.event_signatures = $event_signatures,
                s.family_signatures = $family_signatures,
                s.topic_signatures = $topic_signatures,
                s.family_root_id = coalesce(s.family_root_id, $family_root_id),
                s.last_event_time = $last_event_time,
                s.posts_count = coalesce(s.posts_count, 0) + 1,
                s.updated_at = $updated_at
            """,
            storyline_id=storyline_id,
            centroid_embedding=merged_embedding,
            title=refresh_payload["title"],
            seed_preview=refresh_payload["seed_preview"],
            entity_keys=refresh_payload["entity_keys"],
            relation_signatures=refresh_payload["relation_signatures"],
            all_relation_signatures=refresh_payload["all_relation_signatures"],
            event_signatures=refresh_payload["event_signatures"],
            family_signatures=refresh_payload["family_signatures"],
            topic_signatures=refresh_payload["topic_signatures"],
            family_root_id=str((best or {}).get("family_root_storyline_id") or storyline_id),
            last_event_time=event_time_iso,
            updated_at=now_iso,
        )
        decision = "assigned_to_existing_storyline"
        explanation = {
            **best,
            "score": round(decision_score, 4),
            "score_source": score_source,
            "pair_score": best_candidate_pair_score if score_source == "pair_catboost" else 0.0,
            "pair_blended_score": pair_blended_score if score_source == "pair_catboost" else 0.0,
            "pair_hybrid_blend": pair_hybrid_blend if score_source == "pair_catboost" else 0.0,
            "hybrid_score": best_candidate_hybrid_score,
            "duplicate_like_match": duplicate_like_match,
            "duplicate_fastpath_ok": duplicate_fastpath_ok,
            "continuity_signal": round(continuity_signal, 4),
            "recent_mentions_count": len((best or {}).get("recent_mentions", []) or []),
            "family_cluster_size": int((best or {}).get("family_cluster_size") or 0),
            "family_signal": round(float((best or {}).get("family_signal", 0.0) or 0.0), 4),
            "family_boost": round(float((best or {}).get("family_boost", 0.0) or 0.0), 4),
            "family_preferred_root": bool((best or {}).get("family_preferred_root")),
            "family_oldest_seed_id": (best or {}).get("family_oldest_seed_id") or "",
            "family_root_storyline_id": (best or {}).get("family_root_storyline_id") or storyline_id,
            "family_rerouted_to_root": bool((best or {}).get("family_rerouted_to_root")),
            "storyline_title_refreshed": bool(refresh_payload.get("title_refreshed")),
            "storyline_prev_title_score": float(refresh_payload.get("prev_title_score", 0.0) or 0.0),
            "storyline_incoming_title_score": float(refresh_payload.get("incoming_title_score", 0.0) or 0.0),
            "storyline_title": resolved_storyline_title,
        }
    else:
        # Guardrail: low-confidence event should not aggressively spawn new storyline.
        # If we have at least one plausible candidate, prefer forced assignment.
        can_force_assign = _force_assign_ok(
            best,
            decision_score=decision_score,
            force_assign_min_score=force_assign_min_score,
            structural_signal_ok=structural_signal_ok,
            duplicate_like_match=duplicate_like_match,
            duplicate_fastpath_ok=duplicate_fastpath_ok,
            continuity_signal=continuity_signal,
            assign_blocked_by_event_only_low_conf=assign_blocked_by_event_only_low_conf,
            assign_blocked_by_digest_seed=assign_blocked_by_digest_seed,
            assign_blocked_by_ad_seed=assign_blocked_by_ad_seed,
        )
        can_attach_within_family = _within_family_attach_ok(
            best,
            family_matcher_meta=family_matcher_meta,
            continuity_signal=continuity_signal,
            pair_blended_score=pair_blended_score,
            duplicate_like_match=duplicate_like_match,
            assign_blocked_by_event_only_low_conf=assign_blocked_by_event_only_low_conf,
            assign_blocked_by_digest_seed=assign_blocked_by_digest_seed,
            assign_blocked_by_ad_seed=assign_blocked_by_ad_seed,
        )
        if can_force_assign and low_confidence_event and allow_new_storyline_creation and best:
            # Low-confidence general-news frames can look deceptively close by country/common frame.
            # Do not force-attach them to old storylines unless there is specific lexical,
            # relation, duplicate, or useful-facet evidence of continuity.
            comps = best.get("components", {}) or {}
            weak_low_conf_force_assign = bool(
                decision_score < assign_threshold
                and not duplicate_like_match
                and not duplicate_fastpath_ok
                and float(comps.get("relation_overlap", 0.0) or 0.0) <= 1e-9
                and max(
                    float(comps.get("tfidf_post_seed_cosine", 0.0) or 0.0),
                    float(comps.get("tfidf_post_peer_max_cosine", 0.0) or 0.0),
                ) < STORYLINE_FORCE_ASSIGN_MIN_TFIDF
                and float(comps.get("useful_facet_overlap_ratio", 0.0) or 0.0) <= 1e-9
                and float(comps.get("shared_registry_entity_count", 0.0) or 0.0) <= 1e-9
            )
            if weak_low_conf_force_assign:
                can_force_assign = False
        llm_arbiter_result: dict[str, Any] | None = None
        llm_arbiter_gate_open = _should_try_llm_storyline_arbiter(
            best,
            decision_score=decision_score,
            assign_threshold=assign_threshold,
            continuity_signal=continuity_signal,
            allow_new_storyline_creation=allow_new_storyline_creation,
        )
        if llm_arbiter_gate_open:
            try:
                llm_arbiter_result = await arbitrate_storyline_candidate(
                    post_text=source_text,
                    title_hint=title_hint,
                    best_candidate={
                        **dict(best or {}),
                        "score_source": score_source,
                        "family_signal": round(float((best or {}).get("family_signal", 0.0) or 0.0), 4),
                    },
                    decision_score=decision_score,
                    assign_threshold=assign_threshold,
                )
            except Exception as exc:
                log.warning(
                    "storyline_llm_arbiter_failed",
                    post_id=post_id,
                    best_candidate_storyline_id=str((best or {}).get("storyline_id") or ""),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                llm_arbiter_result = None
        if _prefer_existing_storyline_over_unassigned(
            low_confidence_event=low_confidence_event,
            allow_new_storyline_creation=allow_new_storyline_creation,
            can_force_assign=can_force_assign,
            concise_official_update=concise_official_update,
        ):
            storyline_id = str(best["storyline_id"])
            prev = next((c for c in candidates if str(c.get("id")) == storyline_id), {}) or {}
            prev_count = int(prev.get("posts_count") or 0)
            merged_embedding = _merge_embedding(
                prev=prev.get("centroid_embedding") or [],
                incoming=event_embedding,
                n_prev=prev_count,
            )
            refresh_payload = _refresh_storyline_payload(
                prev=prev,
                incoming_title=title_hint,
                incoming_source_text=source_text,
                incoming_entity_keys=entity_keys,
                incoming_relation_signatures=relation_signatures,
                incoming_all_relation_signatures=all_relation_signatures,
                incoming_event_signatures=event_signatures,
                incoming_topic_signatures=current_topic_signatures,
            )
            resolved_storyline_title = str(refresh_payload.get("title") or resolved_storyline_title)
            final_storyline_payload = dict(refresh_payload)

            await session.run(
                """
                MATCH (s:Storyline {id: $storyline_id})
                SET s.centroid_embedding = $centroid_embedding,
                    s.title = $title,
                    s.seed_preview = $seed_preview,
                    s.entity_keys = $entity_keys,
                    s.relation_signatures = $relation_signatures,
                    s.all_relation_signatures = $all_relation_signatures,
                    s.event_signatures = $event_signatures,
                    s.family_signatures = $family_signatures,
                    s.topic_signatures = $topic_signatures,
                    s.family_root_id = coalesce(s.family_root_id, $family_root_id),
                    s.last_event_time = $last_event_time,
                    s.posts_count = coalesce(s.posts_count, 0) + 1,
                    s.updated_at = $updated_at
                """,
                storyline_id=storyline_id,
                centroid_embedding=merged_embedding,
                title=refresh_payload["title"],
                seed_preview=refresh_payload["seed_preview"],
                entity_keys=refresh_payload["entity_keys"],
                relation_signatures=refresh_payload["relation_signatures"],
                all_relation_signatures=refresh_payload["all_relation_signatures"],
                event_signatures=refresh_payload["event_signatures"],
                family_signatures=refresh_payload["family_signatures"],
                topic_signatures=refresh_payload["topic_signatures"],
                family_root_id=str((best or {}).get("family_root_storyline_id") or storyline_id),
                last_event_time=event_time_iso,
                updated_at=now_iso,
            )
            decision = (
                "forced_assign_low_event_confidence"
                if low_confidence_event
                else "forced_assign_story_update_only"
            )
            explanation = {
                "score": round(float(best.get("score", 0.0)), 4),
                "hybrid_score": best_candidate_hybrid_score,
                "pair_score": best_candidate_pair_score if score_source == "pair_catboost" else 0.0,
                "score_source": score_source,
                "components": best.get("components", {}),
                "contributions": best.get("contributions", {}),
                "reason": (
                    "low_event_confidence_guardrail"
                    if low_confidence_event
                    else "seed_creation_disabled_force_assign"
                ),
                "event_confidence": round(float(event_confidence), 4),
                "pair_blended_score": pair_blended_score if score_source == "pair_catboost" else 0.0,
                "pair_hybrid_blend": pair_hybrid_blend if score_source == "pair_catboost" else 0.0,
                "new_storyline_min_event_confidence": STORYLINE_NEW_MIN_EVENT_CONFIDENCE,
                "allow_new_storyline_creation": bool(allow_new_storyline_creation),
                "concise_official_update": concise_official_update,
                "force_assign_min_score": force_assign_min_score,
                "force_assign_min_entity_overlap": STORYLINE_FORCE_ASSIGN_MIN_ENTITY_OVERLAP,
                "force_assign_min_relation_overlap": STORYLINE_FORCE_ASSIGN_MIN_RELATION_OVERLAP,
                "force_assign_min_structural_overlap": STORYLINE_FORCE_ASSIGN_MIN_STRUCTURAL_OVERLAP,
                "force_assign_min_continuity": STORYLINE_FORCE_ASSIGN_MIN_CONTINUITY,
                "force_assign_min_family_signal": STORYLINE_FORCE_ASSIGN_MIN_FAMILY_SIGNAL,
                "force_assign_min_tfidf": STORYLINE_FORCE_ASSIGN_MIN_TFIDF,
                "force_assign_min_useful_facet": STORYLINE_FORCE_ASSIGN_MIN_USEFUL_FACET,
                "assign_min_score_no_structural_low_conf": no_structural_low_conf_min_score,
                "duplicate_like_match": duplicate_like_match,
                "duplicate_fastpath_ok": duplicate_fastpath_ok,
                "continuity_signal": round(continuity_signal, 4),
                "recent_mentions_count": len((best or {}).get("recent_mentions", []) or []),
                "family_cluster_size": int((best or {}).get("family_cluster_size") or 0),
                "family_signal": round(float((best or {}).get("family_signal", 0.0) or 0.0), 4),
                "family_boost": round(float((best or {}).get("family_boost", 0.0) or 0.0), 4),
                "family_preferred_root": bool((best or {}).get("family_preferred_root")),
                "family_oldest_seed_id": (best or {}).get("family_oldest_seed_id") or "",
                "family_root_storyline_id": (best or {}).get("family_root_storyline_id") or storyline_id,
                "family_rerouted_to_root": bool((best or {}).get("family_rerouted_to_root")),
                "seed_quality_score": round(float((best or {}).get("seed_quality_score", 0.0) or 0.0), 4),
                "is_retrievable_storyline": bool((best or {}).get("is_retrievable_storyline")),
                "storyline_title_refreshed": bool(refresh_payload.get("title_refreshed")),
                "storyline_prev_title_score": float(refresh_payload.get("prev_title_score", 0.0) or 0.0),
                "storyline_incoming_title_score": float(refresh_payload.get("incoming_title_score", 0.0) or 0.0),
                "storyline_title": resolved_storyline_title,
            }
        elif can_attach_within_family and not assign_blocked_by_concise_official_low_conf:
            storyline_id = str(best["storyline_id"])
            prev = next((c for c in candidates if str(c.get("id")) == storyline_id), {}) or {}
            prev_count = int(prev.get("posts_count") or 0)
            merged_embedding = _merge_embedding(
                prev=prev.get("centroid_embedding") or [],
                incoming=event_embedding,
                n_prev=prev_count,
            )
            refresh_payload = _refresh_storyline_payload(
                prev=prev,
                incoming_title=title_hint,
                incoming_source_text=source_text,
                incoming_entity_keys=entity_keys,
                incoming_relation_signatures=relation_signatures,
                incoming_all_relation_signatures=all_relation_signatures,
                incoming_event_signatures=event_signatures,
                incoming_topic_signatures=current_topic_signatures,
            )
            resolved_storyline_title = str(refresh_payload.get("title") or resolved_storyline_title)
            final_storyline_payload = dict(refresh_payload)

            await session.run(
                """
                MATCH (s:Storyline {id: $storyline_id})
                SET s.centroid_embedding = $centroid_embedding,
                    s.title = $title,
                    s.seed_preview = $seed_preview,
                    s.entity_keys = $entity_keys,
                    s.relation_signatures = $relation_signatures,
                    s.all_relation_signatures = $all_relation_signatures,
                    s.event_signatures = $event_signatures,
                    s.family_signatures = $family_signatures,
                    s.topic_signatures = $topic_signatures,
                    s.family_root_id = coalesce(s.family_root_id, $family_root_id),
                    s.last_event_time = $last_event_time,
                    s.posts_count = coalesce(s.posts_count, 0) + 1,
                    s.updated_at = $updated_at
                """,
                storyline_id=storyline_id,
                centroid_embedding=merged_embedding,
                title=refresh_payload["title"],
                seed_preview=refresh_payload["seed_preview"],
                entity_keys=refresh_payload["entity_keys"],
                relation_signatures=refresh_payload["relation_signatures"],
                all_relation_signatures=refresh_payload["all_relation_signatures"],
                event_signatures=refresh_payload["event_signatures"],
                family_signatures=refresh_payload["family_signatures"],
                topic_signatures=refresh_payload["topic_signatures"],
                family_root_id=str((best or {}).get("family_root_storyline_id") or storyline_id),
                last_event_time=event_time_iso,
                updated_at=now_iso,
            )
            decision = "assigned_within_family_bridge"
            explanation = {
                **best,
                "score": round(float(best.get("score", 0.0)), 4),
                "hybrid_score": best_candidate_hybrid_score,
                "pair_score": best_candidate_pair_score if score_source == "pair_catboost" else 0.0,
                "pair_blended_score": pair_blended_score if score_source == "pair_catboost" else 0.0,
                "pair_hybrid_blend": pair_hybrid_blend if score_source == "pair_catboost" else 0.0,
                "score_source": score_source,
                "reason": "within_family_bridge_attach",
                "event_confidence": round(float(event_confidence or 0.0), 4),
                "duplicate_like_match": duplicate_like_match,
                "duplicate_fastpath_ok": duplicate_fastpath_ok,
                "continuity_signal": round(continuity_signal, 4),
                "within_family_attach": True,
                "within_family_attach_min_root_score": STORYLINE_WITHIN_FAMILY_ATTACH_MIN_ROOT_SCORE,
                "within_family_attach_min_pair_score": STORYLINE_WITHIN_FAMILY_ATTACH_MIN_PAIR_SCORE,
                "within_family_attach_min_continuity": STORYLINE_WITHIN_FAMILY_ATTACH_MIN_CONTINUITY,
                "within_family_attach_min_entity": STORYLINE_WITHIN_FAMILY_ATTACH_MIN_ENTITY,
                "within_family_attach_min_event": STORYLINE_WITHIN_FAMILY_ATTACH_MIN_EVENT,
                "recent_mentions_count": len((best or {}).get("recent_mentions", []) or []),
                "family_cluster_size": int((best or {}).get("family_cluster_size") or 0),
                "family_signal": round(float((best or {}).get("family_signal", 0.0) or 0.0), 4),
                "family_boost": round(float((best or {}).get("family_boost", 0.0) or 0.0), 4),
                "family_preferred_root": bool((best or {}).get("family_preferred_root")),
                "family_oldest_seed_id": (best or {}).get("family_oldest_seed_id") or "",
                "family_root_storyline_id": (best or {}).get("family_root_storyline_id") or storyline_id,
                "family_rerouted_to_root": bool((best or {}).get("family_rerouted_to_root")),
                "storyline_title_refreshed": bool(refresh_payload.get("title_refreshed")),
                "storyline_prev_title_score": float(refresh_payload.get("prev_title_score", 0.0) or 0.0),
                "storyline_incoming_title_score": float(refresh_payload.get("incoming_title_score", 0.0) or 0.0),
                "storyline_title": resolved_storyline_title,
            }
        elif (
            not assign_blocked_by_concise_official_low_conf
            and llm_arbiter_gate_open
            and isinstance(llm_arbiter_result, dict)
            and bool(llm_arbiter_result.get("same_story"))
            and float(llm_arbiter_result.get("confidence", 0.0) or 0.0) >= STORYLINE_LLM_ARBITER_MIN_CONFIDENCE
        ):
            storyline_id = str(best["storyline_id"])
            prev = next((c for c in candidates if str(c.get("id")) == storyline_id), {}) or {}
            prev_count = int(prev.get("posts_count") or 0)
            merged_embedding = _merge_embedding(
                prev=prev.get("centroid_embedding") or [],
                incoming=event_embedding,
                n_prev=prev_count,
            )
            refresh_payload = _refresh_storyline_payload(
                prev=prev,
                incoming_title=title_hint,
                incoming_source_text=source_text,
                incoming_entity_keys=entity_keys,
                incoming_relation_signatures=relation_signatures,
                incoming_all_relation_signatures=all_relation_signatures,
                incoming_event_signatures=event_signatures,
                incoming_topic_signatures=current_topic_signatures,
            )
            resolved_storyline_title = str(refresh_payload.get("title") or resolved_storyline_title)
            final_storyline_payload = dict(refresh_payload)

            await session.run(
                """
                MATCH (s:Storyline {id: $storyline_id})
                SET s.centroid_embedding = $centroid_embedding,
                    s.title = $title,
                    s.seed_preview = $seed_preview,
                    s.entity_keys = $entity_keys,
                    s.relation_signatures = $relation_signatures,
                    s.all_relation_signatures = $all_relation_signatures,
                    s.event_signatures = $event_signatures,
                    s.family_signatures = $family_signatures,
                    s.topic_signatures = $topic_signatures,
                    s.family_root_id = coalesce(s.family_root_id, $family_root_id),
                    s.last_event_time = $last_event_time,
                    s.posts_count = coalesce(s.posts_count, 0) + 1,
                    s.updated_at = $updated_at
                """,
                storyline_id=storyline_id,
                centroid_embedding=merged_embedding,
                title=refresh_payload["title"],
                seed_preview=refresh_payload["seed_preview"],
                entity_keys=refresh_payload["entity_keys"],
                relation_signatures=refresh_payload["relation_signatures"],
                all_relation_signatures=refresh_payload["all_relation_signatures"],
                event_signatures=refresh_payload["event_signatures"],
                family_signatures=refresh_payload["family_signatures"],
                topic_signatures=refresh_payload["topic_signatures"],
                family_root_id=str((best or {}).get("family_root_storyline_id") or storyline_id),
                last_event_time=event_time_iso,
                updated_at=now_iso,
            )
            decision = "assigned_by_llm_arbiter"
            explanation = {
                **best,
                "score": round(float(best.get("score", 0.0)), 4),
                "hybrid_score": best_candidate_hybrid_score,
                "pair_score": best_candidate_pair_score if score_source == "pair_catboost" else 0.0,
                "pair_blended_score": pair_blended_score if score_source == "pair_catboost" else 0.0,
                "pair_hybrid_blend": pair_hybrid_blend if score_source == "pair_catboost" else 0.0,
                "score_source": score_source,
                "reason": "llm_same_story_override_new_storyline",
                "event_confidence": round(float(event_confidence or 0.0), 4),
                "duplicate_like_match": duplicate_like_match,
                "duplicate_fastpath_ok": duplicate_fastpath_ok,
                "continuity_signal": round(continuity_signal, 4),
                "recent_mentions_count": len((best or {}).get("recent_mentions", []) or []),
                "family_cluster_size": int((best or {}).get("family_cluster_size") or 0),
                "family_signal": round(float((best or {}).get("family_signal", 0.0) or 0.0), 4),
                "family_boost": round(float((best or {}).get("family_boost", 0.0) or 0.0), 4),
                "family_preferred_root": bool((best or {}).get("family_preferred_root")),
                "family_oldest_seed_id": (best or {}).get("family_oldest_seed_id") or "",
                "family_root_storyline_id": (best or {}).get("family_root_storyline_id") or storyline_id,
                "family_rerouted_to_root": bool((best or {}).get("family_rerouted_to_root")),
                "llm_arbiter_used": True,
                "llm_arbiter_confidence": round(float(llm_arbiter_result.get("confidence", 0.0) or 0.0), 4),
                "llm_arbiter_reason": str(llm_arbiter_result.get("reason") or "").strip()[:300],
                "llm_arbiter_gate_min_score": STORYLINE_LLM_ARBITER_MIN_SCORE,
                "llm_arbiter_gate_max_margin": STORYLINE_LLM_ARBITER_MAX_MARGIN,
                "llm_arbiter_gate_min_embedding": STORYLINE_LLM_ARBITER_MIN_EMBEDDING,
                "llm_arbiter_gate_min_event_overlap": STORYLINE_LLM_ARBITER_MIN_EVENT_OVERLAP,
                "llm_arbiter_gate_min_family_signal": STORYLINE_LLM_ARBITER_MIN_FAMILY_SIGNAL,
                "llm_arbiter_gate_min_continuity": STORYLINE_LLM_ARBITER_MIN_CONTINUITY,
                "storyline_title_refreshed": bool(refresh_payload.get("title_refreshed")),
                "storyline_prev_title_score": float(refresh_payload.get("prev_title_score", 0.0) or 0.0),
                "storyline_incoming_title_score": float(refresh_payload.get("incoming_title_score", 0.0) or 0.0),
                "storyline_title": resolved_storyline_title,
            }
        elif allow_new_storyline_creation:
            storyline_id = f"storyline_{post_id}"
            seed_digest_score = _digest_score(source_text)
            seed_is_digest = _is_digest_like_text(source_text)
            seed_ad_score = _ad_score(source_text)
            seed_is_advertising = bool(source_is_advertising) or _is_ad_like_text(source_text)
            seed_quality = float(classify_storyline_content(source_text).get("seed_quality_score", 0.0) or 0.0)
            await session.run(
                """
                MERGE (s:Storyline {id: $storyline_id})
                ON CREATE SET
                    s.title = $title,
                    s.created_at = $created_at
                SET s.centroid_embedding = $centroid_embedding,
                    s.entity_keys = $entity_keys,
                    s.relation_signatures = $relation_signatures,
                    s.all_relation_signatures = $all_relation_signatures,
                    s.event_signatures = $event_signatures,
                    s.family_signatures = $family_signatures,
                    s.topic_signatures = $topic_signatures,
                    s.family_root_id = $family_root_id,
                    s.seed_preview = $seed_preview,
                    s.seed_is_digest = $seed_is_digest,
                    s.seed_digest_score = $seed_digest_score,
                    s.seed_is_advertising = $seed_is_advertising,
                    s.seed_ad_score = $seed_ad_score,
                    s.seed_quality_score = $seed_quality_score,
                    s.is_retrievable_seed = $is_retrievable_seed,
                    s.last_event_time = $last_event_time,
                    s.posts_count = coalesce(s.posts_count, 0) + 1,
                    s.updated_at = $updated_at
                """,
                storyline_id=storyline_id,
                title=resolved_storyline_title,
                created_at=now_iso,
                centroid_embedding=event_embedding,
                entity_keys=sorted(entity_keys),
                relation_signatures=sorted(relation_signatures),
                all_relation_signatures=sorted(all_relation_signatures),
                event_signatures=sorted(event_signatures),
                family_signatures=sorted(current_family_signatures),
                topic_signatures=sorted(current_topic_signatures),
                family_root_id=storyline_id,
                seed_preview=(source_text or title_hint or "")[:500],
                seed_is_digest=seed_is_digest,
                seed_digest_score=seed_digest_score,
                seed_is_advertising=seed_is_advertising,
                seed_ad_score=seed_ad_score,
                seed_quality_score=seed_quality,
                is_retrievable_seed=seed_quality >= STORYLINE_MIN_RETRIEVABLE_SEED_QUALITY,
                last_event_time=event_time_iso,
                updated_at=now_iso,
            )
            resolved_storyline_title = title_hint or f"Storyline {post_id}"
            final_storyline_payload = {
                "title": resolved_storyline_title,
                "seed_preview": (source_text or title_hint or "")[:500],
                "entity_keys": sorted(entity_keys),
                "relation_signatures": sorted(relation_signatures),
                "all_relation_signatures": sorted(all_relation_signatures),
                "event_signatures": sorted(event_signatures),
                "family_signatures": sorted(current_family_signatures),
                "topic_signatures": sorted(current_topic_signatures),
            }
            decision = "new_storyline_created"
            explanation = {
                "score": round(best["score"], 4) if best else 0.0,
                "hybrid_score": best_candidate_hybrid_score,
                "pair_score": best_candidate_pair_score if score_source == "pair_catboost" else 0.0,
                "score_source": score_source,
                "components": (best or {}).get("components", {}),
                "contributions": (best or {}).get("contributions", {}),
                "reason": "below_threshold_or_no_candidates",
                "event_confidence": round(float(event_confidence or 0.0), 4),
                "pair_blended_score": pair_blended_score if score_source == "pair_catboost" else 0.0,
                "pair_hybrid_blend": pair_hybrid_blend if score_source == "pair_catboost" else 0.0,
                "seed_is_digest": seed_is_digest,
                "seed_digest_score": round(seed_digest_score, 4),
                "seed_is_advertising": seed_is_advertising,
                "seed_ad_score": round(seed_ad_score, 4),
                "concise_official_update": concise_official_update,
                "assign_blocked_by_concise_official_low_conf": assign_blocked_by_concise_official_low_conf,
                "guardrail_structural_signal_ok": structural_signal_ok,
                "assign_blocked_by_low_conf_no_structural": assign_blocked_by_low_conf_no_structural,
                "assign_blocked_by_event_only_low_conf": assign_blocked_by_event_only_low_conf,
                "assign_blocked_by_digest_seed": assign_blocked_by_digest_seed,
                "assign_blocked_by_ad_seed": assign_blocked_by_ad_seed,
                "assign_min_score_no_structural_low_conf": no_structural_low_conf_min_score,
                "duplicate_like_match": duplicate_like_match,
                "duplicate_fastpath_ok": duplicate_fastpath_ok,
                "continuity_signal": round(continuity_signal, 4),
                "recent_mentions_count": len((best or {}).get("recent_mentions", []) or []),
                "family_cluster_size": int((best or {}).get("family_cluster_size") or 0),
                "family_signal": round(float((best or {}).get("family_signal", 0.0) or 0.0), 4),
                "family_boost": round(float((best or {}).get("family_boost", 0.0) or 0.0), 4),
                "family_preferred_root": bool((best or {}).get("family_preferred_root")),
                "family_oldest_seed_id": (best or {}).get("family_oldest_seed_id") or "",
                "family_root_storyline_id": storyline_id,
                "family_rerouted_to_root": False,
                "llm_arbiter_used": llm_arbiter_gate_open,
                "llm_arbiter_same_story": bool((llm_arbiter_result or {}).get("same_story")) if llm_arbiter_result else False,
                "llm_arbiter_confidence": round(float((llm_arbiter_result or {}).get("confidence", 0.0) or 0.0), 4)
                if llm_arbiter_result
                else 0.0,
                "llm_arbiter_reason": str((llm_arbiter_result or {}).get("reason") or "").strip()[:300]
                if llm_arbiter_result
                else "",
            }
        else:
            storyline_id = ""
            decision = "story_update_only_unassigned"
            explanation = {
                "score": round(best["score"], 4) if best else 0.0,
                "hybrid_score": best_candidate_hybrid_score,
                "pair_score": best_candidate_pair_score if score_source == "pair_catboost" else 0.0,
                "score_source": score_source,
                "components": (best or {}).get("components", {}),
                "contributions": (best or {}).get("contributions", {}),
                "reason": "seed_creation_disabled_for_story_update_only",
                "event_confidence": round(float(event_confidence or 0.0), 4),
                "pair_blended_score": pair_blended_score if score_source == "pair_catboost" else 0.0,
                "pair_hybrid_blend": pair_hybrid_blend if score_source == "pair_catboost" else 0.0,
                "allow_new_storyline_creation": False,
                "duplicate_like_match": duplicate_like_match,
                "duplicate_fastpath_ok": duplicate_fastpath_ok,
                "continuity_signal": round(continuity_signal, 4),
                "recent_mentions_count": len((best or {}).get("recent_mentions", []) or []),
                "family_cluster_size": int((best or {}).get("family_cluster_size") or 0),
                "family_signal": round(float((best or {}).get("family_signal", 0.0) or 0.0), 4),
                "family_boost": round(float((best or {}).get("family_boost", 0.0) or 0.0), 4),
                "family_preferred_root": bool((best or {}).get("family_preferred_root")),
                "family_oldest_seed_id": (best or {}).get("family_oldest_seed_id") or "",
                "family_root_storyline_id": "",
                "family_rerouted_to_root": False,
            }

    if family_matcher_meta:
        explanation.update(family_matcher_meta)

    family_root_storyline_id = str(explanation.get("family_root_storyline_id") or storyline_id)
    story_family_id = _family_id_for_root(family_root_storyline_id)

    if not storyline_id:
        flat_components = explanation.get("components", {}) or {}
        return {
            "enabled": True,
            "storyline_id": "",
            "story_family_id": "",
            "story_family_root_storyline_id": "",
            "story_episode_id": "",
            "storyline_seed_post_id": "",
            "decision": decision,
            "score": round(float(explanation.get("score", 0.0)), 4),
            "threshold": assign_threshold,
            "explanation": json.dumps(explanation, ensure_ascii=False),
            "components": explanation.get("components", {}),
            "contributions": explanation.get("contributions", {}),
            "reason": explanation.get("reason", ""),
            "best_candidate_storyline_id": best_candidate_storyline_id,
            "best_candidate_score": round(decision_score, 4),
            "best_candidate_hybrid_score": best_candidate_hybrid_score,
            "best_candidate_pair_score": best_candidate_pair_score,
            "score_source": score_source,
            "pair_blended_score": pair_blended_score if score_source == "pair_catboost" else 0.0,
            "pair_hybrid_blend": pair_hybrid_blend if score_source == "pair_catboost" else 0.0,
            "continuity_signal": round(continuity_signal, 4),
            "recent_mentions_count": len((best or {}).get("recent_mentions", []) or []),
            "family_cluster_size": int((best or {}).get("family_cluster_size") or 0),
            "family_signal": round(float((best or {}).get("family_signal", 0.0) or 0.0), 4),
            "family_boost": round(float((best or {}).get("family_boost", 0.0) or 0.0), 4),
            "family_preferred_root": bool((best or {}).get("family_preferred_root")),
            "family_oldest_seed_id": (best or {}).get("family_oldest_seed_id") or "",
            "family_root_storyline_id": "",
            "family_rerouted_to_root": False,
            "family_matcher_ready": bool(explanation.get("family_matcher_ready")),
            "family_matcher_mode": explanation.get("family_matcher_mode", ""),
            "family_matcher_threshold": float(explanation.get("family_matcher_threshold", 0.0) or 0.0),
            "family_matcher_fallback_min": float(explanation.get("family_matcher_fallback_min", 0.0) or 0.0),
            "family_matcher_selected_root_id": explanation.get("family_matcher_selected_root_id", ""),
            "family_matcher_selected_root_score": float(explanation.get("family_matcher_selected_root_score", 0.0) or 0.0),
            "family_matcher_best_score": float(explanation.get("family_matcher_best_score", 0.0) or 0.0),
            "family_matcher_candidate_count_before": int(explanation.get("family_matcher_candidate_count_before", 0) or 0),
            "family_matcher_candidate_count_after": int(explanation.get("family_matcher_candidate_count_after", 0) or 0),
            "shared_registry_entity_count": float(flat_components.get("shared_registry_entity_count", 0.0) or 0.0),
            "facet_label_match_count": float(flat_components.get("facet_label_match_count", 0.0) or 0.0),
            "facet_type_match_count": float(flat_components.get("facet_type_match_count", 0.0) or 0.0),
            "useful_facet_match_count": float(flat_components.get("useful_facet_match_count", 0.0) or 0.0),
            "facet_label_overlap_ratio": float(flat_components.get("facet_label_overlap_ratio", 0.0) or 0.0),
            "facet_type_overlap_ratio": float(flat_components.get("facet_type_overlap_ratio", 0.0) or 0.0),
            "useful_facet_overlap_ratio": float(flat_components.get("useful_facet_overlap_ratio", 0.0) or 0.0),
            "broad_entity_no_useful_facet_ratio": float(flat_components.get("broad_entity_no_useful_facet_ratio", 0.0) or 0.0),
            "facet_post_max_similarity": float(flat_components.get("facet_post_max_similarity", 0.0) or 0.0),
            "facet_seed_max_similarity": float(flat_components.get("facet_seed_max_similarity", 0.0) or 0.0),
            "facet_shared_entities": explanation.get("facet_shared_entities", ""),
        }

    selected_prev_candidate = next(
        (c for c in candidates if str(c.get("id") or "") == storyline_id),
        {},
    ) or {}
    storyline_payload = final_storyline_payload or {
        "title": resolved_storyline_title,
        "seed_preview": (source_text or title_hint or "")[:500],
        "entity_keys": sorted(entity_keys),
        "family_signatures": sorted(current_family_signatures),
        "topic_signatures": sorted(current_topic_signatures),
    }
    base_macro_topic_payload = _macro_topic_payload(
        title=str(storyline_payload.get("title") or resolved_storyline_title or "").strip(),
        seed_preview=str(storyline_payload.get("seed_preview") or source_text or "").strip(),
        entity_keys={str(x or "").strip() for x in (storyline_payload.get("entity_keys") or []) if str(x or "").strip()},
        family_signatures={
            str(x or "").strip()
            for x in (storyline_payload.get("family_signatures") or [])
            if str(x or "").strip()
        },
        topic_signatures={
            str(x or "").strip()
            for x in (storyline_payload.get("topic_signatures") or [])
            if str(x or "").strip()
        },
    )
    preferred_macro_topic_id = str(selected_prev_candidate.get("macro_topic_id") or "").strip()
    macro_topic_candidates = await _fetch_macro_topic_candidates(
        session=session,
        topic_signatures={str(x or "").strip() for x in (base_macro_topic_payload.get("topic_signatures") or []) if str(x or "").strip()},
        anchor_entities={str(x or "").strip() for x in (base_macro_topic_payload.get("anchor_entities") or []) if str(x or "").strip()},
        preferred_macro_topic_id=preferred_macro_topic_id,
    )
    aligned_macro_topic = _select_macro_topic_alignment(
        base_macro_topic_payload,
        macro_topic_candidates,
        preferred_macro_topic_id=preferred_macro_topic_id,
    )
    macro_topic_payload = _macro_topic_payload(
        title=str(storyline_payload.get("title") or resolved_storyline_title or "").strip(),
        seed_preview=str(storyline_payload.get("seed_preview") or source_text or "").strip(),
        entity_keys={str(x or "").strip() for x in (storyline_payload.get("entity_keys") or []) if str(x or "").strip()},
        family_signatures={
            str(x or "").strip()
            for x in (storyline_payload.get("family_signatures") or [])
            if str(x or "").strip()
        },
        topic_signatures={
            str(x or "").strip()
            for x in (storyline_payload.get("topic_signatures") or [])
            if str(x or "").strip()
        },
        existing_macro_topic_id=str((aligned_macro_topic or {}).get("macro_topic_id") or preferred_macro_topic_id),
    )
    base_episode_payload = _episode_payload(
        macro_topic_id=macro_topic_payload["macro_topic_id"],
        title=str(storyline_payload.get("title") or resolved_storyline_title or "").strip(),
        seed_preview=str(storyline_payload.get("seed_preview") or source_text or "").strip(),
        entity_keys={str(x or "").strip() for x in (storyline_payload.get("entity_keys") or []) if str(x or "").strip()},
        family_signatures={
            str(x or "").strip()
            for x in (storyline_payload.get("family_signatures") or [])
            if str(x or "").strip()
        },
        topic_signatures={
            str(x or "").strip()
            for x in (storyline_payload.get("topic_signatures") or [])
            if str(x or "").strip()
        },
    )
    preferred_episode_id = str(selected_prev_candidate.get("story_episode_id") or "").strip()
    episode_candidates = await _fetch_episode_candidates(
        session=session,
        macro_topic_id=macro_topic_payload["macro_topic_id"],
        episode_signatures={
            str(x or "").strip()
            for x in (base_episode_payload.get("episode_signatures") or [])
            if str(x or "").strip()
        },
        anchor_entities={
            str(x or "").strip()
            for x in (base_episode_payload.get("anchor_entities") or [])
            if str(x or "").strip()
        },
        preferred_episode_id=preferred_episode_id,
    )
    aligned_episode = _select_episode_alignment(
        base_episode_payload,
        episode_candidates,
        preferred_episode_id=preferred_episode_id,
    )
    episode_payload = _episode_payload(
        macro_topic_id=macro_topic_payload["macro_topic_id"],
        title=str(storyline_payload.get("title") or resolved_storyline_title or "").strip(),
        seed_preview=str(storyline_payload.get("seed_preview") or source_text or "").strip(),
        entity_keys={str(x or "").strip() for x in (storyline_payload.get("entity_keys") or []) if str(x or "").strip()},
        family_signatures={
            str(x or "").strip()
            for x in (storyline_payload.get("family_signatures") or [])
            if str(x or "").strip()
        },
        topic_signatures={
            str(x or "").strip()
            for x in (storyline_payload.get("topic_signatures") or [])
            if str(x or "").strip()
        },
        existing_episode_id=str((aligned_episode or {}).get("episode_id") or preferred_episode_id),
    )

    await session.run(
        """
        MERGE (f:StoryFamily {id: $family_id})
        ON CREATE SET
            f.root_storyline_id = $family_root_storyline_id,
            f.title = $family_title,
            f.created_at = $created_at
        SET f.root_storyline_id = $family_root_storyline_id,
            f.title = coalesce(f.title, $family_title),
            f.family_signatures = $family_signatures,
            f.topic_signatures = $topic_signatures,
            f.updated_at = $updated_at
        WITH f
        MATCH (s:Storyline {id: $storyline_id})
        SET s.family_id = $family_id,
            s.family_root_id = $family_root_storyline_id
        MERGE (s)-[:BELONGS_TO]->(f)
        WITH f
        MATCH (em:EventMention {id: $event_mention_id})
        MERGE (em)-[:PART_OF_FAMILY]->(f)
        WITH f
        MATCH (p:Post {post_id: $post_id})
        MERGE (p)-[:SOURCE_OF_FAMILY]->(f)
        """,
        family_id=story_family_id,
        family_root_storyline_id=family_root_storyline_id,
        family_title=resolved_storyline_title or family_root_storyline_id,
        family_signatures=sorted(current_family_signatures),
        topic_signatures=sorted(current_topic_signatures),
        created_at=now_iso,
        updated_at=now_iso,
        storyline_id=storyline_id,
        event_mention_id=event_mention_id,
        post_id=post_id,
    )

    await session.run(
        """
        MERGE (t:MacroTopic {id: $macro_topic_id})
        ON CREATE SET
            t.created_at = $created_at,
            t.first_seen_at = $first_seen_at,
            t.canonical_title = $canonical_title,
            t.title_score = $title_score
        SET t.updated_at = $updated_at,
            t.last_seen_at = CASE
                WHEN $last_seen_at = '' THEN t.last_seen_at
                WHEN t.last_seen_at IS NULL OR $last_seen_at > t.last_seen_at THEN $last_seen_at
                ELSE t.last_seen_at
            END,
            t.first_seen_at = CASE
                WHEN $first_seen_at = '' THEN t.first_seen_at
                WHEN t.first_seen_at IS NULL OR $first_seen_at < t.first_seen_at THEN $first_seen_at
                ELSE t.first_seen_at
            END,
            t.canonical_title = CASE
                WHEN coalesce(t.canonical_title, '') = '' OR $title_score >= coalesce(t.title_score, 0.0)
                THEN $canonical_title
                ELSE t.canonical_title
            END,
            t.title_score = CASE
                WHEN coalesce(t.canonical_title, '') = '' OR $title_score >= coalesce(t.title_score, 0.0)
                THEN $title_score
                ELSE coalesce(t.title_score, 0.0)
            END,
            t.topic_signatures = reduce(acc = [], item IN (coalesce(t.topic_signatures, []) + $topic_signatures) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            t.anchor_entities = reduce(acc = [], item IN (coalesce(t.anchor_entities, []) + $anchor_entities) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            t.topic_aliases = reduce(acc = [], item IN (coalesce(t.topic_aliases, []) + $topic_aliases) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            t.basis_components = reduce(acc = [], item IN (coalesce(t.basis_components, []) + $basis_components) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            t.representative_storyline_ids = reduce(acc = [], item IN (coalesce(t.representative_storyline_ids, []) + [$storyline_id]) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            t.representative_post_ids = reduce(acc = [], item IN (coalesce(t.representative_post_ids, []) + [$post_id]) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            t.seed_preview = CASE
                WHEN coalesce(t.seed_preview, '') = '' THEN $seed_preview
                ELSE t.seed_preview
            END
        WITH t
        MATCH (s:Storyline {id: $storyline_id})
        OPTIONAL MATCH (s)-[old_story_rel:BELONGS_TO_TOPIC]->(old_story_topic:MacroTopic)
        WHERE old_story_topic.id <> $macro_topic_id
        DELETE old_story_rel
        SET s.macro_topic_id = $macro_topic_id,
            s.macro_topic_title = t.canonical_title
        MERGE (s)-[:BELONGS_TO_TOPIC]->(t)
        WITH t
        MATCH (f:StoryFamily {id: $family_id})
        OPTIONAL MATCH (f)-[old_family_rel:ALIGNS_WITH_TOPIC]->(old_family_topic:MacroTopic)
        WHERE old_family_topic.id <> $macro_topic_id
        DELETE old_family_rel
        MERGE (f)-[:ALIGNS_WITH_TOPIC]->(t)
        WITH t
        MATCH (em:EventMention {id: $event_mention_id})
        OPTIONAL MATCH (em)-[old_event_rel:PART_OF_TOPIC]->(old_event_topic:MacroTopic)
        WHERE old_event_topic.id <> $macro_topic_id
        DELETE old_event_rel
        MERGE (em)-[:PART_OF_TOPIC]->(t)
        WITH t
        MATCH (p:Post {post_id: $post_id})
        OPTIONAL MATCH (p)-[old_post_rel:SOURCE_OF_TOPIC]->(old_post_topic:MacroTopic)
        WHERE old_post_topic.id <> $macro_topic_id
        DELETE old_post_rel
        MERGE (p)-[:SOURCE_OF_TOPIC]->(t)
        """,
        macro_topic_id=macro_topic_payload["macro_topic_id"],
        canonical_title=macro_topic_payload["canonical_title"],
        title_score=float(macro_topic_payload["title_score"] or 0.0),
        topic_signatures=macro_topic_payload["topic_signatures"],
        anchor_entities=macro_topic_payload["anchor_entities"],
        topic_aliases=macro_topic_payload["topic_aliases"],
        basis_components=macro_topic_payload["basis_components"],
        seed_preview=macro_topic_payload["seed_preview"],
        created_at=now_iso,
        updated_at=now_iso,
        first_seen_at=event_time_iso,
        last_seen_at=event_time_iso,
        storyline_id=storyline_id,
        family_id=story_family_id,
        event_mention_id=event_mention_id,
        post_id=post_id,
    )

    await session.run(
        """
        MERGE (e:Episode {id: $episode_id})
        ON CREATE SET
            e.created_at = $created_at,
            e.first_seen_at = $first_seen_at,
            e.canonical_title = $canonical_title,
            e.title_score = $title_score,
            e.phase_type = $phase_type
        SET e.updated_at = $updated_at,
            e.last_seen_at = CASE
                WHEN $last_seen_at = '' THEN e.last_seen_at
                WHEN e.last_seen_at IS NULL OR $last_seen_at > e.last_seen_at THEN $last_seen_at
                ELSE e.last_seen_at
            END,
            e.first_seen_at = CASE
                WHEN $first_seen_at = '' THEN e.first_seen_at
                WHEN e.first_seen_at IS NULL OR $first_seen_at < e.first_seen_at THEN $first_seen_at
                ELSE e.first_seen_at
            END,
            e.phase_type = CASE
                WHEN coalesce(e.phase_type, '') = '' AND $phase_type <> '' THEN $phase_type
                ELSE coalesce(e.phase_type, $phase_type)
            END,
            e.canonical_title = CASE
                WHEN coalesce(e.canonical_title, '') = '' OR $title_score >= coalesce(e.title_score, 0.0)
                THEN $canonical_title
                ELSE e.canonical_title
            END,
            e.title_score = CASE
                WHEN coalesce(e.canonical_title, '') = '' OR $title_score >= coalesce(e.title_score, 0.0)
                THEN $title_score
                ELSE coalesce(e.title_score, 0.0)
            END,
            e.episode_signatures = reduce(acc = [], item IN (coalesce(e.episode_signatures, []) + $episode_signatures) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            e.anchor_entities = reduce(acc = [], item IN (coalesce(e.anchor_entities, []) + $anchor_entities) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            e.topic_aliases = reduce(acc = [], item IN (coalesce(e.topic_aliases, []) + $topic_aliases) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            e.basis_components = reduce(acc = [], item IN (coalesce(e.basis_components, []) + $basis_components) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            e.representative_storyline_ids = reduce(acc = [], item IN (coalesce(e.representative_storyline_ids, []) + [$storyline_id]) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            e.representative_post_ids = reduce(acc = [], item IN (coalesce(e.representative_post_ids, []) + [$post_id]) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            e.seed_preview = CASE
                WHEN coalesce(e.seed_preview, '') = '' THEN $seed_preview
                ELSE e.seed_preview
            END
        WITH e
        MATCH (t:MacroTopic {id: $macro_topic_id})
        OPTIONAL MATCH (e)-[old_episode_topic_rel:PART_OF_TOPIC]->(old_episode_topic:MacroTopic)
        WHERE old_episode_topic.id <> $macro_topic_id
        DELETE old_episode_topic_rel
        MERGE (e)-[:PART_OF_TOPIC]->(t)
        WITH e, t
        MATCH (s:Storyline {id: $storyline_id})
        OPTIONAL MATCH (s)-[old_story_episode_rel:BELONGS_TO_EPISODE]->(old_episode:Episode)
        WHERE old_episode.id <> $episode_id
        DELETE old_story_episode_rel
        SET s.story_episode_id = $episode_id,
            s.story_episode_title = e.canonical_title
        MERGE (s)-[:BELONGS_TO_EPISODE]->(e)
        WITH e, t
        MATCH (em:EventMention {id: $event_mention_id})
        OPTIONAL MATCH (em)-[old_event_episode_rel:PART_OF_EPISODE]->(old_event_episode:Episode)
        WHERE old_event_episode.id <> $episode_id
        DELETE old_event_episode_rel
        MERGE (em)-[:PART_OF_EPISODE]->(e)
        WITH e, t
        MATCH (p:Post {post_id: $post_id})
        OPTIONAL MATCH (p)-[old_post_episode_rel:SOURCE_OF_EPISODE]->(old_post_episode:Episode)
        WHERE old_post_episode.id <> $episode_id
        DELETE old_post_episode_rel
        MERGE (p)-[:SOURCE_OF_EPISODE]->(e)
        WITH e, t
        OPTIONAL MATCH (prev_episode:Episode)-[:PART_OF_TOPIC]->(t)
        WHERE prev_episode.id <> e.id
          AND coalesce(prev_episode.last_seen_at, '') < coalesce(e.first_seen_at, '')
        WITH e, prev_episode
        ORDER BY coalesce(prev_episode.last_seen_at, '') DESC
        LIMIT 1
        FOREACH (_ IN CASE WHEN prev_episode IS NULL THEN [] ELSE [1] END |
            MERGE (prev_episode)-[:EARLIER_THAN]->(e)
        )
        """,
        episode_id=episode_payload["episode_id"],
        canonical_title=episode_payload["canonical_title"],
        phase_type=episode_payload["phase_type"],
        title_score=float(episode_payload["title_score"] or 0.0),
        episode_signatures=episode_payload["episode_signatures"],
        anchor_entities=episode_payload["anchor_entities"],
        topic_aliases=episode_payload["topic_aliases"],
        basis_components=episode_payload["basis_components"],
        seed_preview=episode_payload["seed_preview"],
        created_at=now_iso,
        updated_at=now_iso,
        first_seen_at=event_time_iso,
        last_seen_at=event_time_iso,
        macro_topic_id=macro_topic_payload["macro_topic_id"],
        storyline_id=storyline_id,
        event_mention_id=event_mention_id,
        post_id=post_id,
    )

    update_step_payload = _update_step_payload(
        macro_topic_id=macro_topic_payload["macro_topic_id"],
        storyline_id=storyline_id,
        event_mention_id=event_mention_id,
        post_id=post_id,
        event_time=event_time,
        summary=str(source_text or title_hint or resolved_storyline_title or "").strip(),
        event_signature=next(iter(sorted(event_signatures)), ""),
        entity_keys={str(x or "").strip() for x in (storyline_payload.get("entity_keys") or []) if str(x or "").strip()},
        participant_role_signatures=participant_role_signatures,
    )

    await session.run(
        """
        MERGE (u:UpdateStep {id: $step_id})
        ON CREATE SET
            u.created_at = $created_at,
            u.first_seen_at = $first_seen_at,
            u.last_seen_at = $last_seen_at,
            u.canonical_summary = $canonical_summary,
            u.step_signature = $step_signature,
            u.event_signature = $event_signature
        SET u.updated_at = $updated_at,
            u.first_seen_at = CASE
                WHEN $first_seen_at = '' THEN u.first_seen_at
                WHEN u.first_seen_at IS NULL OR $first_seen_at < u.first_seen_at THEN $first_seen_at
                ELSE u.first_seen_at
            END,
            u.last_seen_at = CASE
                WHEN $last_seen_at = '' THEN u.last_seen_at
                WHEN u.last_seen_at IS NULL OR $last_seen_at > u.last_seen_at THEN $last_seen_at
                ELSE u.last_seen_at
            END,
            u.canonical_summary = CASE
                WHEN coalesce(u.canonical_summary, '') = '' THEN $canonical_summary
                ELSE u.canonical_summary
            END,
            u.anchor_entities = reduce(acc = [], item IN (coalesce(u.anchor_entities, []) + $anchor_entities) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            u.role_signatures = reduce(acc = [], item IN (coalesce(u.role_signatures, []) + $role_signatures) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            u.storyline_ids = reduce(acc = [], item IN (coalesce(u.storyline_ids, []) + $storyline_ids) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            u.source_post_ids = reduce(acc = [], item IN (coalesce(u.source_post_ids, []) + $source_post_ids) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END),
            u.event_mention_ids = reduce(acc = [], item IN (coalesce(u.event_mention_ids, []) + $event_mention_ids) |
                CASE WHEN item IN acc THEN acc ELSE acc + item END)
        WITH u
        MATCH (t:MacroTopic {id: $macro_topic_id})
        OPTIONAL MATCH (u)-[old_step_topic_rel:PART_OF_TOPIC]->(old_step_topic:MacroTopic)
        WHERE old_step_topic.id <> $macro_topic_id
        DELETE old_step_topic_rel
        MERGE (u)-[:PART_OF_TOPIC]->(t)
        WITH u, t
        MATCH (s:Storyline {id: $storyline_id})
        MERGE (u)-[:PART_OF_STORYLINE]->(s)
        WITH u, t
        MATCH (e:Episode {id: $episode_id})
        OPTIONAL MATCH (u)-[old_step_episode_rel:PART_OF_EPISODE]->(old_step_episode:Episode)
        WHERE old_step_episode.id <> $episode_id
        DELETE old_step_episode_rel
        MERGE (u)-[:PART_OF_EPISODE]->(e)
        WITH u, t, e
        MATCH (p:Post {post_id: $post_id})
        OPTIONAL MATCH (p)-[old_post_step_rel:PART_OF_STEP]->(old_post_step:UpdateStep)
        WITH u, t, e, p, old_post_step_rel, old_post_step
        FOREACH (_ IN CASE WHEN old_post_step_rel IS NOT NULL AND old_post_step.id <> u.id THEN [1] ELSE [] END |
            DELETE old_post_step_rel
        )
        MERGE (p)-[:PART_OF_STEP]->(u)
        WITH u, t, e
        MATCH (em:EventMention {id: $event_mention_id})
        OPTIONAL MATCH (em)-[old_event_step_rel:PART_OF_STEP]->(old_event_step:UpdateStep)
        WITH u, t, e, em, old_event_step_rel, old_event_step
        FOREACH (_ IN CASE WHEN old_event_step_rel IS NOT NULL AND old_event_step.id <> u.id THEN [1] ELSE [] END |
            DELETE old_event_step_rel
        )
        MERGE (em)-[:PART_OF_STEP]->(u)
        WITH u, t, e
        OPTIONAL MATCH (prev:UpdateStep)-[:PART_OF_TOPIC]->(t)
        WHERE prev.id <> u.id
          AND (
            EXISTS { (prev)-[:PART_OF_EPISODE]->(e) }
            OR NOT EXISTS { (u)-[:PART_OF_EPISODE]->(:Episode) }
          )
          AND coalesce(prev.last_seen_at, '') < coalesce(u.first_seen_at, '')
        WITH u, prev
        ORDER BY coalesce(prev.last_seen_at, '') DESC
        LIMIT 1
        FOREACH (_ IN CASE WHEN prev IS NULL THEN [] ELSE [1] END |
            MERGE (prev)-[:UPDATES]->(u)
        )
        """,
        step_id=update_step_payload["step_id"],
        step_signature=update_step_payload["step_signature"],
        canonical_summary=update_step_payload["canonical_summary"],
        event_signature=update_step_payload["event_signature"],
        anchor_entities=update_step_payload["anchor_entities"],
        role_signatures=update_step_payload["role_signatures"],
        storyline_ids=update_step_payload["storyline_ids"],
        source_post_ids=update_step_payload["source_post_ids"],
        event_mention_ids=update_step_payload["event_mention_ids"],
        first_seen_at=update_step_payload["first_seen_at"],
        last_seen_at=update_step_payload["last_seen_at"],
        created_at=now_iso,
        updated_at=now_iso,
        macro_topic_id=macro_topic_payload["macro_topic_id"],
        episode_id=episode_payload["episode_id"],
        storyline_id=storyline_id,
        post_id=post_id,
        event_mention_id=event_mention_id,
    )

    await session.run(
        """
        MATCH (em:EventMention {id: $event_mention_id})
        MATCH (s:Storyline {id: $storyline_id})
        OPTIONAL MATCH (em)-[old_part_rel:PART_OF]->(old_storyline:Storyline)
        WITH em, s, old_part_rel, old_storyline
        FOREACH (_ IN CASE WHEN old_part_rel IS NOT NULL AND old_storyline.id <> $storyline_id THEN [1] ELSE [] END |
            DELETE old_part_rel
        )
        MERGE (em)-[r:PART_OF]->(s)
        SET r.score = $score,
            r.threshold = $threshold,
            r.decision = $decision,
            r.explanation_json = $explanation_json,
            r.updated_at = $updated_at
        """,
        event_mention_id=event_mention_id,
        storyline_id=storyline_id,
        score=round(float(explanation.get("score", 0.0)), 4),
        threshold=assign_threshold,
        decision=decision,
        explanation_json=json.dumps(explanation, ensure_ascii=False),
        updated_at=now_iso,
    )

    await session.run(
        """
        MATCH (p:Post {post_id: $post_id})
        MATCH (s:Storyline {id: $storyline_id})
        OPTIONAL MATCH (p)-[old_source_rel:SOURCE_OF]->(old_storyline:Storyline)
        WITH p, s, old_source_rel, old_storyline
        FOREACH (_ IN CASE WHEN old_source_rel IS NOT NULL AND old_storyline.id <> $storyline_id THEN [1] ELSE [] END |
            DELETE old_source_rel
        )
        MERGE (p)-[:SOURCE_OF]->(s)
        """,
        post_id=post_id,
        storyline_id=storyline_id,
    )

    flat_components = explanation.get("components", {}) or {}
    return {
        "enabled": True,
        "storyline_id": storyline_id,
        "story_family_id": story_family_id,
        "story_family_root_storyline_id": family_root_storyline_id,
        "macro_topic_id": macro_topic_payload["macro_topic_id"],
        "macro_topic_title": macro_topic_payload["canonical_title"],
        "story_episode_id": episode_payload["episode_id"],
        "story_episode_title": episode_payload["canonical_title"],
        "update_step_id": update_step_payload["step_id"],
        "storyline_seed_post_id": _storyline_seed_post_id(storyline_id),
        "decision": decision,
        "score": round(float(explanation.get("score", 0.0)), 4),
        "threshold": assign_threshold,
        "explanation": json.dumps(explanation, ensure_ascii=False),
        "components": explanation.get("components", {}),
        "contributions": explanation.get("contributions", {}),
        "reason": explanation.get("reason", ""),
        "best_candidate_storyline_id": best_candidate_storyline_id,
        "best_candidate_score": round(decision_score, 4),
        "best_candidate_hybrid_score": best_candidate_hybrid_score,
        "best_candidate_pair_score": best_candidate_pair_score,
        "score_source": score_source,
        "pair_blended_score": pair_blended_score if score_source == "pair_catboost" else 0.0,
        "pair_hybrid_blend": pair_hybrid_blend if score_source == "pair_catboost" else 0.0,
        "continuity_signal": round(continuity_signal, 4),
        "recent_mentions_count": len((best or {}).get("recent_mentions", []) or []),
        "family_cluster_size": int((best or {}).get("family_cluster_size") or 0),
        "family_signal": round(float((best or {}).get("family_signal", 0.0) or 0.0), 4),
        "family_boost": round(float((best or {}).get("family_boost", 0.0) or 0.0), 4),
        "family_preferred_root": bool((best or {}).get("family_preferred_root")),
        "family_oldest_seed_id": (best or {}).get("family_oldest_seed_id") or "",
        "family_root_storyline_id": explanation.get("family_root_storyline_id", storyline_id),
        "family_rerouted_to_root": bool(explanation.get("family_rerouted_to_root")),
        "family_matcher_ready": bool(explanation.get("family_matcher_ready")),
        "family_matcher_mode": explanation.get("family_matcher_mode", ""),
        "family_matcher_threshold": float(explanation.get("family_matcher_threshold", 0.0) or 0.0),
        "family_matcher_fallback_min": float(explanation.get("family_matcher_fallback_min", 0.0) or 0.0),
        "family_matcher_selected_root_id": explanation.get("family_matcher_selected_root_id", ""),
        "family_matcher_selected_root_score": float(explanation.get("family_matcher_selected_root_score", 0.0) or 0.0),
        "family_matcher_best_score": float(explanation.get("family_matcher_best_score", 0.0) or 0.0),
        "family_matcher_candidate_count_before": int(explanation.get("family_matcher_candidate_count_before", 0) or 0),
        "family_matcher_candidate_count_after": int(explanation.get("family_matcher_candidate_count_after", 0) or 0),
        "shared_registry_entity_count": float(flat_components.get("shared_registry_entity_count", 0.0) or 0.0),
        "facet_label_match_count": float(flat_components.get("facet_label_match_count", 0.0) or 0.0),
        "facet_type_match_count": float(flat_components.get("facet_type_match_count", 0.0) or 0.0),
        "useful_facet_match_count": float(flat_components.get("useful_facet_match_count", 0.0) or 0.0),
        "facet_label_overlap_ratio": float(flat_components.get("facet_label_overlap_ratio", 0.0) or 0.0),
        "facet_type_overlap_ratio": float(flat_components.get("facet_type_overlap_ratio", 0.0) or 0.0),
        "useful_facet_overlap_ratio": float(flat_components.get("useful_facet_overlap_ratio", 0.0) or 0.0),
        "broad_entity_no_useful_facet_ratio": float(flat_components.get("broad_entity_no_useful_facet_ratio", 0.0) or 0.0),
        "facet_post_max_similarity": float(flat_components.get("facet_post_max_similarity", 0.0) or 0.0),
        "facet_seed_max_similarity": float(flat_components.get("facet_seed_max_similarity", 0.0) or 0.0),
        "facet_shared_entities": explanation.get("facet_shared_entities", ""),
    }
