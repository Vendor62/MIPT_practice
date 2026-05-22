import os
import re
import asyncio
import threading
import json
import structlog
from collections import Counter

from sqlalchemy import text
from datetime import datetime, timezone
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from gliner import GLiNER
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer

from embeddings_service.celery_app import celery_app
from embeddings_service.neo4j_client import driver

from embeddings_service.text_filters import (
    classify_storyline_content,
    clean_for_ner,
    clean_for_re,
    has_ad_marker,
    is_promotional_text,
    sanitize_pipeline_text,
)
from embeddings_service.entity_normalizer import normalize_entity_text, partition_entities
from embeddings_service.relation_validator import filter_relations
from embeddings_service.relation_scorer import apply_relation_model_gate
from embeddings_service.event_mention import build_event_mention_lite
from embeddings_service.storyline_resolver import resolve_and_write_storyline

log = structlog.get_logger()
# PATCH_MARKER_MREBEL_PIPELINE_2026_02_23

# Torch device: -1 = CPU, 0+ = CUDA index.
def _ie_torch_device() -> int:
    raw = os.getenv("IE_TORCH_DEVICE", "-1").strip()
    try:
        return int(raw)
    except ValueError:
        return -1


# ─── Конфиг ───────────────────────────────────────────────
GLINER_MODEL = os.getenv("GLINER_MODEL", "urchade/gliner_multi-v2.1")
REBEL_MODEL  = os.getenv("REBEL_MODEL",  "Babelscape/mrebel-base")
EMBED_MODEL  = os.getenv("IE_EMBED_MODEL", "intfloat/multilingual-e5-large")
REBEL_MAX_LENGTH = int(os.getenv("REBEL_MAX_LENGTH", "256"))
REBEL_NUM_BEAMS = int(os.getenv("REBEL_NUM_BEAMS", "4"))
REBEL_NUM_RETURN_SEQUENCES = int(os.getenv("REBEL_NUM_RETURN_SEQUENCES", "2"))
REBEL_NO_REPEAT_NGRAM_SIZE = int(os.getenv("REBEL_NO_REPEAT_NGRAM_SIZE", "3"))
REBEL_MAX_CHUNKS = int(os.getenv("REBEL_MAX_CHUNKS", "3"))
REBEL_CHUNK_CHARS = int(os.getenv("REBEL_CHUNK_CHARS", "700"))
REBEL_CHUNK_OVERLAP = int(os.getenv("REBEL_CHUNK_OVERLAP", "120"))
RELATION_REJECT_LOG_SAMPLE = int(os.getenv("RELATION_REJECT_LOG_SAMPLE", "120"))
GLINER_BASE_THRESHOLD = float(os.getenv("GLINER_BASE_THRESHOLD", "0.50"))
GLINER_DOMAIN_THRESHOLD = float(os.getenv("GLINER_DOMAIN_THRESHOLD", "0.42"))
DISPATCH_QUEUE = os.getenv("DISPATCH_QUEUE", "dispatch_queue")

DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST", "db")
DB_PORT     = os.getenv("DB_PORT", "5432")
POSTGRES_DB = os.getenv("DB_NAME")

engine = create_async_engine(
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}",
    pool_pre_ping=True,
)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# ─── Онтология сущностей ───────────────────────────────────
BASE_ENTITY_TYPES = [
    "person",
    "organization",
    "location",
    "event",
    "product",
    "law",
]

DOMAIN_ENTITY_TYPES = {
    "politics":   ["political party", "election", "treaty", "sanction", "diplomat"],
    "economics":  ["company", "stock", "currency", "deal", "bankruptcy"],
    "military":   ["military unit", "weapon", "territory", "operation", "ceasefire"],
    "sports":     ["team", "athlete", "tournament", "transfer", "coach"],
    "technology": ["startup", "patent", "algorithm", "data breach"],
    "science":    ["discovery", "experiment", "institution", "publication"],
    "disasters":  ["natural disaster", "casualty", "rescue operation"],
}

# ─── Ленивая загрузка моделей ──────────────────────────────
_gliner    = None
_rebel_pipe = None
_embedder  = None


def get_gliner() -> GLiNER:
    global _gliner
    if _gliner is None:
        log.info("loading_gliner", model=GLINER_MODEL, device=_ie_torch_device())
        _gliner = GLiNER.from_pretrained(GLINER_MODEL, map_location="cuda" if _ie_torch_device() >= 0 else "cpu")
    return _gliner

def get_rebel():
    """
    mREBEL (Babelscape/mrebel-base) в нашем окружении сидит на M2M100,
    и корректнее всего дергать его через pipeline translation_xx_to_yy
    с src_lang/tgt_lang (см. model card).
    """
    global _rebel_pipe
    if _rebel_pipe is None:
        dev = _ie_torch_device()
        log.info("loading_rebel", model=REBEL_MODEL, device=dev)
        _rebel_pipe = pipeline(
            "translation_xx_to_yy",
            model=REBEL_MODEL,
            tokenizer=REBEL_MODEL,
            device=dev,
        )
    return _rebel_pipe

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        dev = _ie_torch_device()
        log.info("loading_embedder", model=EMBED_MODEL, device=dev)
        _embedder = SentenceTransformer(EMBED_MODEL, device=("cuda" if dev >= 0 else "cpu"))
    return _embedder


# ─── Event loop для async в Celery ────────────────────────
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread = None
_loop_ready = threading.Event()


def _start_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop_ready.set()
    _loop.run_forever()


def _ensure_loop():
    global _loop_thread
    if _loop_thread and _loop_ready.is_set():
        return
    _loop_thread = threading.Thread(target=_start_loop, daemon=True)
    _loop_thread.start()
    _loop_ready.wait()


def _run_async(coro):
    _ensure_loop()
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# ─── Извлечение сущностей (GLiNER) ────────────────────────

def _dedup_extracted_entities(items: list[dict]) -> list[dict]:
    seen: dict[tuple[str, str, int | None, int | None], dict] = {}
    for item in items:
        key = (
            normalize_entity_text(str(item.get("text") or "")),
            str(item.get("label") or "").strip().lower(),
            item.get("start") if isinstance(item.get("start"), int) else None,
            item.get("end") if isinstance(item.get("end"), int) else None,
        )
        current = seen.get(key)
        if current is None or float(item.get("score") or 0.0) > float(current.get("score") or 0.0):
            seen[key] = item
    return list(seen.values())


def extract_entities(text: str, extra_types: list[str] | None = None) -> list[dict]:
    gliner = get_gliner()
    cleaned = clean_for_ner(text)
    core_entities = gliner.predict_entities(cleaned, BASE_ENTITY_TYPES, threshold=GLINER_BASE_THRESHOLD)
    domain_entities: list[dict] = []
    extra = [str(x or "").strip() for x in (extra_types or []) if str(x or "").strip()]
    if extra:
        domain_entities = gliner.predict_entities(cleaned, extra, threshold=GLINER_DOMAIN_THRESHOLD)
    return _dedup_extracted_entities([*(core_entities or []), *(domain_entities or [])])


# ─── Извлечение триплетов (mREBEL) ────────────────────────
_SPECIAL_RE = re.compile(r"^__[^ ]+__$")  # __en__, __tn__, __uk__, __zu__, etc.

def _clean_decoded(decoded: str) -> list[str]:
    s = (
        decoded.replace("</s>", "")
               .replace("<s>", "")
               .replace("<pad>", "")
               .strip()
    )
    return s.split() if s else []


def _parse_mrebel_output(decoded: str) -> list[dict]:
    toks = _clean_decoded(decoded)
    if not toks:
        return []

    # отрезаем лидирующие спец-теги
    while toks and _SPECIAL_RE.match(toks[0]):
        toks.pop(0)
    if not toks:
        return []

    # все спец-теги считаем разделителями полей
    seps = [i for i, t in enumerate(toks) if _SPECIAL_RE.match(t)]
    if len(seps) < 2:
        return []

    triplets = []
    cursor = 0
    while True:
        try:
            i1 = next(i for i in range(cursor, len(toks)) if _SPECIAL_RE.match(toks[i]))
            i2 = next(i for i in range(i1 + 1, len(toks)) if _SPECIAL_RE.match(toks[i]))
        except StopIteration:
            break

        head = " ".join(toks[cursor:i1]).strip()
        tail = " ".join(toks[i1 + 1:i2]).strip()

        try:
            i3 = next(i for i in range(i2 + 1, len(toks)) if _SPECIAL_RE.match(toks[i]))
            rel = " ".join(toks[i2 + 1:i3]).strip()
            cursor = i3 + 1
        except StopIteration:
            rel = " ".join(toks[i2 + 1:]).strip()
            cursor = len(toks)

        # важно: не требуем rel (для устойчивости)
        if head and tail:
            triplets.append({"head": head, "tail": tail, "relation": rel or "related_to"})

        if cursor >= len(toks):
            break

    return triplets


def _dedup_triplets(triplets: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for t in triplets:
        head = normalize_entity_text(t.get("head", ""))
        tail = normalize_entity_text(t.get("tail", ""))
        rel = (t.get("relation") or "").strip().lower()
        key = (head, rel, tail)
        if not head or not tail or not rel or key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _chunk_text_for_re(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= REBEL_CHUNK_CHARS:
        return [t]

    chunks = []
    start = 0
    limit = max(REBEL_CHUNK_CHARS, 200)
    overlap = max(0, min(REBEL_CHUNK_OVERLAP, limit // 2))
    n = len(t)

    while start < n and len(chunks) < REBEL_MAX_CHUNKS:
        end = min(n, start + limit)
        if end < n:
            window = t[start:end]
            boundary = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
            if boundary > limit // 2:
                end = start + boundary + 1
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(start + 1, end - overlap)

    return chunks


def extract_relations(text: str, entities: list[dict]) -> list[dict]:
    if not entities:
        return []

    pipe = get_rebel()
    cleaned = clean_for_re(text)
    chunks = _chunk_text_for_re(cleaned)
    decoded_items = []
    raw_triplets = []
    for chunk in chunks:
        results = pipe(
            chunk,
            src_lang="ru",
            tgt_lang="ru",
            return_tensors=True,
            return_text=False,
            do_sample=False,
            max_length=REBEL_MAX_LENGTH,
            num_beams=REBEL_NUM_BEAMS,
            num_return_sequences=REBEL_NUM_RETURN_SEQUENCES,
            no_repeat_ngram_size=REBEL_NO_REPEAT_NGRAM_SIZE,
        )
        for res in results:
            ids = res.get("translation_token_ids")
            if ids is None:
                continue
            decoded = pipe.tokenizer.batch_decode([ids], skip_special_tokens=False)[0]
            decoded_items.append(decoded)
            raw_triplets.extend(_parse_mrebel_output(decoded))

    raw_triplets = _dedup_triplets(raw_triplets)

    log.info(
        "rebel_extraction",
        chunks=len(chunks),
        decoded_samples=decoded_items[:3],
        decoded_count=len(decoded_items),
        raw=len(raw_triplets),
    )
    return raw_triplets

# ─── Запись в Neo4j ───────────────────────────────────────

def embed_passage(text: str) -> list[float]:
    model = get_embedder()
    return model.encode(f"passage: {text}", normalize_embeddings=True).tolist()


def _match_relation_entity(
    relation_value: str,
    relation_key: str | None,
    entities: list[dict],
) -> dict | None:
    if relation_key:
        key_norm = normalize_entity_text(relation_key)
        for entity in entities:
            entity_key = normalize_entity_text(entity.get("canonical_key", ""))
            if entity_key and entity_key == key_norm:
                return entity

    value_norm = normalize_entity_text(relation_value)
    if not value_norm:
        return None

    for entity in entities:
        name_norm = normalize_entity_text(entity.get("text", ""))
        raw_norm = normalize_entity_text(entity.get("raw_text", ""))
        if value_norm in {name_norm, raw_norm}:
            return entity

    return None


def _relation_signature(rel: dict) -> str:
    head = normalize_entity_text(str(rel.get("head_key") or rel.get("head") or ""))
    tail = normalize_entity_text(str(rel.get("tail_key") or rel.get("tail") or ""))
    relation = str(rel.get("relation") or "")
    return f"{relation}:{head}->{tail}"


def _storyline_title_hint(entities: list[dict], relations: list[dict], post_id: int) -> str:
    def _entity_title_score(entity: dict) -> float:
        text = str(entity.get("text") or "").strip()
        if not text:
            return -1.0
        label = str(entity.get("label") or entity.get("original_label") or "").strip().lower()
        subtype = str(entity.get("entity_subtype") or "").strip().lower()
        score = 0.0
        if entity.get("is_graph_core_entity"):
            score += 4.0
        if entity.get("is_storyline_anchor_entity") or entity.get("storyline_anchor_eligible"):
            score += 3.0
        if entity.get("rag_anchor_eligible"):
            score += 2.2
        if entity.get("is_event_argument_entity"):
            score += 1.0
        if label in {"organization", "product", "person"}:
            score += 2.0
        elif label in {"event", "law"}:
            score += 1.4
        elif label == "location":
            score += 0.5
        if subtype:
            score += 0.8
        if re.search(r"[A-Za-z0-9]", text):
            score += 0.5
        if len(text) >= 6:
            score += 0.25
        return score

    seen: set[str] = set()
    ranked: list[tuple[float, str]] = []
    for entity in entities:
        name = str(entity.get("text") or "").strip()
        if not name:
            continue
        norm = normalize_entity_text(name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        ranked.append((_entity_title_score(entity), name))

    ranked.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    names = [name for score, name in ranked if score > 0][:2]
    rel = str(relations[0].get("relation") or "general") if relations else "general"
    if names:
        return f"{', '.join(names[:2])} [{rel}]"
    return f"Storyline {post_id} [{rel}]"


def _entity_storage_key(entity: dict) -> str:
    label = str(entity.get("label") or entity.get("original_label") or "").strip().lower()
    canonical_key = normalize_entity_text(str(entity.get("canonical_key") or ""))
    if canonical_key:
        return canonical_key
    text = normalize_entity_text(str(entity.get("text") or entity.get("raw_text") or ""))
    return f"{label}:{text}" if label and text else text


def _entity_lookup_keys(entity: dict) -> set[str]:
    keys = set()
    for value in (
        entity.get("canonical_key"),
        entity.get("text"),
        entity.get("raw_text"),
        entity.get("normalized_text"),
    ):
        norm = normalize_entity_text(str(value or ""))
        if norm:
            keys.add(norm)
    storage_key = _entity_storage_key(entity)
    if storage_key:
        keys.add(storage_key)
    return keys


def _index_entities(entities: list[dict]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for entity in entities:
        for key in _entity_lookup_keys(entity):
            index[key] = entity
    return index


def _merge_entity_payload(target: dict, incoming: dict) -> dict:
    incoming_decision = str(incoming.get("decision") or "").strip().lower()
    target_decision = str(target.get("decision") or "").strip().lower()
    if incoming_decision == "promote" and target_decision != "promote":
        target.update(incoming)
    else:
        for key, value in incoming.items():
            current = target.get(key)
            if key not in target or current is None or current == "":
                target[key] = value
    return target


def _relation_endpoint_entity(rel: dict, side: str, entity_index: dict[str, dict]) -> dict | None:
    for candidate in (
        rel.get(f"{side}_key"),
        rel.get(f"{side}_canonical"),
        rel.get(side),
    ):
        norm = normalize_entity_text(str(candidate or ""))
        if not norm:
            continue
        entity = entity_index.get(norm)
        if entity:
            return entity

    label = str(rel.get(f"{side}_type") or "").strip().lower()
    text = normalize_entity_text(str(rel.get(f"{side}_canonical") or rel.get(side) or ""))
    if not label or not text:
        return None

    canonical_key = str(rel.get(f"{side}_key") or f"{label}:{text.replace(' ', '_')}")
    entity = {
        "text": text,
        "raw_text": str(rel.get(side) or text),
        "normalized_text": text,
        "label": label,
        "original_label": label,
        "canonical_key": canonical_key,
        "decision": str(rel.get(f"{side}_decision") or "mention"),
        "reason": str(rel.get(f"{side}_reason") or "relation_argument_stub"),
        "materialized_from_relation": True,
    }
    for key in _entity_lookup_keys(entity):
        entity_index[key] = entity
    return entity


def _prepare_story_entities(
    *,
    core_entities: list[dict],
    support_entities: list[dict],
    anchor_entities: list[dict],
    relations: list[dict],
) -> tuple[list[dict], dict[str, object], dict[str, dict]]:
    entity_index = _index_entities(support_entities)
    for entity in anchor_entities:
        for key in _entity_lookup_keys(entity):
            entity_index[key] = entity
    for entity in core_entities:
        for key in _entity_lookup_keys(entity):
            entity_index[key] = entity

    prepared: dict[str, dict] = {}
    core_storage_keys = {_entity_storage_key(entity) for entity in core_entities if _entity_storage_key(entity)}
    anchor_storage_keys = {_entity_storage_key(entity) for entity in anchor_entities if _entity_storage_key(entity)}
    relation_argument_storage_keys: set[str] = set()

    def upsert_entity(
        entity: dict,
        *,
        is_core: bool,
        is_event_argument: bool,
        is_storyline_anchor: bool,
    ) -> None:
        storage_key = _entity_storage_key(entity)
        if not storage_key:
            return
        current = prepared.get(storage_key)
        if current is None:
            current = dict(entity)
            current["signal_roles"] = set()
            prepared[storage_key] = current
        else:
            _merge_entity_payload(current, entity)
            current.setdefault("signal_roles", set())

        if is_core:
            current["signal_roles"].add("graph_core_entity")
        if is_storyline_anchor:
            current["signal_roles"].add("storyline_anchor_entity")
        if is_event_argument:
            current["signal_roles"].add("event_argument_entity")
            current["signal_roles"].add("storyline_support_entity")
        current["is_graph_core_entity"] = "graph_core_entity" in current["signal_roles"]
        current["is_storyline_anchor_entity"] = "storyline_anchor_entity" in current["signal_roles"]
        current["is_event_argument_entity"] = "event_argument_entity" in current["signal_roles"]
        current["is_storyline_support_entity"] = "storyline_support_entity" in current["signal_roles"]
        current["primary_signal_role"] = (
            "graph_core_entity"
            if current["is_graph_core_entity"]
            else "storyline_anchor_entity"
            if current["is_storyline_anchor_entity"]
            else "event_argument_entity"
            if current["is_event_argument_entity"]
            else "storyline_support_entity"
        )

    for entity in core_entities:
        upsert_entity(entity, is_core=True, is_event_argument=False, is_storyline_anchor=True)

    for entity in anchor_entities:
        upsert_entity(
            entity,
            is_core=_entity_storage_key(entity) in core_storage_keys,
            is_event_argument=False,
            is_storyline_anchor=True,
        )

    accepted_relations_total = len(relations)
    accepted_relations_with_both_args_materialized = 0
    accepted_relations_storyline_only = 0

    for rel in relations:
        head_entity = _relation_endpoint_entity(rel, "head", entity_index)
        tail_entity = _relation_endpoint_entity(rel, "tail", entity_index)
        if head_entity and tail_entity:
            accepted_relations_with_both_args_materialized += 1
        if not rel.get("write_to_graph", True):
            accepted_relations_storyline_only += 1

        for entity in (head_entity, tail_entity):
            if not entity:
                continue
            storage_key = _entity_storage_key(entity)
            if storage_key:
                relation_argument_storage_keys.add(storage_key)
            upsert_entity(
                entity,
                is_core=storage_key in core_storage_keys,
                is_event_argument=True,
                is_storyline_anchor=storage_key in anchor_storage_keys,
            )

    materialized_entities: list[dict] = []
    event_argument_entities_from_mentions = 0
    for storage_key, entity in prepared.items():
        signal_roles = sorted(entity.get("signal_roles", set()))
        entity["signal_roles"] = signal_roles
        if storage_key in relation_argument_storage_keys and storage_key not in core_storage_keys:
            event_argument_entities_from_mentions += 1
        materialized_entities.append(entity)

    signal_stats: dict[str, object] = {
        "core_entities_count": sum(1 for entity in materialized_entities if entity.get("is_graph_core_entity")),
        "storyline_anchor_entities_count": sum(
            1 for entity in materialized_entities if entity.get("is_storyline_anchor_entity")
        ),
        "support_entities_count": sum(
            1
            for entity in materialized_entities
            if entity.get("is_storyline_support_entity") and not entity.get("is_graph_core_entity")
        ),
        "event_argument_entities_count": len(relation_argument_storage_keys),
        "event_argument_entities_from_mentions": event_argument_entities_from_mentions,
        "accepted_relations_total": accepted_relations_total,
        "accepted_relations_with_both_args_materialized": accepted_relations_with_both_args_materialized,
        "accepted_relations_storyline_only": accepted_relations_storyline_only,
        "event_arguments_from_promote": sum(
            1 for storage_key in relation_argument_storage_keys if storage_key in core_storage_keys
        ),
        "event_arguments_from_mention": event_argument_entities_from_mentions,
        "all_relation_signatures": sorted({_relation_signature(rel) for rel in relations}),
        "storyline_relation_signatures": sorted(
            {
                _relation_signature(rel)
                for rel in relations
                if str(rel.get("utility_bucket") or "") in {"storyline_core", "storyline_support"}
            }
        ),
        "storyline_core_relation_count": sum(
            1 for rel in relations if str(rel.get("utility_bucket") or "") == "storyline_core"
        ),
        "storyline_support_relation_count": sum(
            1 for rel in relations if str(rel.get("utility_bucket") or "") == "storyline_support"
        ),
        "background_relation_count": sum(
            1 for rel in relations if str(rel.get("utility_bucket") or "") == "background_static"
        ),
        "storyline_entity_keys": sorted(
            {
                normalize_entity_text(str(entity.get("canonical_key") or entity.get("text") or ""))
                for entity in materialized_entities
                if str(entity.get("canonical_key") or entity.get("text") or "").strip()
            }
        ),
    }
    return materialized_entities, signal_stats, entity_index


async def _write_to_neo4j(
    post_id: int,
    text: str,
    timestamp: datetime | None,
    core_entities: list[dict],
    support_entities: list[dict],
    relations: list[dict],
    *,
    source_is_advertising: bool = False,
    content_gate_meta: dict | None = None,
) -> None:
    materialized_entities, signal_stats, entity_index = _prepare_story_entities(
        core_entities=core_entities,
        support_entities=support_entities,
        anchor_entities=[
            entity
            for entity in support_entities
            if entity.get("storyline_anchor_eligible") or entity.get("rag_anchor_eligible")
        ],
        relations=relations,
    )
    event_id = f"event_{post_id}"
    event_mention = build_event_mention_lite(
        post_id=post_id,
        text=text,
        timestamp=timestamp,
        entities=materialized_entities,
        relations=relations,
        entity_usage_stats=signal_stats,
    )
    event_mention_id = event_mention["id"]
    event_emb = embed_passage(text)
    ts_iso = timestamp.isoformat() if timestamp else None
    now_iso = datetime.now(timezone.utc).isoformat()
    content_gate_meta = content_gate_meta or {}
    content_type = str(content_gate_meta.get("content_type") or "news_story_candidate")
    content_state = str(content_gate_meta.get("content_state") or "seed_candidate")
    content_gate_reason = str(content_gate_meta.get("reason") or "")
    content_gate_score = float(content_gate_meta.get("gate_score", 0.0) or 0.0)
    is_storyline_candidate = bool(content_gate_meta.get("is_storyline_candidate", True))
    is_seed_candidate = bool(content_gate_meta.get("is_seed_candidate", True))
    seed_quality_score = float(content_gate_meta.get("seed_quality_score", 0.0) or 0.0)
    content_wrapper_peeled = bool(content_gate_meta.get("wrapper_peeled", False))

    async with driver.session() as session:
        # Post
        await session.run(
            """
            MERGE (p:Post {post_id: $post_id})
            SET p.published_at = $ts,
                p.summary = $summary,
                p.storyline_content_type = $storyline_content_type,
                p.storyline_content_state = $storyline_content_state,
                p.storyline_content_gate_reason = $storyline_content_gate_reason,
                p.storyline_content_gate_score = $storyline_content_gate_score,
                p.storyline_seed_quality_score = $storyline_seed_quality_score
            """,
            post_id=post_id,
            ts=ts_iso,
            summary=(text or "")[:500],
            storyline_content_type=content_type,
            storyline_content_state=content_state,
            storyline_content_gate_reason=content_gate_reason,
            storyline_content_gate_score=content_gate_score,
            storyline_seed_quality_score=seed_quality_score,
        )

        # Event + Post->Event
        await session.run(
            """
            MERGE (e:Event {id: $event_id})
            SET e.summary = $summary,
                e.embedding = $embedding,
                e.published_at = $published_at,
                e.recorded_at = $recorded_at
            WITH e
            MATCH (p:Post {post_id: $post_id})
            MERGE (p)-[:DESCRIBES]->(e)
            """,
            event_id=event_id,
            summary=text[:500],
            embedding=event_emb,
            published_at=ts_iso,
            recorded_at=now_iso,
            post_id=post_id,
        )

        # EventMention-lite + Post/Event links (dual-write, backward compatible)
        await session.run(
            """
            MERGE (em:EventMention {id: $event_mention_id})
            SET em.post_id = $post_id,
                em.event_type = $event_type,
                em.event_type_coarse = $event_type_coarse,
                em.event_subtype = $event_subtype,
                em.event_signature = $event_signature,
                em.summary = $summary,
                em.event_time = $event_time,
                em.event_time_text = $event_time_text,
                em.event_time_resolved = $event_time_resolved,
                em.update_kind = $update_kind,
                em.status = $status,
                em.actors = $actors,
                em.places = $places,
                em.participant_keys = $participant_keys,
                em.participant_role_signatures = $participant_role_signatures,
                em.location_keys = $location_keys,
                em.entities_count = $entities_count,
                em.relations_count = $relations_count,
                em.storyline_core_relation_count = $storyline_core_relation_count,
                em.storyline_support_relation_count = $storyline_support_relation_count,
                em.background_relation_count = $background_relation_count,
                em.shadow_relation_count = $shadow_relation_count,
                em.participants_json = $participants_json,
                em.confidence_breakdown_json = $confidence_breakdown_json,
                em.core_entities_count = $core_entities_count,
                em.support_entities_count = $support_entities_count,
                em.event_argument_entities_count = $event_argument_entities_count,
                em.event_argument_entities_from_mentions = $event_argument_entities_from_mentions,
                em.storyline_content_type = $storyline_content_type,
                em.storyline_content_state = $storyline_content_state,
                em.storyline_content_gate_reason = $storyline_content_gate_reason,
                em.storyline_content_gate_score = $storyline_content_gate_score,
                em.storyline_seed_quality_score = $storyline_seed_quality_score,
                em.confidence = $confidence,
                em.updated_at = $updated_at
            WITH em
            MATCH (p:Post {post_id: $post_id})
            MERGE (p)-[:DESCRIBES]->(em)
            WITH em
            MATCH (e:Event {id: $event_id})
            MERGE (e)-[:HAS_MENTION]->(em)
            """,
            event_mention_id=event_mention_id,
            event_id=event_id,
            post_id=post_id,
            event_type=event_mention.get("event_type"),
            event_type_coarse=event_mention.get("event_type_coarse"),
            event_subtype=event_mention.get("event_subtype"),
            event_signature=event_mention.get("event_signature"),
            summary=event_mention.get("summary"),
            event_time=event_mention.get("time"),
            event_time_text=event_mention.get("event_time_text"),
            event_time_resolved=event_mention.get("event_time_resolved"),
            update_kind=event_mention.get("update_kind"),
            status=event_mention.get("status"),
            actors=event_mention.get("actors"),
            places=event_mention.get("places"),
            participant_keys=event_mention.get("participant_keys"),
            participant_role_signatures=event_mention.get("participant_role_signatures"),
            location_keys=event_mention.get("location_keys"),
            entities_count=event_mention.get("entities_count"),
            relations_count=event_mention.get("relations_count"),
            storyline_core_relation_count=event_mention.get("storyline_core_relation_count", 0),
            storyline_support_relation_count=event_mention.get(
                "storyline_support_relation_count", 0
            ),
            background_relation_count=event_mention.get("background_relation_count", 0),
            shadow_relation_count=event_mention.get("shadow_relation_count", 0),
            participants_json=json.dumps(event_mention.get("participants", []), ensure_ascii=False),
            confidence_breakdown_json=json.dumps(
                event_mention.get("confidence_breakdown", {}),
                ensure_ascii=False,
            ),
            core_entities_count=event_mention.get("core_entities_count", 0),
            support_entities_count=event_mention.get("support_entities_count", 0),
            event_argument_entities_count=event_mention.get("event_argument_entities_count", 0),
            event_argument_entities_from_mentions=event_mention.get(
                "event_argument_entities_from_mentions", 0
            ),
            storyline_content_type=content_type,
            storyline_content_state=content_state,
            storyline_content_gate_reason=content_gate_reason,
            storyline_content_gate_score=content_gate_score,
            storyline_seed_quality_score=seed_quality_score,
            confidence=event_mention.get("confidence"),
            updated_at=now_iso,
        )

        # Batch entity upserts + links to reduce network round-trips to Neo4j.
        entity_rows = []
        for entity in materialized_entities:
            entity_rows.append(
                {
                    "entity_id": f"{entity['label']}_{entity['text'].lower().replace(' ', '_')}",
                    "name": entity["text"],
                    "type": entity["label"],
                    "subtype": entity.get("entity_subtype"),
                    "canonical_key": entity.get("canonical_key"),
                    "raw_text": entity.get("raw_text"),
                    "decision": entity.get("decision"),
                    "reason": entity.get("reason"),
                    "is_graph_core_entity": bool(entity.get("is_graph_core_entity")),
                    "is_storyline_anchor_entity": bool(entity.get("is_storyline_anchor_entity")),
                    "is_event_argument_entity": bool(entity.get("is_event_argument_entity")),
                    "is_storyline_support_entity": bool(entity.get("is_storyline_support_entity")),
                    "entity_support_level": entity.get("primary_signal_role"),
                }
            )

        if entity_rows:
            await session.run(
                """
                UNWIND $rows AS row
                MERGE (en:Entity {id: row.entity_id})
                ON CREATE SET
                    en.name = row.name,
                    en.type = row.type,
                    en.subtype = row.subtype,
                    en.canonical_key = row.canonical_key,
                    en.raw_text = row.raw_text,
                    en.last_seen_decision = row.decision,
                    en.last_seen_reason = row.reason,
                    en.is_graph_core_entity = row.is_graph_core_entity,
                    en.is_storyline_anchor_entity = row.is_storyline_anchor_entity,
                    en.is_event_argument_entity = row.is_event_argument_entity,
                    en.is_storyline_support_entity = row.is_storyline_support_entity
                ON MATCH SET
                    en.name = row.name,
                    en.type = row.type,
                    en.subtype = coalesce(row.subtype, en.subtype),
                    en.canonical_key = row.canonical_key,
                    en.raw_text = coalesce(en.raw_text, row.raw_text),
                    en.last_seen_decision = row.decision,
                    en.last_seen_reason = row.reason,
                    en.is_graph_core_entity = coalesce(en.is_graph_core_entity, false) OR row.is_graph_core_entity,
                    en.is_storyline_anchor_entity = coalesce(en.is_storyline_anchor_entity, false) OR row.is_storyline_anchor_entity,
                    en.is_event_argument_entity = coalesce(en.is_event_argument_entity, false) OR row.is_event_argument_entity,
                    en.is_storyline_support_entity = coalesce(en.is_storyline_support_entity, false) OR row.is_storyline_support_entity
                WITH en, row
                MATCH (ev:Event {id: $event_id})
                MERGE (ev)-[r1:INVOLVES {role: row.type}]->(en)
                SET r1.entity_support_level = row.entity_support_level,
                    r1.entity_decision = row.decision
                WITH en, row
                MATCH (em:EventMention {id: $event_mention_id})
                MERGE (em)-[r2:INVOLVES {role: row.type}]->(en)
                SET r2.entity_support_level = row.entity_support_level,
                    r2.entity_decision = row.decision
                """,
                rows=entity_rows,
                event_id=event_id,
                event_mention_id=event_mention_id,
            )

        # Batch relation upserts to reduce network round-trips to Neo4j.
        relation_rows = []
        for rel in relations:
            if not rel.get("write_to_graph", True):
                continue
            head_entity = _relation_endpoint_entity(rel, "head", entity_index)
            tail_entity = _relation_endpoint_entity(rel, "tail", entity_index)
            if not head_entity or not tail_entity:
                continue
            relation_rows.append(
                {
                    "head_id": f"{head_entity['label']}_{head_entity['text'].lower().replace(' ', '_')}",
                    "tail_id": f"{tail_entity['label']}_{tail_entity['text'].lower().replace(' ', '_')}",
                    "relation": rel["relation"],
                    "confidence": rel.get("confidence"),
                    "reason_group": rel.get("reason_group"),
                    "schema_version": rel.get("schema_version"),
                }
            )

        if relation_rows:
            await session.run(
                """
                UNWIND $rows AS row
                MATCH (h:Entity {id: row.head_id})
                MATCH (t:Entity {id: row.tail_id})
                MERGE (h)-[r:RELATED_TO {relation: row.relation}]->(t)
                ON CREATE SET
                    r.source_post_id = $post_id,
                    r.created_at = $created_at,
                    r.confidence = row.confidence,
                    r.reason_group = row.reason_group,
                    r.schema_version = row.schema_version
                ON MATCH SET
                    r.last_seen_post_id = $post_id,
                    r.updated_at = $created_at,
                    r.confidence = coalesce(row.confidence, r.confidence),
                    r.reason_group = coalesce(row.reason_group, r.reason_group),
                    r.schema_version = coalesce(row.schema_version, r.schema_version)
                """,
                rows=relation_rows,
                post_id=post_id,
                created_at=now_iso,
            )

        if is_storyline_candidate:
            storyline_meta = await resolve_and_write_storyline(
                session=session,
                post_id=post_id,
                event_mention_id=event_mention_id,
                event_time=timestamp,
                event_embedding=event_emb,
                entity_keys=set(signal_stats.get("storyline_entity_keys") or []),
                relation_signatures=set(signal_stats.get("storyline_relation_signatures") or []),
                all_relation_signatures=set(signal_stats.get("all_relation_signatures") or []),
                event_signatures=set(event_mention.get("event_signatures") or []),
                participant_role_signatures=set(event_mention.get("participant_role_signatures") or []),
                location_keys=set(event_mention.get("location_keys") or []),
                title_hint=_storyline_title_hint(materialized_entities, relations, post_id),
                source_text=text,
                source_is_advertising=source_is_advertising,
                event_confidence=event_mention.get("confidence"),
                allow_new_storyline_creation=is_seed_candidate,
            )
        else:
            storyline_meta = {
                "enabled": True,
                "storyline_id": "",
                "story_family_id": "",
                "story_family_root_storyline_id": "",
                "story_episode_id": "",
                "storyline_seed_post_id": "",
                "decision": "skipped_non_story_content",
                "score": 0.0,
                "threshold": 0.0,
                "reason": content_gate_reason or "non_story_content_gate",
                "best_candidate_storyline_id": "",
                "best_candidate_score": 0.0,
                "best_candidate_hybrid_score": 0.0,
                "best_candidate_pair_score": 0.0,
                "pair_blended_score": 0.0,
                "pair_hybrid_blend": 0.0,
                "score_source": "content_gate",
                "continuity_signal": 0.0,
                "recent_mentions_count": 0,
                "family_cluster_size": 0,
                "family_signal": 0.0,
                "family_boost": 0.0,
                "family_preferred_root": False,
                "family_oldest_seed_id": "",
                "family_root_storyline_id": "",
                "family_rerouted_to_root": False,
                "family_matcher_ready": False,
                "family_matcher_mode": "",
                "family_matcher_threshold": 0.0,
                "family_matcher_fallback_min": 0.0,
                "family_matcher_selected_root_id": "",
                "family_matcher_selected_root_score": 0.0,
                "family_matcher_best_score": 0.0,
                "family_matcher_candidate_count_before": 0,
                "family_matcher_candidate_count_after": 0,
                "components": {},
                "contributions": {},
            }

    log.info(
        "ie_signal_bridge",
        post_id=post_id,
        core_entities_count=signal_stats.get("core_entities_count"),
        support_entities_count=signal_stats.get("support_entities_count"),
        event_argument_entities_count=signal_stats.get("event_argument_entities_count"),
        event_argument_entities_from_mentions=signal_stats.get("event_argument_entities_from_mentions"),
        accepted_relations_total=signal_stats.get("accepted_relations_total"),
        accepted_relations_with_both_args_materialized=signal_stats.get(
            "accepted_relations_with_both_args_materialized"
        ),
        accepted_relations_storyline_only=signal_stats.get("accepted_relations_storyline_only"),
        event_arguments_from_promote=signal_stats.get("event_arguments_from_promote"),
        event_arguments_from_mention=signal_stats.get("event_arguments_from_mention"),
        storyline_core_relation_count=signal_stats.get("storyline_core_relation_count"),
        background_relation_count=signal_stats.get("background_relation_count"),
    )

    log.info(
        "ie_content_gate",
        post_id=post_id,
        content_type=content_type,
        content_state=content_state,
        content_gate_reason=content_gate_reason,
        content_gate_score=content_gate_score,
        is_storyline_candidate=is_storyline_candidate,
        is_seed_candidate=is_seed_candidate,
        seed_quality_score=seed_quality_score,
        content_wrapper_peeled=content_wrapper_peeled,
    )

    log.info(
        "ie_storyline_assignment",
        post_id=post_id,
        event_mention_id=event_mention_id,
        storyline_id=storyline_meta.get("storyline_id"),
        story_family_id=storyline_meta.get("story_family_id"),
        story_family_root_storyline_id=storyline_meta.get("story_family_root_storyline_id"),
        story_episode_id=storyline_meta.get("story_episode_id"),
        storyline_seed_post_id=storyline_meta.get("storyline_seed_post_id"),
        decision=storyline_meta.get("decision"),
        score=storyline_meta.get("score"),
        threshold=storyline_meta.get("threshold"),
        reason=storyline_meta.get("reason"),
        best_candidate_storyline_id=storyline_meta.get("best_candidate_storyline_id"),
        best_candidate_score=storyline_meta.get("best_candidate_score"),
        best_candidate_hybrid_score=storyline_meta.get("best_candidate_hybrid_score"),
        best_candidate_pair_score=storyline_meta.get("best_candidate_pair_score"),
        pair_blended_score=storyline_meta.get("pair_blended_score"),
        pair_hybrid_blend=storyline_meta.get("pair_hybrid_blend"),
        score_source=storyline_meta.get("score_source"),
        continuity_signal=storyline_meta.get("continuity_signal"),
        recent_mentions_count=storyline_meta.get("recent_mentions_count"),
        family_cluster_size=storyline_meta.get("family_cluster_size"),
        family_signal=storyline_meta.get("family_signal"),
        family_boost=storyline_meta.get("family_boost"),
        family_preferred_root=storyline_meta.get("family_preferred_root"),
        family_oldest_seed_id=storyline_meta.get("family_oldest_seed_id"),
        family_root_storyline_id=storyline_meta.get("family_root_storyline_id"),
        family_rerouted_to_root=storyline_meta.get("family_rerouted_to_root"),
        family_matcher_ready=storyline_meta.get("family_matcher_ready"),
        family_matcher_mode=storyline_meta.get("family_matcher_mode"),
        family_matcher_threshold=storyline_meta.get("family_matcher_threshold"),
        family_matcher_fallback_min=storyline_meta.get("family_matcher_fallback_min"),
        family_matcher_selected_root_id=storyline_meta.get("family_matcher_selected_root_id"),
        family_matcher_selected_root_score=storyline_meta.get("family_matcher_selected_root_score"),
        family_matcher_best_score=storyline_meta.get("family_matcher_best_score"),
        family_matcher_candidate_count_before=storyline_meta.get("family_matcher_candidate_count_before"),
        family_matcher_candidate_count_after=storyline_meta.get("family_matcher_candidate_count_after"),
        content_type=content_type,
        content_state=content_state,
        content_gate_reason=content_gate_reason,
        content_gate_score=content_gate_score,
        is_storyline_candidate=is_storyline_candidate,
        is_seed_candidate=is_seed_candidate,
        seed_quality_score=seed_quality_score,
        content_wrapper_peeled=content_wrapper_peeled,
        components=storyline_meta.get("components"),
        contributions=storyline_meta.get("contributions"),
        event_mention_confidence=event_mention.get("confidence"),
    )

    log.info(
        "graph_written",
        post_id=post_id,
        event_mention_id=event_mention_id,
        event_mention_type=event_mention.get("event_type"),
        event_mention_confidence=event_mention.get("confidence"),
        storyline_id=storyline_meta.get("storyline_id"),
        story_family_id=storyline_meta.get("story_family_id"),
        storyline_decision=storyline_meta.get("decision"),
        storyline_score=storyline_meta.get("score"),
        storyline_content_type=content_type,
        storyline_content_state=content_state,
        entities=len(materialized_entities),
        relations=len(relations),
        core_entities_count=signal_stats.get("core_entities_count"),
        support_entities_count=signal_stats.get("support_entities_count"),
        event_argument_entities_count=signal_stats.get("event_argument_entities_count"),
    )
# ─── Celery таски ─────────────────────────────────────────

@celery_app.task(name="ie.process_post", bind=True, max_retries=3)
def process_post(self, payload: dict):
    async def _mark_processing(post_id: int) -> bool:
        async with async_session() as session:
            result = await session.execute(
                text("""
                    UPDATE posts
                    SET ie_status = 'processing',
                        ie_started_at = now(),
                        ie_error = NULL
                    WHERE id = :post_id
                      AND ie_status = 'queued'
                    RETURNING id
                """),
                {"post_id": post_id},
            )
            row = result.first()
            await session.commit()
            return row is not None

    async def _mark_done(post_id: int) -> bool:
        async with async_session() as session:
            result = await session.execute(
                text("""
                    UPDATE posts
                    SET ie_status = 'done',
                        ie_processed_at = now(),
                        ie_error = NULL
                    WHERE id = :post_id
                      AND ie_status = 'processing'
                    RETURNING id
                """),
                {"post_id": post_id},
            )
            row = result.first()
            await session.commit()
            return row is not None

    async def _mark_failed(post_id: int, err: str) -> bool:
        async with async_session() as session:
            result = await session.execute(
                text("""
                    UPDATE posts
                    SET ie_status = 'failed',
                        ie_error = :err,
                        ie_retry_count = ie_retry_count + 1
                    WHERE id = :post_id
                      AND ie_status IN ('queued', 'processing')
                    RETURNING id
                """),
                {"post_id": post_id, "err": err[:2000]},
            )
            row = result.first()
            await session.commit()
            return row is not None

    async def _work():
        post_id = payload.get("post_id")
        text_ = payload.get("text")
        raw_text = payload.get("raw_text") or text_
        content_gate_meta = classify_storyline_content(raw_text or text_ or "")
        pipeline_text = str(content_gate_meta.get("pipeline_text") or sanitize_pipeline_text(raw_text or text_ or ""))
        source_is_advertising = (
            has_ad_marker(raw_text or text_ or "")
            or is_promotional_text(raw_text or text_ or "")
            or str(content_gate_meta.get("content_type") or "") == "promo_or_partner"
        )
        ts_raw = payload.get("timestamp")
        domains = payload.get("domains", [])
        timestamp = datetime.fromisoformat(ts_raw) if ts_raw else None

        if not post_id:
            return None

        claimed = await _mark_processing(post_id)
        if not claimed:
            log.info("ie_skip_already_claimed", post_id=post_id)
            return None

        if not text_:
            await _mark_failed(post_id, "empty_text_payload")
            log.warning("ie_skip_empty_text", post_id=post_id)
            return None

        extra_types = []
        for domain in domains:
            extra_types.extend(DOMAIN_ENTITY_TYPES.get(domain, []))

        clean_text = clean_for_ner(pipeline_text)
        clean_raw_text = clean_for_re(raw_text)

        raw_entities = extract_entities(clean_text, extra_types)
        promoted_entities, kept_mentions, rejected_candidates = partition_entities(
            raw_entities,
            full_text=pipeline_text,
            post_id=post_id,
        )
        all_candidates = promoted_entities + kept_mentions + rejected_candidates

        rule_counts = Counter(
            c.get("rule_decision", c.get("decision", "unknown")) for c in all_candidates
        )
        final_counts = Counter(c.get("decision", "unknown") for c in all_candidates)
        reason_counts = Counter(c.get("reason", "unknown") for c in all_candidates)
        override_count = sum(1 for c in all_candidates if c.get("override_applied"))
        model_scored_count = sum(
            1 for c in all_candidates if c.get("decision_model_proba") is not None
        )

        log.info(
            "ie_entity_decision_summary",
            post_id=post_id,
            candidates_total=len(all_candidates),
            model_scored_count=model_scored_count,
            model_scored_ratio=round(model_scored_count / len(all_candidates), 4)
            if all_candidates
            else 0.0,
            override_count=override_count,
            rule_counts=dict(rule_counts),
            final_counts=dict(final_counts),
            top_final_reasons=reason_counts.most_common(10),
        )

        relation_link_entities = promoted_entities + kept_mentions
        raw_relations = extract_relations(clean_raw_text, relation_link_entities)
        rule_relations, rejected_relations = filter_relations(raw_relations, relation_link_entities)
        relations, model_rejected_relations, model_gate_meta = apply_relation_model_gate(rule_relations)
        rejected_relations = rejected_relations + model_rejected_relations

        log.info(
            "ie_entities_promoted",
            post_id=post_id,
            ents=[
                {
                    "text": e["text"],
                    "label": e["label"],
                    "reason": e.get("reason"),
                    "canonical_key": e.get("canonical_key"),
                }
                for e in promoted_entities
            ],
        )

        log.info(
            "ie_entities_mentions",
            post_id=post_id,
            mentions=[
                {
                    "text": e["text"],
                    "label": e["label"],
                    "reason": e.get("reason"),
                }
                for e in kept_mentions[:50]
            ],
            mentions_count=len(kept_mentions),
        )

        log.info(
            "ie_entities_rejected",
            post_id=post_id,
            rejected=[
                {
                    "text": e["text"],
                    "label": e["label"],
                    "reason": e.get("reason"),
                }
                for e in rejected_candidates[:50]
            ],
            rejected_count=len(rejected_candidates),
        )

        log.info(
            "ie_relations",
            post_id=post_id,
            candidate_count_before_filter=len(raw_relations),
            rule_accept_count=len(rule_relations),
            model_gate_enabled=model_gate_meta.get("enabled"),
            model_gate_threshold=model_gate_meta.get("threshold"),
            model_gate_accepted_count=model_gate_meta.get("accepted_relations"),
            model_gate_rejected_count=model_gate_meta.get("rejected_relations"),
            provisional_count=sum(1 for r in relations if r.get("provisional")),
            graph_writable_count=sum(1 for r in relations if r.get("write_to_graph", True)),
            rels=relations,
        )

        log.info(
            "ie_relations_rejected",
            post_id=post_id,
            rejected=rejected_relations[:RELATION_REJECT_LOG_SAMPLE],
            rejected_count=len(rejected_relations),
        )

        await _write_to_neo4j(
            post_id,
            pipeline_text or text_,
            timestamp,
            promoted_entities,
            relation_link_entities,
            relations,
            source_is_advertising=source_is_advertising,
            content_gate_meta=content_gate_meta,
        )

        done = await _mark_done(post_id)
        if not done:
            log.warning("ie_done_status_not_updated", post_id=post_id)
        else:
            dispatch_result = celery_app.send_task(
                "app.tasks.dispatch_storyline_updates_for_post",
                args=[post_id],
                queue=DISPATCH_QUEUE,
            )
            log.info(
                "storyline_dispatch_enqueued_after_ie",
                post_id=post_id,
                queue=DISPATCH_QUEUE,
                task_id=getattr(dispatch_result, "id", None),
            )

        return post_id

    post_id = payload.get("post_id")
    try:
        return _run_async(_work())
    except Exception as exc:
        if post_id:
            _run_async(_mark_failed(post_id, str(exc)))
        log.error("ie_pipeline_error", post_id=post_id, exc=str(exc))
        raise self.retry(exc=exc, countdown=120)

@celery_app.task(name="ie.scan_unprocessed_posts")
def scan_unprocessed_posts():
    return run_ie_scan_once()

def run_ie_scan_once() -> int:
    min_post_id = int(os.getenv("IE_SCAN_MIN_POST_ID", "0"))
    scan_batch_size = max(1, int(os.getenv("IE_SCAN_BATCH_SIZE", "10")))

    async def _scan():
        async with async_session() as session:
            rows = (
                await session.execute(
                    text("""
                        WITH picked AS (
                            SELECT id
                            FROM posts
                            WHERE id > :min_post_id
                              AND content IS NOT NULL
                              AND (
                                ie_status IS NULL
                                OR ie_status = 'pending'
                                OR (ie_status = 'failed' AND ie_retry_count < 3)
                              )
                            ORDER BY id ASC
                            LIMIT :scan_batch_size
                            FOR UPDATE SKIP LOCKED
                        )
                        UPDATE posts p
                        SET ie_status = 'queued',
                            ie_enqueued_at = now()
                        FROM picked
                        WHERE p.id = picked.id
                        RETURNING p.id, p.processed_content, p.content, p.timestamp
                    """),
                    {"min_post_id": min_post_id, "scan_batch_size": scan_batch_size},
                )
            ).all()
            await session.commit()

        queued = 0
        for post_id, processed_content, content, timestamp in rows:
            process_post.apply_async(
                args=[{
                    "post_id": post_id,
                    "text": processed_content or content,
                    "raw_text": content,
                    "timestamp": timestamp.isoformat() if timestamp else None,
                }],
                queue="ie_queue",
            )
            queued += 1

        log.info(
            "ie_scan_done",
            queued=queued,
            min_post_id=min_post_id,
            scan_batch_size=scan_batch_size,
        )
        return queued

    return _run_async(_scan())
