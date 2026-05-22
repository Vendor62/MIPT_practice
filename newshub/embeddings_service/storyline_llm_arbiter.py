from __future__ import annotations

import json
import os
from typing import Any

import structlog

try:
    import aiohttp
except ModuleNotFoundError:  # pragma: no cover - lightweight test fallback
    class _AiohttpFallback:
        class ClientError(Exception):
            pass

        class ClientTimeout:
            def __init__(self, total=None):
                self.total = total

        class ClientSession:
            def __init__(self, *args, **kwargs):
                self.closed = False

            async def close(self):
                self.closed = True

            def post(self, *args, **kwargs):
                raise RuntimeError("aiohttp is not installed")

    aiohttp = _AiohttpFallback()


log = structlog.get_logger()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "45"))
STORYLINE_LLM_ARBITER_TIMEOUT = int(os.getenv("STORYLINE_LLM_ARBITER_TIMEOUT", "20"))

_session: aiohttp.ClientSession | None = None


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=min(DEEPSEEK_TIMEOUT, STORYLINE_LLM_ARBITER_TIMEOUT))
        _session = aiohttp.ClientSession(timeout=timeout)
    return _session


async def _post_chat(payload: dict[str, Any]) -> dict[str, Any]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    session = await _get_session()
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload) as resp:
        body_text = await resp.text()
        if resp.status >= 500 or resp.status == 429:
            raise RuntimeError(f"DeepSeek HTTP {resp.status}: {body_text[:500]}")
        if resp.status >= 400:
            raise RuntimeError(f"DeepSeek HTTP {resp.status}: {body_text[:500]}")
        try:
            return json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"DeepSeek invalid JSON: {body_text[:300]}") from exc


def _extract_json_object(content: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


async def arbitrate_storyline_candidate(
    *,
    post_text: str,
    title_hint: str | None,
    best_candidate: dict[str, Any],
    decision_score: float,
    assign_threshold: float,
) -> dict[str, Any] | None:
    if not DEEPSEEK_API_KEY:
        return None

    recent_mentions = []
    for item in (best_candidate.get("recent_mentions") or [])[:3]:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or "").strip()
        if summary:
            recent_mentions.append(summary[:220])

    comps = dict(best_candidate.get("components") or {})
    compact_candidate = {
        "storyline_id": str(best_candidate.get("storyline_id") or "").strip()[:80],
        "family_root_storyline_id": str(best_candidate.get("family_root_storyline_id") or "").strip()[:80],
        "storyline_title": str(best_candidate.get("title") or "").strip()[:180],
        "storyline_seed_preview": str(best_candidate.get("seed_preview") or "").strip()[:400],
        "posts_count": int(best_candidate.get("candidate_posts_count") or 0),
        "score_source": str(best_candidate.get("score_source") or "").strip()[:40],
        "decision_score": round(float(decision_score or 0.0), 4),
        "assign_threshold": round(float(assign_threshold or 0.0), 4),
        "family_signal": round(float(best_candidate.get("family_signal", 0.0) or 0.0), 4),
        "recent_mentions": recent_mentions,
        "components": {
            "embedding_similarity": round(float(comps.get("embedding_similarity", 0.0) or 0.0), 4),
            "entity_overlap": round(float(comps.get("entity_overlap", 0.0) or 0.0), 4),
            "relation_overlap": round(float(comps.get("relation_overlap", 0.0) or 0.0), 4),
            "event_overlap": round(float(comps.get("event_overlap", 0.0) or 0.0), 4),
            "duplicate_text_overlap": round(float(comps.get("duplicate_text_overlap", 0.0) or 0.0), 4),
            "tfidf_post_seed_cosine": round(float(comps.get("tfidf_post_seed_cosine", 0.0) or 0.0), 4),
            "tfidf_post_peer_max_cosine": round(float(comps.get("tfidf_post_peer_max_cosine", 0.0) or 0.0), 4),
            "family_signature_overlap": round(float(comps.get("family_signature_overlap", 0.0) or 0.0), 4),
            "topic_signature_overlap": round(float(comps.get("topic_signature_overlap", 0.0) or 0.0), 4),
            "exemplar_role_overlap": round(float(comps.get("exemplar_role_overlap", 0.0) or 0.0), 4),
            "exemplar_event_frame_overlap": round(float(comps.get("exemplar_event_frame_overlap", 0.0) or 0.0), 4),
        },
    }

    system = (
        "Ты арбитр story tracking для новостных постов.\n"
        "Нужно решить только один вопрос: является ли новый пост продолжением уже существующего сюжета.\n"
        "Смотри на причинно-смысловую непрерывность сюжета, а не только на общую тему.\n"
        "Если это тот же инцидент, тот же проект, та же операция, то same_story=true.\n"
        "Если это просто похожая тема, похожая отрасль или пересечение по общим сущностям, то same_story=false.\n"
        "Не придумывай фактов. Верни только JSON без markdown.\n"
        "Схема: "
        "{\"same_story\":true,\"confidence\":0.0,\"reason\":\"...\"}"
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "post_title": str(title_hint or "").strip()[:200],
                        "post_text": str(post_text or "").strip()[:1800],
                        "candidate_storyline": compact_candidate,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 260,
        "response_format": {"type": "json_object"},
    }
    data = await _post_chat(payload)
    content = str((((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or "").strip()
    result = _extract_json_object(content)
    if not isinstance(result, dict):
        log.warning("storyline_llm_arbiter.invalid_json", body_preview=content[:300])
        return None
    return result
