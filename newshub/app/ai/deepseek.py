import os
import re
import json
import asyncio
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

DEEPSEEK_GROUP_MAX_TOKENS = int(os.getenv("DEEPSEEK_GROUP_MAX_TOKENS", "900"))
DEEPSEEK_INSTRUCTION_MAX_TOKENS = int(os.getenv("DEEPSEEK_INSTRUCTION_MAX_TOKENS", "220"))
DEEPSEEK_INSTRUCTION_CONTENT_MAX_CHARS = int(os.getenv("DEEPSEEK_INSTRUCTION_CONTENT_MAX_CHARS", "1800"))
DEEPSEEK_BETA_API_URL = os.getenv("DEEPSEEK_BETA_API_URL", "https://api.deepseek.com/beta/chat/completions")
DEEPSEEK_ROUTER_MAX_TOKENS = int(os.getenv("DEEPSEEK_ROUTER_MAX_TOKENS", "420"))
DEEPSEEK_ROUTER_CONTEXT_MAX_CHARS = int(os.getenv("DEEPSEEK_ROUTER_CONTEXT_MAX_CHARS", "2400"))

# Thinking mode (DeepSeek V4 Pro)
DEEPSEEK_THINKING_MODEL = os.getenv("DEEPSEEK_THINKING_MODEL", "deepseek-v4-pro")
DEEPSEEK_GROUP_THINKING_ENABLED = os.getenv("DEEPSEEK_GROUP_THINKING_ENABLED", "true").lower() == "true"
# Отдельный таймаут для группировки — thinking mode думает дольше обычного
DEEPSEEK_GROUP_TIMEOUT = int(os.getenv("DEEPSEEK_GROUP_TIMEOUT", "300"))


def _parse_optional_timeout_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"", "0", "none", "null", "off", "false", "disabled"}:
        return 0
    return int(value)


# 0/none/off disables the per-request client timeout for thinking merge.
DEEPSEEK_GROUP_TIMEOUT_A1 = _parse_optional_timeout_env("DEEPSEEK_GROUP_TIMEOUT_A1", DEEPSEEK_GROUP_TIMEOUT)
DEEPSEEK_GROUP_TIMEOUT_A2 = _parse_optional_timeout_env(
    "DEEPSEEK_GROUP_TIMEOUT_A2", max(240, min(DEEPSEEK_GROUP_TIMEOUT, 300))
)
DEEPSEEK_GROUP_TIMEOUT_A3 = _parse_optional_timeout_env(
    "DEEPSEEK_GROUP_TIMEOUT_A3", max(180, min(DEEPSEEK_GROUP_TIMEOUT, 240))
)
DEEPSEEK_GROUP_THINKING_ATTEMPTS = int(os.getenv("DEEPSEEK_GROUP_THINKING_ATTEMPTS", "3"))

_session: aiohttp.ClientSession | None = None


class DeepseekRetryableError(RuntimeError):
    pass


class DeepseekAuthError(RuntimeError):
    pass


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=DEEPSEEK_TIMEOUT)
        _session = aiohttp.ClientSession(timeout=timeout)
    return _session


async def close_session() -> None:
    global _session
    if _session and not _session.closed:
        await _session.close()
    _session = None


async def arbitrate_storyline_follow_target(
    *,
    post_text: str,
    assigned_context: dict | None,
    recovery_candidates: list[dict],
) -> dict | None:
    if not DEEPSEEK_API_KEY:
        log.warning("DEEPSEEK_API_KEY not set, skipping storyline follow target arbitration")
        return None
    if not recovery_candidates:
        return None

    compact_candidates: list[dict] = []
    for item in recovery_candidates[:8]:
        compact_candidates.append(
            {
                "storyline_id": str(item.get("storyline_id") or "").strip()[:80],
                "title": str(item.get("title") or item.get("storyline_title") or "").strip()[:160],
                "seed_preview": str(item.get("seed_preview") or item.get("storyline_seed_preview") or "").strip()[:320],
                "posts_count": int(item.get("posts_count") or 0),
                "story_family_id": str(item.get("story_family_id") or "").strip()[:80],
                "family_root_storyline_id": str(item.get("family_root_storyline_id") or "").strip()[:80],
                "token_overlap": int(item.get("token_overlap") or 0),
                "entity_token_overlap": int(item.get("entity_token_overlap") or 0),
                "signature_token_overlap": int(item.get("signature_token_overlap") or 0),
                "recovery_score": int(item.get("recovery_score") or 0),
                "anchor_trust_score": float(item.get("anchor_trust_score") or 0.0),
            }
        )

    assigned_payload = {
        "storyline_id": str((assigned_context or {}).get("storyline_id") or "").strip()[:80],
        "storyline_title": str((assigned_context or {}).get("storyline_title") or "").strip()[:160],
        "storyline_seed_preview": str((assigned_context or {}).get("storyline_seed_preview") or "").strip()[:320],
        "story_family_id": str((assigned_context or {}).get("story_family_id") or "").strip()[:80],
        "family_root_storyline_id": str((assigned_context or {}).get("family_root_storyline_id") or "").strip()[:80],
        "anchor_trust_score": float((assigned_context or {}).get("anchor_trust_score") or 0.0),
    }

    system = (
        "Ты арбитр новостного сторитрекинга.\n"
        "На входе текст конкретного поста, уже назначенный storyline и несколько recovery-кандидатов.\n"
        "Нужно определить, соответствует ли assigned storyline тексту поста.\n"
        "Если не соответствует, выбери лучший recovery candidate.\n"
        "Не ориентируйся на общие сущности вроде городов или стран, ищи именно тот же сюжет.\n"
        "Верни ТОЛЬКО валидный JSON без markdown.\n"
        "Схема: {\"assigned_is_valid\":true,\"selected_storyline_id\":\"storyline_123\",\"reason\":\"...\"}\n"
        "Если ни один кандидат не подходит, оставь selected_storyline_id пустым.\n"
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "post_text": str(post_text or "").strip()[:1500],
                        "assigned_context": assigned_payload,
                        "recovery_candidates": compact_candidates,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }
    data = await _post_chat(payload)
    content = data["choices"][0]["message"]["content"].strip()
    return _extract_balanced_json_object(content)


async def arbitrate_user_storyline_follow_match(
    *,
    post_text: str,
    assigned_context: dict | None,
    follow_candidates: list[dict],
) -> dict | None:
    if not DEEPSEEK_API_KEY:
        log.warning("DEEPSEEK_API_KEY not set, skipping user storyline follow arbitration")
        return None
    if not follow_candidates:
        return None

    compact_candidates: list[dict] = []
    for item in follow_candidates[:6]:
        compact_candidates.append(
            {
                "family_root_storyline_id": str(item.get("family_root_storyline_id") or "").strip()[:80],
                "storyline_id": str(item.get("storyline_id") or "").strip()[:80],
                "user_follow_title": str(item.get("user_follow_title") or "").strip()[:180],
                "storyline_title": str(item.get("storyline_title") or item.get("title") or "").strip()[:180],
                "storyline_seed_preview": str(item.get("storyline_seed_preview") or item.get("seed_preview") or "").strip()[:360],
                "macro_topic_title": str(item.get("macro_topic_title") or "").strip()[:160],
                "posts_count": int(item.get("posts_count") or 0),
                "score": round(float(item.get("score") or 0.0), 4),
                "marker_hits": int(item.get("marker_hits") or 0),
                "post_token_overlap": int(item.get("post_token_overlap") or 0),
                "assigned_entity_overlap": int(item.get("assigned_entity_overlap") or 0),
                "assigned_signature_overlap": int(item.get("assigned_signature_overlap") or 0),
                "assigned_text_overlap": int(item.get("assigned_text_overlap") or 0),
                "focus_title_tokens": list(item.get("focus_title_tokens") or [])[:12],
                "focus_hits": list(item.get("focus_hits") or [])[:12],
                "focus_strong_hits": list(item.get("focus_strong_hits") or [])[:12],
                "focus_broad_hits": list(item.get("focus_broad_hits") or [])[:12],
            }
        )

    assigned_payload = {
        "storyline_id": str((assigned_context or {}).get("storyline_id") or "").strip()[:80],
        "storyline_title": str((assigned_context or {}).get("storyline_title") or "").strip()[:180],
        "storyline_seed_preview": str((assigned_context or {}).get("storyline_seed_preview") or "").strip()[:320],
        "family_root_storyline_id": str((assigned_context or {}).get("family_root_storyline_id") or "").strip()[:80],
        "macro_topic_title": str((assigned_context or {}).get("macro_topic_title") or "").strip()[:160],
    }

    system = (
        "Ты арбитр подписок на новостные сюжеты.\n"
        "На входе текст нового поста, storyline, к которому он уже прикреплён системой, и shortlist сюжетов, на которые подписан пользователь.\n"
        "Нужно ответить на вопрос: является ли новый пост продолжением одного или нескольких подписанных пользователем сюжетов.\n"
        "Смотри на причинно-смысловую непрерывность, конкретный проект, инцидент, объект, программу, операцию.\n"
        "Не опирайся только на общий канал, страну, войну, географию или широкую тему.\n"
        "Если у кандидата есть user_follow_title, это название, выбранное/подтвержденное пользователем; считай его главным смыслом подписки.\n"
        "Если user_follow_title конфликтует с graph storyline_title или seed_preview, не матчь пост только по graph title/общим сущностям.\n"
        "Сначала мысленно выдели главное событие нового поста и главное событие user_follow_title.\n"
        "Матч разрешен только если совпадает именно ядро события: тот же конфликт/дело/решение/этап/объект, а не просто общий актор или страна.\n"
        "Если пост формально про переговоры/заявление третьих сторон, но предмет переговоров явно тот же объект подписки (например программа, конфликт, санкции, перемирие), это может быть продолжением.\n"
        "Упоминание якоря подписки как фонового пункта в посте про другой главный сюжет не считается продолжением.\n"
        "Если focus_hits состоят только из слишком широких focus_broad_hits и нет focus_strong_hits, будь особенно строгим и обычно возвращай matched=false.\n"
        "Если подходят несколько подписанных сюжетов, выбери все подходящие family_root_storyline_id.\n"
        "Если ни один подписанный сюжет не подходит, selected_family_root_storyline_ids должен быть пустым списком, а selected_family_root_storyline_id пустой строкой.\n"
        "Верни ТОЛЬКО валидный JSON без markdown.\n"
        "В reason кратко укажи post_main_event, follow_main_event и почему это same/different.\n"
        "Схема: {\"matched\":true,\"selected_family_root_storyline_ids\":[\"storyline_123\"],\"selected_family_root_storyline_id\":\"storyline_123\",\"confidence\":0.0,\"reason\":\"post_main_event=...; follow_main_event=...; ...\"}\n"
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "post_text": str(post_text or "").strip()[:1800],
                        "assigned_storyline": assigned_payload,
                        "follow_candidates": compact_candidates,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 520,
        "response_format": {"type": "json_object"},
    }
    data = await _post_chat(payload)
    content = data["choices"][0]["message"]["content"].strip()
    return _extract_balanced_json_object(content)


async def summarize_related_storyline_branches(
    *,
    anchor: dict,
    candidates: list[dict],
    max_branches: int = 5,
) -> dict | None:
    if not DEEPSEEK_API_KEY:
        log.warning("DEEPSEEK_API_KEY not set, skipping related storyline branch summarization")
        return None
    if not candidates:
        return None

    compact_candidates: list[dict] = []
    for item in candidates[:12]:
        compact_candidates.append(
            {
                "storyline_id": str(item.get("storyline_id") or "").strip()[:80],
                "family_root_storyline_id": str(item.get("family_root_storyline_id") or "").strip()[:80],
                "story_family_id": str(item.get("story_family_id") or "").strip()[:80],
                "title": str(item.get("title") or item.get("storyline_title") or "").strip()[:180],
                "seed_preview": str(item.get("seed_preview") or item.get("storyline_seed_preview") or "").strip()[:420],
                "macro_topic_title": str(item.get("macro_topic_title") or "").strip()[:180],
                "posts_count": int(item.get("posts_count") or 0),
                "related_score": round(float(item.get("related_score") or 0.0), 4),
                "same_macro_topic": int(item.get("same_macro_topic") or 0),
                "family_match": int(item.get("family_match") or 0),
                "shared_entity_count": int(item.get("shared_entity_count") or 0),
                "shared_signature_count": int(item.get("shared_signature_count") or 0),
                "shared_topic_count": int(item.get("shared_topic_count") or 0),
                "matched_tokens": [str(x or "").strip()[:40] for x in (item.get("matched_tokens") or [])[:8]],
                "example_posts": [str(x or "").strip()[:520] for x in (item.get("example_posts") or [])[:3] if str(x or "").strip()],
            }
        )

    anchor_payload = {
        "storyline_id": str((anchor or {}).get("storyline_id") or "").strip()[:80],
        "family_root_storyline_id": str((anchor or {}).get("family_root_storyline_id") or "").strip()[:80],
        "storyline_title": str((anchor or {}).get("storyline_title") or "").strip()[:180],
        "storyline_seed_preview": str((anchor or {}).get("storyline_seed_preview") or "").strip()[:420],
        "focus_post_text": str((anchor or {}).get("focus_post_text") or "").strip()[:900],
        "macro_topic_title": str((anchor or {}).get("macro_topic_title") or "").strip()[:180],
        "story_episode_title": str((anchor or {}).get("story_episode_title") or "").strip()[:180],
    }

    system = (
        "Ты редактор навигации по новостному графу.\n"
        "На входе текущий сюжет и shortlist соседних storyline из графа.\n"
        "focus_post_text внутри anchor_storyline — это конкретная карточка, под которой пользователь нажал кнопку; считай ее главным контекстом запроса.\n"
        "Нужно выбрать похожие или соседние ветки, которые помогают лучше разобраться в текущем сюжете.\n"
        "Это НЕ задача найти тот же самый сюжет: дубли и очередные procedural updates того же кейса нужно отбрасывать.\n"
        "Не показывай как соседние сюжеты новые детали того же дела/обыска/ареста/залога/суда/заявления адвоката, если это продолжение той же истории с теми же главными фигурантами.\n"
        "Хорошие соседние ветки — это последствия, политический контекст, другие затронутые персонажи, смежные расследования, реакция институтов или отдельная линия внутри той же большой темы.\n"
        "Не включай кандидата, если связь только по широкой стране, городу, ведомству, персоне или общей теме без конкретной смысловой связи.\n"
        "Предпочитай кандидатов, которые связаны с focus_post_text конкретными обстоятельствами, участниками, решениями, последствиями или временной линией.\n"
        "Если несколько кандидатов описывают один смысл, оставь один лучший.\n"
        "Если кандидатов несколько и они не явные дубли, выбери 3-5 веток; не возвращай только одну ветку при наличии нескольких осмысленных кандидатов.\n"
        "Для каждого выбранного кандидата придумай новый человеческий title сам: не копируй технические title вида '[general]', не оставляй служебные пометки.\n"
        "description должен быть пересказом сути соседнего сюжета, а не цитатой или обрезанным куском одного поста.\n"
        "Не смешивай в title и description несколько независимых апдейтов; выдели центральную линию кандидата.\n"
        "Если example_posts внутри одного кандидата выглядят как набор разных новостей, выбери центральную линию по seed_preview/title и не добавляй посторонние детали.\n"
        "why_related используй только для внутреннего объяснения: кратко и без цитирования macro_topic_title.\n"
        "Опирайся на seed_preview и example_posts, но сжимай их в 1-2 предложения.\n"
        "Не придумывай фактов вне переданных title/seed_preview/macro_topic_title/example_posts.\n"
        "Пиши по-русски, нейтрально, без кликбейта.\n"
        "Если хороший соседний сюжет только один, верни один; если есть несколько осмысленных, верни до max_branches.\n"
        "Верни ТОЛЬКО валидный JSON без markdown.\n"
        "relation_type должен быть одним из: same_family, same_macro_topic, parallel_development, shared_entities, background_context.\n"
        "Схема: {\"branches\":[{\"storyline_id\":\"...\",\"family_root_storyline_id\":\"...\",\"title\":\"...\",\"description\":\"...\",\"relation_type\":\"same_macro_topic\",\"why_related\":\"...\",\"confidence\":0.0}]}\n"
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "anchor_storyline": anchor_payload,
                        "candidate_storylines": compact_candidates,
                        "max_branches": max(1, min(int(max_branches or 5), 6)),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
    }
    data = await _post_chat(payload)
    content = data["choices"][0]["message"]["content"].strip()
    obj = _extract_balanced_json_object(content)
    if not isinstance(obj, dict):
        return None
    branches = obj.get("branches")
    if not isinstance(branches, list):
        return None
    return {"branches": branches[: max(1, min(int(max_branches or 5), 6))]}


async def generate_storyline_title(
    *,
    seed_post_text: str,
    current_title: str | None = None,
) -> str | None:
    if not DEEPSEEK_API_KEY:
        return None
    text = str(seed_post_text or "").strip()
    if not text:
        return None

    system = (
        "Ты придумываешь короткие человеческие названия новостных сюжетов.\n"
        "На входе сид-пост сюжета и, возможно, текущее сыроватое системное название.\n"
        "Нужно придумать короткое, понятное, нейтральное название сюжета на русском языке.\n"
        "Название должно быть конкретным, без кликбейта, обычно 2-6 слов.\n"
        "Избегай кавычек, двоеточий, служебных пометок вроде [general], слов storyline/story/сюжет.\n"
        "Не придумывай фактов, которых нет в тексте.\n"
        "Верни ТОЛЬКО валидный JSON без markdown.\n"
        "Схема: {\"title\":\"...\"}\n"
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "seed_post_text": text[:2200],
                        "current_title": str(current_title or "").strip()[:180],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 120,
        "response_format": {"type": "json_object"},
    }
    data = await _post_chat(payload)
    content = data["choices"][0]["message"]["content"].strip()
    obj = _extract_balanced_json_object(content) or {}
    title = str(obj.get("title") or "").strip()
    if not title:
        return None
    return re.sub(r"\s+", " ", title).strip()[:120]


async def extract_storyline_anchor_profile(*, post_text: str) -> dict | None:
    if not DEEPSEEK_API_KEY:
        log.warning("DEEPSEEK_API_KEY not set, skipping storyline anchor profile extraction")
        return None
    text = str(post_text or "").strip()
    if not text:
        return None

    system = (
        "Ты помощник новостного сторитрекинга.\n"
        "На входе текст одного новостного поста.\n"
        "Нужно извлечь компактный retrieval profile для поиска связанных публикаций по тому же сюжету.\n"
        "Если пост относится к более широкому долгоживущему сюжету, программе, кампании, войне или проекту, выдели этот macro-topic отдельно.\n"
        "Но если пост стоит на пересечении широкого фона и более узкой конкретной ветки, отдавай приоритет узкой ветке, которая лучше помогает найти похожие публикации по тому же конкретному смыслу.\n"
        "Не выбирай слишком широкий macro-topic вроде просто 'война', 'конфликт' или 'эскалация', если в тексте есть более конкретная ветка: блокировка Telegram, новая волна мобилизации, миссия программы, конкретный закон, конкретный инцидент.\n"
        "Отделяй core-topic от побочных упоминаний.\n"
        "Не добавляй фактов, которых нет в тексте.\n"
        "Верни ТОЛЬКО валидный JSON без markdown.\n"
        "Схема:\n"
        "{"
        "\"story_title\":\"...\","
        "\"core_summary\":\"...\","
        "\"macro_topic\":\"...\","
        "\"topic_terms\":[\"...\"],"
        "\"topic_aliases\":[\"...\"],"
        "\"must_terms\":[\"...\"],"
        "\"supporting_terms\":[\"...\"],"
        "\"downweight_terms\":[\"...\"],"
        "\"search_phrases\":[\"...\"]"
        "}\n"
        "Требования:\n"
        "- macro_topic: пусто или короткое имя именно того уровня сюжета, который полезен для поиска похожих публикаций; не делай его слишком широким\n"
        "- topic_terms: 1-6 терминов полезной сюжетной ветки, а не только самого широкого фона\n"
        "- topic_aliases: 0-6 альтернативных названий/вариантов записи этой сюжетной ветки\n"
        "- must_terms: 2-6 самых важных якорей сюжета\n"
        "- supporting_terms: 2-8 дополнительных уточняющих терминов\n"
        "- downweight_terms: широкие или побочные термины, по которым нельзя тащить сюжет в одиночку\n"
        "- search_phrases: 1-4 короткие фразы для поиска того же события\n"
    )

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text[:2500]},
        ],
        "temperature": 0.1,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }
    data = await _post_chat(payload)
    content = data["choices"][0]["message"]["content"].strip()
    return _extract_balanced_json_object(content)


async def evaluate_instruction_filter(
    *,
    source_title: str | None,
    source_link: str | None,
    post_title: str | None,
    post_text: str | None,
    prompt_text: str,
) -> dict | None:
    if not DEEPSEEK_API_KEY:
        raise DeepseekAuthError("DEEPSEEK_API_KEY not set")

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты AI-фильтр live-ленты новостей.\n"
                    "На входе инструкция пользователя и один пост.\n"
                    "Нужно решить, соответствует ли пост инструкции пользователя.\n"
                    "Верни ТОЛЬКО JSON без markdown.\n"
                    "Схема: {\"decision\":\"allow\"|\"block\",\"reason_short\":\"...\"}\n"
                    "reason_short должен быть коротким, понятным и без технического жаргона.\n"
                    "Если сомневаешься, ориентируйся строго на смысл инструкции пользователя.\n"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_prompt": str(prompt_text or "").strip(),
                        "post": {
                            "source": str(source_title or "").strip()[:200],
                            "source_link": str(source_link or "").strip()[:300],
                            "title": str(post_title or "").strip()[:300],
                            "content": str(post_text or "").strip()[:DEEPSEEK_INSTRUCTION_CONTENT_MAX_CHARS],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": DEEPSEEK_INSTRUCTION_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    data = await _post_chat(payload)
    content = data["choices"][0]["message"]["content"].strip()
    result = _extract_balanced_json_object(content)
    if not isinstance(result, dict):
        raise RuntimeError("DeepSeek returned empty instruction-filter payload")
    return result


def _router_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "resolve_bot_action",
            "description": "Resolve a Telegram bot settings or FAQ request into one allowlisted intent with structured slots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [
                            "show_settings_overview",
                            "show_subscriptions",
                            "add_subscriptions",
                            "remove_subscriptions",
                            "set_feed_filter",
                            "toggle_forwarding",
                            "toggle_summary",
                            "digest_enable",
                            "digest_disable",
                            "digest_send_now",
                            "digest_set_time",
                            "digest_set_offset",
                            "set_language",
                            "show_billing",
                            "show_help_topic",
                            "set_global_instruction_filter",
                        ],
                    },
                    "slots": {
                        "type": "object",
                        "properties": {
                            "filter_mode": {"type": "string"},
                            "time_hhmm": {"type": "string"},
                            "utc_offset": {"type": "string"},
                            "language_code": {"type": "string"},
                            "links": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "handles": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "value_bool": {"type": "boolean"},
                            "prompt_text": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                    "needs_clarification": {"type": "boolean"},
                    "clarify_question": {"type": "string"},
                    "faq_topic": {
                        "type": "string",
                        "enum": ["", "general", "settings", "subscriptions", "billing", "forwarding", "filters", "digest", "language", "summary", "ai_filter", "storyline_tracking"],
                    },
                    "proposed_user_message": {"type": "string"},
                },
                "required": [
                    "intent",
                    "slots",
                    "needs_clarification",
                    "clarify_question",
                    "faq_topic",
                    "proposed_user_message",
                ],
                "additionalProperties": False,
            },
        },
    }


def _router_system_prompt(*, retry_mode: bool = False) -> str:
    retry_line = (
        "The previous attempt was invalid. Return exactly one valid JSON object matching the schema.\n"
        if retry_mode
        else ""
    )
    return (
        "You are the NLU router for a Telegram bot that manages settings, subscriptions, billing status, and FAQ.\n"
        "Decide only one allowlisted intent.\n"
        "Never invent database changes, SQL, payment creation, or free-form chat.\n"
        "If the user asks about the service, map to show_help_topic.\n"
        "If the user asks about settings, subscriptions, billing, news forwarding, digest, language, AI filter, storyline tracking, or summarization, use show_help_topic with the matching faq_topic.\n"
        "If the user asks specifically about filter modes or the difference between filters, use show_help_topic with faq_topic='filters'.\n"
        "If the user asks what summarization is, how it works, or why it is needed, use show_help_topic with faq_topic='summary'.\n"
        "If the user asks to show, send, keep, hide, block, or exclude news by concrete free-form topics/categories/interests, use set_global_instruction_filter and put the user's concise instruction in slots.prompt_text.\n"
        "Do not use set_global_instruction_filter for vague quality words only, such as important, top, best, main, важное, главное, лучшее, without a concrete topic/category. For those, use set_feed_filter with needs_clarification=true and ask whether the user means the regular Only hot posts filter or an AI filter rule.\n"
        "Examples: 'show only economy and politics news', 'do not send sports', 'показывай только новости по экономике и политике'.\n"
        "If the user asks for a setting change but misses a required slot, set needs_clarification=true and ask one short question.\n"
        "If the request is outside the supported scope, use show_help_topic with faq_topic='general'.\n"
        "Use add_subscriptions/remove_subscriptions only for explicit add/remove wording. Bare links are handled outside the model.\n"
        "Return concise proposed_user_message in the user's language.\n"
        "Allowed intents: show_settings_overview, show_subscriptions, add_subscriptions, remove_subscriptions, "
        "set_feed_filter, toggle_forwarding, toggle_summary, digest_enable, digest_disable, digest_send_now, "
        "digest_set_time, digest_set_offset, set_language, show_billing, show_help_topic, set_global_instruction_filter.\n"
        "Return JSON only.\n"
        + retry_line
    )


def _router_user_payload(*, text: str, context: dict) -> str:
    compact_context = json.dumps(context, ensure_ascii=False)[:DEEPSEEK_ROUTER_CONTEXT_MAX_CHARS]
    return json.dumps(
        {
            "user_text": str(text or "").strip()[:900],
            "context": _extract_balanced_json_object(compact_context) or context,
        },
        ensure_ascii=False,
    )


def _extract_router_tool_args(data: dict) -> dict | None:
    try:
        message = data["choices"][0]["message"]
    except Exception:
        return None
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return None
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        if function.get("name") != "resolve_bot_action":
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            continue
        try:
            parsed = json.loads(arguments)
        except Exception:
            parsed = _extract_balanced_json_object(arguments)
        if isinstance(parsed, dict):
            return parsed
    return None


async def resolve_bot_router_intent(*, text: str, context: dict) -> dict | None:
    if not DEEPSEEK_API_KEY:
        raise DeepseekAuthError("DEEPSEEK_API_KEY not set")

    user_payload = _router_user_payload(text=text, context=context)

    tool_payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": _router_system_prompt()},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0.0,
        "max_tokens": DEEPSEEK_ROUTER_MAX_TOKENS,
        "tools": [_router_tool_schema()],
        "tool_choice": {"type": "function", "function": {"name": "resolve_bot_action"}},
    }

    try:
        data = await _post_chat(tool_payload, api_url_override=DEEPSEEK_BETA_API_URL)
        tool_args = _extract_router_tool_args(data)
        if isinstance(tool_args, dict):
            return tool_args
    except Exception as exc:
        log.warning("deepseek.router.tool_call_failed", error=repr(exc))

    for attempt in range(2):
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": _router_system_prompt(retry_mode=attempt > 0)},
                {
                    "role": "user",
                    "content": (
                        user_payload
                        + '\nReturn this JSON shape exactly: '
                        + '{"intent":"show_help_topic","slots":{},"needs_clarification":false,'
                        + '"clarify_question":"","faq_topic":"general","proposed_user_message":"..."}'
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": DEEPSEEK_ROUTER_MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        data = await _post_chat(payload)
        content = data["choices"][0]["message"]["content"].strip()
        result = _extract_balanced_json_object(content)
        if isinstance(result, dict) and result.get("intent"):
            return result

    raise DeepseekRetryableError("DeepSeek router returned invalid payload twice")

async def _post_chat(
    payload: dict,
    api_url_override: str | None = None,
    timeout_override: int | None = None,
) -> dict:
    if not DEEPSEEK_API_KEY:
        raise DeepseekAuthError("DEEPSEEK_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    session = await _get_session()
    if timeout_override is not None and timeout_override <= 0:
        request_timeout = aiohttp.ClientTimeout(total=None)
    elif timeout_override is not None:
        request_timeout = aiohttp.ClientTimeout(total=timeout_override)
    else:
        request_timeout = None
    try:
        target_url = api_url_override or DEEPSEEK_API_URL
        async with session.post(
            target_url, json=payload, headers=headers, timeout=request_timeout
        ) as resp:
            body_text = await resp.text()

            if resp.status in (401, 403):
                log.error("deepseek.http_auth_error", status=resp.status, body_preview=body_text[:300])
                raise DeepseekAuthError(f"DeepSeek HTTP {resp.status}: {body_text[:500]}")

            if resp.status == 429 or resp.status >= 500:
                log.warning("deepseek.http_retryable", status=resp.status, body_preview=body_text[:300])
                raise DeepseekRetryableError(f"DeepSeek HTTP {resp.status}: {body_text[:500]}")

            if resp.status != 200:
                log.error("deepseek.http_error", status=resp.status, body_preview=body_text[:300])
                raise RuntimeError(f"DeepSeek HTTP {resp.status}: {body_text[:500]}")

            try:
                return json.loads(body_text)
            except Exception as e:
                log.error("deepseek.invalid_json_response", body_preview=body_text[:300])
                raise RuntimeError(f"DeepSeek: invalid JSON response: {body_text[:500]}") from e

    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
        # str(e) может быть пустым, поэтому repr/type
        log.warning("deepseek.network_error", error_type=type(e).__name__, error_repr=repr(e))
        raise DeepseekRetryableError(f"{type(e).__name__}: {repr(e)}") from e


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if t.startswith("```"):
        # убираем первую строку ``` или ```json
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        # убираем закрывающие ```
        t = re.sub(r"\n```$", "", t.strip())
    return t.strip()


def _extract_balanced_json_object(text: str) -> dict | None:
    """
    Пытаемся извлечь первый полноценный JSON-объект { ... } из текста,
    корректно работая с вложенными {} (regex тут ненадёжен).
    """
    t = _strip_code_fences(text)
    if not t:
        return None

    # 1) пробуем целиком
    try:
        return json.loads(t)
    except Exception:
        pass

    # 2) вырезаем первый сбалансированный { ... }
    start = t.find("{")
    if start < 0:
        return None

    depth = 0
    end = -1
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end < 0:
        return None

    candidate = t[start:end]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _compact_digest_story_summary(text: str, *, max_chars: int) -> str:
    sample = re.sub(r"\s+", " ", str(text or "").strip())
    if len(sample) <= max_chars:
        return sample
    clipped = sample[:max_chars].rstrip(" ,.;:")
    return clipped + "..."


def _looks_like_digest_title_enumeration(title: str) -> bool:
    sample = re.sub(r"\s+", " ", str(title or "").strip())
    if not sample:
        return False
    comma_count = sample.count(",")
    if comma_count >= 2:
        return True
    lowered = sample.lower()
    if lowered.count(" и ") >= 2 and len(sample) > 90:
        return True
    return False


def _infer_digest_thematic_label(rough_stories: list[dict], idx_list: list[int]) -> str:
    snippets: list[str] = []
    for idx in idx_list:
        if 0 <= idx < len(rough_stories):
            story = rough_stories[idx]
            snippets.append(str(story.get("title") or ""))
            snippets.append(str(story.get("summary") or ""))
    corpus = " ".join(snippets).lower()

    category_rules = [
        (
            "Культура и светские новости",
            [
                "оскар",
                "книжн",
                "ярмарк",
                "фильм",
                "сериал",
                "аниме",
                "режисс",
                "автор",
                "культур",
                "музык",
                "артист",
                "шоу",
                "звезд",
                "vip",
                "1win",
                "ufc",
                "пётр ян",
            ],
        ),
        (
            "Международные конфликты и безопасность",
            [
                "иран",
                "сша",
                "израил",
                "бпла",
                "дрон",
                "пво",
                "атак",
                "удар",
                "аэродром",
                "перемир",
                "армии",
                "демобилиз",
                "зеленск",
                "всу",
                "военн",
                "конфликт",
            ],
        ),
        (
            "Политика и государственное регулирование",
            [
                "роскомнадзор",
                "штраф",
                "протокол",
                "генпрокурат",
                "изъят",
                "закон",
                "чиновник",
                "евросою",
                "вступлен",
                "реформ",
                "государ",
                "регулир",
            ],
        ),
        (
            "Технологии и интернет",
            [
                "cloudflare",
                "vpn",
                "приложени",
                "робот",
                "мессенджер",
                "чип",
                "данных",
                "сервер",
                "технолог",
                "интернет",
            ],
        ),
        (
            "Происшествия и криминал",
            [
                "дтп",
                "пожар",
                "кримин",
                "мошенн",
                "погиб",
                "посадка",
                "авар",
                "нападен",
                "покушен",
                "происшеств",
            ],
        ),
        (
            "Экономика и бизнес",
            [
                "пошлин",
                "актив",
                "бизнес",
                "рынок",
                "снек",
                "чипсы",
                "ассортимент",
                "компан",
                "финанс",
                "эконом",
            ],
        ),
    ]

    best_label = "Разное: тематическая подборка"
    best_score = 0
    for label, keywords in category_rules:
        score = sum(1 for keyword in keywords if keyword in corpus)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def _normalize_digest_group_title(title: str, rough_stories: list[dict], idx_list: list[int]) -> tuple[str, bool]:
    clean_title = str(title or "").strip()[:140]
    if len(idx_list) <= 1:
        return clean_title or "Сюжет", False
    if not _looks_like_digest_title_enumeration(clean_title):
        return clean_title or "Сюжет", False
    return _infer_digest_thematic_label(rough_stories, idx_list), True


def _build_digest_indexed_story(
    rough_story: dict,
    *,
    idx: int,
    summary_limit: int,
    include_meta: bool,
) -> dict:
    item = {
        "idx": idx,
        "title": str(rough_story.get("title") or "").strip()[:140],
        "summary": _compact_digest_story_summary(rough_story.get("summary") or "", max_chars=summary_limit),
    }
    if include_meta:
        item["posts_count"] = len(rough_story.get("post_ids") or [])
    return item


def _validate_digest_merge_groups(
    groups: list[dict],
    rough_stories: list[dict],
    *,
    fill_uncovered: bool,
) -> dict:
    total = len(rough_stories)
    claimed_idx: set[int] = set()
    stories_out: list[dict] = []
    duplicate_hits = 0
    invalid_hits = 0

    for group in groups:
        raw_idx = group.get("idx") or []
        deduped_idx: list[int] = []
        local_seen: set[int] = set()

        for raw_value in raw_idx:
            try:
                idx = int(raw_value)
            except (TypeError, ValueError):
                invalid_hits += 1
                continue
            if idx < 0 or idx >= total:
                invalid_hits += 1
                continue
            if idx in local_seen:
                duplicate_hits += 1
                continue
            if idx in claimed_idx:
                duplicate_hits += 1
                continue
            local_seen.add(idx)
            claimed_idx.add(idx)
            deduped_idx.append(idx)

        if not deduped_idx:
            continue

        merged_post_ids: list[int] = []
        seen_post_ids: set[int] = set()
        for idx in deduped_idx:
            for pid in (rough_stories[idx].get("post_ids") or []):
                try:
                    pid_int = int(pid)
                except (TypeError, ValueError):
                    continue
                if pid_int in seen_post_ids:
                    continue
                seen_post_ids.add(pid_int)
                merged_post_ids.append(pid_int)

        if not merged_post_ids:
            continue

        title = str(group.get("title") or "").strip()[:140]
        if not title:
            if len(deduped_idx) == 1:
                title = str(rough_stories[deduped_idx[0]].get("title") or "").strip()[:140]
            else:
                title = "Сюжет"
        title, title_was_normalized = _normalize_digest_group_title(title, rough_stories, deduped_idx)

        summary = str(group.get("summary") or "").strip()[:900]
        if not summary and deduped_idx:
            summary = str(rough_stories[deduped_idx[0]].get("summary") or "").strip()[:900]

        stories_out.append(
            {
                "title": title,
                "summary": summary,
                "post_ids": merged_post_ids,
                "_merge_idx": deduped_idx,
                "_title_normalized": title_was_normalized,
            }
        )

    uncovered_idx = [idx for idx in range(total) if idx not in claimed_idx]
    salvaged_singletons = 0
    if fill_uncovered:
        for idx in uncovered_idx:
            source_story = rough_stories[idx]
            post_ids: list[int] = []
            for pid in (source_story.get("post_ids") or []):
                try:
                    post_ids.append(int(pid))
                except (TypeError, ValueError):
                    continue
            if not post_ids:
                continue
            stories_out.append(
                {
                    "title": str(source_story.get("title") or "").strip()[:140] or "Сюжет",
                    "summary": str(source_story.get("summary") or "").strip()[:900],
                    "post_ids": post_ids,
                    "_merge_idx": [idx],
                }
            )
            salvaged_singletons += 1

    return {
        "stories": stories_out,
        "coverage_ratio": (len(claimed_idx) / total) if total else 0.0,
        "covered_idx": len(claimed_idx),
        "total_idx": total,
        "uncovered_idx": uncovered_idx,
        "duplicate_hits": duplicate_hits,
        "invalid_hits": invalid_hits,
        "salvaged_singletons": salvaged_singletons,
    }


def detect_source_language(text: str) -> str | None:
    sample = (text or "").strip()
    if not sample:
        return None

    cyrillic = sum(len(word) for word in re.findall(r"[А-Яа-яЁё]{4,}", sample))
    latin = sum(len(word) for word in re.findall(r"[A-Za-z]{4,}", sample))

    if cyrillic == 0 and latin == 0:
        return None
    if cyrillic >= latin * 1.2:
        return "ru"
    if latin >= cyrillic * 1.2:
        return "en"
    return None


def build_summary_prompt(source_lang: str | None) -> str:
    if source_lang == "ru":
        return (
            "Суммируй новость в 1-3 предложениях на русском языке. "
            "Сохрани ключевые факты и не добавляй оценки. "
            "Не переводи текст на другой язык. "
            "Без эмодзи."
        )
    if source_lang == "en":
        return (
            "Summarize the news in 1-3 sentences in English. "
            "Preserve key facts and do not add opinions. "
            "Do not translate it into another language. "
            "No emojis."
        )
    return (
        "Summarize the news in 1-3 sentences. "
        "Use the same language as the source text and do not translate it. "
        "Preserve key facts and do not add opinions. "
        "No emojis."
    )


async def summarize(text: str, source_lang: str | None = None) -> str | None:
    if not DEEPSEEK_API_KEY:
        log.warning("DEEPSEEK_API_KEY not set, skipping summarization")
        return None

    resolved_lang = source_lang or detect_source_language(text)
    prompt = build_summary_prompt(resolved_lang)

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "max_tokens": 350,
    }

    try:
        data = await _post_chat(payload)
        return data["choices"][0]["message"]["content"].strip()
    except (DeepseekAuthError, DeepseekRetryableError):
        raise
    except Exception as e:
        log.exception("DeepSeek summarize failed", error_repr=repr(e))
        return None


async def group_digest_stories(cards: list[dict], max_stories: int = 12) -> dict | None:
    """
    cards: [{id:int, source:str, url:str|None, text:str}]
    Return: {"stories":[{"title":str,"summary":str,"post_ids":[int,...]}]}

    Uses DeepSeek V4 Pro with thinking mode when DEEPSEEK_GROUP_THINKING_ENABLED=true,
    otherwise falls back to deepseek-chat.
    """
    if not DEEPSEEK_API_KEY:
        log.warning("DEEPSEEK_API_KEY not set, skipping digest grouping")
        return None

    def _compact(src_cards: list[dict]) -> list[dict]:
        out: list[dict] = []
        for c in src_cards:
            try:
                pid = int(c.get("id"))
            except Exception:
                continue
            out.append({
                "id": pid,
                "source": str(c.get("source") or "").strip()[:60],
                "url": str(c.get("url") or "").strip()[:200],
                # Увеличен лимит текста: 160 → 280, чтобы LLM лучше понимал суть поста
                "text": str(c.get("text") or "").strip()[:280],
            })
        return out

    system = (
        "Ты опытный новостной редактор. Получи список постов из Telegram-каналов и сгруппируй их в сюжеты дайджеста.\n\n"
        "ПРАВИЛА ГРУППИРОВКИ:\n"
        "1. Один сюжет = одно конкретное событие, инцидент или развитие темы.\n"
        "2. Несколько постов об одном событии (включая апдейты и продолжения) — объединяй в один сюжет.\n"
        "3. СТРОГО ЗАПРЕЩЕНО объединять в один сюжет посты на разные темы только потому, что они «важные» или оба о событиях.\n"
        "4. Если между постами нет явной тематической связи (общее событие / персонаж / место + одна история) — это разные сюжеты.\n"
        "5. Широкие темы типа «война», «экономика», «политика» — НЕ являются сюжетом. Сюжет — конкретный эпизод внутри темы.\n"
        "6. Лучше больше точных узких сюжетов, чем один широкий с несвязанными темами.\n\n"
        "ЗАГОЛОВОК (title): конкретный, 3–8 слов. Пример: «Взрыв на рынке в Кабуле», не «Происшествие за рубежом».\n"
        "САММАРИ (summary): 1–3 предложения, только факты из постов, нейтрально, без домыслов и эмодзи.\n\n"
        "Верни ТОЛЬКО валидный JSON без Markdown и пояснений.\n"
        "Схема: {\"stories\":[{\"title\":\"...\",\"summary\":\"...\",\"post_ids\":[1,2,3]}]}\n"
        "post_ids: только входные id; каждый id — не более чем в одном сюжете.\n"
    )

    use_thinking = DEEPSEEK_GROUP_THINKING_ENABLED

    # ---------- вспомогательные функции ----------

    async def _chat_batch(src_cards: list[dict], *, stories_limit: int, batch_label: str = "") -> list[dict]:
        """Быстрая черновая группировка одного батча через deepseek-chat."""
        compact = _compact(src_cards)
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": system + f"Ограничение: не больше {stories_limit} сюжетов.\n",
                },
                {"role": "user", "content": json.dumps({"cards": compact}, ensure_ascii=False)},
            ],
            "temperature": 0.2,
            "max_tokens": 1800,
            "response_format": {"type": "json_object"},
        }
        data = await _post_chat(payload)
        content = data["choices"][0]["message"]["content"].strip()
        obj = _extract_balanced_json_object(content) or {}
        raw = obj.get("stories") or []
        result = raw if isinstance(raw, list) else []
        if not result:
            log.warning(
                "digest.grouping.chat_batch_empty",
                batch=batch_label,
                cards=len(compact),
                content_preview=content[:300],
            )
        return result

    async def _thinking_merge(
        rough_stories: list[dict],
        *,
        stories_limit: int,
        summary_limit: int,
        timeout_seconds: int,
        include_meta: bool,
        fill_uncovered: bool,
        attempt_label: str,
    ) -> dict | None:
        """
        Второй проход: thinking-модель получает только заголовки + саммари черновых сюжетов
        с их порядковыми номерами (idx). Её задача — только решить КАКИЕ сюжеты объединить.
        post_ids Python склеивает сам по возвращённым idx — модель их не трогает.
        """
        merge_system = (
            "Ты главный редактор ежедневного новостного дайджеста.\n"
            "Тебе дан пронумерованный список черновых сюжетов, полученных из нескольких\n"
            "независимых батчей обработки постов из Telegram-каналов.\n\n"

            "ТВОЯ ЗАДАЧА — собрать из этих черновиков финальный дайджест высокого качества.\n\n"

            "═══ ШАГ 1: ДЕДУПЛИКАЦИЯ ═══\n"
            "Внимательно проверь каждую пару сюжетов — не описывают ли они одно событие\n"
            "с разных сторон или в разных формулировках. Объединяй такие пары ВСЕГДА.\n"
            "Критерии дубликата (хотя бы один):\n"
            "  • одно и то же решение/указ/постановление (например, запрет ввоза чего-либо)\n"
            "  • одно и то же происшествие (одна локация + одно действие)\n"
            "  • одно и то же лицо + одно и то же действие/событие\n"
            "  • новость + реакция/уточнение к той же новости\n"
            "Примеры:\n"
            "  «Запрет ввоза Starlink» ≡ «Запрет на ввоз спутниковых терминалов»\n"
            "  «Меликов уходит с поста» ≡ «Путин объявил об отставке главы Дагестана»\n"
            "  «Атака БПЛА в Туапсе» ≡ «ПВО отразила дроны над Туапсе»\n\n"

            "═══ ШАГ 2: ТЕМАТИЧЕСКОЕ ОБЪЕДИНЕНИЕ ═══\n"
            "После дедупликации объедини тематически однородные мелкие сюжеты в блоки.\n"
            "Хорошие блоки — это когда читатель сразу понимает, почему новости вместе:\n"
            "  • «Военные события» — атаки БПЛА, обстрелы, сводки с фронта\n"
            "  • «Криминальные происшествия» — бытовые преступления, задержания\n"
            "  • «Наука и технологии» — запуски ракет, ИИ, научные открытия\n"
            "  • «Политические инициативы» — законопроекты, предложения чиновников\n"
            "  • «Экономика» — курсы валют, зарплаты, отчёты компаний\n"
            "  • «Зарубежные новости» — события за пределами России без военной повестки\n"
            "  • «Культура и спорт» — искусство, кино, соревнования\n"
            "  • «Погода» — прогнозы, природные явления\n\n"
            "ЖЁСТКИЕ ЗАПРЕТЫ на объединение:\n"
            "  ✗ военные действия / война + наука / технологии / спорт\n"
            "  ✗ Украина / боевые действия + любая мирная тема\n"
            "  ✗ российская внутренняя политика + зарубежные события\n"
            "  ✗ криминал + политика / законодательство\n"
            "  ✗ никогда не объединяй новости только потому что «не знаю куда ещё поставить»\n\n"
            "ПРАВИЛО РАЗНОГО:\n"
            "Если несколько сюжетов не вписываются ни в один связный блок — создай группу\n"
            "с заголовком «Разное» и помести их туда. Это честнее, чем придумывать\n"
            "искусственный общий знаменатель для несвязанных новостей.\n\n"

            "═══ ПРАВИЛО ПОЛНОТЫ ═══\n"
            "Каждый входной idx ОБЯЗАН попасть ровно в одну группу. Ни один сюжет не теряется.\n\n"

            "═══ ТРЕБОВАНИЯ К ТЕКСТУ ═══\n"
            "- title: цепкий журналистский заголовок. Для одиночного события — конкретный факт.\n"
            "  Для тематического блока — обобщающий. Максимум 10 слов.\n"
            "- summary: плотный фактический текст. Для одиночного события — 2–3 предложения\n"
            "  с ключевыми деталями. Для блока — перечисли 3–5 событий в 2–3 предложениях.\n"
            "  Никаких оценок, только факты.\n\n"

            "Верни ТОЛЬКО валидный JSON без Markdown-обёртки:\n"
            "{\"groups\":[{\"idx\":[0,5,12],\"title\":\"...\",\"summary\":\"...\"}]}\n\n"
            "- idx: массив входных индексов (0-based), которые объединяются в эту группу\n"
            f"- Итого групп в output: не более {stories_limit}\n"
        )

        # Отправляем только индекс + заголовок + саммари — без post_ids
        indexed = [
            _build_digest_indexed_story(
                s,
                idx=i,
                summary_limit=summary_limit,
                include_meta=include_meta,
            )
            for i, s in enumerate(rough_stories)
        ]

        payload: dict = {
            "model": DEEPSEEK_THINKING_MODEL,
            "messages": [
                {"role": "system", "content": merge_system},
                {"role": "user", "content": json.dumps({"stories": indexed}, ensure_ascii=False)},
            ],
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
            "max_tokens": 32000,
        }
        log.info(
            "digest.grouping.thinking_merge_start",
            attempt=attempt_label,
            rough_stories=len(rough_stories),
            stories_limit=stories_limit,
            summary_limit=summary_limit,
            include_meta=include_meta,
            client_timeout_disabled=timeout_seconds <= 0,
            timeout_seconds=None if timeout_seconds <= 0 else timeout_seconds,
        )
        data = await _post_chat(payload, timeout_override=timeout_seconds)
        message = data["choices"][0]["message"]

        raw_content = message.get("content") or ""
        if not raw_content.strip():
            raw_content = message.get("reasoning_content") or ""
            if raw_content.strip():
                log.warning("digest.grouping.thinking_content_empty_using_reasoning")

        content = raw_content.strip()
        log.debug("digest.grouping.thinking_merge_response", content_preview=content[:300])

        obj = _extract_balanced_json_object(content)
        if not obj or "groups" not in obj:
            log.error(
                "digest.grouping.thinking_merge_invalid_json",
                attempt=attempt_label,
                content_preview=content[:500],
            )
            return None

        validation = _validate_digest_merge_groups(
            list(obj.get("groups") or []),
            rough_stories,
            fill_uncovered=fill_uncovered,
        )
        result_stories = validation["stories"]

        # Логируем сырые группы из thinking — чтобы видеть что модель решила
        for gi, group in enumerate(obj.get("groups") or []):
            log.info(
                "digest.grouping.thinking_group_raw",
                gi=gi,
                idx=group.get("idx"),
                title=(group.get("title") or "")[:80],
                merged=len(group.get("idx") or []) > 1,
            )

        log.info(
            "digest.grouping.thinking_merge_done",
            attempt=attempt_label,
            groups_in=len(obj.get("groups") or []),
            stories_out=len(result_stories),
            covered_idx=validation["covered_idx"],
            total_idx=validation["total_idx"],
            uncovered_idx=len(validation["uncovered_idx"]),
            duplicate_hits=validation["duplicate_hits"],
            invalid_hits=validation["invalid_hits"],
            salvaged_singletons=validation["salvaged_singletons"],
            coverage_ratio=round(validation["coverage_ratio"], 4),
        )
        return validation

    # ---------- основная логика ----------

    try:
        src = (cards or [])
        if not src:
            return None

        if use_thinking:
            # Двухпроходный пайплайн:
            # Проход 1 — deepseek-chat быстро группирует ВСЕ карточки батчами (полное покрытие).
            # Проход 2 — thinking-модель получает только черновые сюжеты (маленький вход)
            #            и делает финальное слияние дублей и чистку.

            CHAT_BATCH = 15
            rough_stories: list[dict] = []
            for i in range(0, len(src), CHAT_BATCH):
                batch = src[i : i + CHAT_BATCH]
                batch_label = f"b{i // CHAT_BATCH}"
                try:
                    batch_stories = await _chat_batch(batch, stories_limit=10, batch_label=batch_label)
                    rough_stories.extend(batch_stories)
                    log.info(
                        "digest.grouping.chat_batch_ok",
                        batch_idx=i // CHAT_BATCH,
                        cards=len(batch),
                        stories=len(batch_stories),
                    )
                except DeepseekRetryableError as e:
                    log.warning(
                        "digest.grouping.chat_batch_failed",
                        batch_idx=i // CHAT_BATCH,
                        error=str(e),
                    )

            if not rough_stories:
                raise DeepseekRetryableError("chat pass returned no stories")

            # Логируем все черновые сюжеты — чтобы видеть что дошло до thinking merge
            for ri, rs in enumerate(rough_stories):
                pids = rs.get("post_ids") or []
                log.info(
                    "digest.grouping.rough_story",
                    idx=ri,
                    title=(rs.get("title") or "")[:80],
                    post_ids=pids,
                    sources_count=len(pids),
                )

            log.info(
                "digest.grouping.chat_pass_done",
                rough_stories=len(rough_stories),
                cards_total=len(src),
            )

            merge_plans = [
                {
                    "attempt_label": "a1_full",
                    "summary_limit": 400,
                    "timeout_seconds": DEEPSEEK_GROUP_TIMEOUT_A1,
                    "include_meta": True,
                    "fill_uncovered": True,
                    "stories_limit": min(max_stories, 12),
                },
                {
                    "attempt_label": "a2_compact",
                    "summary_limit": 240,
                    "timeout_seconds": DEEPSEEK_GROUP_TIMEOUT_A2,
                    "include_meta": True,
                    "fill_uncovered": True,
                    "stories_limit": min(max_stories + 2, 14),
                },
                {
                    "attempt_label": "a3_minified",
                    "summary_limit": 140,
                    "timeout_seconds": DEEPSEEK_GROUP_TIMEOUT_A3,
                    "include_meta": False,
                    "fill_uncovered": True,
                    "stories_limit": min(max_stories + 4, 16),
                },
            ][: max(1, DEEPSEEK_GROUP_THINKING_ATTEMPTS)]

            merged = None
            merge_failures: list[str] = []
            for merge_plan in merge_plans:
                try:
                    merged = await _thinking_merge(rough_stories, **merge_plan)
                except DeepseekRetryableError as e:
                    merge_failures.append(f"{merge_plan['attempt_label']}:{e}")
                    log.warning(
                        "digest.grouping.thinking_merge_failed",
                        attempt=merge_plan["attempt_label"],
                        error=str(e),
                    )
                    merged = None
                except Exception as e:
                    merge_failures.append(f"{merge_plan['attempt_label']}:{type(e).__name__}")
                    log.warning(
                        "digest.grouping.thinking_merge_unexpected_failure",
                        attempt=merge_plan["attempt_label"],
                        error_repr=repr(e),
                    )
                    merged = None

                if merged and (merged.get("stories") or []):
                    break

            if merged and (merged.get("stories") or []):
                for si, st in enumerate(merged.get("stories") or []):
                    pids = st.get("post_ids") or []
                    log.info(
                        "digest.grouping.merged_story",
                        idx=si,
                        title=(st.get("title") or "")[:80],
                        post_ids=pids,
                        sources_count=len(pids),
                        merged_idx=st.get("_merge_idx") or [],
                        title_normalized=bool(st.get("_title_normalized")),
                    )
                    st.pop("_merge_idx", None)
                    st.pop("_title_normalized", None)
                log.info(
                    "digest.grouping.thinking_ok",
                    model=DEEPSEEK_THINKING_MODEL,
                    rough_stories_in=len(rough_stories),
                    stories_found=len((merged.get("stories") or [])),
                    merge_failures=merge_failures,
                )
                return {"stories": merged.get("stories") or []}

            # Fallback: если thinking упал — вернём черновые сюжеты как есть
            log.warning(
                "digest.grouping.thinking_fallback_to_rough",
                rough_stories=len(rough_stories),
                merge_failures=merge_failures,
            )
            return {"stories": rough_stories}

        else:
            # Только deepseek-chat, батчами
            CHAT_BATCH = 15
            all_stories: list[dict] = []
            for i in range(0, min(len(src), 50), CHAT_BATCH):
                batch = src[i : i + CHAT_BATCH]
                batch_label = f"b{i // CHAT_BATCH}"
                try:
                    batch_stories = await _chat_batch(batch, stories_limit=10, batch_label=batch_label)
                    all_stories.extend(batch_stories)
                except DeepseekRetryableError as e:
                    log.warning("digest.grouping.chat_batch_failed", batch_idx=i // CHAT_BATCH, error=str(e))

            if not all_stories:
                raise DeepseekRetryableError("deepseek-chat returned no stories on all batches")
            return {"stories": all_stories}

    except DeepseekAuthError as e:
        log.error("DeepSeek grouping auth error", error=str(e))
        return None
    except DeepseekRetryableError:
        raise
    except Exception as e:
        log.exception("DeepSeek grouping failed", error_repr=repr(e))
        return None


async def generate_digest_headline(stories: list[dict]) -> str | None:
    """
    stories: [{"title": str, "summary": str, "sources_count": int}] — LLM-сюжеты дайджеста.
    Возвращает одну фразу — главное событие дня, как лид на первой полосе.
    """
    if not DEEPSEEK_API_KEY:
        log.warning("DEEPSEEK_API_KEY not set, skipping digest headline")
        return None
    if not stories:
        return None

    system = (
        "Ты выпускающий редактор информационного агентства. "
        "Тебе дан список сюжетов из дайджеста за 24 часа — заголовки, краткие саммари и число источников.\n\n"
        "ЗАДАЧА: выбери ОДИН — самый значимый и резонансный сюжет дня. "
        "Напиши для него одну ёмкую фразу-анонс, как лид на первой полосе газеты.\n\n"
        "КРИТЕРИИ ВЫБОРА:\n"
        "- широкое освещение (sources_count: чем больше источников, тем важнее)\n"
        "- конкретное событие с ощутимыми последствиями, а не фоновый процесс\n"
        "- неожиданность или высокая общественная значимость\n\n"
        "ПРАВИЛО ПРИОРИТЕТА:\n"
        "- если один сюжет заметно опережает остальные по sources_count, по умолчанию выбирай его\n"
        "- отступай от этого только если другой сюжет явно более важен по последствиям и общественной значимости\n"
        "- не выбирай мягкий или развлекательный сюжет, если в списке есть крупное политическое, военное, экономическое или аварийное событие с большим охватом\n\n"
        "ТРЕБОВАНИЯ К ФРАЗЕ:\n"
        "- ровно 1 предложение, 8–18 слов\n"
        "- конкретно: факты, имена, места — если они есть в данных\n"
        "- нейтральный тон, без оценок, превосходных степеней и восклицаний\n"
        "- без эмодзи, без кавычек в начале и конце\n"
        "- верни ТОЛЬКО текст фразы — ничего больше, никаких пояснений"
    )

    compact = []
    for i, s in enumerate(stories[:10], 1):
        compact.append({
            "n": i,
            "sources_count": int(s.get("sources_count") or 0),
            "title": str(s.get("title") or "").strip()[:120],
            "summary": str(s.get("summary") or s.get("text") or "").strip()[:200],
        })

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"stories": compact}, ensure_ascii=False)},
        ],
        "temperature": 0.1,
        "max_tokens": 80,
    }

    try:
        data = await _post_chat(payload)
        result = data["choices"][0]["message"]["content"].strip()
        # Убираем случайные кавычки по краям, если модель всё же добавила
        result = result.strip('"\'«»')
        return result if result else None
    except (DeepseekAuthError, DeepseekRetryableError):
        raise
    except Exception as e:
        log.exception("DeepSeek digest headline failed", error_repr=repr(e))
        return None


def _normalize_topic(topic: str) -> str:
    t = re.sub(r"\s+", " ", (topic or "").strip().lower())
    return t[:64]


async def extract_post_topics(text: str, max_topics: int = 10) -> list[str]:
    """
    Returns a list of short topic tags for a post.
    Fallback strategy is handled by caller (e.g. unknown).
    """
    if not DEEPSEEK_API_KEY:
        log.warning("DEEPSEEK_API_KEY not set, skipping topic extraction")
        return []

    system = (
        "Ты классификатор тем новостного поста.\n"
        "Верни ТОЛЬКО валидный JSON без markdown.\n"
        "Формат: {\"topics\":[\"тема1\",\"тема2\", ...]}\n"
        "Требования:\n"
        "- от 0 до 10 коротких тем\n"
        "- каждая тема: 1-3 слова\n"
        "- без эмодзи и пунктуации\n"
        "- не дублируй темы\n"
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": (text or "")[:4000]},
        ],
        "temperature": 0.1,
        "max_tokens": 220,
        "response_format": {"type": "json_object"},
    }

    try:
        data = await _post_chat(payload)
        content = data["choices"][0]["message"]["content"].strip()
        obj = _extract_balanced_json_object(content) or {}
        raw_topics = obj.get("topics", [])
        if not isinstance(raw_topics, list):
            return []

        result: list[str] = []
        seen: set[str] = set()
        for item in raw_topics:
            if not isinstance(item, str):
                continue
            topic = _normalize_topic(item)
            if not topic or topic in seen:
                continue
            seen.add(topic)
            result.append(topic)
            if len(result) >= max_topics:
                break
        return result
    except DeepseekAuthError:
        raise
    except DeepseekRetryableError:
        raise
    except Exception as e:
        log.exception("DeepSeek topic extraction failed", error_repr=repr(e))
        return []


async def arbitrate_storyline_timeline(
    *,
    anchor: dict,
    candidate_storylines: list[dict],
    candidate_macro_topics: list[dict] | None = None,
    mixed_topic_mode: bool = False,
    anchor_profile: dict | None = None,
    groups: list[dict],
    max_items: int = 10,
) -> dict | None:
    if not DEEPSEEK_API_KEY:
        log.warning("DEEPSEEK_API_KEY not set, skipping storyline timeline arbitration")
        return None

    compact_groups: list[dict] = []
    for group in groups:
        group_id = str(group.get("group_id") or "").strip()
        if not group_id:
            continue
        sources = []
        for source in group.get("sources") or []:
            sources.append(
                {
                    "post_id": int(source.get("post_id") or 0),
                    "timestamp": str(source.get("timestamp") or "").strip()[:40],
                    "source": str(source.get("source") or "").strip()[:80],
                    "url": str(source.get("url") or "").strip()[:240],
                }
            )
        compact_groups.append(
            {
                "group_id": group_id,
                "macro_topic_id": str(group.get("macro_topic_id") or "").strip()[:80],
                "macro_topic_title": str(group.get("macro_topic_title") or "").strip()[:160],
                "start_timestamp": str(group.get("start_timestamp") or "").strip()[:40],
                "end_timestamp": str(group.get("end_timestamp") or "").strip()[:40],
                "post_count": int(group.get("post_count") or 0),
                "source_count": int(group.get("source_count") or 0),
                "storyline_ids": [str(item or "").strip()[:80] for item in (group.get("storyline_ids") or [])][:12],
                "retrieval_origins": [str(item or "").strip()[:32] for item in (group.get("retrieval_origins") or [])][:6],
                "representative_text": str(group.get("representative_text") or "").strip()[:900],
                "sources": sources[:20],
            }
        )

    compact_storylines: list[dict] = []
    for item in candidate_storylines:
        compact_storylines.append(
            {
                "storyline_id": str(item.get("storyline_id") or "").strip()[:80],
                "title": str(item.get("title") or "").strip()[:160],
                "seed_preview": str(item.get("seed_preview") or "").strip()[:260],
                "posts_count": int(item.get("posts_count") or 0),
                "story_family_id": str(item.get("story_family_id") or "").strip()[:80],
                "family_root_storyline_id": str(item.get("family_root_storyline_id") or "").strip()[:80],
                "family_match": int(item.get("family_match") or 0),
                "shared_entity_count": int(item.get("shared_entity_count") or 0),
                "shared_signature_count": int(item.get("shared_signature_count") or 0),
                "token_overlap": int(item.get("token_overlap") or 0),
            }
        )

    compact_macro_topics: list[dict] = []
    for item in candidate_macro_topics or []:
        compact_macro_topics.append(
            {
                "macro_topic_id": str(item.get("macro_topic_id") or "").strip()[:80],
                "macro_topic_title": str(item.get("macro_topic_title") or "").strip()[:160],
                "best_retrieval_score": float(item.get("best_retrieval_score") or 0.0),
                "same_macro_topic": int(item.get("same_macro_topic") or 0),
                "shared_entity_count": int(item.get("shared_entity_count") or 0),
                "shared_topic_count": int(item.get("shared_topic_count") or 0),
                "token_overlap": int(item.get("token_overlap") or 0),
                "topic_token_overlap": int(item.get("topic_token_overlap") or 0),
                "must_overlap": int(item.get("must_overlap") or 0),
                "must_phrase_overlap": int(item.get("must_phrase_overlap") or 0),
                "negative_overlap": int(item.get("negative_overlap") or 0),
                "negative_phrase_overlap": int(item.get("negative_phrase_overlap") or 0),
                "storyline_ids": [str(x or "").strip()[:80] for x in (item.get("storyline_ids") or [])][:12],
                "storyline_titles": [str(x or "").strip()[:160] for x in (item.get("storyline_titles") or [])][:8],
            }
        )

    selector_system = (
        "Ты редактор новостного сторитрекинга.\n"
        "На входе якорный сюжет, набор candidate macro-topics и candidate groups, уже частично схлопнутых по дубликатам.\n"
        "Выбери только те group_id, которые относятся к тому же большому сюжету, что и anchor.\n"
        "Если пост смешанный и стоит на пересечении нескольких близких веток, допускается сохранить группы из нескольких macro-topic, но только если они действительно описывают связанный общий контекст якорного поста.\n"
        "Не включай ветку только потому, что совпадает широкий человек, страна, война или география. Нужен branch-specific сигнал: Telegram/Max, мобилизация, конкретный запуск миссии, паводок, закон, катастрофа и т.п.\n"
        "Если mixed_topic_mode=true, ориентируйся прежде всего на must_terms, search_phrases и strongest candidate_macro_topics с must_overlap/must_phrase_overlap.\n"
        "Нужно отсечь нерелевантный шум и ложные merge.\n"
        "Верни ТОЛЬКО валидный JSON без markdown.\n"
        "Схема: {\"selected_group_ids\":[\"group_0001\",\"group_0002\"]}\n"
        "Если сомневаешься, лучше НЕ включать group.\n"
    )

    final_system = (
        "Ты редактор новостного сторитрекинга.\n"
        "На входе якорный сюжет, похожие storyline-кластеры, candidate macro-topics и уже схлопнутые candidate groups постов.\n"
        "Часть clusters/groups может быть нерелевантной или ложным merge.\n"
        "Некоторые посты стоят на пересечении двух-трех больших сюжетов. В таком случае не пытайся насильно сделать один ложный сюжет: выбирай только те ветки, которые реально нужны для объяснения якорного поста.\n"
        "Если mixed_topic_mode=true, можно собрать историю из 2-3 близких веток, но только если они вместе объясняют anchor-пост. Не добавляй ветки только из-за широкой войны, широкой географии или общего персонажа.\n"
        "Твоя задача: собрать один связный сюжетный контекст вокруг anchor и построить краткую хронологию.\n"
        "Верни ТОЛЬКО валидный JSON без markdown.\n"
        "Схема:\n"
        "{"
        "\"story_title\":\"...\","
        "\"overview\":\"...\","
        "\"topic_mode\":\"single|mixed\","
        "\"relevant_macro_topic_ids\":[\"macro_topic_1\"],"
        "\"selected_storyline_ids\":[\"storyline_1\"],"
        "\"items\":[{\"group_ids\":[\"group_0001\"],\"summary\":\"...\"}]"
        "}\n"
        "Требования:\n"
        f"- items: не больше {max_items}\n"
        "- используй только group_id из входных groups\n"
        "- можно объединять несколько group_id в один хронологический шаг, если это один и тот же апдейт разными словами\n"
        "- отбрасывай groups, которые не относятся к одному и тому же сюжету\n"
        "- items должны отражать хронологическое развитие сюжета\n"
        "- раннюю launch/start/prep/founding step включай только если она явно относится к тому же branch-specific сюжету, что и anchor; общей институции, географии или широкого контекста недостаточно\n"
        "- summary у каждого item: 1 короткое предложение, нейтрально, без домыслов\n"
        "- overview: 2-4 предложения, нейтрально, без воды\n"
        "- если mixed_topic_mode=true и веток несколько, в overview коротко объясни, как они связаны\n"
        "- если есть сомнения, лучше отбрасывай нерелевантный пост\n"
    )

    anchor_payload = {
        "storyline_id": str(anchor.get("storyline_id") or "").strip(),
        "storyline_title": str(anchor.get("storyline_title") or "").strip(),
        "storyline_seed_preview": str(anchor.get("storyline_seed_preview") or "").strip()[:300],
        "story_family_id": str(anchor.get("story_family_id") or "").strip(),
        "family_root_storyline_id": str(anchor.get("family_root_storyline_id") or "").strip(),
        "macro_topic_id": str(anchor.get("macro_topic_id") or "").strip(),
        "macro_topic_title": str(anchor.get("macro_topic_title") or "").strip()[:160],
        "mixed_topic_mode": bool(mixed_topic_mode),
        "anchor_profile": {
            "macro_topic": str((anchor_profile or {}).get("macro_topic") or "").strip()[:160],
            "must_terms": [str(x or "").strip()[:120] for x in ((anchor_profile or {}).get("must_terms") or [])][:8],
            "search_phrases": [str(x or "").strip()[:160] for x in ((anchor_profile or {}).get("search_phrases") or [])][:6],
            "topic_terms": [str(x or "").strip()[:120] for x in ((anchor_profile or {}).get("topic_terms") or [])][:8],
            "downweight_terms": [str(x or "").strip()[:120] for x in ((anchor_profile or {}).get("downweight_terms") or [])][:8],
        },
    }

    async def _select_relevant_group_ids(batch_groups: list[dict]) -> set[str]:
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": selector_system},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "anchor": anchor_payload,
                            "candidate_storylines": compact_storylines,
                            "candidate_macro_topics": compact_macro_topics,
                            "groups": batch_groups,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 700,
            "response_format": {"type": "json_object"},
        }
        data = await _post_chat(payload)
        content = data["choices"][0]["message"]["content"].strip()
        obj = _extract_balanced_json_object(content) or {}
        return {
            str(item or "").strip()
            for item in (obj.get("selected_group_ids") or [])
            if str(item or "").strip()
        }

    async def _build_final_timeline(final_groups: list[dict]) -> dict | None:
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": final_system},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "anchor": anchor_payload,
                            "candidate_storylines": compact_storylines,
                            "candidate_macro_topics": compact_macro_topics,
                            "groups": final_groups,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.2,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"},
        }
        data = await _post_chat(payload)
        content = data["choices"][0]["message"]["content"].strip()
        return _extract_balanced_json_object(content)

    try:
        filtered_groups = compact_groups
        if len(compact_groups) > 24:
            selected_ids: set[str] = set()
            for idx in range(0, len(compact_groups), 18):
                batch = compact_groups[idx : idx + 18]
                selected_ids.update(await _select_relevant_group_ids(batch))
            if selected_ids:
                filtered_groups = [item for item in compact_groups if str(item.get("group_id") or "") in selected_ids]

        obj = await _build_final_timeline(filtered_groups)
        if not obj:
            log.error(
                "deepseek.storyline_timeline.invalid_json",
                group_count=len(filtered_groups),
                total_group_count=len(compact_groups),
            )
            return None
        return obj
    except DeepseekAuthError:
        raise
    except DeepseekRetryableError:
        raise
    except Exception as e:
        log.exception("deepseek.storyline_timeline.failed", error_repr=repr(e))
        return None
