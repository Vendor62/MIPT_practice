import re

URL_RE = re.compile(r"https?://\S+|t\.me/\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
AD_HASHTAG_RE = re.compile(r"(^|\s)#реклама\b", re.IGNORECASE)
NEWSLETTER_WRAPPER_RE = re.compile(
    r"^\s*(?:утренняя|вечерняя|дневная)\s+рассылка\b[:\-–—]*\s*",
    re.IGNORECASE,
)
NEWSLETTER_DIVIDER_RE = re.compile(r"(?:_{3,}|-{3,}|={3,}).*$", re.DOTALL)
NEWSLETTER_CTA_RE = re.compile(
    r"(?:чтобы\s+не\s+остаться\s+без\s+новост|"
    r"не\s+остаться\s+без\s+новост|"
    r"после\s+блокировки\s+telegram|"
    r"подпишитесь\s+на\s+рассылк).*?$",
    re.IGNORECASE | re.DOTALL,
)
VIDEO_STUB_RE = re.compile(r"^\s*(?:аудио|видео|audio|video)\s*\|", re.IGNORECASE)
SOCIAL_FOOTER_RE = re.compile(
    r"(?:^|\s)(?:telegram|телеграм|max|vk|вк|youtube|yt|rutube|ok)\s*"
    r"(?:\|\s*(?:telegram|телеграм|max|vk|вк|youtube|yt|rutube|ok)\s*){1,5}$",
    re.IGNORECASE,
)
CTA_TAIL_RE = re.compile(
    r"(?:подписывайт(?:есь|еcь)|подпишитесь|смотрите\s+видео|полное\s+видео|"
    r"ссылка\s+в\s+описании|по\s+ссылке|перейти\s+по\s+ссылке)\b.*$",
    re.IGNORECASE | re.DOTALL,
)
LOW_SIGNAL_PROMO_RE = re.compile(
    r"\b(vpn|впн|maxswap|status\s*vpn|skyzit|shiv(?:a|aru)\s*vpn)\b",
    re.IGNORECASE,
)
URL_HEAVY_RE = re.compile(r"(https?://\S+|t\.me/\S+)", re.IGNORECASE)
PROMO_CUE_RE = re.compile(
    r"\b("
    r"акци(?:я|и|ю|ей)|розыгрыш|приз(?:ы|а|ов)?|выигра(?:ть|й|йте|л)|"
    r"закаж(?:и|ите|ть|те)|заказ(?:ать|е|ом)?|покупк(?:а|у|и|ой)?|"
    r"скидк(?:а|и|у|ой)?|кешбэк|кэшбэк|qr-?код|приложени(?:е|и|я)|"
    r"для участия|участвуй|получи|получить|дарим|разыгрываем|"
    r"наклейк(?:и|а|у)?|стикер(?:ы|ов|а)?|промокод|бонус(?:ы|ов)?|"
    r"партнер(?:ов|ы|а)?|подписчик(?:ов|ам|и)?|оформлени(?:е|и)|"
    r"эквайринг|терминал"
    r")\b",
    re.IGNORECASE,
)
BRAND_STYLE_RE = re.compile(
    r"\b([A-ZА-ЯЁ][A-ZА-ЯЁa-zа-яё]+(?:[-—][A-ZА-ЯЁa-zа-яё]+)+|"
    r"[A-Za-z]{3,}\d*|[А-ЯЁ][а-яё]+-[А-ЯЁ][а-яё]+)\b"
)
MULTISPACE_RE = re.compile(r"\s+")
QUOTE_RE = re.compile(r"[\"'`«»„“]+")
STAR_RE = re.compile(r"\*+")
BRACKETS_RE = re.compile(r"\[[^\]]+\]|\([^\)]*\)")
LEADIN_RE = re.compile(
    r"^(как сообщает|по данным|по информации|сообщает|пишет|отмечает)\s+",
    re.IGNORECASE,
)
FOREIGN_AGENT_DISCLAIMER_FULL_RE = re.compile(
    r"настоящий\s+материал.*?решени(?:ем|я)\s+еспч",
    re.IGNORECASE | re.DOTALL,
)
FOREIGN_AGENT_DISCLAIMER_PREFIX_RE = re.compile(
    r"^\s*настоящий\s+материал[^\n]*\n",
    re.IGNORECASE | re.MULTILINE,
)
FOREIGN_AGENT_DISCLAIMER_INLINE_RE = re.compile(
    r"\s*настоящий\s+материал\s*\(?(?:информация)?\)?\s*"
    r"произведен\s+и\s*(?:\([^)]+\)\s*)?распространен.*?"
    r"иностранн(?:ым|ого)\s+агент(?:ом|а).*?(?:18\+|$)",
    re.IGNORECASE | re.DOTALL,
)
DIGEST_CUE_RE = re.compile(
    r"(главн(ые|ое)\s+новост|дайджест|итоги\s+(дня|недели)|"
    r"главное\s+к\s+этому\s+часу|что\s+известно|сводка|коротко\s+о\s+главном)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,}")
CONCISE_OFFICIAL_EVENT_RE = re.compile(
    r"\b("
    r"провед(?:ет|ёт|ут|утcя|ется|утcя|ётcя|ено|ена|ены|ил[аи]?|или)|"
    r"заяв(?:ил|ила|или|ляет|ят)|сообщ(?:ил|ила|или|ает|ают)|"
    r"объяв(?:ил|ила|или|ят|ляет)|подпис(?:ал|ала|али|ывает|ан[аоы]?)|"
    r"поруч(?:ил|ила|или|ит|ено)|утверд(?:ил|ила|или|ят|ен[аоы]?)|"
    r"обсуд(?:ил|ила|или|ят|ит)|встрет(?:ится|ятся|ился|илась|ились)|"
    r"назнач(?:ил|ила|или|ат|ен[аоы]?)|отмен(?:ил|ила|или|ят|ен[аоы]?)|"
    r"внес(?:ет|ёт|ли|ла|ен[аоы]?)|прин(?:ял|яла|яли|имут|ят)"
    r")\b",
    re.IGNORECASE,
)
CONCISE_OFFICIAL_ANCHOR_RE = re.compile(
    r"\b("
    r"президент|премьер|министр|глава|губернатор|мэр|сенатор|депутат|"
    r"путин|песков|мишустин|лавров|трамп|зеленск\w*|"
    r"кремл\w*|правительств\w*|кабмин\w*|госдум\w*|совфед\w*|"
    r"минфин|мид|минобороны|цб|центробанк|фрс|белый\s+дом"
    r")\b",
    re.IGNORECASE,
)
CONCISE_OFFICIAL_ATTRIBUTION_RE = re.compile(
    r"(?:[—–-]\s*[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё-]{3,}\s*$|"
    r"\b(?:песков|кремл\w*|правительств\w*|кабмин\w*|минфин|мид|"
    r"минобороны|цб|центробанк|белый\s+дом|пресс-служб\w*)\b)",
    re.IGNORECASE,
)


def has_ad_marker(text: str) -> bool:
    return bool(AD_HASHTAG_RE.search(text or ""))


def promotional_text_score(text: str) -> float:
    raw = text or ""
    lowered = raw.lower()
    promo_hits = len(PROMO_CUE_RE.findall(lowered))
    brand_hits = len(BRAND_STYLE_RE.findall(raw))
    score = 0.0
    if has_ad_marker(raw):
        score += 1.0
    score += min(0.75, promo_hits * 0.18)
    score += min(0.30, brand_hits * 0.06)
    if "для участия" in lowered:
        score += 0.20
    if "выиграть" in lowered or "призы" in lowered:
        score += 0.18
    if "qr-код" in lowered or "qr код" in lowered:
        score += 0.12
    return min(1.0, score)


def is_promotional_text(text: str) -> bool:
    return promotional_text_score(text) >= 0.55


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def seed_quality_score(
    *,
    pipeline_text: str,
    token_count: int,
    removed_ratio: float,
    promo_score: float,
    digest_like: bool,
    audio_video_stub: bool,
    wrapper_peeled: bool,
) -> float:
    text_len = len((pipeline_text or "").strip())
    score = 0.0

    score += min(0.40, token_count * 0.02)
    score += min(0.20, text_len / 600.0)

    if token_count >= 12:
        score += 0.15
    if token_count >= 25:
        score += 0.10

    score -= min(0.30, max(0.0, removed_ratio - 0.15))
    score -= min(0.35, promo_score * 0.60)
    if digest_like:
        score -= 0.22
    if audio_video_stub:
        score -= 0.25
    if wrapper_peeled:
        score -= 0.08

    if token_count <= 2 or text_len < 12:
        score -= 0.60
    elif token_count <= 5 or text_len < 40:
        score -= 0.20

    return round(_clamp01(score), 4)


def _is_concise_official_update(
    *,
    raw: str,
    pipeline: str,
    token_count: int,
    url_count: int,
    promo_score: float,
    digest_like: bool,
    audio_video_stub: bool,
    removed_ratio: float,
) -> bool:
    if token_count < 6 or token_count > 14:
        return False
    if len((pipeline or "").strip()) < 35:
        return False
    if url_count > 0 or promo_score >= 0.20 or digest_like or audio_video_stub:
        return False
    if removed_ratio >= 0.20:
        return False

    hay = pipeline or raw or ""
    has_event = bool(CONCISE_OFFICIAL_EVENT_RE.search(hay))
    has_anchor = bool(CONCISE_OFFICIAL_ANCHOR_RE.search(hay))
    has_attribution = bool(CONCISE_OFFICIAL_ATTRIBUTION_RE.search(raw or hay))
    return has_event and has_anchor and has_attribution


def _strip_footer_noise(text: str) -> str:
    t = text or ""
    t = SOCIAL_FOOTER_RE.sub(" ", t)
    t = CTA_TAIL_RE.sub(" ", t)
    return t


def peel_story_wrappers(text: str) -> str:
    t = text or ""
    t = NEWSLETTER_WRAPPER_RE.sub("", t)
    t = NEWSLETTER_DIVIDER_RE.sub(" ", t)
    t = NEWSLETTER_CTA_RE.sub(" ", t)
    return MULTISPACE_RE.sub(" ", t).strip()


def strip_service_blocks(text: str) -> str:
    t = peel_story_wrappers(text or "")
    t = FOREIGN_AGENT_DISCLAIMER_FULL_RE.sub(" ", t)
    t = FOREIGN_AGENT_DISCLAIMER_PREFIX_RE.sub(" ", t)
    t = FOREIGN_AGENT_DISCLAIMER_INLINE_RE.sub(" ", t)
    t = AD_HASHTAG_RE.sub(" ", t)
    t = _strip_footer_noise(t)
    return t


def sanitize_pipeline_text(text: str) -> str:
    t = strip_service_blocks(text)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t


def classify_storyline_content(text: str) -> dict[str, object]:
    raw = text or ""
    peeled_raw = peel_story_wrappers(raw)
    pipeline = sanitize_pipeline_text(peeled_raw)
    lowered_raw = raw.lower()

    removed_ratio = 0.0
    raw_len = max(1, len((peeled_raw or raw).strip()))
    removed_ratio = max(0.0, min(1.0, 1.0 - (len(pipeline) / raw_len)))
    url_count = len(URL_HEAVY_RE.findall(raw))
    token_count = _token_count(pipeline)
    promo_score = promotional_text_score(raw)
    digest_like = bool(DIGEST_CUE_RE.search(raw))
    audio_video_stub = bool(VIDEO_STUB_RE.search(raw))
    footer_noise = bool(SOCIAL_FOOTER_RE.search(raw) or CTA_TAIL_RE.search(raw))
    low_signal_promo = bool(LOW_SIGNAL_PROMO_RE.search(raw))
    foreign_agent_visible = "настоящий материал" in lowered_raw or "иностранного агента" in lowered_raw
    wrapper_peeled = peeled_raw.strip() != raw.strip()
    seed_quality = seed_quality_score(
        pipeline_text=pipeline,
        token_count=token_count,
        removed_ratio=removed_ratio,
        promo_score=promo_score,
        digest_like=digest_like,
        audio_video_stub=audio_video_stub,
        wrapper_peeled=wrapper_peeled,
    )
    concise_official_update = _is_concise_official_update(
        raw=raw,
        pipeline=pipeline,
        token_count=token_count,
        url_count=url_count,
        promo_score=promo_score,
        digest_like=digest_like,
        audio_video_stub=audio_video_stub,
        removed_ratio=removed_ratio,
    )

    content_type = "news_story_candidate"
    content_state = "seed_candidate"
    reason = ""
    gate_score = 0.0

    if token_count <= 2 or len(pipeline.strip()) < 12:
        content_type = "boilerplate_heavy"
        content_state = "non_story"
        reason = "too_short_low_information"
        gate_score = 1.0
    elif promo_score >= 0.55 or (audio_video_stub and (promo_score >= 0.30 or low_signal_promo)):
        content_type = "promo_or_partner"
        content_state = "non_story"
        reason = "promo_markers"
        gate_score = max(promo_score, 0.55)
    elif audio_video_stub and (url_count >= 1 or token_count < 45):
        content_type = "audio_video_stub"
        content_state = "non_story"
        reason = "audio_video_stub"
        gate_score = 0.75
    elif digest_like:
        content_type = "digest_or_roundup"
        content_state = "story_update_only" if token_count >= 8 else "non_story"
        reason = "digest_markers"
        gate_score = 0.70
    elif removed_ratio >= 0.30 and token_count < 35:
        content_type = "boilerplate_heavy"
        content_state = "story_update_only" if token_count >= 8 else "non_story"
        reason = "boilerplate_ratio"
        gate_score = round(removed_ratio, 4)

    if content_state == "seed_candidate" and seed_quality < 0.45:
        if concise_official_update:
            reason = "concise_official_update"
            gate_score = 0.0
        else:
            content_state = "story_update_only" if token_count >= 8 else "non_story"
            if not reason:
                reason = "low_seed_quality"
                gate_score = round(max(gate_score, 1.0 - seed_quality), 4)

    is_storyline_candidate = content_state != "non_story"
    is_seed_candidate = content_state == "seed_candidate"

    return {
        "content_type": content_type,
        "content_state": content_state,
        "reason": reason,
        "gate_score": round(gate_score, 4),
        "is_storyline_candidate": is_storyline_candidate,
        "is_seed_candidate": is_seed_candidate,
        "seed_quality_score": seed_quality,
        "pipeline_text": pipeline,
        "peeled_text": peeled_raw,
        "removed_ratio": round(removed_ratio, 4),
        "token_count": token_count,
        "url_count": url_count,
        "promo_score": round(promo_score, 4),
        "audio_video_stub": audio_video_stub,
        "digest_like": digest_like,
        "footer_noise": footer_noise,
        "foreign_agent_visible": foreign_agent_visible,
        "wrapper_peeled": wrapper_peeled,
        "concise_official_update": concise_official_update,
    }

def _base_clean(text: str) -> str:
    t = sanitize_pipeline_text(text)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = BRACKETS_RE.sub(" ", t)
    t = STAR_RE.sub("", t)
    t = QUOTE_RE.sub("", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

def clean_for_ner(text: str) -> str:
    return _base_clean(text)

def clean_for_re(text: str) -> str:
    t = _base_clean(text)
    t = LEADIN_RE.sub("", t)
    return t
