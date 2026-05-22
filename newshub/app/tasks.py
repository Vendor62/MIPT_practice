import asyncio
import hashlib
import json
import os
import random
import re
import threading
import html
from pathlib import Path
from datetime import datetime, timedelta, timezone

import structlog
from aiogram import Bot, types
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from sqlalchemy import case, func, select, text, update
from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError

from app.celery_app import celery_app
from app.experiments import ensure_aa_assignment, record_aa_exposure
from app.models import (
    Click,
    Community,
    DigestRun,
    Post,
    PostTopic,
    Reaction,
    User,
    UserBillingState,
    UserCommunity,
    UserEmbeddingProfile,
    UserKeywordStat,
    UserModelState,
    UserNewsTimeSurvey,
    PaymentEvent,
    PaymentOrder,
    SubscriptionRequest,
    UserStorylineFollow,
    StorylineUpdateEvent,
    UserDeliveryPreference,
    get_session,
)
from app.delivery_prefs import (
    PLUS_BOT_TOKEN,
    MAIN_BOT_TOKEN,
    resolve_delivery_route,
    resolve_delivery_route_from_pref,
    bulk_load_delivery_prefs,
)
from app.notifications import (
    build_tracking_link,
    render_digest_pages,
    send_digest_first_page,
    send_digest_to_user,
    send_post_to_user,
    source_link_tracking_enabled,
)
from app.storyline_mvp import (
    get_storyline_context_for_post,
    get_storyline_context_for_storyline_id,
    get_update_step_for_post,
)
from app.storytracking_rollout import storytracking_allowed_for_user
from app.ai.deepseek import (
    summarize,
    DeepseekRetryableError,
    group_digest_stories,
    generate_digest_headline,
    arbitrate_user_storyline_follow_match,
    generate_storyline_title,
)
from celery.exceptions import SoftTimeLimitExceeded
# TODO[showcase]: stub after removing payments/promo/userbot/subscribe modules from public version
# from app.subscription_operations import ensure_subscription_request_operation
from app.i18n import button_text, get_user_locale, t
from app.news_time_survey import NEWS_TIME_FOLLOWUP_DELAY, NEWS_TIME_OPTION_KEYS, followup_due_at
from app.instruction_filters import (
    INSTRUCTION_FILTER_ENABLED,
    ResolvedInstructionRule,
    get_or_create_instruction_decision,
    merge_instruction_and_reco,
    resolve_effective_instruction_rules_for_users,
    user_has_instruction_filter_access,
)
from app.reco import (
    FEATURE_NAMES,
    baseline_score,
    blend_scores,
    cosine_similarity,
    decision_for_mode,
    logistic_probability,
    reco_error_fallback_decision,
    shadow_mode_decisions,
)
from app.reco_embeddings import (
    RECO_EMBEDDING_SOURCE,
    get_post_embedding,
    get_post_embeddings_dual_probe,
)
from app.topic_worker import process_post_topics
# TODO[showcase]: stub after removing payments/promo/userbot/subscribe modules from public version
# from app.billing import recalculate_user_billing_state, resolve_premium_entitlement
# from app.payments.service import (
#     enforce_degrade_if_needed,
#     repair_confirmed_payment_entitlements,
#     sync_recent_pending_payments,
# )


log = structlog.get_logger()

DIGEST_TASK_SOFT_TIME_LIMIT = int(os.getenv("DIGEST_TASK_SOFT_TIME_LIMIT", "3300"))
DIGEST_TASK_TIME_LIMIT = int(os.getenv("DIGEST_TASK_TIME_LIMIT", "3600"))
DIGEST_SENDING_STALE_MINUTES = int(
    os.getenv("DIGEST_SENDING_STALE_MINUTES", str(max(15, (DIGEST_TASK_TIME_LIMIT // 60) + 5)))
)

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_ready = threading.Event()

SEND_CONCURRENCY = int(os.getenv("SEND_CONCURRENCY", "10"))
DEEPSEEK_MAX_RETRIES = int(os.getenv("DEEPSEEK_MAX_RETRIES", "5"))
DB_MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "3"))
SUMMARY_MIN_CHARS = int(os.getenv("SUMMARY_MIN_CHARS", "200"))
FILTER_COLD_START_MIN_EVENTS = int(os.getenv("FILTER_COLD_START_MIN_EVENTS", "8"))
FILTER_THRESHOLD_NOT_INTERESTING = float(os.getenv("FILTER_THRESHOLD_NOT_INTERESTING", "0.35"))
FILTER_THRESHOLD_ONLY_FIRE = float(os.getenv("FILTER_THRESHOLD_ONLY_FIRE", "0.75"))
FILTER_EXPLORATION_EPSILON = float(os.getenv("FILTER_EXPLORATION_EPSILON", "0.07"))
RECO_BLEND_ALPHA = float(os.getenv("RECO_BLEND_ALPHA", "0.7"))
RECO_ENABLE_BASELINE = os.getenv("RECO_ENABLE_BASELINE", "true").lower() == "true"
RECO_ENABLE_LOGISTIC = os.getenv("RECO_ENABLE_LOGISTIC", "true").lower() == "true"
RECO_ENABLE_BANDIT = os.getenv("RECO_ENABLE_BANDIT", "true").lower() == "true"
RECO_DRY_RUN = os.getenv("RECO_DRY_RUN", "false").lower() == "true"
RECO_DUAL_PROBE_SAMPLE_RATE = float(os.getenv("RECO_DUAL_PROBE_SAMPLE_RATE", "0.1"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
BOT_PLUS_USERNAME = (os.getenv("BOT_PLUS_USERNAME", "") or "").strip().lstrip("@")
PREMIUM_WEEK_REMINDER_LOG_PATH = os.getenv(
    "PREMIUM_WEEK_REMINDER_LOG_PATH",
    "/app/logs/premium_week_reminders.jsonl",
)
PREMIUM_WEEK_REMINDER_EVENT = "premium_week_after_first_subscription"
NEWS_TIME_SURVEY_REDELIVERY_LIMIT = max(1, int(os.getenv("NEWS_TIME_SURVEY_REDELIVERY_LIMIT", "500")))
NEWS_TIME_BASELINE_REDELIVERY_LIMIT = max(1, int(os.getenv("NEWS_TIME_BASELINE_REDELIVERY_LIMIT", "500")))
NEWS_TIME_CURRENT_REDELIVERY_LIMIT = max(1, int(os.getenv("NEWS_TIME_CURRENT_REDELIVERY_LIMIT", "500")))
STORYLINE_DISPATCH_DELAY_SECONDS = max(0, int(os.getenv("STORYLINE_DISPATCH_DELAY_SECONDS", "180")))
STORYLINE_DISPATCH_IE_RETRY_SECONDS = max(10, int(os.getenv("STORYLINE_DISPATCH_IE_RETRY_SECONDS", "120")))
STORYLINE_DISPATCH_IE_MAX_RETRIES = max(0, int(os.getenv("STORYLINE_DISPATCH_IE_MAX_RETRIES", "6")))

_SENT_END_RE = re.compile(r"[.!?…]\s")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.|t\.me/)\S+")
_TG_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{3,}")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+"
)
_STORYLINE_TOKEN_RE = re.compile(r"[a-zа-яё0-9-]{4,}", re.IGNORECASE)
_STORYLINE_FOLLOW_FALLBACK_STOPWORDS = {
    "сюжет",
    "новость",
    "новости",
    "история",
    "обновление",
    "программа",
    "модуль",
    "система",
    "системы",
    "станция",
    "станции",
    "миссия",
    "экипаж",
    "проект",
    "после",
    "снова",
    "сегодня",
    "завтра",
    "вчера",
    "также",
    "через",
    "между",
    "когда",
    "пост",
}
_STORYLINE_FOLLOW_WEAK_TITLE_TOKENS = {
    "дело",
    "суд",
    "сюжет",
    "новость",
    "обновление",
    "подозревается",
    "обвиняется",
    "заявил",
    "заявила",
    "заявили",
    "млн",
    "млрд",
    "миллион",
    "миллионов",
    "миллиард",
    "миллиардов",
}
_STORYLINE_FOLLOW_GENERIC_FOCUS_TOKENS = {
    "блокировк",
    "данн",
    "друг",
    "зарубежн",
    "иностранн",
    "компан",
    "нарушен",
    "пользовател",
    "росс",
    "российск",
    "россиян",
    "сервис",
    "сервер",
    "хранен",
}
_STORYLINE_SIGNATURE_TECH_TOKENS = {
    "entity",
    "event",
    "frame",
    "general",
    "kind",
    "law",
    "location",
    "news",
    "organization",
    "pair",
    "person",
    "product",
    "subtype",
    "term",
    "topic",
    "update",
    "этого",
    "организация",
    "персона",
    "продукт",
}
_STORYLINE_WEAK_ANCHOR_MARKERS = _STORYLINE_FOLLOW_GENERIC_FOCUS_TOKENS | {
    "американск",
    "официальн",
    "популярн",
}
_DIGEST_NOISE_LINE_RE = re.compile(
    r"(?i)^\s*(?:[^\wа-яА-Я]{0,3}\s*)?(?:соцсети|источник|source|via|credits?)\b"
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _schedule_storyline_dispatch_after_ie(
    post_id: int,
    *,
    reason: str,
    countdown: int | None = None,
) -> None:
    delay = STORYLINE_DISPATCH_DELAY_SECONDS if countdown is None else max(0, int(countdown))
    try:
        celery_app.send_task(
            "app.tasks.dispatch_storyline_updates_for_post",
            args=[int(post_id)],
            queue="dispatch_queue",
            countdown=delay,
        )
        log.info(
            "storyline_dispatch_scheduled",
            post_id=int(post_id),
            countdown=delay,
            reason=reason,
        )
    except Exception as exc:
        log.warning(
            "storyline_dispatch_schedule_failed",
            post_id=int(post_id),
            countdown=delay,
            reason=reason,
            error=str(exc),
        )


async def _send_billing_message(
    telegram_id: int,
    text: str,
    *,
    reply_markup: types.InlineKeyboardMarkup | None = None,
) -> bool:
    if not TELEGRAM_BOT_TOKEN:
        return False
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        await bot.send_message(chat_id=telegram_id, text=text, reply_markup=reply_markup)
        return True
    except Exception as exc:
        log.warning("billing.reminder_send_failed", telegram_id=telegram_id, error=str(exc))
        return False
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass


async def _send_storyline_followup_message(telegram_id: int, text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN:
        return False
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        await bot.send_message(chat_id=telegram_id, text=text)
        return True
    except Exception as exc:
        log.warning("storyline.followup_send_failed", telegram_id=telegram_id, error=str(exc))
        return False
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass


async def _send_digest_plus_delivery_notice(telegram_id: int) -> bool:
    if not TELEGRAM_BOT_TOKEN:
        return False
    target = f"@{BOT_PLUS_USERNAME}" if BOT_PLUS_USERNAME else "второго бота"
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        await bot.send_message(
            chat_id=telegram_id,
            text=f"Дайджест отправлен в {target}",
        )
        return True
    except Exception as exc:
        log.warning("digest.plus_notice_send_failed", telegram_id=telegram_id, error=str(exc))
        return False
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass


def _news_time_saved_keyboard(locale: str) -> types.InlineKeyboardMarkup:
    return types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text=button_text("news_time_saved_yes", locale),
                    callback_data="news_time:saved:yes",
                ),
                types.InlineKeyboardButton(
                    text=button_text("news_time_saved_no", locale),
                    callback_data="news_time:saved:no",
                ),
            ]
        ]
    )


def _news_time_baseline_keyboard(locale: str) -> types.InlineKeyboardMarkup:
    buttons = [
        types.InlineKeyboardButton(
            text=button_text(option_key, locale),
            callback_data=f"news_time:opt:{option_key}",
        )
        for option_key in NEWS_TIME_OPTION_KEYS
    ]
    rows = [buttons[idx : idx + 3] for idx in range(0, len(buttons), 3)]
    return types.InlineKeyboardMarkup(inline_keyboard=rows)


async def _send_news_time_followup_prompt(*, telegram_id: int, locale: str) -> tuple[bool, str | None]:
    if not TELEGRAM_BOT_TOKEN:
        return False
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        await bot.send_message(
            chat_id=telegram_id,
            text=t(locale, "news_time_saved_question"),
            reply_markup=_news_time_saved_keyboard(locale),
        )
        return True, None
    except TelegramForbiddenError as exc:
        log.warning(
            "news_time_survey.redelivery_forbidden",
            telegram_id=telegram_id,
            error=str(exc),
        )
        return False, str(exc)
    except Exception as exc:
        log.warning(
            "news_time_survey.redelivery_send_failed",
            telegram_id=telegram_id,
            error=str(exc),
        )
        return False, str(exc)
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass


async def _send_news_time_baseline_prompt(*, telegram_id: int, locale: str) -> tuple[bool, str | None]:
    if not TELEGRAM_BOT_TOKEN:
        return False
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        await bot.send_message(
            chat_id=telegram_id,
            text=t(locale, "news_time_baseline_question"),
            reply_markup=_news_time_baseline_keyboard(locale),
        )
        return True, None
    except TelegramForbiddenError as exc:
        log.warning(
            "news_time_survey.baseline_redelivery_forbidden",
            telegram_id=telegram_id,
            error=str(exc),
        )
        return False, str(exc)
    except Exception as exc:
        log.warning(
            "news_time_survey.baseline_redelivery_send_failed",
            telegram_id=telegram_id,
            error=str(exc),
        )
        return False, str(exc)
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass


async def _send_news_time_current_prompt(*, telegram_id: int, locale: str) -> tuple[bool, str | None]:
    if not TELEGRAM_BOT_TOKEN:
        return False
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        await bot.send_message(
            chat_id=telegram_id,
            text=t(locale, "news_time_current_question"),
            reply_markup=_news_time_baseline_keyboard(locale),
        )
        return True, None
    except TelegramForbiddenError as exc:
        log.warning(
            "news_time_survey.current_redelivery_forbidden",
            telegram_id=telegram_id,
            error=str(exc),
        )
        return False, str(exc)
    except Exception as exc:
        log.warning(
            "news_time_survey.current_redelivery_send_failed",
            telegram_id=telegram_id,
            error=str(exc),
        )
        return False, str(exc)
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass


def _storyline_title_needs_humanization(value: str | None) -> bool:
    title = str(value or "").strip()
    if not title:
        return True
    lowered = title.lower()
    if lowered.startswith("storyline_"):
        return True
    if "[general]" in lowered or "[seed" in lowered:
        return True
    if len(title) < 4:
        return True
    return False


async def _resolve_storyline_human_title(
    *,
    follow: UserStorylineFollow,
    fallback_title: str,
    seed_text: str,
) -> str:
    current = str(getattr(follow, "storyline_title", None) or "").strip()
    if current and not _storyline_title_needs_humanization(current):
        return current

    candidate = None
    try:
        candidate = await generate_storyline_title(
            seed_post_text=seed_text,
            current_title=current or fallback_title,
        )
    except Exception as exc:
        log.warning(
            "storyline.title_generation_failed",
            follow_id=getattr(follow, "id", None),
            family_root_storyline_id=str(getattr(follow, "family_root_storyline_id", "") or ""),
            error=str(exc),
        )

    return str(candidate or current or fallback_title).strip()[:500] or fallback_title


def _build_storyline_update_keyboard(
    *,
    locale: str,
    post_id: int,
    family_root_storyline_id: str,
    timeline_anchor_post_id: int | None = None,
) -> types.InlineKeyboardMarkup:
    timeline_callback = f"storytimelinecard:{post_id}"
    if timeline_anchor_post_id and int(timeline_anchor_post_id) > 0 and int(timeline_anchor_post_id) != int(post_id):
        timeline_callback = f"storytimelinecard:{post_id}:{int(timeline_anchor_post_id)}"
    return types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text=t(locale, "storyline_timeline_button"),
                    callback_data=timeline_callback,
                ),
                types.InlineKeyboardButton(
                    text=t(locale, "storyline_unfollow_button"),
                    callback_data=f"storyunfollowroot:{post_id}:{family_root_storyline_id}",
                ),
            ]
        ]
    )


def _storyline_update_source_key(source: dict[str, object]) -> tuple[int, str]:
    return (
        int(source.get("post_id") or 0),
        str(source.get("url") or "").strip(),
    )


def _merge_storyline_update_sources(*source_lists: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()
    for source_list in source_lists:
        for raw_source in source_list or []:
            source = {
                "post_id": int(raw_source.get("post_id") or 0),
                "label": str(raw_source.get("label") or raw_source.get("source") or "").strip(),
                "url": str(raw_source.get("url") or "").strip(),
                "timestamp": str(raw_source.get("timestamp") or "").strip(),
            }
            key = _storyline_update_source_key(source)
            if key in seen:
                continue
            seen.add(key)
            merged.append(source)
    return merged


_STORYLINE_CARD_TRAILING_CTA_RE = re.compile(
    r"(?i)\b(?:подписаться|подпишись|подписывайтесь|подписывайся|подписка|subscribe|follow)\b"
)


def _clean_storyline_update_summary_text(value: str | None) -> str:
    text_value = _EMOJI_RE.sub("", str(value or ""))
    text_value = text_value.replace("\ufe0f", "").replace("\u200d", "")
    text_value = text_value.replace("\u00a0", " ")
    lines = [line.strip() for line in text_value.splitlines()]

    while lines and not lines[-1]:
        lines.pop()

    # Source CTAs usually live in the last one-two lines; keep body text intact.
    while lines and _STORYLINE_CARD_TRAILING_CTA_RE.search(lines[-1]):
        lines.pop()
        while lines and not lines[-1]:
            lines.pop()

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned or str(value or "").strip()


def _render_storyline_update_card_text(
    *,
    locale: str,
    title: str,
    summary_text: str,
    sources: list[dict[str, object]],
    telegram_id: int,
) -> str:
    safe_title = html.escape(str(title or "").strip())
    safe_summary = html.escape(_clean_storyline_update_summary_text(summary_text))
    tracking_enabled = source_link_tracking_enabled()

    source_lines: list[str] = []
    for source in _merge_storyline_update_sources(sources):
        raw_post_id = int(source.get("post_id") or 0)
        source_url = str(source.get("url") or "").strip()
        source_label = str(source.get("label") or source.get("source") or t(locale, "notif_source")).strip()
        tracked_source_url = (
            build_tracking_link(raw_post_id, telegram_id, source="storyline_card")
            if tracking_enabled and raw_post_id > 0 and source_url
            else source_url
        )
        if tracked_source_url:
            source_lines.append(f'- <a href="{html.escape(tracked_source_url)}">{html.escape(source_label)}</a>')
        else:
            source_lines.append(f"- {html.escape(source_label)}")

    if not source_lines:
        source_lines.append(f"- {html.escape(t(locale, 'notif_source'))}")

    sources_header = "Источники:" if locale == "ru" else "Sources:"
    sources_block = "\n".join([html.escape(sources_header), *source_lines])
    return (
        f"<b>{t(locale, 'storyline_update_card_header')}</b>\n"
        f"<i>{safe_title}</i>\n\n"
        f"{safe_summary}\n\n"
        f"{sources_block}\n\n"
        f"<i>Отслеживание сюжетов находится в фазе тестирования и активной доработки. Могут возникать ошибки.</i>"
    )


async def _send_storyline_update_card(
    *,
    telegram_id: int,
    locale: str,
    title: str,
    summary_text: str,
    source_url: str | None,
    source_label: str | None,
    post_id: int,
    family_root_storyline_id: str,
    timeline_anchor_post_id: int | None = None,
    bot_token_override: str | None = None,
    delivery_target: str = "main",
) -> dict[str, object] | None:
    resolved_token = str(bot_token_override or TELEGRAM_BOT_TOKEN or "").strip()
    if not resolved_token:
        return None
    bot = Bot(token=resolved_token)
    try:
        text = _render_storyline_update_card_text(
            locale=locale,
            title=title,
            summary_text=summary_text,
            sources=[
                {
                    "post_id": int(post_id),
                    "label": source_label or t(locale, "notif_source"),
                    "url": source_url or "",
                }
            ],
            telegram_id=int(telegram_id),
        )
        msg = await bot.send_message(
            chat_id=telegram_id,
            text=text,
            parse_mode="HTML",
            disable_web_page_preview=False,
            reply_markup=_build_storyline_update_keyboard(
                locale=locale,
                post_id=post_id,
                family_root_storyline_id=family_root_storyline_id,
                timeline_anchor_post_id=timeline_anchor_post_id,
            ),
        )
        return {
            "message_id": int(getattr(msg, "message_id", 0) or 0) or None,
            "delivery_result": f"storyline_update_card:{delivery_target}",
        }
    except Exception as exc:
        log.warning("storyline.card_send_failed", telegram_id=telegram_id, post_id=post_id, error=str(exc))
        return None
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass


def _storyline_event_token_for_target(delivery_target: str, fallback_token: str | None = None) -> str:
    target = str(delivery_target or "").strip()
    if target == "plus" and PLUS_BOT_TOKEN:
        return PLUS_BOT_TOKEN
    if target in {"main", "main_fallback"} and MAIN_BOT_TOKEN:
        return MAIN_BOT_TOKEN
    return str(fallback_token or TELEGRAM_BOT_TOKEN or MAIN_BOT_TOKEN or "").strip()


async def _edit_storyline_update_card(
    *,
    telegram_id: int,
    message_id: int,
    locale: str,
    title: str,
    summary_text: str,
    sources: list[dict[str, object]],
    post_id: int,
    family_root_storyline_id: str,
    timeline_anchor_post_id: int | None,
    bot_token_override: str | None,
) -> bool:
    resolved_token = str(bot_token_override or TELEGRAM_BOT_TOKEN or "").strip()
    if not resolved_token or message_id <= 0:
        return False
    bot = Bot(token=resolved_token)
    try:
        await bot.edit_message_text(
            chat_id=int(telegram_id),
            message_id=int(message_id),
            text=_render_storyline_update_card_text(
                locale=locale,
                title=title,
                summary_text=summary_text,
                sources=sources,
                telegram_id=int(telegram_id),
            ),
            parse_mode="HTML",
            disable_web_page_preview=False,
            reply_markup=_build_storyline_update_keyboard(
                locale=locale,
                post_id=int(post_id),
                family_root_storyline_id=family_root_storyline_id,
                timeline_anchor_post_id=timeline_anchor_post_id,
            ),
        )
        return True
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc).lower():
            return True
        log.warning(
            "storyline.card_edit_bad_request",
            telegram_id=telegram_id,
            message_id=message_id,
            post_id=post_id,
            error=str(exc),
        )
        return False
    except Exception as exc:
        log.warning(
            "storyline.card_edit_failed",
            telegram_id=telegram_id,
            message_id=message_id,
            post_id=post_id,
            error=str(exc),
        )
        return False
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass


def _normalize_storyline_event_text(value: str | None) -> str:
    text_value = str(value or "").lower()
    text_value = re.sub(r"https?://\S+", " ", text_value)
    text_value = re.sub(r"@\w+", " ", text_value)
    text_value = re.sub(r"[^\w\sа-яё-]", " ", text_value, flags=re.IGNORECASE)
    tokens = [
        token.strip("-")
        for token in re.findall(r"[a-zа-яё0-9-]{4,}", text_value, flags=re.IGNORECASE)
    ]
    stopwords = {
        "сообщают", "заявил", "заявила", "заявили", "будет", "после", "перед", "этого",
        "этой", "который", "которая", "которые", "украины", "украине", "украина",
    }
    filtered = [token for token in tokens if token and token not in stopwords]
    return " ".join(filtered[:24])


def _fallback_storyline_event_key(post: Post, summary_text: str) -> str:
    ts = getattr(post, "timestamp", None)
    if isinstance(ts, datetime):
        bucket_start_hour = (ts.hour // 6) * 6
        bucket = ts.replace(hour=bucket_start_hour, minute=0, second=0, microsecond=0).isoformat()
    else:
        bucket = "unknown"
    normalized = _normalize_storyline_event_text(
        summary_text or getattr(post, "summary", None) or getattr(post, "content", None)
    )
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16] if normalized else f"post_{int(post.id)}"
    return f"fallback:{bucket}:{digest}"


def _storyline_event_similarity(current_text: str | None, existing_text: str | None) -> dict[str, object]:
    current_tokens = set(_normalize_storyline_event_text(current_text).split())
    existing_tokens = set(_normalize_storyline_event_text(existing_text).split())
    if not current_tokens or not existing_tokens:
        return {"matched": False, "jaccard": 0.0, "overlap": 0.0, "shared": []}

    shared = current_tokens & existing_tokens
    union = current_tokens | existing_tokens
    jaccard = len(shared) / max(1, len(union))
    overlap = len(shared) / max(1, min(len(current_tokens), len(existing_tokens)))
    # Same-follow recent updates often arrive as short paraphrases from different channels.
    # Keep the default gate strict, but allow a high-shared-token gray zone so
    # near-identical reposts do not produce separate storyline cards.
    matched = len(shared) >= 5 and (
        jaccard >= 0.42
        or overlap >= 0.62
        or (len(shared) >= 8 and jaccard >= 0.36 and overlap >= 0.52)
    )
    return {
        "matched": matched,
        "jaccard": round(jaccard, 4),
        "overlap": round(overlap, 4),
        "shared": sorted(shared)[:12],
    }


async def _send_or_merge_storyline_update_card(
    *,
    user: User,
    follow: UserStorylineFollow,
    locale: str,
    title: str,
    summary_text: str,
    source_url: str | None,
    source_label: str | None,
    post: Post,
    family_root_storyline_id: str,
    event_key: str,
    route_token: str | None,
    delivery_target: str,
) -> dict[str, object] | None:
    summary_text = _clean_storyline_update_summary_text(summary_text)
    follow_id = int(getattr(follow, "id", 0) or 0)
    if follow_id <= 0:
        return await _send_storyline_update_card(
            telegram_id=user.telegram_id,
            locale=locale,
            title=title,
            summary_text=summary_text,
            source_url=source_url,
            source_label=source_label,
            post_id=int(post.id),
            family_root_storyline_id=family_root_storyline_id,
            timeline_anchor_post_id=int(getattr(follow, "source_post_id", 0) or 0) or None,
            bot_token_override=route_token,
            delivery_target=delivery_target,
        )

    current_source = {
        "post_id": int(post.id),
        "label": str(source_label or t(locale, "notif_source")).strip(),
        "url": str(source_url or "").strip(),
        "timestamp": getattr(post, "timestamp", None).isoformat() if getattr(post, "timestamp", None) else "",
    }
    initial_sources = _merge_storyline_update_sources([current_source])
    family_root = str(getattr(follow, "family_root_storyline_id", "") or family_root_storyline_id).strip()
    canonical_post_id = int(getattr(follow, "source_post_id", 0) or 0) or int(post.id)
    timeline_anchor_post_id = int(getattr(follow, "source_post_id", 0) or 0) or None
    delivery_target_key = str(delivery_target or "main")
    effective_event_key = str(event_key or "").strip() or _fallback_storyline_event_key(post, summary_text)

    async for event_session in get_session():
        inserted_event_id = None
        try:
            recent_rows = await event_session.execute(
                text(
                    """
                    SELECT id, event_key, title, summary, last_seen_at
                    FROM storyline_update_events
                    WHERE user_id = :user_id
                      AND follow_id = :follow_id
                      AND family_root_storyline_id = :family_root_storyline_id
                      AND delivery_target = :delivery_target
                      AND last_seen_at >= now() - interval '8 hours'
                    ORDER BY last_seen_at DESC
                    LIMIT 25
                    """
                ),
                {
                    "user_id": int(user.id),
                    "follow_id": follow_id,
                    "family_root_storyline_id": family_root,
                    "delivery_target": delivery_target_key,
                },
            )
            best_match: tuple[str, dict[str, object]] | None = None
            for recent in recent_rows.mappings():
                recent_event_key = str(recent.get("event_key") or "").strip()
                if not recent_event_key or recent_event_key == effective_event_key:
                    continue
                similarity = _storyline_event_similarity(
                    summary_text,
                    str(recent.get("summary") or recent.get("title") or ""),
                )
                if similarity.get("matched"):
                    best_match = (recent_event_key, similarity)
                    break
            if best_match:
                previous_event_key, similarity = best_match
                log.info(
                    "storyline.update_event_similar_recent_match",
                    post_id=post.id,
                    user_id=user.id,
                    follow_id=follow_id,
                    event_key=effective_event_key,
                    matched_event_key=previous_event_key,
                    jaccard=similarity.get("jaccard"),
                    overlap=similarity.get("overlap"),
                    shared=similarity.get("shared"),
                )
                effective_event_key = previous_event_key

            result = await event_session.execute(
                text(
                    """
                    INSERT INTO storyline_update_events (
                        user_id, follow_id, family_root_storyline_id, event_key,
                        canonical_post_id, last_post_id, delivery_target,
                        title, summary, sources, source_count
                    )
                    VALUES (
                        :user_id, :follow_id, :family_root_storyline_id, :event_key,
                        :canonical_post_id, :last_post_id, :delivery_target,
                        :title, :summary, CAST(:sources AS jsonb), :source_count
                    )
                    ON CONFLICT (
                        user_id, follow_id, family_root_storyline_id, event_key, delivery_target
                    ) DO NOTHING
                    RETURNING id
                    """
                ),
                {
                    "user_id": int(user.id),
                    "follow_id": follow_id,
                    "family_root_storyline_id": family_root,
                    "event_key": effective_event_key,
                    "canonical_post_id": canonical_post_id,
                    "last_post_id": int(post.id),
                    "delivery_target": delivery_target_key,
                    "title": str(title or "").strip(),
                    "summary": str(summary_text or "").strip(),
                    "sources": json.dumps(initial_sources, ensure_ascii=False),
                    "source_count": len(initial_sources),
                },
            )
            inserted_event_id = result.scalar_one_or_none()
            await event_session.commit()
        except Exception:
            await event_session.rollback()
            log.exception(
                "storyline.update_event_claim_failed",
                post_id=post.id,
                user_id=user.id,
                follow_id=follow_id,
                event_key=effective_event_key,
            )
            inserted_event_id = None

        if inserted_event_id:
            delivery = await _send_storyline_update_card(
                telegram_id=user.telegram_id,
                locale=locale,
                title=title,
                summary_text=summary_text,
                source_url=source_url,
                source_label=source_label,
                post_id=int(post.id),
                family_root_storyline_id=family_root,
                timeline_anchor_post_id=timeline_anchor_post_id,
                bot_token_override=route_token,
                delivery_target=delivery_target,
            )
            if not delivery:
                try:
                    await event_session.execute(
                        text("DELETE FROM storyline_update_events WHERE id = :event_id"),
                        {"event_id": int(inserted_event_id)},
                    )
                    await event_session.commit()
                except Exception:
                    await event_session.rollback()
                return None

            message_id = int((delivery or {}).get("message_id") or 0) or None
            try:
                await event_session.execute(
                    text(
                        """
                        UPDATE storyline_update_events
                        SET telegram_message_id = :message_id,
                            delivery_result = :delivery_result,
                            updated_at = now(),
                            last_seen_at = now()
                        WHERE id = :event_id
                        """
                    ),
                    {
                        "event_id": int(inserted_event_id),
                        "message_id": message_id,
                        "delivery_result": str((delivery or {}).get("delivery_result") or ""),
                    },
                )
                await event_session.commit()
            except Exception:
                await event_session.rollback()
                log.warning(
                    "storyline.update_event_message_marker_failed",
                    event_id=int(inserted_event_id),
                    post_id=post.id,
                    user_id=user.id,
                )
            return delivery

        try:
            row = await event_session.execute(
                text(
                    """
                    SELECT id, telegram_message_id, delivery_target, title, summary, sources
                    FROM storyline_update_events
                    WHERE user_id = :user_id
                      AND follow_id = :follow_id
                      AND family_root_storyline_id = :family_root_storyline_id
                      AND event_key = :event_key
                      AND delivery_target = :delivery_target
                    LIMIT 1
                    """
                ),
                {
                    "user_id": int(user.id),
                    "follow_id": follow_id,
                    "family_root_storyline_id": family_root,
                    "event_key": effective_event_key,
                    "delivery_target": delivery_target_key,
                },
            )
            existing = row.mappings().first()
            if not existing:
                return None
            existing_sources = existing.get("sources") or []
            if isinstance(existing_sources, str):
                try:
                    existing_sources = json.loads(existing_sources)
                except json.JSONDecodeError:
                    existing_sources = []
            merged_sources = _merge_storyline_update_sources(list(existing_sources or []), initial_sources)
            await event_session.execute(
                text(
                    """
                    UPDATE storyline_update_events
                    SET sources = CAST(:sources AS jsonb),
                        source_count = :source_count,
                        last_post_id = :last_post_id,
                        last_seen_at = now(),
                        updated_at = now()
                    WHERE id = :event_id
                    """
                ),
                {
                    "event_id": int(existing["id"]),
                    "sources": json.dumps(merged_sources, ensure_ascii=False),
                    "source_count": len(merged_sources),
                    "last_post_id": int(post.id),
                },
            )
            await event_session.commit()
        except Exception:
            await event_session.rollback()
            log.exception(
                "storyline.update_event_merge_failed",
                post_id=post.id,
                user_id=user.id,
                follow_id=follow_id,
                event_key=effective_event_key,
            )
            return None

        message_id = int(existing.get("telegram_message_id") or 0)
        token = _storyline_event_token_for_target(str(existing.get("delivery_target") or delivery_target), route_token)
        edited = False
        if message_id > 0:
            edited = await _edit_storyline_update_card(
                telegram_id=int(user.telegram_id),
                message_id=message_id,
                locale=locale,
                title=str(existing.get("title") or title),
                summary_text=str(existing.get("summary") or summary_text),
                sources=merged_sources,
                post_id=int(post.id),
                family_root_storyline_id=family_root,
                timeline_anchor_post_id=timeline_anchor_post_id,
                bot_token_override=token,
            )
        return {
            "message_id": message_id or None,
            "delivery_result": "storyline_update_card:merged_edit" if edited else "storyline_update_card:merged",
            "merged": True,
            "source_count": len(merged_sources),
        }

    return None


def _premium_week_reminder_keyboard(locale: str) -> types.InlineKeyboardMarkup:
    return types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text=t(locale, "premium_week_learn_more"),
                    callback_data="premium_week:billing",
                )
            ]
        ]
    )


def _read_premium_week_reminder_user_ids(path: str | Path | None = None) -> set[int]:
    marker_path = Path(path or PREMIUM_WEEK_REMINDER_LOG_PATH)
    if not marker_path.exists():
        return set()
    sent: set[int] = set()
    for line in marker_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") != PREMIUM_WEEK_REMINDER_EVENT:
            continue
        try:
            sent.add(int(record["user_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sent


def _mark_premium_week_reminder_sent(
    *,
    user_id: int,
    telegram_id: int,
    first_subscription_at: datetime,
    sent_at: datetime,
    path: str | Path | None = None,
) -> None:
    marker_path = Path(path or PREMIUM_WEEK_REMINDER_LOG_PATH)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": PREMIUM_WEEK_REMINDER_EVENT,
        "user_id": int(user_id),
        "telegram_id": int(telegram_id),
        "first_subscription_at": first_subscription_at.isoformat(),
        "sent_at": sent_at.isoformat(),
    }
    with marker_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _start_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop_ready.set()
    _loop.run_forever()


def _subscribed_user_ids_subquery():
    return select(UserCommunity.user_id).distinct().subquery()


def _ensure_loop():
    global _loop_thread
    if _loop_thread and _loop_ready.is_set():
        return
    _loop_thread = threading.Thread(target=_start_loop, daemon=True)
    _loop_thread.start()
    _loop_ready.wait()


def _run_async(coro, timeout=None):
    _ensure_loop()
    assert _loop is not None
    fut = asyncio.run_coroutine_threadsafe(coro, _loop)
    return fut.result(timeout=timeout)

def smart_trim(text: str, max_len: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text

    cut = text[:max_len].rstrip()

    # Пытаемся резать по концу предложения в хвосте
    tail = cut[max(0, len(cut) - 120):]
    ends = list(_SENT_END_RE.finditer(tail))
    if ends:
        cut = cut[:len(cut) - len(tail) + ends[-1].end()].rstrip()
    else:
        # иначе по последнему пробелу
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0].rstrip()

    return cut + "…"


def _storyline_follow_fallback_tokens(*values: object) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in _STORYLINE_TOKEN_RE.findall(str(value or "").lower()):
            normalized = token.strip("-")
            if (
                len(normalized) < 4
                or normalized in _STORYLINE_FOLLOW_FALLBACK_STOPWORDS
            ):
                continue
            tokens.add(normalized)
    return tokens


def _storyline_loose_token_base(token: str) -> str:
    normalized = str(token or "").strip("-").lower()
    if len(normalized) < 5:
        return normalized
    for suffix in (
        "ыми", "ими", "ого", "ему", "ами", "ями", "иях", "ией", "иям",
        "ии", "ие", "ия", "ом", "ем", "ах", "ях", "ый", "ий", "ая",
        "ое", "ые", "ых", "их", "ую", "юю", "а", "я", "е", "и", "у",
        "ю", "ы",
    ):
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 4:
            return normalized[: -len(suffix)]
    return normalized


def _storyline_follow_title_focus_tokens(title: str | None) -> set[str]:
    tokens = _storyline_follow_fallback_tokens(title)
    return {
        base
        for token in tokens
        for base in [_storyline_loose_token_base(token)]
        if token not in _STORYLINE_FOLLOW_WEAK_TITLE_TOKENS
        and base not in _STORYLINE_FOLLOW_WEAK_TITLE_TOKENS
        and not token.isdigit()
    }


async def _storyline_recent_focus_token_counts(
    session,
    tokens: set[str],
    *,
    lookback_hours: int = 12,
    limit: int = 1200,
) -> dict[str, int]:
    focus_tokens = {str(token or "").strip().lower() for token in tokens if str(token or "").strip()}
    if not focus_tokens:
        return {}
    try:
        rows = await session.execute(
            text(
                """
                SELECT coalesce(summary, processed_content, content, title, '') AS text_value
                FROM posts
                WHERE timestamp >= now() - (:lookback_hours * interval '1 hour')
                ORDER BY timestamp DESC
                LIMIT :limit
                """
            ),
            {"lookback_hours": int(lookback_hours), "limit": int(limit)},
        )
    except Exception:
        log.warning("storyline.recent_focus_token_counts_failed", tokens=sorted(focus_tokens)[:20])
        return {}

    counts = {token: 0 for token in focus_tokens}
    for row in rows:
        text_value = row[0]
        doc_tokens = {
            _storyline_loose_token_base(token)
            for token in _storyline_follow_fallback_tokens(text_value)
        }
        for token in focus_tokens & doc_tokens:
            counts[token] += 1
    return counts


def _storyline_follow_title_focus_evidence(
    post_text: str | None,
    title: str | None,
    recent_token_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    title_tokens = _storyline_follow_title_focus_tokens(title)
    if not title_tokens:
        return {"matched": True, "hits": [], "strong_hits": [], "broad_hits": [], "title_tokens": []}
    post_tokens = {
        _storyline_loose_token_base(token)
        for token in _storyline_follow_fallback_tokens(post_text)
    }
    hits = title_tokens & post_tokens
    counts = recent_token_counts or {}
    # A token is broad when it appears in many recent posts. This catches global actors/topics
    # without hardcoding names, while allowing rare entities to anchor a storyline by themselves.
    broad_hits = {token for token in hits if int(counts.get(token, 0) or 0) >= 60}
    generic_hits = hits & _STORYLINE_FOLLOW_GENERIC_FOCUS_TOKENS
    specific_title_tokens = title_tokens - _STORYLINE_FOLLOW_GENERIC_FOCUS_TOKENS
    specific_hits = hits & specific_title_tokens
    strong_hits = hits - broad_hits - generic_hits
    required_hits = len(title_tokens) if len(title_tokens) <= 2 else 2
    matched = bool(strong_hits or len(hits) >= required_hits)
    if specific_title_tokens and not specific_hits:
        matched = False
    return {
        "matched": matched,
        "hits": sorted(hits),
        "strong_hits": sorted(strong_hits),
        "broad_hits": sorted(broad_hits),
        "generic_hits": sorted(generic_hits),
        "specific_hits": sorted(specific_hits),
        "title_tokens": sorted(title_tokens),
    }


def _storyline_follow_title_has_specific_evidence(post_text: str | None, title: str | None) -> bool:
    return bool(_storyline_follow_title_focus_evidence(post_text, title).get("matched"))


def _normalize_storyline_text(value: str | None) -> str:
    text = str(value or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^\w\sа-яё-]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _storyline_named_anchor_terms(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9-]{2,}", raw):
        token = match.group(0).strip().lower()
        if len(token) < 4 or token in _STORYLINE_FOLLOW_FALLBACK_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _storyline_anchor_markers_from_context(context: dict[str, object] | None, *, limit: int = 5) -> list[str]:
    if not context:
        return []
    markers: list[str] = []
    seen: set[str] = set()
    for part in [
        str(context.get("storyline_title") or "").strip(),
        str(context.get("storyline_seed_preview") or "").strip(),
        str(context.get("macro_topic_title") or "").strip(),
    ]:
        for token in _storyline_named_anchor_terms(part):
            base = _storyline_loose_token_base(token)
            if base in _STORYLINE_WEAK_ANCHOR_MARKERS:
                continue
            if token in seen:
                continue
            seen.add(token)
            markers.append(token)
            if len(markers) >= limit:
                return markers
    return markers


def _storyline_text_marker_hits(text: str | None, markers: list[str]) -> int:
    hay = _normalize_storyline_text(text)
    if not hay or not markers:
        return 0
    padded = f" {hay} "
    hits = 0
    for marker in markers:
        normalized_marker = _normalize_storyline_text(marker)
        if not normalized_marker:
            continue
        marker_parts = [part for part in normalized_marker.split() if len(part) >= 4]
        if not marker_parts:
            continue
        phrase = " ".join(marker_parts)
        if f" {phrase} " in padded:
            hits += 1
    return hits


def _storyline_signature_tokens(context: dict[str, object] | None) -> set[str]:
    if not context:
        return set()
    tokens: set[str] = set()
    for raw in [
        *(context.get("family_signatures") or []),
        *(context.get("topic_signatures") or []),
        *(context.get("macro_topic_signatures") or []),
    ]:
        value = str(raw or "").strip().lower()
        if not value:
            continue
        # Signature labels are structural ("topic_pair", "event_frame", etc.).
        # Only the payload after separators should influence follow matching.
        value = re.sub(r"^[a-z_]+:", "", value)
        value = value.replace("::", " ").replace(":", " ").replace("_", " ")
        for token in _STORYLINE_TOKEN_RE.findall(value):
            base = _storyline_loose_token_base(token.strip("-"))
            if (
                len(base) < 4
                or base in _STORYLINE_FOLLOW_FALLBACK_STOPWORDS
                or base in _STORYLINE_SIGNATURE_TECH_TOKENS
            ):
                continue
            tokens.add(base)
    return tokens


def _storyline_context_text_tokens(context: dict[str, object] | None) -> set[str]:
    if not context:
        return set()
    return _storyline_follow_fallback_tokens(
        str(context.get("storyline_title") or "").strip(),
        str(context.get("storyline_seed_preview") or "").strip(),
        str(context.get("macro_topic_title") or "").strip(),
    )


def _score_follow_context_candidate(
    *,
    post_text: str,
    assigned_context: dict[str, object] | None,
    follow_context: dict[str, object] | None,
) -> dict[str, object]:
    post_tokens = _storyline_follow_fallback_tokens(post_text)
    assigned_entity_tokens = _storyline_anchor_entity_tokens(assigned_context)
    assigned_signature_tokens = _storyline_signature_tokens(assigned_context)
    assigned_text_tokens = _storyline_context_text_tokens(assigned_context)

    follow_entity_tokens = _storyline_anchor_entity_tokens(follow_context)
    follow_signature_tokens = _storyline_signature_tokens(follow_context)
    follow_text_tokens = _storyline_context_text_tokens(follow_context)
    follow_markers = _storyline_anchor_markers_from_context(follow_context, limit=5)

    marker_hits = _storyline_text_marker_hits(post_text, follow_markers)
    post_token_overlap = len(post_tokens & follow_text_tokens)
    assigned_entity_overlap = len(assigned_entity_tokens & follow_entity_tokens)
    assigned_signature_overlap = len(assigned_signature_tokens & follow_signature_tokens)
    assigned_text_overlap = len(assigned_text_tokens & follow_text_tokens)

    title_match = bool(
        str((assigned_context or {}).get("storyline_title") or "").strip()
        and str((follow_context or {}).get("storyline_title") or "").strip()
        and str((assigned_context or {}).get("storyline_title") or "").strip().lower()
        == str((follow_context or {}).get("storyline_title") or "").strip().lower()
    )
    score = (
        marker_hits * 6
        + post_token_overlap * 2
        + assigned_entity_overlap * 5
        + assigned_signature_overlap * 2
        + assigned_text_overlap
        + (4 if title_match else 0)
    )
    return {
        "marker_hits": marker_hits,
        "post_token_overlap": post_token_overlap,
        "assigned_entity_overlap": assigned_entity_overlap,
        "assigned_signature_overlap": assigned_signature_overlap,
        "assigned_text_overlap": assigned_text_overlap,
        "title_match": title_match,
        "score": float(score),
    }


def _storyline_follow_candidate_has_direct_evidence(score_info: dict[str, object]) -> bool:
    """Avoid sending broad signature-only overlaps to the LLM arbiter."""
    marker_hits = int(score_info.get("marker_hits") or 0)
    post_token_overlap = int(score_info.get("post_token_overlap") or 0)
    assigned_entity_overlap = int(score_info.get("assigned_entity_overlap") or 0)
    assigned_text_overlap = int(score_info.get("assigned_text_overlap") or 0)
    title_match = bool(score_info.get("title_match"))
    return bool(
        title_match
        or marker_hits > 0
        or post_token_overlap >= 5
        or assigned_entity_overlap >= 2
        or (assigned_entity_overlap >= 1 and assigned_text_overlap >= 4)
    )


def _storyline_anchor_entity_tokens(context: dict[str, object] | None) -> set[str]:
    if not context:
        return set()
    tokens: set[str] = set()
    for key in (context.get("entity_keys") or []):
        raw = str(key or "").strip().lower()
        if ":" not in raw:
            continue
        label, value = raw.split(":", 1)
        if label not in {"product", "organization", "event", "law"}:
            continue
        for token in _STORYLINE_TOKEN_RE.findall(value.replace("_", " ")):
            normalized = token.strip("-")
            if (
                len(normalized) < 5
                or normalized in _STORYLINE_FOLLOW_FALLBACK_STOPWORDS
            ):
                continue
            tokens.add(normalized)
    return tokens


def _storyline_anchor_entity_keys(context: dict[str, object] | None) -> set[str]:
    if not context:
        return set()
    keys: set[str] = set()
    for key in (context.get("entity_keys") or []):
        raw = str(key or "").strip().lower()
        if ":" not in raw:
            continue
        label, value = raw.split(":", 1)
        if label not in {"product", "organization", "event", "law"}:
            continue
        value = value.strip()
        if not value:
            continue
        keys.add(f"{label}:{value}")
    return keys


def _storyline_sibling_storyline_ids(context: dict[str, object] | None) -> set[str]:
    if not context:
        return set()
    ids: set[str] = set()
    for item in (context.get("sibling_candidates") or []):
        if not isinstance(item, dict):
            continue
        storyline_id = str(item.get("storyline_id") or "").strip()
        if storyline_id:
            ids.add(storyline_id)
    return ids


def _storyline_contexts_are_close_for_follow(
    anchor_context: dict[str, object] | None,
    candidate_context: dict[str, object] | None,
) -> bool:
    if not anchor_context or not candidate_context:
        return False

    anchor_storyline_id = str(anchor_context.get("storyline_id") or "").strip()
    anchor_root_id = str(anchor_context.get("family_root_storyline_id") or "").strip()
    candidate_storyline_id = str(candidate_context.get("storyline_id") or "").strip()
    candidate_root_id = str(candidate_context.get("family_root_storyline_id") or "").strip()
    anchor_neighbor_ids = _storyline_sibling_storyline_ids(anchor_context)
    candidate_neighbor_ids = _storyline_sibling_storyline_ids(candidate_context)

    if candidate_root_id and candidate_root_id in anchor_neighbor_ids:
        return True
    if anchor_root_id and anchor_root_id in candidate_neighbor_ids:
        return True
    if candidate_storyline_id and candidate_storyline_id in anchor_neighbor_ids:
        return True
    if anchor_storyline_id and anchor_storyline_id in candidate_neighbor_ids:
        return True

    anchor_macro_topic_id = str(anchor_context.get("macro_topic_id") or "").strip()
    candidate_macro_topic_id = str(candidate_context.get("macro_topic_id") or "").strip()
    same_macro_topic = bool(
        anchor_macro_topic_id
        and candidate_macro_topic_id
        and anchor_macro_topic_id == candidate_macro_topic_id
    )

    anchor_entity_keys = _storyline_anchor_entity_keys(anchor_context)
    candidate_entity_keys = _storyline_anchor_entity_keys(candidate_context)
    shared_entity_keys = anchor_entity_keys & candidate_entity_keys
    if same_macro_topic and shared_entity_keys:
        return True

    anchor_tokens = _storyline_anchor_entity_tokens(anchor_context)
    candidate_tokens = _storyline_anchor_entity_tokens(candidate_context)
    overlap = anchor_tokens & candidate_tokens
    strong_overlap = {
        token for token in overlap
        if len(token) >= 7 or "-" in token or any(ch.isdigit() for ch in token)
    }

    if not same_macro_topic:
        return False

    anchor_signature_tokens = _storyline_signature_tokens(anchor_context)
    candidate_signature_tokens = _storyline_signature_tokens(candidate_context)
    shared_signature_tokens = anchor_signature_tokens & candidate_signature_tokens

    if strong_overlap and shared_signature_tokens:
        return True
    return False


async def _load_storyline_follow_targets(
    session,
    *,
    post_id: int,
) -> tuple[dict[str, object] | None, list[tuple[User, UserStorylineFollow]]]:
    context = await get_storyline_context_for_post(post_id)
    if not context:
        return None, []

    family_root_storyline_id = str(
        context.get("family_root_storyline_id") or context.get("storyline_id") or ""
    ).strip()
    if not family_root_storyline_id:
        return context, []

    result = await session.execute(
        select(User, UserStorylineFollow)
        .join(UserStorylineFollow, UserStorylineFollow.user_id == User.id)
        .where(User.feed_filter != "digest_only")
        .where(UserStorylineFollow.family_root_storyline_id == family_root_storyline_id)
        .where(UserStorylineFollow.is_active.is_(True))
        .order_by(UserStorylineFollow.created_at.asc(), UserStorylineFollow.id.asc())
    )
    pairs = list(result.all())
    if not pairs:
        post_row = await session.execute(
            select(Post.summary, Post.processed_content, Post.content, Post.title, Post.community_id).where(Post.id == int(post_id))
        )
        post_payload = post_row.first()
        current_post_text = ""
        current_post_community_id: int | None = None
        if post_payload:
            current_post_text = (
                str(post_payload[0] or "").strip()
                or str(post_payload[1] or "").strip()
                or str(post_payload[2] or "").strip()
                or str(post_payload[3] or "").strip()
            )
            try:
                current_post_community_id = int(post_payload[4]) if post_payload[4] is not None else None
            except Exception:
                current_post_community_id = None
        fallback_result = await session.execute(
            select(User, UserStorylineFollow)
            .join(UserStorylineFollow, UserStorylineFollow.user_id == User.id)
            .where(User.feed_filter != "digest_only")
            .where(UserStorylineFollow.is_active.is_(True))
            .order_by(UserStorylineFollow.created_at.asc(), UserStorylineFollow.id.asc())
        )
        fallback_pairs_all = list(fallback_result.all())
        if not fallback_pairs_all:
            return context, []
        root_context_cache: dict[str, dict[str, object] | None] = {}
        candidate_pairs_by_root: dict[str, list[tuple[User, UserStorylineFollow]]] = {}
        shortlisted_candidates: list[dict[str, object]] = []
        for user, follow in fallback_pairs_all:
            follow_root_id = str(getattr(follow, "family_root_storyline_id", "") or "").strip()
            if not follow_root_id or follow_root_id == family_root_storyline_id:
                continue
            if follow_root_id not in root_context_cache:
                root_context_cache[follow_root_id] = await get_storyline_context_for_storyline_id(follow_root_id)
            follow_context = root_context_cache.get(follow_root_id)
            if not follow_context:
                continue
            candidate_pairs_by_root.setdefault(follow_root_id, []).append((user, follow))
        if not candidate_pairs_by_root:
            return context, []

        def _follow_user_title_for_root(follow_root_id: str) -> str:
            for _user, follow in candidate_pairs_by_root.get(follow_root_id, []):
                title = str(getattr(follow, "storyline_title", None) or "").strip()
                if title:
                    return title
            return ""

        all_follow_focus_tokens: set[str] = set()
        for follow_root_id in candidate_pairs_by_root:
            all_follow_focus_tokens |= _storyline_follow_title_focus_tokens(
                _follow_user_title_for_root(follow_root_id)
            )
        recent_focus_token_counts = await _storyline_recent_focus_token_counts(
            session,
            all_follow_focus_tokens,
        )

        deterministic_root_ids: list[str] = []
        evidence_dropped_root_ids: list[str] = []
        for follow_root_id, follow_context in root_context_cache.items():
            if follow_root_id not in candidate_pairs_by_root or not follow_context:
                continue
            if _storyline_contexts_are_close_for_follow(context, follow_context):
                deterministic_root_ids.append(follow_root_id)
                score_info = _score_follow_context_candidate(
                    post_text=current_post_text,
                    assigned_context=context,
                    follow_context=follow_context,
                )
                user_follow_title = _follow_user_title_for_root(follow_root_id)
                focus_evidence = _storyline_follow_title_focus_evidence(
                    current_post_text,
                    user_follow_title,
                    recent_focus_token_counts,
                )
                has_direct_evidence = _storyline_follow_candidate_has_direct_evidence(score_info) or bool(
                    focus_evidence.get("matched")
                )
                if not has_direct_evidence:
                    evidence_dropped_root_ids.append(follow_root_id)
                    continue
                shortlisted_candidates.append(
                    {
                        "storyline_id": str(follow_context.get("storyline_id") or "").strip(),
                        "family_root_storyline_id": follow_root_id,
                        "user_follow_title": user_follow_title,
                        "storyline_title": str(follow_context.get("storyline_title") or "").strip(),
                        "storyline_seed_preview": str(follow_context.get("storyline_seed_preview") or "").strip(),
                        "macro_topic_title": str(follow_context.get("macro_topic_title") or "").strip(),
                        "posts_count": int(follow_context.get("posts_count") or 0),
                        "focus_hits": focus_evidence.get("hits") or [],
                        "focus_strong_hits": focus_evidence.get("strong_hits") or [],
                        "focus_broad_hits": focus_evidence.get("broad_hits") or [],
                        "focus_title_tokens": focus_evidence.get("title_tokens") or [],
                        **score_info,
                    }
                )

        if not shortlisted_candidates:
            for follow_root_id, follow_context in root_context_cache.items():
                if follow_root_id not in candidate_pairs_by_root or not follow_context:
                    continue
                score_info = _score_follow_context_candidate(
                    post_text=current_post_text,
                    assigned_context=context,
                    follow_context=follow_context,
                )
                score = float(score_info.get("score") or 0.0)
                user_follow_title = _follow_user_title_for_root(follow_root_id)
                focus_evidence = _storyline_follow_title_focus_evidence(
                    current_post_text,
                    user_follow_title,
                    recent_focus_token_counts,
                )
                if score < 5.0 and not focus_evidence.get("matched"):
                    continue
                has_direct_evidence = _storyline_follow_candidate_has_direct_evidence(score_info) or bool(
                    focus_evidence.get("matched")
                )
                if not has_direct_evidence:
                    evidence_dropped_root_ids.append(follow_root_id)
                    continue
                shortlisted_candidates.append(
                    {
                        "storyline_id": str(follow_context.get("storyline_id") or "").strip(),
                        "family_root_storyline_id": follow_root_id,
                        "user_follow_title": user_follow_title,
                        "storyline_title": str(follow_context.get("storyline_title") or "").strip(),
                        "storyline_seed_preview": str(follow_context.get("storyline_seed_preview") or "").strip(),
                        "macro_topic_title": str(follow_context.get("macro_topic_title") or "").strip(),
                        "posts_count": int(follow_context.get("posts_count") or 0),
                        "focus_hits": focus_evidence.get("hits") or [],
                        "focus_strong_hits": focus_evidence.get("strong_hits") or [],
                        "focus_broad_hits": focus_evidence.get("broad_hits") or [],
                        "focus_title_tokens": focus_evidence.get("title_tokens") or [],
                        **score_info,
                    }
                )

        if not shortlisted_candidates:
            return context, []

        shortlisted_candidates.sort(
            key=lambda item: (
                -float(item.get("score") or 0.0),
                -int(item.get("marker_hits") or 0),
                -int(item.get("assigned_entity_overlap") or 0),
                -int(item.get("assigned_signature_overlap") or 0),
                -int(item.get("posts_count") or 0),
            )
        )
        shortlisted_candidates = shortlisted_candidates[:5]

        log.info(
            "storyline_follow_targets_llm_shortlist",
            post_id=post_id,
            storyline_id=str(context.get("storyline_id") or ""),
            family_root_storyline_id=family_root_storyline_id,
            candidate_root_ids=[
                str(item.get("family_root_storyline_id") or "") for item in shortlisted_candidates
            ],
            deterministic_root_ids=deterministic_root_ids,
            evidence_dropped_root_ids=sorted(set(evidence_dropped_root_ids))[:20],
        )

        selected_root_ids: list[str] = []
        arbitration_reason = ""
        arbitration_answered = False
        try:
            arbitration = await arbitrate_user_storyline_follow_match(
                post_text=current_post_text,
                assigned_context=context,
                follow_candidates=shortlisted_candidates,
            )
        except DeepseekRetryableError as exc:
            arbitration = None
            log.warning(
                "storyline_follow_targets_llm_retryable",
                post_id=post_id,
                error=str(exc),
            )
        except Exception as exc:
            arbitration = None
            log.warning(
                "storyline_follow_targets_llm_failed",
                post_id=post_id,
                error=str(exc),
            )

        if isinstance(arbitration, dict):
            arbitration_answered = True
            raw_selected_root_ids = arbitration.get("selected_family_root_storyline_ids")
            if isinstance(raw_selected_root_ids, list):
                for value in raw_selected_root_ids:
                    root_id = str(value or "").strip()
                    if root_id and root_id not in selected_root_ids:
                        selected_root_ids.append(root_id)
            legacy_selected_root_id = str(arbitration.get("selected_family_root_storyline_id") or "").strip()
            if legacy_selected_root_id and legacy_selected_root_id not in selected_root_ids:
                selected_root_ids.append(legacy_selected_root_id)
            if not bool(arbitration.get("matched")):
                selected_root_ids = []
            selected_root_ids = [
                root_id for root_id in selected_root_ids
                if root_id in candidate_pairs_by_root
            ]
            arbitration_reason = str(arbitration.get("reason") or "").strip()[:240]
            log.info(
                "storyline_follow_targets_llm_selected",
                post_id=post_id,
                storyline_id=str(context.get("storyline_id") or ""),
                family_root_storyline_id=family_root_storyline_id,
                selected_root_id=selected_root_ids[0] if selected_root_ids else "",
                selected_root_ids=selected_root_ids,
                matched=bool(arbitration.get("matched")),
                confidence=float(arbitration.get("confidence") or 0.0),
                reason=arbitration_reason,
            )

        if not selected_root_ids and not arbitration_answered and len(shortlisted_candidates) == 1:
            only_candidate = shortlisted_candidates[0]
            if float(only_candidate.get("score") or 0.0) >= 9.0:
                root_id = str(only_candidate.get("family_root_storyline_id") or "").strip()
                if root_id:
                    selected_root_ids = [root_id]
                    arbitration_reason = "single_high_score_fallback"

        if selected_root_ids:
            pairs = []
            seen_pair_keys: set[tuple[int, int]] = set()
            for selected_root_id in selected_root_ids:
                for user, follow in candidate_pairs_by_root.get(selected_root_id, []):
                    pair_key = (int(getattr(user, "id", 0) or 0), int(getattr(follow, "id", 0) or 0))
                    if pair_key in seen_pair_keys:
                        continue
                    seen_pair_keys.add(pair_key)
                    pairs.append((user, follow))
            log.info(
                "storyline_follow_targets_fallback_branches",
                post_id=post_id,
                storyline_id=str(context.get("storyline_id") or ""),
                family_root_storyline_id=family_root_storyline_id,
                fallback_root_ids=selected_root_ids,
                pairs_count=len(pairs),
                reason=arbitration_reason,
            )
        else:
            return context, []

    filtered: list[tuple[User, UserStorylineFollow]] = []
    for user, follow in pairs:
        if not storytracking_allowed_for_user(user):
            continue
        if int(getattr(follow, "source_post_id", 0) or 0) == int(post_id):
            continue
        filtered.append((user, follow))
    return context, filtered


async def _dispatch_storyline_updates_for_post(
    session,
    *,
    post: Post,
    community: Community,
) -> dict[str, int | str | None]:
    from app.audit import audit

    storyline_context, storyline_follow_pairs = await _load_storyline_follow_targets(
        session,
        post_id=int(post.id),
    )
    if not storyline_context or not storyline_follow_pairs:
        return {
            "ok": 0,
            "fail": 0,
            "followup_ok": 0,
            "followup_fail": 0,
            "skipped_duplicate": 0,
            "family_root_storyline_id": None,
        }

    storyline_follow_users = [user for user, _follow in storyline_follow_pairs]
    delivery_prefs_by_user_id = await bulk_load_delivery_prefs(
        session,
        [int(user.id) for user in storyline_follow_users],
    )
    summary_text = None
    try:
        if post.content:
            content_text = post.content.strip()
            if len(content_text) >= SUMMARY_MIN_CHARS:
                summary_text = post.summary
                if not summary_text:
                    summary_text = await summarize(post.content)
                    post.summary = summary_text
                    post.summary_created_at = datetime.now(timezone.utc) if summary_text else None
                    await session.commit()
    except Exception as e:
        log.error(
            "storyline_summary_failed_continue",
            post_id=post.id,
            community_id=community.id,
            error=str(e),
            exc_info=True,
        )
        audit(
            "storyline_dispatch.summary_failed",
            post_id=post.id,
            community_id=community.id,
            error=str(e),
        )
        summary_text = None

    content_link = post.content_link or {}
    original_url = content_link.get("url")
    if not original_url:
        msg_id = content_link.get("id")
        base = (community.link or "").rstrip("/")
        if base and msg_id:
            original_url = f"{base}/{int(msg_id)}"
            content_link["url"] = original_url
            post.content_link = content_link
            await session.commit()

    sem = asyncio.Semaphore(SEND_CONCURRENCY)
    storyline_title = (
        str((storyline_context or {}).get("storyline_title") or "").strip()
        or str((storyline_context or {}).get("family_root_storyline_id") or "").strip()
        or str((storyline_context or {}).get("storyline_id") or "").strip()
        or f"storyline_{post.id}"
    )
    family_root_storyline_id = str(
        (storyline_context or {}).get("family_root_storyline_id") or ""
    ).strip()
    seed_text = (
        str((storyline_context or {}).get("storyline_seed_preview") or "").strip()
        or str(post.content or "").strip()
    )
    storyline_card_summary = (
        str(summary_text or "").strip()
        or str(post.content or "").strip()
        or t("ru", "notif_news_without_text")
    )
    storyline_card_summary = _clean_storyline_update_summary_text(storyline_card_summary)
    if len(storyline_card_summary) > 1500:
        storyline_card_summary = storyline_card_summary[:1497].rstrip() + "..."
    source_label = (community.name or post.title or t("ru", "notif_source")).strip()
    update_step = await get_update_step_for_post(int(post.id))
    update_step_id = str((update_step or {}).get("step_id") or "").strip()
    storyline_update_event_key = (
        f"step:{update_step_id}"
        if update_step_id
        else _fallback_storyline_event_key(post, storyline_card_summary)
    )
    follow_titles_by_id: dict[int, str] = {}
    dirty_titles = False
    for _user, follow in storyline_follow_pairs:
        resolved_title = await _resolve_storyline_human_title(
            follow=follow,
            fallback_title=storyline_title,
            seed_text=seed_text,
        )
        follow_id = int(getattr(follow, "id", 0) or 0)
        if follow_id:
            follow_titles_by_id[follow_id] = resolved_title
        current_title = str(getattr(follow, "storyline_title", None) or "").strip()
        if resolved_title and resolved_title != current_title:
            follow.storyline_title = resolved_title
            session.add(follow)
            dirty_titles = True
    if dirty_titles:
        await session.commit()

    async def _claim_delivery(user_id: int) -> bool:
        claim = await session.execute(
            text(
                """
                INSERT INTO dispatch_deliveries (post_id, user_id, attempt_count)
                VALUES (:post_id, :user_id, 1)
                ON CONFLICT (post_id, user_id) DO NOTHING
                RETURNING post_id
                """
            ),
            {"post_id": int(post.id), "user_id": int(user_id)},
        )
        return claim.scalar_one_or_none() is not None

    async def _release_delivery(user_id: int) -> None:
        await session.execute(
            text(
                "DELETE FROM dispatch_deliveries "
                "WHERE post_id = :post_id AND user_id = :user_id "
                "AND sent_at IS NULL"
            ),
            {"post_id": int(post.id), "user_id": int(user_id)},
        )

    async def _mark_delivery_sent(user_id: int, delivery: dict | None) -> None:
        message_id = None
        if isinstance(delivery, dict):
            raw_mid = delivery.get("message_id")
            if raw_mid is not None:
                try:
                    message_id = int(raw_mid)
                except (TypeError, ValueError):
                    message_id = None
        await session.execute(
            text(
                """
                UPDATE dispatch_deliveries
                SET sent_at = now(),
                    telegram_message_id = :mid,
                    attempt_count = GREATEST(attempt_count, 2)
                WHERE post_id = :post_id AND user_id = :user_id
                """
            ),
            {
                "post_id": int(post.id),
                "user_id": int(user_id),
                "mid": message_id,
            },
        )

    regular_delivery_needs_storyline_user_ids: set[int] = set()
    for user, _follow in storyline_follow_pairs:
        row = await session.execute(
            text(
                """
                SELECT attempt_count
                FROM dispatch_deliveries
                WHERE post_id = :post_id
                  AND user_id = :user_id
                  AND sent_at IS NOT NULL
                LIMIT 1
                """
            ),
            {"post_id": int(post.id), "user_id": int(user.id)},
        )
        attempt_count = row.scalar_one_or_none()
        if attempt_count is not None and int(attempt_count or 0) < 2:
            regular_delivery_needs_storyline_user_ids.add(int(user.id))

    async def _mark_storyline_card_sent(user_id: int, delivery: dict | None = None) -> None:
        message_id = None
        if isinstance(delivery, dict):
            raw_mid = delivery.get("message_id")
            if raw_mid is not None:
                try:
                    message_id = int(raw_mid)
                except (TypeError, ValueError):
                    message_id = None
        params = {
            "post_id": int(post.id),
            "user_id": int(user_id),
            "mid": message_id,
        }
        if message_id is not None:
            stmt = """
                UPDATE dispatch_deliveries
                SET attempt_count = GREATEST(attempt_count, 2),
                    telegram_message_id = :mid
                WHERE post_id = :post_id AND user_id = :user_id
            """
        else:
            stmt = """
                UPDATE dispatch_deliveries
                SET attempt_count = GREATEST(attempt_count, 2)
                WHERE post_id = :post_id AND user_id = :user_id
            """
        await session.execute(text(stmt), params)

    log.info(
        "Storyline dispatch start",
        post_id=post.id,
        community_id=community.id,
        family_root_storyline_id=family_root_storyline_id,
        users_count=len(storyline_follow_pairs),
    )
    audit(
        "storyline_dispatch.start",
        post_id=post.id,
        community_id=community.id,
        family_root_storyline_id=family_root_storyline_id,
        users_count=len(storyline_follow_pairs),
        trigger="post_ie",
    )

    async def _send_storyline_with_sem(user: User, follow: UserStorylineFollow):
        async with sem:
            locale = get_user_locale(user)
            follow_title = follow_titles_by_id.get(int(getattr(follow, "id", 0) or 0), storyline_title)
            route = resolve_delivery_route_from_pref(
                pref=delivery_prefs_by_user_id.get(int(user.id)),
                delivery_kind="storyline",
            )
            audit(
                "storyline_send.attempt",
                post_id=post.id,
                community_id=community.id,
                user_id=user.id,
                telegram_id=getattr(user, "telegram_id", None),
                family_root_storyline_id=str(getattr(follow, "family_root_storyline_id", "") or ""),
                storyline_title=follow_title,
                delivery_target=route.target,
                delivery_route_reason=route.reason,
                trigger="post_ie",
            )
            try:
                delivery = await _send_or_merge_storyline_update_card(
                    user=user,
                    follow=follow,
                    locale=locale,
                    title=follow_title,
                    summary_text=storyline_card_summary,
                    source_url=original_url,
                    source_label=source_label,
                    post=post,
                    family_root_storyline_id=str(getattr(follow, "family_root_storyline_id", "") or family_root_storyline_id),
                    event_key=storyline_update_event_key,
                    route_token=route.token,
                    delivery_target=route.target,
                )
                if not delivery and route.target == "plus" and MAIN_BOT_TOKEN:
                    delivery = await _send_or_merge_storyline_update_card(
                        user=user,
                        follow=follow,
                        locale=locale,
                        title=follow_title,
                        summary_text=storyline_card_summary,
                        source_url=original_url,
                        source_label=source_label,
                        post=post,
                        family_root_storyline_id=str(getattr(follow, "family_root_storyline_id", "") or family_root_storyline_id),
                        event_key=storyline_update_event_key,
                        route_token=MAIN_BOT_TOKEN,
                        delivery_target="main_fallback",
                    )
                if not delivery:
                    raise RuntimeError("storyline_card_send_failed")
                audit(
                    "storyline_send.success",
                    post_id=post.id,
                    community_id=community.id,
                    user_id=user.id,
                    telegram_id=getattr(user, "telegram_id", None),
                    delivery_result=(delivery or {}).get("delivery_result"),
                    telegram_message_id=(delivery or {}).get("message_id"),
                    trigger="post_ie",
                )
                return delivery or {}
            except Exception as e:
                audit(
                    "storyline_send.fail",
                    post_id=post.id,
                    community_id=community.id,
                    user_id=user.id,
                    telegram_id=getattr(user, "telegram_id", None),
                    error=str(e),
                    error_type=type(e).__name__,
                    trigger="post_ie",
                )
                log.error(
                    "Storyline send failed",
                    post_id=post.id,
                    community_id=community.id,
                    user_id=user.id,
                    telegram_id=getattr(user, "telegram_id", None),
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                return e

    async def _send_storyline_followup_with_sem(user: User, follow: UserStorylineFollow):
        async with sem:
            locale = get_user_locale(user)
            follow_title = follow_titles_by_id.get(int(getattr(follow, "id", 0) or 0), storyline_title)
            route = resolve_delivery_route_from_pref(
                pref=delivery_prefs_by_user_id.get(int(user.id)),
                delivery_kind="storyline",
            )
            audit(
                "storyline_followup.attempt",
                post_id=post.id,
                community_id=community.id,
                user_id=user.id,
                telegram_id=getattr(user, "telegram_id", None),
                family_root_storyline_id=str(getattr(follow, "family_root_storyline_id", "") or ""),
                storyline_title=follow_title,
                delivery_target=route.target,
                delivery_route_reason=route.reason,
                trigger="post_ie",
            )
            delivery = await _send_or_merge_storyline_update_card(
                user=user,
                follow=follow,
                locale=locale,
                title=follow_title,
                summary_text=storyline_card_summary,
                source_url=original_url,
                source_label=source_label,
                post=post,
                family_root_storyline_id=str(getattr(follow, "family_root_storyline_id", "") or family_root_storyline_id),
                event_key=storyline_update_event_key,
                route_token=route.token,
                delivery_target=route.target,
            )
            if not delivery and route.target == "plus" and MAIN_BOT_TOKEN:
                delivery = await _send_or_merge_storyline_update_card(
                    user=user,
                    follow=follow,
                    locale=locale,
                    title=follow_title,
                    summary_text=storyline_card_summary,
                    source_url=original_url,
                    source_label=source_label,
                    post=post,
                    family_root_storyline_id=str(getattr(follow, "family_root_storyline_id", "") or family_root_storyline_id),
                    event_key=storyline_update_event_key,
                    route_token=MAIN_BOT_TOKEN,
                    delivery_target="main_fallback",
                )
            if delivery:
                audit(
                    "storyline_followup.success",
                    post_id=post.id,
                    community_id=community.id,
                    user_id=user.id,
                    telegram_id=getattr(user, "telegram_id", None),
                    delivery_result=(delivery or {}).get("delivery_result"),
                    telegram_message_id=(delivery or {}).get("message_id"),
                    trigger="post_ie",
                )
                return delivery or {}
            audit(
                "storyline_followup.fail",
                post_id=post.id,
                community_id=community.id,
                user_id=user.id,
                telegram_id=getattr(user, "telegram_id", None),
                trigger="post_ie",
            )
            return RuntimeError("storyline_followup_send_failed")

    claimed_storyline_pairs: list[tuple[User, UserStorylineFollow]] = []
    followup_storyline_pairs: list[tuple[User, UserStorylineFollow]] = []
    skipped_storyline_duplicate = 0
    for user, follow in storyline_follow_pairs:
        if await _claim_delivery(int(user.id)):
            claimed_storyline_pairs.append((user, follow))
        else:
            skipped_storyline_duplicate += 1
            if int(user.id) in regular_delivery_needs_storyline_user_ids:
                followup_storyline_pairs.append((user, follow))
    await session.commit()

    if skipped_storyline_duplicate:
        audit(
            "storyline_dispatch.skipped_duplicate",
            post_id=post.id,
            community_id=community.id,
            skipped=skipped_storyline_duplicate,
            trigger="post_ie",
        )
        log.info(
            "Storyline dispatch duplicate claims skipped",
            post_id=post.id,
            community_id=community.id,
            skipped=skipped_storyline_duplicate,
        )

    if claimed_storyline_pairs:
        storyline_results = await asyncio.gather(
            *[_send_storyline_with_sem(user, follow) for user, follow in claimed_storyline_pairs],
            return_exceptions=True,
        )
        for (user, _follow), result in zip(claimed_storyline_pairs, storyline_results, strict=False):
            try:
                if isinstance(result, dict):
                    await _mark_delivery_sent(int(user.id), result)
                else:
                    await _release_delivery(int(user.id))
            except Exception as e:
                log.warning(
                    "storyline_dispatch.delivery_update_failed",
                    post_id=post.id,
                    user_id=user.id,
                    error=str(e),
                )
        try:
            await session.commit()
        except Exception:
            await session.rollback()
    else:
        storyline_results = []

    if followup_storyline_pairs:
        storyline_followup_results = await asyncio.gather(
            *[
                _send_storyline_followup_with_sem(user, follow)
                for user, follow in followup_storyline_pairs
            ],
            return_exceptions=True,
        )
        for (user, _follow), result in zip(followup_storyline_pairs, storyline_followup_results, strict=False):
            if not isinstance(result, dict):
                continue
            try:
                await _mark_storyline_card_sent(int(user.id))
            except Exception as e:
                log.warning(
                    "storyline_dispatch.followup_marker_update_failed",
                    post_id=post.id,
                    user_id=user.id,
                    error=str(e),
                )
        try:
            await session.commit()
        except Exception:
            await session.rollback()
    else:
        storyline_followup_results = []

    storyline_ok = sum(1 for r in storyline_results if isinstance(r, dict))
    storyline_fail = sum(1 for r in storyline_results if not isinstance(r, dict))
    storyline_followup_ok = sum(1 for r in storyline_followup_results if isinstance(r, dict))
    storyline_followup_fail = sum(1 for r in storyline_followup_results if not isinstance(r, dict))

    audit(
        "storyline_dispatch.finish",
        post_id=post.id,
        community_id=community.id,
        users_count=len(claimed_storyline_pairs),
        ok=storyline_ok,
        fail=storyline_fail,
        followup_ok=storyline_followup_ok,
        followup_fail=storyline_followup_fail,
        skipped_duplicate=skipped_storyline_duplicate,
        family_root_storyline_id=family_root_storyline_id,
        trigger="post_ie",
    )
    log.info(
        "Storyline dispatch finish",
        post_id=post.id,
        community_id=community.id,
        users_count=len(claimed_storyline_pairs),
        ok=storyline_ok,
        fail=storyline_fail,
        followup_ok=storyline_followup_ok,
        followup_fail=storyline_followup_fail,
        skipped_duplicate=skipped_storyline_duplicate,
        family_root_storyline_id=family_root_storyline_id,
    )
    return {
        "ok": storyline_ok,
        "fail": storyline_fail,
        "followup_ok": storyline_followup_ok,
        "followup_fail": storyline_followup_fail,
        "skipped_duplicate": skipped_storyline_duplicate,
        "family_root_storyline_id": family_root_storyline_id or None,
    }


def sanitize_digest_text(text: str, max_len: int = 250) -> str:
    raw = html.unescape((text or "").strip())
    if not raw:
        return ""

    cleaned_lines: list[str] = []
    for line in raw.splitlines():
        s = (line or "").strip()
        if not s:
            continue

        # Remove all direct links inside post text.
        s = _URL_RE.sub("", s)
        s = s.strip(" \t\r\n()[]<>")
        if not s:
            continue

        # Drop pure Telegram handle lines like "@channel".
        if _TG_HANDLE_RE.fullmatch(s):
            continue

        # Drop common source-credit tails (e.g. "📹 соцсети ...").
        if _DIGEST_NOISE_LINE_RE.match(s):
            continue

        # Remove handles inside remaining text and collapse spaces.
        s = _TG_HANDLE_RE.sub("", s)
        s = _EMOJI_RE.sub("", s)
        s = re.sub(r"\s{2,}", " ", s).strip(" -–—|•")
        if not s:
            continue

        cleaned_lines.append(s)

    if not cleaned_lines:
        return ""

    compact = " ".join(cleaned_lines).strip()
    return smart_trim(compact, max_len=max_len) if len(compact) > max_len else compact

def normalize_source_name(source: str | None) -> str:
    s = (source or "").strip()
    if not s:
        return "Источник"
    # нормальный @username
    if s.startswith("@") and len(s) > 1:
        return s
    # самые неприятные случаи
    if s in {"c", "@c"}:
        return "Канал"
    return s

async def _run_with_session(coro):
    async for session in get_session():
        return await coro(session)


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _to_float_vector(raw) -> list[float]:
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for item in raw:
        try:
            out.append(float(item))
        except Exception:
            return []
    return out


async def _feature_vector_for_user_post(
    session,
    user: User,
    post: Post,
) -> tuple[dict[str, float], int, "UserModelState | None"]:
    post_keywords = [str(k).strip().lower() for k in (post.keywords or []) if isinstance(k, str)]
    post_keywords = [k for k in post_keywords if k]
    model_state = await session.get(UserModelState, user.id)
    reset_at = getattr(model_state, "reset_at", None)
    if reset_at and reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)

    topics_res = await session.execute(select(PostTopic.topic).where(PostTopic.post_id == post.id))
    post_topics = [str(r[0]).strip().lower() for r in topics_res.all() if r[0]]

    keyword_scores: list[float] = []
    if post_keywords:
        kw_res = await session.execute(
            select(UserKeywordStat.keyword, UserKeywordStat.score).where(
                UserKeywordStat.user_id == user.id,
                UserKeywordStat.keyword.in_(post_keywords),
            )
        )
        kw_map = {str(k).lower(): float(s or 0.0) for k, s in kw_res.all()}
        for kw in post_keywords:
            keyword_scores.append(max(0.0, kw_map.get(kw, 0.0)))

    topic_scores: list[float] = []
    if post_topics:
        topic_kw_res = await session.execute(
            select(UserKeywordStat.keyword, UserKeywordStat.score).where(
                UserKeywordStat.user_id == user.id,
                UserKeywordStat.keyword.in_(post_topics),
            )
        )
        topic_map = {str(k).lower(): float(s or 0.0) for k, s in topic_kw_res.all()}
        for topic in post_topics:
            topic_scores.append(max(0.0, topic_map.get(topic, 0.0)))

    source_reactions_stmt = (
        select(func.count(Reaction.id), func.coalesce(func.sum(
            case(
                (Reaction.reaction == "fire", 1.0),
                (Reaction.reaction == "up", 0.6),
                (Reaction.reaction == "down", -0.8),
                else_=0.0,
            )
        ), 0.0))
        .select_from(Reaction)
        .join(Post, Post.id == Reaction.post_id)
        .where(Reaction.user_id == user.id, Post.community_id == post.community_id)
    )
    if reset_at:
        source_reactions_stmt = source_reactions_stmt.where(Reaction.updated_at >= reset_at)
    source_reactions_res = await session.execute(source_reactions_stmt)
    source_reaction_count, source_reaction_score = source_reactions_res.one()

    source_clicks_stmt = select(func.coalesce(func.sum(Click.click_count), 0)).where(
        Click.user_id == user.id,
        Click.community_id == post.community_id,
    )
    if reset_at:
        source_clicks_stmt = source_clicks_stmt.where(Click.first_clicked_at >= reset_at)
    source_clicks_res = await session.execute(source_clicks_stmt)
    source_clicks_total = int(source_clicks_res.scalar_one() or 0)
    source_affinity = max(0.0, min(1.0, (float(source_reaction_score or 0.0) + 0.2 * source_clicks_total) / 5.0))

    post_reactions_res = await session.execute(
        select(func.count(Reaction.id), func.coalesce(func.sum(
            case(
                (Reaction.reaction == "fire", 1.0),
                (Reaction.reaction == "up", 0.5),
                (Reaction.reaction == "down", -0.6),
                else_=0.0,
            )
        ), 0.0)).where(Reaction.post_id == post.id)
    )
    post_reaction_count, post_reaction_sum = post_reactions_res.one()
    popularity = max(0.0, min(1.0, (float(post_reaction_sum or 0.0) + float(post_reaction_count or 0.0) * 0.2) / 6.0))

    freshness = 0.4
    if post.timestamp:
        ts = post.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_hours = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0)
        freshness = max(0.0, min(1.0, 1.0 - (age_hours / 72.0)))

    engagement = max(0.0, min(1.0, float(getattr(user, "engagement_score", 0.0) or 0.0) / 20.0))
    has_summary = 1.0 if bool(post.summary) else 0.0
    has_media = 1.0 if bool(post.media_path) else 0.0

    embedding_similarity = 0.0
    post_embedding_row = await get_post_embedding(session, int(post.id))
    if post_embedding_row:
        profile = await session.get(UserEmbeddingProfile, user.id)
        if profile and profile.dim and isinstance(profile.vector, list):
            embedding_similarity = cosine_similarity(
                _to_float_vector(profile.vector),
                _to_float_vector(post_embedding_row),
            )

    features = {
        "keyword_affinity": max(0.0, min(1.0, _avg(keyword_scores) / 3.0)) if keyword_scores else 0.0,
        "topic_affinity": max(0.0, min(1.0, _avg(topic_scores) / 3.0)) if topic_scores else 0.0,
        "source_affinity": source_affinity,
        "post_popularity": popularity,
        "freshness": freshness,
        "engagement": engagement,
        "has_summary": has_summary,
        "has_media": has_media,
        "embedding_similarity": embedding_similarity,
    }

    events_count = int(source_reaction_count or 0) + int(source_clicks_total or 0)
    return features, events_count, model_state

@celery_app.task(
    name="app.tasks.sync_channel_history_task",
    autoretry_for=(SQLAlchemyError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": DB_MAX_RETRIES},
)
def sync_channel_history_task(group_handle: str):
    # Telethon синк теперь делает только newshub-userbot.service
    log.warning(
        "sync_channel_history_task is disabled (handled by userbot service)",
        group_handle=group_handle,
    )
    return None

@celery_app.task(
    name="app.tasks.summarize_post_task", 
    bind=True,
    task_time_limit=600,       
    task_soft_time_limit=540,  
)
def summarize_post_task(self, post_id: int):
    async def _task():
        async def _work(session):
            result = await session.execute(select(Post).where(Post.id == post_id))
            post = result.scalar_one_or_none()
            if not post or not post.content:
                return None
            content_text = post.content.strip()
            if len(content_text) < SUMMARY_MIN_CHARS:
                return None

            if post.summary:
                return post.summary

            summary_text = await summarize(post.content)
            post.summary = summary_text
            post.summary_created_at = datetime.now(timezone.utc) if summary_text else None
            await session.commit()
            return summary_text

        return await _run_with_session(_work)

    try:
        return _run_async(_task(), timeout=580)  # ← добавь timeout чуть меньше task_time_limit
    except SoftTimeLimitExceeded:  # ← добавь обработку
        log.warning("summarize_post_task soft time limit exceeded", post_id=post_id)
        raise self.retry(countdown=120, max_retries=2)
    except SQLAlchemyError as e:
        raise self.retry(
            exc=e,
            countdown=min(60, 2 ** self.request.retries),
            max_retries=DB_MAX_RETRIES,
        )
    except DeepseekRetryableError as e:
        raise self.retry(
            exc=e,
            countdown=min(60, 2 ** self.request.retries),
            max_retries=DEEPSEEK_MAX_RETRIES,
        )


@celery_app.task(
    name="app.tasks.extract_post_topics_task",
    bind=True,
    task_time_limit=240,
    task_soft_time_limit=200,
)
def extract_post_topics_task(self, post_id: int):
    async def _task():
        async def _work(session):
            return await process_post_topics(session, post_id)

        return await _run_with_session(_work)

    try:
        return _run_async(_task(), timeout=220)
    except SoftTimeLimitExceeded:
        log.warning("extract_post_topics_task soft time limit exceeded", post_id=post_id)
        raise self.retry(countdown=30, max_retries=3)
    except SQLAlchemyError as e:
        raise self.retry(
            exc=e,
            countdown=min(60, 2 ** self.request.retries),
            max_retries=DB_MAX_RETRIES,
        )
    except DeepseekRetryableError as e:
        raise self.retry(
            exc=e,
            countdown=min(60, 2 ** self.request.retries),
            max_retries=DEEPSEEK_MAX_RETRIES,
        )


@celery_app.task(name="app.tasks.dispatch_post_to_subscribers", bind=True)
def dispatch_post_to_subscribers(self, post_id: int):
    async def _task():
        async def _work(session):
            from app.audit import audit

            result = await session.execute(select(Post).where(Post.id == post_id))
            post = result.scalar_one_or_none()
            if not post:
                log.warning("Post not found for dispatch", post_id=post_id)
                audit("dispatch.post_not_found", post_id=post_id)
                return None
            post.dispatch_started_at = datetime.now(timezone.utc)
            post.dispatch_error = None
            session.add(post)
            await session.commit()

            community_result = await session.execute(
                select(Community).where(Community.id == post.community_id)
            )
            community = community_result.scalar_one_or_none()
            if not community:
                log.warning("Community not found for post", post_id=post_id, community_id=getattr(post, "community_id", None))
                audit("dispatch.community_not_found", post_id=post_id, community_id=getattr(post, "community_id", None))
                post.dispatch_finished_at = datetime.now(timezone.utc)
                post.dispatch_error = "community_not_found"
                session.add(post)
                await session.commit()
                return None

            _schedule_storyline_dispatch_after_ie(int(post.id), reason="live_dispatch")

            subscribers_result = await session.execute(
                select(User)
                .join(UserCommunity, UserCommunity.user_id == User.id)
                .where(UserCommunity.community_id == community.id)
            )
            
            users = subscribers_result.scalars().all()

            if not users:
                pending_subs_res = await session.execute(
                    select(func.count(SubscriptionRequest.id)).where(
                        SubscriptionRequest.group_link == community.link,
                        SubscriptionRequest.status.in_(["queued", "joining", "joined", "syncing"]),
                    )
                )
                incomplete_subscriptions = int(pending_subs_res.scalar_one() or 0)
                log.info(
                    "Dispatch skipped: no subscribers",
                    post_id=post.id,
                    community_id=community.id,
                )
                audit(
                    "dispatch.no_subscribers",
                    post_id=post.id,
                    community_id=community.id,
                    incomplete_subscriptions=incomplete_subscriptions,
                )
                if incomplete_subscriptions > 0:
                    audit(
                        "subscription_not_completed_at_post_time",
                        post_id=post.id,
                        community_id=community.id,
                        group_link=community.link,
                        incomplete_subscriptions=incomplete_subscriptions,
                    )
                post.dispatch_finished_at = datetime.now(timezone.utc)
                session.add(post)
                await session.commit()
                return None

            user_send_list: list[User] = []
            user_decisions: dict[int, dict] = {}
            instruction_rules: dict[int, ResolvedInstructionRule] = {}
            premium_access_cache: dict[int, bool] = {}
            if INSTRUCTION_FILTER_ENABLED:
                instruction_rules = await resolve_effective_instruction_rules_for_users(
                    session,
                    user_ids=[int(user.id) for user in users if getattr(user, "id", None) is not None],
                    community_id=int(community.id),
                )
            for user in users:
                mode = (getattr(user, "feed_filter", "all") or "all").strip().lower()
                if mode == "digest_only":
                    user_decisions[user.id] = {
                        "send": False,
                        "reason": "digest_only",
                        "subscription_status": "completed",
                        "shadow": shadow_mode_decisions(
                            score=0.0,
                            threshold_not_interesting=FILTER_THRESHOLD_NOT_INTERESTING,
                            threshold_only_fire=FILTER_THRESHOLD_ONLY_FIRE,
                            cold_start=True,
                        ),
                    }
                    continue

                features: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
                events_count = 0
                baseline = 0.0
                logistic_prob = 0.5
                final_score = 0.0
                reason = "all_mode"
                send = True
                subscription_status = "completed"
                shadow = shadow_mode_decisions(
                    score=0.0,
                    threshold_not_interesting=FILTER_THRESHOLD_NOT_INTERESTING,
                    threshold_only_fire=FILTER_THRESHOLD_ONLY_FIRE,
                    cold_start=True,
                )
                try:
                    sub_req_res = await session.execute(
                        select(SubscriptionRequest.status)
                        .where(
                            SubscriptionRequest.user_id == user.id,
                            SubscriptionRequest.group_link == community.link,
                        )
                        .order_by(SubscriptionRequest.id.desc())
                        .limit(1)
                    )
                    subscription_status = sub_req_res.scalar_one_or_none() or "completed"

                    features, events_count, model_state = await _feature_vector_for_user_post(session, user, post)
                    baseline = baseline_score(features) if RECO_ENABLE_BASELINE else 0.5

                    if model_state and RECO_ENABLE_LOGISTIC:
                        weights = model_state.weights if isinstance(model_state.weights, dict) else {}
                        logistic_prob = logistic_probability(weights, float(model_state.bias or 0.0), features)
                    final_score = (
                        blend_scores(baseline, logistic_prob, RECO_BLEND_ALPHA)
                        if RECO_ENABLE_LOGISTIC
                        else baseline
                    )

                    shadow = shadow_mode_decisions(
                        score=final_score,
                        threshold_not_interesting=FILTER_THRESHOLD_NOT_INTERESTING,
                        threshold_only_fire=FILTER_THRESHOLD_ONLY_FIRE,
                        cold_start=(events_count < FILTER_COLD_START_MIN_EVENTS),
                    )

                    send, reason = decision_for_mode(
                        mode=mode,
                        score=final_score,
                        threshold_not_interesting=FILTER_THRESHOLD_NOT_INTERESTING,
                        threshold_only_fire=FILTER_THRESHOLD_ONLY_FIRE,
                        cold_start=(events_count < FILTER_COLD_START_MIN_EVENTS),
                        epsilon=(FILTER_EXPLORATION_EPSILON if RECO_ENABLE_BANDIT else 0.0),
                    )

                    if (
                        RECO_EMBEDDING_SOURCE == "dual"
                        and RECO_DUAL_PROBE_SAMPLE_RATE > 0
                        and random.random() < RECO_DUAL_PROBE_SAMPLE_RATE
                    ):
                        profile = await session.get(UserEmbeddingProfile, user.id)
                        if profile and profile.dim and isinstance(profile.vector, list):
                            sql_emb, neo4j_emb = await get_post_embeddings_dual_probe(session, int(post.id))
                            if sql_emb and neo4j_emb:
                                sim_sql = cosine_similarity(_to_float_vector(profile.vector), _to_float_vector(sql_emb))
                                sim_neo4j = cosine_similarity(
                                    _to_float_vector(profile.vector),
                                    _to_float_vector(neo4j_emb),
                                )
                                probe_features = dict(features)
                                probe_features["embedding_similarity"] = sim_neo4j
                                probe_baseline = baseline_score(probe_features) if RECO_ENABLE_BASELINE else 0.5
                                probe_logistic = logistic_prob
                                if model_state and RECO_ENABLE_LOGISTIC:
                                    weights = model_state.weights if isinstance(model_state.weights, dict) else {}
                                    probe_logistic = logistic_probability(
                                        weights,
                                        float(model_state.bias or 0.0),
                                        probe_features,
                                    )
                                probe_final = (
                                    blend_scores(probe_baseline, probe_logistic, RECO_BLEND_ALPHA)
                                    if RECO_ENABLE_LOGISTIC
                                    else probe_baseline
                                )
                                probe_send, _ = decision_for_mode(
                                    mode=mode,
                                    score=probe_final,
                                    threshold_not_interesting=FILTER_THRESHOLD_NOT_INTERESTING,
                                    threshold_only_fire=FILTER_THRESHOLD_ONLY_FIRE,
                                    cold_start=(events_count < FILTER_COLD_START_MIN_EVENTS),
                                    epsilon=0.0,
                                )
                                log.info(
                                    "reco_dual_decision_probe",
                                    post_id=post.id,
                                    user_id=user.id,
                                    sim_sql=round(float(sim_sql), 6),
                                    sim_neo4j=round(float(sim_neo4j), 6),
                                    sim_delta=round(float(sim_neo4j - sim_sql), 6),
                                    final_score_sql_path=round(float(final_score), 6),
                                    final_score_neo4j_path=round(float(probe_final), 6),
                                    score_delta=round(float(probe_final - final_score), 6),
                                    send_sql_path=bool(send),
                                    send_neo4j_path=bool(probe_send),
                                    decision_changed=bool(send != probe_send),
                                )
                    if RECO_DRY_RUN:
                        send = True
                        reason = f"dry_run_{reason}"
                except Exception as e:
                    log.warning(
                        "dispatch.reco_failed_fallback_all",
                        post_id=post.id,
                        user_id=user.id,
                        error=str(e),
                    )
                    send, reason = reco_error_fallback_decision()

                user_decisions[user.id] = {
                    "reco_send": send,
                    "send": send,
                    "mode": mode,
                    "reason": reason,
                    "events_count": events_count,
                    "baseline": baseline,
                    "logistic_prob": logistic_prob,
                    "final_score": final_score,
                    "subscription_status": subscription_status,
                    "shadow": shadow,
                    "instruction_scope": None,
                    "instruction_decision": None,
                    "instruction_reason": None,
                    "instruction_latency_ms": None,
                    "instruction_error_code": None,
                    "soft_bypass": False,
                    "system_notice": None,
                }
                audit(
                    "dispatch.filter_decision",
                    post_id=post.id,
                    community_id=community.id,
                    user_id=user.id,
                    subscription_status=subscription_status,
                    mode=mode,
                    send=send,
                    reason=reason,
                    events_count=events_count,
                    baseline=baseline,
                    logistic_prob=logistic_prob,
                    final_score=final_score,
                    would_send_all=bool(shadow["all"]["send"]),
                    would_reason_all=str(shadow["all"]["reason"]),
                    would_send_not_interesting=bool(shadow["not_interesting"]["send"]),
                    would_reason_not_interesting=str(shadow["not_interesting"]["reason"]),
                    would_send_only_fire=bool(shadow["only_fire"]["send"]),
                    would_reason_only_fire=str(shadow["only_fire"]["reason"]),
                    would_send_digest_only=bool(shadow["digest_only"]["send"]),
                    would_reason_digest_only=str(shadow["digest_only"]["reason"]),
                )
                resolved_rule = instruction_rules.get(int(user.id))
                if resolved_rule:
                    premium_active = premium_access_cache.get(int(user.id))
                    if premium_active is None:
                        premium_active = await user_has_instruction_filter_access(session, int(user.id))
                        premium_access_cache[int(user.id)] = premium_active
                    if premium_active:
                        instruction_eval = await get_or_create_instruction_decision(
                            session,
                            user=user,
                            post_id=int(post.id),
                            community=community,
                            source_title=community.name or post.title,
                            post_title=post.title,
                            post_text=post.content,
                            resolved_rule=resolved_rule,
                        )
                        final_send, soft_bypass = merge_instruction_and_reco(
                            reco_send=bool(send),
                            instruction_decision=instruction_eval.decision,
                        )
                        instruction_reason = instruction_eval.reason_short
                        if soft_bypass and send:
                            instruction_reason = "Instruction filter temporarily bypassed"
                            user_decisions[user.id]["system_notice"] = t(
                                get_user_locale(user),
                                "instruction_soft_bypass_notice",
                            )
                        user_decisions[user.id].update(
                            {
                                "send": final_send,
                                "instruction_scope": resolved_rule.scope,
                                "instruction_decision": instruction_eval.decision,
                                "instruction_reason": instruction_reason,
                                "instruction_latency_ms": instruction_eval.latency_ms,
                                "instruction_error_code": instruction_eval.error_code,
                                "soft_bypass": soft_bypass,
                            }
                        )
                        audit(
                            "dispatch.instruction_decision",
                            post_id=post.id,
                            community_id=community.id,
                            user_id=user.id,
                            rule_scope=resolved_rule.scope,
                            prompt_hash=resolved_rule.prompt_hash,
                            reco_allow=bool(send),
                            llm_allow=bool(instruction_eval.llm_allow),
                            final_allow=bool(final_send),
                            soft_bypass=bool(soft_bypass),
                            decision=instruction_eval.decision,
                            reason_short=instruction_reason,
                            latency_ms=instruction_eval.latency_ms,
                            error_code=instruction_eval.error_code,
                        )
                    else:
                        audit(
                            "dispatch.instruction_skipped_non_premium",
                            post_id=post.id,
                            community_id=community.id,
                            user_id=user.id,
                            rule_scope=resolved_rule.scope,
                        )
                if user_decisions[user.id]["send"]:
                    user_send_list.append(user)

            if not user_send_list:
                log.info(
                    "Dispatch skipped: filtered for all users",
                    post_id=post.id,
                    community_id=community.id,
                    users_count=len(users),
                )
                audit(
                    "dispatch.filtered_all",
                    post_id=post.id,
                    community_id=community.id,
                    users_count=len(users),
                )
                post.dispatch_finished_at = datetime.now(timezone.utc)
                session.add(post)
                await session.commit()
                return None
            
            log.info(
                "Dispatch start",
                post_id=post.id,
                community_id=community.id,
                users_count=len(user_send_list),
            )

            # if not users:
            #     log.info("No subscribers for community", community_id=community.id, post_id=post.id)
            #     audit("dispatch.no_subscribers", post_id=post.id, community_id=community.id)
            #     return None

            # summary
            summary_text = None
            try:
                summary_audience = list(users)
                if any(bool(getattr(u, "summary_enabled", False)) for u in summary_audience) and post.content:
                    content_text = post.content.strip()
                    if len(content_text) < SUMMARY_MIN_CHARS:
                        log.info("Skip summarization for short post", post_id=post.id, length=len(content_text))
                        summary_text = None
                    else:
                        summary_text = post.summary
                        if not summary_text:
                            summary_text = await summarize(post.content)
                            post.summary = summary_text
                            post.summary_created_at = datetime.now(timezone.utc) if summary_text else None
                            await session.commit()
            except Exception as e:
                # не валим рассылку из-за суммаризации; просто логируем
                log.error("Summarization failed, continue without summary", post_id=post.id, error=str(e), exc_info=True)
                audit("dispatch.summary_failed", post_id=post.id, community_id=community.id, error=str(e))
                summary_text = None

            content_link = post.content_link or {}
            original_url = content_link.get("url")

            if not original_url:
                msg_id = content_link.get("id")  # это telegram message.id [userbot.py]
                base = (community.link or "").rstrip("/")
                if base and msg_id:
                    original_url = f"{base}/{int(msg_id)}"
                    # опционально: записать обратно в БД, чтобы в будущем url уже был заполнен
                    content_link["url"] = original_url
                    post.content_link = content_link
                    await session.commit()

            sem = asyncio.Semaphore(SEND_CONCURRENCY)

            async def _send_with_sem(user: User):
                async with sem:
                    decision = user_decisions.get(user.id, {})
                    audit(
                        "send.attempt",
                        post_id=post.id,
                        community_id=community.id,
                        user_id=user.id,
                        subscription_status=decision.get("subscription_status"),
                        telegram_id=getattr(user, "telegram_id", None),
                        tg_username=(getattr(user, "username", None) or None),
                        first_name=(getattr(user, "first_name", None) or None),
                        last_name=(getattr(user, "last_name", None) or None),
                        summary_enabled=bool(getattr(user, "summary_enabled", False)),
                        reco_mode=decision.get("mode"),
                        reco_reason=decision.get("reason"),
                        reco_score=decision.get("final_score"),
                        instruction_scope=decision.get("instruction_scope"),
                        instruction_decision=decision.get("instruction_decision"),
                        instruction_reason=decision.get("instruction_reason"),
                        soft_bypass=bool(decision.get("soft_bypass")),
                    )
                    log.info(
                        "Send attempt",
                        post_id=post.id,
                        community_id=community.id,
                        user_id=user.id,
                        telegram_id=getattr(user, "telegram_id", None),
                        tg_username=(getattr(user, "username", None) or None),
                    )

                    try:
                        locale = get_user_locale(user)
                        delivery = await send_post_to_user(
                            telegram_id=user.telegram_id,
                            user_id=user.id,
                            tg_username=user.username,
                            post_id=post.id,
                            text=post.content,
                            original_url=original_url,
                            source_title=post.title,
                            summary_enabled=bool(user.summary_enabled),
                            summary_text=summary_text if user.summary_enabled else None,
                            media_path=post.media_path,
                            media_mime=post.media_mime,
                            media_items=post.media_items,
                            locale=locale,
                            system_notice=decision.get("system_notice"),
                            storytracking_enabled=storytracking_allowed_for_user(user),
                        )

                        if (delivery or {}).get("delivery_result") == "blocked_by_user":
                            previous_filter = (getattr(user, "feed_filter", "all") or "all").strip().lower()
                            user.feed_filter = "digest_only"
                            user.last_live_feed_filter = previous_filter
                            session.add(user)
                            audit(
                                "dispatch.user_marked_digest_only_blocked_bot",
                                post_id=post.id,
                                community_id=community.id,
                                user_id=user.id,
                                telegram_id=getattr(user, "telegram_id", None),
                                previous_filter=previous_filter,
                                new_filter="digest_only",
                            )
                            log.info(
                                "Dispatch disabled for blocked Telegram user",
                                post_id=post.id,
                                community_id=community.id,
                                user_id=user.id,
                                telegram_id=getattr(user, "telegram_id", None),
                            )
                            return {"message_id": None, "delivery_result": "blocked_by_user"}

                        audit(
                            "send.success",
                            post_id=post.id,
                            community_id=community.id,
                            user_id=user.id,
                            subscription_status=decision.get("subscription_status"),
                            telegram_id=getattr(user, "telegram_id", None),
                            delivery_result=(delivery or {}).get("delivery_result"),
                            telegram_message_id=(delivery or {}).get("message_id"),
                        )
                        log.info(
                            "Send success",
                            post_id=post.id,
                            community_id=community.id,
                            user_id=user.id,
                            telegram_id=getattr(user, "telegram_id", None),
                        )
                        return delivery or {}

                    except TelegramForbiddenError as e:
                        previous_filter = (getattr(user, "feed_filter", "all") or "all").strip().lower()
                        user.feed_filter = "digest_only"
                        user.last_live_feed_filter = previous_filter
                        session.add(user)
                        audit(
                            "dispatch.user_marked_digest_only_blocked_bot",
                            post_id=post.id,
                            community_id=community.id,
                            user_id=user.id,
                            telegram_id=getattr(user, "telegram_id", None),
                            error=str(e),
                            error_type=type(e).__name__,
                            previous_filter=previous_filter,
                            new_filter="digest_only",
                        )
                        log.info(
                            "Dispatch disabled for blocked Telegram user",
                            post_id=post.id,
                            community_id=community.id,
                            user_id=user.id,
                            telegram_id=getattr(user, "telegram_id", None),
                        )
                        return {"message_id": None, "delivery_result": "blocked_by_user"}

                    except Exception as e:
                        audit(
                            "send.fail",
                            post_id=post.id,
                            community_id=community.id,
                            user_id=user.id,
                            telegram_id=getattr(user, "telegram_id", None),
                            error=str(e),
                            error_type=type(e).__name__,
                        )
                        log.error(
                            "Send failed",
                            post_id=post.id,
                            community_id=community.id,
                            user_id=user.id,
                            telegram_id=getattr(user, "telegram_id", None),
                            error=str(e),
                            error_type=type(e).__name__,
                            exc_info=True,
                        )
                        return e  # чтобы собрать статистику ниже

            # Idempotency guard: claim each (post_id, user_id) in the
            # dispatch_deliveries outbox before sending. On Celery retry the
            # duplicate claim returns no rows → that user is skipped and we
            # avoid double-sending. On send failure the row is deleted so the
            # retry can re-attempt.
            async def _claim_delivery(user_id: int) -> bool:
                claim = await session.execute(
                    text(
                        """
                        INSERT INTO dispatch_deliveries (post_id, user_id, attempt_count)
                        VALUES (:post_id, :user_id, 1)
                        ON CONFLICT (post_id, user_id) DO NOTHING
                        RETURNING post_id
                        """
                    ),
                    {"post_id": int(post.id), "user_id": int(user_id)},
                )
                return claim.scalar_one_or_none() is not None

            async def _release_delivery(user_id: int) -> None:
                await session.execute(
                    text(
                        "DELETE FROM dispatch_deliveries "
                        "WHERE post_id = :post_id AND user_id = :user_id "
                        "AND sent_at IS NULL"
                    ),
                    {"post_id": int(post.id), "user_id": int(user_id)},
                )

            async def _mark_delivery_sent(user_id: int, delivery: dict | None) -> None:
                message_id = None
                if isinstance(delivery, dict):
                    raw_mid = delivery.get("message_id")
                    if raw_mid is not None:
                        try:
                            message_id = int(raw_mid)
                        except (TypeError, ValueError):
                            message_id = None
                await session.execute(
                    text(
                        """
                        UPDATE dispatch_deliveries
                        SET sent_at = now(),
                            telegram_message_id = :mid
                        WHERE post_id = :post_id AND user_id = :user_id
                        """
                    ),
                    {
                        "post_id": int(post.id),
                        "user_id": int(user_id),
                        "mid": message_id,
                    },
                )

            claimed_users: list[User] = []
            skipped_duplicate = 0
            for u in user_send_list:
                if await _claim_delivery(int(u.id)):
                    claimed_users.append(u)
                else:
                    skipped_duplicate += 1
            await session.commit()

            if skipped_duplicate:
                audit(
                    "dispatch.skipped_duplicate",
                    post_id=post.id,
                    community_id=community.id,
                    skipped=skipped_duplicate,
                )
                log.info(
                    "Dispatch duplicate claims skipped",
                    post_id=post.id,
                    community_id=community.id,
                    skipped=skipped_duplicate,
                )

            if claimed_users:
                results = await asyncio.gather(
                    *[_send_with_sem(u) for u in claimed_users],
                    return_exceptions=True,
                )
                sent_at = datetime.now(timezone.utc)
                for user, result in zip(claimed_users, results, strict=False):
                    try:
                        if isinstance(result, dict):
                            variant = await ensure_aa_assignment(session, user_id=int(user.id))
                            await _mark_delivery_sent(int(user.id), result)
                            raw_message_id = result.get("message_id")
                            try:
                                message_id = int(raw_message_id) if raw_message_id is not None else None
                            except (TypeError, ValueError):
                                message_id = None
                            content_chars = len(post.content or "")
                            summary_chars = len(summary_text or "")
                            has_summary = bool(user.summary_enabled and summary_text and content_chars >= SUMMARY_MIN_CHARS)
                            await record_aa_exposure(
                                session,
                                user_id=int(user.id),
                                post_id=int(post.id),
                                variant=variant,
                                sent_at=sent_at,
                                telegram_message_id=message_id,
                                delivery_result=str(result.get("delivery_result") or "").strip() or None,
                                render_mode="summary" if has_summary else "full",
                                content_chars=content_chars,
                                summary_chars=summary_chars if has_summary else 0,
                                has_summary=has_summary,
                            )
                            audit(
                                "experiment.aa_exposure",
                                post_id=post.id,
                                community_id=community.id,
                                user_id=user.id,
                                telegram_id=getattr(user, "telegram_id", None),
                                variant=variant,
                                render_mode="summary" if has_summary else "full",
                            )
                        else:
                            await _release_delivery(int(user.id))
                    except Exception as e:
                        log.warning(
                            "dispatch.delivery_experiment_update_failed",
                            post_id=post.id,
                            user_id=user.id,
                            error=str(e),
                        )
                try:
                    await session.commit()
                except Exception:
                    await session.rollback()
            else:
                results = []

            ok = sum(1 for r in results if isinstance(r, dict))
            fail = sum(1 for r in results if not isinstance(r, dict))
            storyline_ok = 0
            storyline_fail = 0

            audit(
                "dispatch.finish",
                post_id=post.id,
                community_id=community.id,
                users_count=len(claimed_users),
                ok=ok,
                fail=fail,
                storyline_ok=storyline_ok,
                storyline_fail=storyline_fail,
                skipped_duplicate=skipped_duplicate,
            )
            log.info(
                "Dispatch finish",
                post_id=post.id,
                community_id=community.id,
                users_count=len(claimed_users),
                ok=ok,
                fail=fail,
                storyline_ok=storyline_ok,
                storyline_fail=storyline_fail,
                filtered=len(users) - len(user_send_list),
                skipped_duplicate=skipped_duplicate,
            )
            post.dispatch_finished_at = datetime.now(timezone.utc)
            total_fail = fail + storyline_fail
            post.dispatch_error = None if total_fail == 0 else f"partial_fail:{total_fail}"
            session.add(post)
            await session.commit()
            return None

        return await _run_with_session(_work)

    try:
        return _run_async(_task())
    except SQLAlchemyError as e:
        raise self.retry(
            exc=e,
            countdown=min(60, 2 ** self.request.retries),
            max_retries=DB_MAX_RETRIES,
        )
    except DeepseekRetryableError as e:
        raise self.retry(
            exc=e,
            countdown=min(60, 2 ** self.request.retries),
            max_retries=DEEPSEEK_MAX_RETRIES,
        )


@celery_app.task(name="app.tasks.dispatch_storyline_updates_for_post", bind=True)
def dispatch_storyline_updates_for_post(self, post_id: int):
    async def _work(session):
        from app.audit import audit

        post = await session.get(Post, int(post_id))
        if not post:
            log.warning("storyline_dispatch_post_not_found", post_id=post_id)
            audit("storyline_dispatch.post_not_found", post_id=post_id)
            return None

        community = await session.get(Community, int(post.community_id))
        if not community:
            log.warning(
                "storyline_dispatch_community_not_found",
                post_id=post_id,
                community_id=getattr(post, "community_id", None),
            )
            audit(
                "storyline_dispatch.community_not_found",
                post_id=post_id,
                community_id=getattr(post, "community_id", None),
            )
            return None

        status_row = await session.execute(
            text("SELECT ie_status FROM posts WHERE id = :post_id"),
            {"post_id": int(post_id)},
        )
        ie_status = str(status_row.scalar_one_or_none() or "").strip().lower()
        if ie_status != "done":
            log.info(
                "storyline_dispatch_waiting_for_ie",
                post_id=post_id,
                ie_status=ie_status or None,
                retry=self.request.retries,
                max_retries=STORYLINE_DISPATCH_IE_MAX_RETRIES,
            )
            audit(
                "storyline_dispatch.waiting_for_ie",
                post_id=post_id,
                community_id=getattr(post, "community_id", None),
                ie_status=ie_status or None,
                retry=self.request.retries,
                max_retries=STORYLINE_DISPATCH_IE_MAX_RETRIES,
            )
            if self.request.retries < STORYLINE_DISPATCH_IE_MAX_RETRIES:
                raise self.retry(
                    countdown=STORYLINE_DISPATCH_IE_RETRY_SECONDS,
                    max_retries=STORYLINE_DISPATCH_IE_MAX_RETRIES,
                )
            return {
                "ok": 0,
                "fail": 0,
                "followup_ok": 0,
                "followup_fail": 0,
                "skipped_duplicate": 0,
                "family_root_storyline_id": None,
                "reason": "ie_not_done",
            }

        return await _dispatch_storyline_updates_for_post(
            session,
            post=post,
            community=community,
        )

    async def _task():
        return await _run_with_session(_work)

    try:
        return _run_async(_task())
    except SQLAlchemyError as e:
        raise self.retry(
            exc=e,
            countdown=min(60, 2 ** self.request.retries),
            max_retries=DB_MAX_RETRIES,
        )
    except DeepseekRetryableError as e:
        raise self.retry(
            exc=e,
            countdown=min(60, 2 ** self.request.retries),
            max_retries=DEEPSEEK_MAX_RETRIES,
        )


@celery_app.task(name="app.tasks.detect_missing_dispatch")
def detect_missing_dispatch():
    async def _task():
        async def _work(session):
            from app.audit import audit

            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(seconds=60)
            orphan_marked = (
                await session.execute(
                    update(Post)
                    .where(
                        Post.dispatch_enqueued_at.is_not(None),
                        Post.dispatch_enqueued_at < cutoff,
                        Post.dispatch_started_at.is_(None),
                        Post.community_id.is_(None),
                    )
                    .values(
                        dispatch_started_at=now,
                        dispatch_finished_at=now,
                        dispatch_error="community_missing_backlog",
                    )
                )
            ).rowcount or 0
            if orphan_marked:
                log.info("dispatch_missing_orphan_backlog_marked", count=orphan_marked)

            rows = (
                await session.execute(
                    select(Post)
                    .where(
                        Post.dispatch_enqueued_at.is_not(None),
                        Post.dispatch_enqueued_at < cutoff,
                        Post.dispatch_started_at.is_(None),
                        Post.community_id.is_not(None),
                    )
                    .order_by(Post.dispatch_enqueued_at.asc())
                    .limit(100)
                )
            ).scalars().all()

            for post in rows:
                audit(
                    "dispatch_missing_after_post_saved",
                    post_id=post.id,
                    community_id=post.community_id,
                    dispatch_enqueued_at=post.dispatch_enqueued_at.isoformat() if post.dispatch_enqueued_at else None,
                )
            if rows:
                sample_ids = [int(post.id) for post in rows[:10] if getattr(post, "id", None) is not None]
                log.warning(
                    "dispatch_missing_after_post_saved_batch",
                    count=len(rows),
                    sample_post_ids=sample_ids,
                    oldest_dispatch_enqueued_at=rows[0].dispatch_enqueued_at.isoformat()
                    if rows[0].dispatch_enqueued_at
                    else None,
                )
            return len(rows)

        return await _run_with_session(_work)

    return _run_async(_task())


@celery_app.task(name="app.tasks.cleanup_summaries")
def cleanup_summaries():
    summary_ttl_days = int(os.getenv("SUMMARY_TTL_DAYS", "14"))
    if summary_ttl_days <= 0:
        return 0

    async def _task():
        async def _work(session):
            cutoff = datetime.now(timezone.utc) - timedelta(days=summary_ttl_days)
            result = await session.execute(
                update(Post)
                .where(Post.summary_created_at.is_not(None), Post.summary_created_at < cutoff)
                .values(summary=None, summary_created_at=None)
            )
            await session.commit()
            return result.rowcount or 0

        return await _run_with_session(_work)

    return _run_async(_task())


BILLING_BATCH_SIZE = int(os.getenv("BILLING_BATCH_SIZE", "50"))
PAYMENT_SYNC_PENDING_LIMIT = int(os.getenv("PAYMENT_SYNC_PENDING_LIMIT", "50"))
PAYMENT_SYNC_REPAIR_LIMIT = int(os.getenv("PAYMENT_SYNC_REPAIR_LIMIT", "50"))
PAYMENT_SYNC_RECENT_HOURS = int(os.getenv("PAYMENT_SYNC_RECENT_HOURS", "24"))


@celery_app.task(name="app.tasks.billing_monthly_rollover", bind=True)
def billing_monthly_rollover(self):
    async def _task():
        async def _work(session):
            degraded_total = 0
            updated_total = 0
            offset = 0
            while True:
                batch_res = await session.execute(
                    select(User).order_by(User.id).offset(offset).limit(BILLING_BATCH_SIZE)
                )
                batch = batch_res.scalars().all()
                if not batch:
                    break
                for user in batch:
                    try:
                        await recalculate_user_billing_state(session, user.id)
                        removed = await enforce_degrade_if_needed(session, user)
                        degraded_total += int(removed or 0)
                        updated_total += 1
                    except Exception as e:
                        log.warning("billing.rollover.user_failed", user_id=user.id, error=str(e))
                await session.commit()
                offset += BILLING_BATCH_SIZE
            log.info(
                "billing.rollover.done",
                users_updated=updated_total,
                degraded_total=degraded_total,
            )
            return {"users_updated": updated_total, "degraded_total": degraded_total}

        return await _run_with_session(_work)

    return _run_async(_task())


@celery_app.task(name="app.tasks.billing_sync_pending_payments", bind=True)
def billing_sync_pending_payments(self):
    async def _task():
        async def _work(session):
            created_after = _utcnow() - timedelta(hours=max(1, PAYMENT_SYNC_RECENT_HOURS))
            poll_result = await sync_recent_pending_payments(
                session,
                limit=max(1, PAYMENT_SYNC_PENDING_LIMIT),
                created_after=created_after,
            )
            repair_result = await repair_confirmed_payment_entitlements(
                session,
                limit=max(1, PAYMENT_SYNC_REPAIR_LIMIT),
            )
            await session.commit()
            log.info(
                "billing.sync_pending_payments.done",
                poll=poll_result,
                repair=repair_result,
            )
            return {"poll": poll_result, "repair": repair_result}

        return await _run_with_session(_work)

    return _run_async(_task())


@celery_app.task(name="app.tasks.billing_renewal_reminders", bind=True)
def billing_renewal_reminders(self):
    async def _task():
        async def _work(session):
            now = _utcnow()
            users = (
                await session.execute(
                    select(User)
                    .join(PaymentOrder, PaymentOrder.user_id == User.id)
                    .where(PaymentOrder.status == "confirmed")
                    .distinct()
                    .order_by(User.id.asc())
                )
            ).scalars().all()

            sent_total = 0
            skipped_total = 0
            for user in users:
                entitlement = await resolve_premium_entitlement(session, user.id, now_utc=now)
                if not entitlement.active or not entitlement.expires_at:
                    skipped_total += 1
                    continue

                days_left = (entitlement.expires_at.date() - now.date()).days
                if days_left not in {3, 1}:
                    skipped_total += 1
                    continue

                current_order = (
                    await session.execute(
                        select(PaymentOrder)
                        .where(
                            PaymentOrder.user_id == user.id,
                            PaymentOrder.status == "confirmed",
                            PaymentOrder.period_end == entitlement.expires_at,
                        )
                        .order_by(PaymentOrder.confirmed_at.desc(), PaymentOrder.id.desc())
                    )
                ).scalars().first()
                if not current_order:
                    skipped_total += 1
                    continue

                event_type = f"renewal_reminder_{days_left}d"
                event_exists = await session.execute(
                    select(PaymentEvent.id)
                    .where(
                        PaymentEvent.payment_order_id == current_order.id,
                        PaymentEvent.event_type == event_type,
                    )
                    .limit(1)
                )
                if event_exists.scalar_one_or_none() is not None:
                    skipped_total += 1
                    continue

                if not getattr(user, "telegram_id", None):
                    skipped_total += 1
                    continue

                locale = get_user_locale(user)
                reminder_key = "billing_renewal_reminder_3d" if days_left == 3 else "billing_renewal_reminder_1d"
                text = t(
                    locale,
                    reminder_key,
                    expiry_date=entitlement.expires_at.strftime("%Y-%m-%d"),
                )
                sent = await _send_billing_message(user.telegram_id, text)
                if not sent:
                    skipped_total += 1
                    continue

                session.add(
                    PaymentEvent(
                        payment_order_id=current_order.id,
                        event_type=event_type,
                        payload={
                            "user_id": user.id,
                            "telegram_id": user.telegram_id,
                            "days_left": days_left,
                            "premium_expires_at": entitlement.expires_at.isoformat(),
                        },
                    )
                )
                sent_total += 1
                await session.commit()

            log.info(
                "billing.renewal_reminders.done",
                checked_total=len(users),
                sent_total=sent_total,
                skipped_total=skipped_total,
            )
            return {
                "checked_total": len(users),
                "sent_total": sent_total,
                "skipped_total": skipped_total,
            }

        return await _run_with_session(_work)

    return _run_async(_task())


@celery_app.task(name="app.tasks.premium_week_after_first_subscription_reminders", bind=True)
def premium_week_after_first_subscription_reminders(self):
    async def _task():
        async def _work(session):
            now = _utcnow()
            due_before = now - timedelta(days=7)
            sent_user_ids = _read_premium_week_reminder_user_ids()

            first_subscription = (
                select(
                    SubscriptionRequest.user_id.label("user_id"),
                    func.min(SubscriptionRequest.updated_at).label("first_subscription_at"),
                )
                .where(SubscriptionRequest.status == "completed")
                .group_by(SubscriptionRequest.user_id)
                .subquery()
            )
            rows = (
                await session.execute(
                    select(User, first_subscription.c.first_subscription_at)
                    .join(first_subscription, first_subscription.c.user_id == User.id)
                    .where(first_subscription.c.first_subscription_at <= due_before)
                    .order_by(User.id.asc())
                )
            ).all()

            sent_total = 0
            skipped_total = 0
            for user, first_subscription_at in rows:
                if int(user.id) in sent_user_ids:
                    skipped_total += 1
                    continue
                if not getattr(user, "telegram_id", None):
                    skipped_total += 1
                    continue

                entitlement = await resolve_premium_entitlement(session, user.id, now_utc=now)
                if entitlement.active:
                    skipped_total += 1
                    continue

                locale = get_user_locale(user)
                sent = await _send_billing_message(
                    user.telegram_id,
                    t(locale, "premium_week_after_first_subscription"),
                    reply_markup=_premium_week_reminder_keyboard(locale),
                )
                if not sent:
                    skipped_total += 1
                    continue

                _mark_premium_week_reminder_sent(
                    user_id=user.id,
                    telegram_id=user.telegram_id,
                    first_subscription_at=first_subscription_at,
                    sent_at=now,
                )
                sent_user_ids.add(int(user.id))
                sent_total += 1
                await session.commit()

            log.info(
                "premium_week_after_first_subscription_reminders.done",
                checked_total=len(rows),
                sent_total=sent_total,
                skipped_total=skipped_total,
            )
            return {
                "checked_total": len(rows),
                "sent_total": sent_total,
                "skipped_total": skipped_total,
            }

        return await _run_with_session(_work)

    return _run_async(_task())


@celery_app.task(name="app.tasks.redeliver_due_news_time_followups", bind=True)
def redeliver_due_news_time_followups(
    self,
    *,
    dry_run: bool = True,
    limit: int | None = None,
    include_asked_unanswered: bool = True,
    only_user_ids: list[int] | None = None,
):
    async def _task():
        async def _work(session):
            from app.audit import audit

            now = _utcnow()
            followup_due_cutoff = now - NEWS_TIME_FOLLOWUP_DELAY
            subscribed = _subscribed_user_ids_subquery()
            effective_limit = int(limit or NEWS_TIME_SURVEY_REDELIVERY_LIMIT)
            if effective_limit <= 0:
                effective_limit = NEWS_TIME_SURVEY_REDELIVERY_LIMIT

            stmt = (
                select(User, UserNewsTimeSurvey)
                .join(subscribed, subscribed.c.user_id == User.id)
                .join(UserNewsTimeSurvey, UserNewsTimeSurvey.user_id == User.id)
                .where(
                    UserNewsTimeSurvey.baseline_answered_at.is_not(None),
                    User.created_at <= followup_due_cutoff,
                    UserNewsTimeSurvey.time_saved_answered_at.is_(None),
                    UserNewsTimeSurvey.followup_abandoned_at.is_(None),
                )
                .order_by(User.created_at.asc(), User.id.asc())
                .limit(effective_limit)
            )
            if only_user_ids:
                normalized_user_ids = [int(user_id) for user_id in only_user_ids if user_id is not None]
                if normalized_user_ids:
                    stmt = stmt.where(User.id.in_(normalized_user_ids))
            if not include_asked_unanswered:
                stmt = stmt.where(UserNewsTimeSurvey.followup_asked_at.is_(None))

            rows = (await session.execute(stmt)).all()

            checked_total = 0
            sent_total = 0
            skipped_total = 0
            failed_total = 0
            due_never_asked = 0
            due_asked_unanswered = 0

            for user, survey in rows:
                checked_total += 1
                asked_before = getattr(survey, "followup_asked_at", None) is not None
                if asked_before:
                    due_asked_unanswered += 1
                else:
                    due_never_asked += 1

                if not getattr(user, "telegram_id", None):
                    skipped_total += 1
                    continue

                if dry_run:
                    continue

                locale = get_user_locale(user)
                audit(
                    "news_time_survey.redelivery_attempt",
                    user_id=user.id,
                    telegram_id=user.telegram_id,
                    followup_due_at=str(getattr(survey, "followup_due_at", None)),
                    asked_before=asked_before,
                )
                sent, send_error = await _send_news_time_followup_prompt(
                    telegram_id=int(user.telegram_id),
                    locale=locale,
                )
                if not sent:
                    failed_total += 1
                    audit(
                        "news_time_survey.redelivery_failed",
                        user_id=user.id,
                        telegram_id=user.telegram_id,
                        followup_due_at=str(getattr(survey, "followup_due_at", None)),
                        asked_before=asked_before,
                        error=send_error,
                    )
                    continue

                sent_at = _utcnow()
                expected_due = followup_due_at(getattr(user, "created_at", sent_at))
                survey.followup_due_at = expected_due
                if getattr(survey, "followup_asked_at", None) is None:
                    survey.followup_asked_at = sent_at
                    audit(
                        "news_time_survey.followup_asked",
                        telegram_id=user.telegram_id,
                        user_id=user.id,
                    )
                else:
                    survey.followup_retry_asked_at = sent_at
                    audit(
                        "news_time_survey.followup_retry_asked",
                        telegram_id=user.telegram_id,
                        user_id=user.id,
                    )
                await session.commit()
                sent_total += 1
                audit(
                    "news_time_survey.redelivery_success",
                    user_id=user.id,
                    telegram_id=user.telegram_id,
                    sent_at=str(sent_at),
                    asked_before=asked_before,
                )

            log.info(
                "news_time_survey.redelivery.done",
                dry_run=bool(dry_run),
                include_asked_unanswered=bool(include_asked_unanswered),
                checked_total=checked_total,
                sent_total=sent_total,
                skipped_total=skipped_total,
                failed_total=failed_total,
                due_never_asked=due_never_asked,
                due_asked_unanswered=due_asked_unanswered,
            )
            return {
                "dry_run": bool(dry_run),
                "include_asked_unanswered": bool(include_asked_unanswered),
                "limit": effective_limit,
                "only_user_ids": [int(user_id) for user_id in only_user_ids] if only_user_ids else None,
                "checked_total": checked_total,
                "sent_total": sent_total,
                "skipped_total": skipped_total,
                "failed_total": failed_total,
                "due_never_asked": due_never_asked,
                "due_asked_unanswered": due_asked_unanswered,
            }

        return await _run_with_session(_work)

    return _run_async(_task())


@celery_app.task(name="app.tasks.redeliver_missing_news_time_baseline", bind=True)
def redeliver_missing_news_time_baseline(
    self,
    *,
    dry_run: bool = True,
    limit: int | None = None,
    only_user_ids: list[int] | None = None,
):
    async def _task():
        async def _work(session):
            from app.audit import audit

            effective_limit = int(limit or NEWS_TIME_BASELINE_REDELIVERY_LIMIT)
            if effective_limit <= 0:
                effective_limit = NEWS_TIME_BASELINE_REDELIVERY_LIMIT
            subscribed = _subscribed_user_ids_subquery()

            rows = (
                await session.execute(
                    select(User, UserNewsTimeSurvey)
                    .join(subscribed, subscribed.c.user_id == User.id)
                    .outerjoin(UserNewsTimeSurvey, UserNewsTimeSurvey.user_id == User.id)
                    .where(
                        or_(
                            UserNewsTimeSurvey.id.is_(None),
                            UserNewsTimeSurvey.baseline_answered_at.is_(None),
                        )
                    )
                    .order_by(User.id.asc())
                    .limit(effective_limit)
                )
            ).all()
            if only_user_ids:
                normalized_user_ids = {int(user_id) for user_id in only_user_ids if user_id is not None}
                if normalized_user_ids:
                    rows = [(user, survey) for user, survey in rows if int(user.id) in normalized_user_ids]

            checked_total = 0
            sent_total = 0
            skipped_total = 0
            failed_total = 0
            missing_survey_row_total = 0
            existing_without_baseline_total = 0

            for user, survey in rows:
                checked_total += 1
                if survey is None:
                    missing_survey_row_total += 1
                else:
                    existing_without_baseline_total += 1

                if not getattr(user, "telegram_id", None):
                    skipped_total += 1
                    continue

                if dry_run:
                    continue

                locale = get_user_locale(user)
                audit(
                    "news_time_survey.baseline_redelivery_attempt",
                    user_id=user.id,
                    telegram_id=user.telegram_id,
                    missing_survey_row=bool(survey is None),
                )
                sent, send_error = await _send_news_time_baseline_prompt(
                    telegram_id=int(user.telegram_id),
                    locale=locale,
                )
                if not sent:
                    failed_total += 1
                    audit(
                        "news_time_survey.baseline_redelivery_failed",
                        user_id=user.id,
                        telegram_id=user.telegram_id,
                        missing_survey_row=bool(survey is None),
                        error=send_error,
                    )
                    continue

                if survey is None:
                    survey = UserNewsTimeSurvey(user_id=user.id)
                    session.add(survey)
                sent_at = _utcnow()
                if survey and getattr(survey, "baseline_asked_at", None) is None:
                    survey.baseline_asked_at = sent_at
                    audit("news_time_survey.baseline_asked", telegram_id=user.telegram_id, user_id=user.id)
                elif survey:
                    survey.baseline_retry_asked_at = sent_at
                    audit("news_time_survey.baseline_retry_asked", telegram_id=user.telegram_id, user_id=user.id)
                await session.commit()
                sent_total += 1
                audit(
                    "news_time_survey.baseline_redelivery_success",
                    user_id=user.id,
                    telegram_id=user.telegram_id,
                    missing_survey_row=bool(survey is None),
                )

            log.info(
                "news_time_survey.baseline_redelivery.done",
                dry_run=bool(dry_run),
                checked_total=checked_total,
                sent_total=sent_total,
                skipped_total=skipped_total,
                failed_total=failed_total,
                missing_survey_row_total=missing_survey_row_total,
                existing_without_baseline_total=existing_without_baseline_total,
            )
            return {
                "dry_run": bool(dry_run),
                "limit": effective_limit,
                "only_user_ids": [int(user_id) for user_id in only_user_ids] if only_user_ids else None,
                "checked_total": checked_total,
                "sent_total": sent_total,
                "skipped_total": skipped_total,
                "failed_total": failed_total,
                "missing_survey_row_total": missing_survey_row_total,
                "existing_without_baseline_total": existing_without_baseline_total,
            }

        return await _run_with_session(_work)

    return _run_async(_task())


@celery_app.task(name="app.tasks.redeliver_news_time_current_question", bind=True)
def redeliver_news_time_current_question(
    self,
    *,
    dry_run: bool = True,
    limit: int | None = None,
    only_user_ids: list[int] | None = None,
):
    async def _task():
        async def _work(session):
            from app.audit import audit

            effective_limit = int(limit or NEWS_TIME_CURRENT_REDELIVERY_LIMIT)
            if effective_limit <= 0:
                effective_limit = NEWS_TIME_CURRENT_REDELIVERY_LIMIT
            subscribed = _subscribed_user_ids_subquery()

            rows = (
                await session.execute(
                    select(User, UserNewsTimeSurvey)
                    .join(subscribed, subscribed.c.user_id == User.id)
                    .join(UserNewsTimeSurvey, UserNewsTimeSurvey.user_id == User.id)
                    .where(
                        UserNewsTimeSurvey.baseline_answered_at.is_not(None),
                        UserNewsTimeSurvey.time_saved_answer == "yes",
                        UserNewsTimeSurvey.time_saved_answered_at.is_not(None),
                        UserNewsTimeSurvey.current_answered_at.is_(None),
                    )
                    .order_by(UserNewsTimeSurvey.time_saved_answered_at.asc(), User.id.asc())
                    .limit(effective_limit)
                )
            ).all()
            if only_user_ids:
                normalized_user_ids = {int(user_id) for user_id in only_user_ids if user_id is not None}
                if normalized_user_ids:
                    rows = [(user, survey) for user, survey in rows if int(user.id) in normalized_user_ids]

            checked_total = 0
            sent_total = 0
            skipped_total = 0
            failed_total = 0

            for user, survey in rows:
                checked_total += 1
                if not getattr(user, "telegram_id", None):
                    skipped_total += 1
                    continue
                if dry_run:
                    continue

                locale = get_user_locale(user)
                audit(
                    "news_time_survey.current_redelivery_attempt",
                    user_id=user.id,
                    telegram_id=user.telegram_id,
                )
                sent, send_error = await _send_news_time_current_prompt(
                    telegram_id=int(user.telegram_id),
                    locale=locale,
                )
                if not sent:
                    failed_total += 1
                    audit(
                        "news_time_survey.current_redelivery_failed",
                        user_id=user.id,
                        telegram_id=user.telegram_id,
                        error=send_error,
                    )
                    continue

                sent_at = _utcnow()
                if getattr(survey, "current_asked_at", None) is None:
                    survey.current_asked_at = sent_at
                    audit("news_time_survey.current_asked", telegram_id=user.telegram_id, user_id=user.id)
                else:
                    survey.current_retry_asked_at = sent_at
                    audit("news_time_survey.current_retry_asked", telegram_id=user.telegram_id, user_id=user.id)
                await session.commit()
                sent_total += 1
                audit(
                    "news_time_survey.current_redelivery_success",
                    user_id=user.id,
                    telegram_id=user.telegram_id,
                    sent_at=str(sent_at),
                )

            log.info(
                "news_time_survey.current_redelivery.done",
                dry_run=bool(dry_run),
                checked_total=checked_total,
                sent_total=sent_total,
                skipped_total=skipped_total,
                failed_total=failed_total,
            )
            return {
                "dry_run": bool(dry_run),
                "limit": effective_limit,
                "only_user_ids": [int(user_id) for user_id in only_user_ids] if only_user_ids else None,
                "checked_total": checked_total,
                "sent_total": sent_total,
                "skipped_total": skipped_total,
                "failed_total": failed_total,
            }

        return await _run_with_session(_work)

    return _run_async(_task())


@celery_app.task(
    name="app.tasks.build_and_send_digest",
    bind=True,
    soft_time_limit=DIGEST_TASK_SOFT_TIME_LIMIT,
    time_limit=DIGEST_TASK_TIME_LIMIT,
)
def build_and_send_digest(self, run_id: int):
    async def _task():
        async def _work(session):
            # Idempotency / anti-duplicate-send guard.
            #
            # Previous implementation loaded the run, set status='sending' and
            # then did TG sends. If the worker crashed anywhere between the
            # first send and the final status='sent', Celery would retry and
            # the user would receive the digest again.
            #
            # Claim the run atomically with a conditional UPDATE. We allow
            # 'queued'/'failed' (normal retry path) and a stale 'sending'
            # (stuck worker recovery).
            claim_res = await session.execute(
                text(
                    """
                    UPDATE digest_runs
                    SET status = 'sending',
                        error = NULL,
                        updated_at = now()
                    WHERE id = :run_id
                      AND (
                            status IN ('queued', 'created', 'failed')
                            OR (
                                status = 'sending'
                                AND updated_at < now() - (:stale_minutes * INTERVAL '1 minute')
                            )
                          )
                    RETURNING id
                    """
                ),
                {"run_id": int(run_id), "stale_minutes": int(DIGEST_SENDING_STALE_MINUTES)},
            )
            claimed = claim_res.scalar_one_or_none()
            await session.commit()

            if claimed is None:
                log.info("digest.build_and_send_skipped_already_in_progress_or_done", run_id=run_id)
                return None

            run = await session.get(DigestRun, run_id)
            if not run:
                return None

            try:
                # user
                user = await session.get(User, run.user_id)
                if not user or not user.telegram_id:
                    run.status = "failed"
                    run.error = "user.telegram_id is empty"
                    await session.commit()
                    return None

                # subscriptions -> community_ids
                comm_ids_res = await session.execute(
                    select(UserCommunity.community_id).where(UserCommunity.user_id == run.user_id)
                )
                community_ids = [r[0] for r in comm_ids_res.all()]
                if not community_ids:
                    run.status = "skipped_empty"
                    run.error = None
                    await session.commit()
                    return None

                # community_id -> source
                comm_res = await session.execute(
                    select(Community.id, Community.link).where(Community.id.in_(community_ids))
                )
                comm_map: dict[int, str] = {}
                for cid, clink in comm_res.all():
                    name = ""
                    if clink and "t.me/" in clink:
                        name = "@" + clink.split("t.me/", 1)[1].strip("/").split("?", 1)[0]
                    comm_map[cid] = name or (clink or "Источник")
                    
                # community_id -> base link (для fallback на конкретный пост: https://t.me/<channel>)
                comm_link_map: dict[int, str] = {}
                for cid, clink in comm_res.all():
                    comm_link_map[cid] = (clink or "").rstrip("/")

                # posts
                posts_res = await session.execute(
                    select(Post)
                    .where(
                        Post.community_id.in_(community_ids),
                        Post.timestamp.is_not(None),
                        Post.timestamp >= run.period_start,
                        Post.timestamp < run.period_end,
                    )
                    .order_by(Post.timestamp.desc())
                    .limit(50)
                )
                posts = posts_res.scalars().all()
                if not posts:
                    run.status = "skipped_empty"
                    run.error = None
                    await session.commit()
                    return None

                # Шаг 0: дотягиваем саммари для постов, у которых его нет.
                # Запускаем параллельно, чтобы не замедлять сборку дайджеста.
                posts_need_summary = [
                    p for p in posts
                    if not (p.summary or "").strip()
                    and len((p.content or "").strip()) >= SUMMARY_MIN_CHARS
                ]
                if posts_need_summary:
                    log.info(
                        "digest.prefetch_summaries.start",
                        run_id=run.id,
                        count=len(posts_need_summary),
                    )
                    sem = asyncio.Semaphore(8)

                    async def _fetch_and_save(post):
                        async with sem:
                            try:
                                text_for_summary = (post.content or "").strip()
                                summary_text = await asyncio.wait_for(
                                    summarize(text_for_summary), timeout=30
                                )
                                if summary_text:
                                    post.summary = summary_text
                                    post.summary_created_at = datetime.now(timezone.utc)
                            except Exception as exc:
                                log.warning(
                                    "digest.prefetch_summaries.error",
                                    post_id=post.id,
                                    error=repr(exc),
                                )

                    await asyncio.gather(*[_fetch_and_save(p) for p in posts_need_summary])
                    await session.commit()
                    log.info(
                        "digest.prefetch_summaries.done",
                        run_id=run.id,
                        fetched=sum(1 for p in posts_need_summary if p.summary),
                    )

                # cards
                cards: list[dict] = []
                for p in posts:
                    content_link = p.content_link or {}
                    original_url = content_link.get("url")

                    # Fallback: если url не сохранился, собираем ссылку на пост из community.link + message.id
                    if not original_url:
                        msg_id = content_link.get("id")  # telegram message.id
                        base = comm_link_map.get(p.community_id, "")
                        if base and msg_id:
                            original_url = f"{base}/{int(msg_id)}"

                    source_raw = comm_map.get(p.community_id, "Источник")
                    source = normalize_source_name(source_raw)

                    summary = (p.summary or "").strip()
                    if summary:
                        # LLM _compact всё равно обрежет до 280 на входе модели;
                        # 450 нужно для полного отображения в "Прочем" без обрыва на полуслове.
                        text_ = sanitize_digest_text(summary, max_len=450)
                    else:
                        content = (p.content or "").strip()
                        text_ = sanitize_digest_text(content, max_len=450) if content else ""

                    if not text_ and not original_url:
                        continue

                    tracking_enabled = source_link_tracking_enabled()
                    tracking_url = (
                        build_tracking_link(int(p.id), int(user.telegram_id), source="digest")
                        if tracking_enabled
                        else ""
                    )
                    digest_url = original_url or tracking_url or ""
                    cards.append(
                        {
                            "id": p.id,
                            "source": source,
                            # В дайджесте по клику пользователь должен попадать сразу на исходный пост.
                            # Трекинг-ссылку держим только как запасной fallback, если original_url не удалось собрать.
                            "url": digest_url,
                            "links": [],
                            "text": text_,
                        }
                    )

                if not cards:
                    run.status = "skipped_empty"
                    run.error = None
                    await session.commit()
                    return None

                # ---- LLM grouping -> send_items ----
                send_items: list[dict] = []

                # индекс по всем карточкам
                by_id: dict[int, dict] = {}
                for c in cards:
                    try:
                        by_id[int(c["id"])] = c
                    except Exception:
                        continue

                # Шаг 1: LLM группировка — все карточки в одном вызове.
                # group_digest_stories внутри ограничивает входной срез (40 для thinking, 15 для chat).
                # Единый вызов позволяет LLM видеть весь контекст и не разбивать один сюжет
                # между батчами, как происходило при BATCH=15.
                all_stories: list[dict] = []
                grouped = None
                try:
                    # group_digest_stories has per-stage DeepSeek timeouts and can
                    # return rough chat stories when thinking merge is unavailable.
                    # A shorter outer wait_for used to cancel that fallback path,
                    # producing source-only digests even after chat grouping worked.
                    grouped = await group_digest_stories(cards, max_stories=10)
                except asyncio.TimeoutError:
                    log.warning("digest.grouping.wait_for_timeout", run_id=run.id, cards=len(cards))
                    grouped = None
                except DeepseekRetryableError as e:
                    log.warning(
                        "digest.grouping.retryable_error",
                        run_id=run.id,
                        cards=len(cards),
                        error=str(e),
                        error_repr=repr(e),
                    )
                    grouped = None
                except Exception as e:
                    log.exception(
                        "digest.grouping.unexpected_error",
                        run_id=run.id,
                        cards=len(cards),
                        error_repr=repr(e),
                    )
                    grouped = None

                stories_raw = grouped.get("stories") if isinstance(grouped, dict) else None
                if isinstance(stories_raw, list) and stories_raw:
                    all_stories.extend(stories_raw)

                log.info(
                    "digest.grouping.result",
                    run_id=run.id,
                    grouped_ok=bool(all_stories),
                    stories_count=len(all_stories),
                    cards_total=len(cards),
                )

                used: set[int] = set()

                # Шаг 2: Собираем LLM-сюжеты (только многоисточниковые ≥2)
                llm_stories: list[dict] = []
                
                if all_stories:
                    for st in all_stories:
                        raw_ids = (
                            st.get("post_ids")
                            or st.get("ids")
                            or st.get("postIds")
                            or st.get("posts")
                            or st.get("items")
                            or []
                        )

                        post_ids: list[int] = []
                        for pid in raw_ids:
                            try:
                                pid = int(pid)
                            except Exception:
                                continue
                            if pid in by_id:
                                post_ids.append(pid)

                        post_ids = list(dict.fromkeys(post_ids))
                        if not post_ids:
                            continue

                        # не даём одному посту попасть в несколько сюжетов
                        post_ids = [pid for pid in post_ids if pid not in used]
                        if not post_ids:
                            continue

                        sources: list[str] = []
                        links: list[str] = []
                        link_labels: dict[str, str] = {}
                        for pid in post_ids:
                            c = by_id[pid]
                            raw_s = (c.get("source") or "").strip()
                            s = normalize_source_name(raw_s)
                            u = (c.get("url") or "").strip()
                            if s and s not in sources:
                                sources.append(s)
                            if u and u not in links:
                                links.append(u)
                            if u and s and u not in link_labels:
                                link_labels[u] = s

                        # Добавляем в send_items ТОЛЬКО если источников ≥2
                        if len(sources) >= 2:
                            used.update(post_ids)
                            
                            title = (st.get("title") or "").strip()[:140] or "Сюжет"
                            summary = (st.get("summary") or "").strip()[:900]
                            if not summary:
                                summary = (by_id[post_ids[0]].get("text") or "").strip()[:900]

                            llm_stories.append({
                                "title": title,
                                "source": " / ".join(sources) if sources else "Источник",
                                "text": summary,
                                "url": links[0] if links else None,
                                "links": links,
                                "link_labels": link_labels,
                                "post_ids": post_ids,
                                "sources_count": len(sources),
                            })
                
                # Сортируем по количеству источников (убывание)
                llm_stories.sort(key=lambda x: x.get("sources_count", 0), reverse=True)

                send_items.extend(
                    [
                        {
                            key: value
                            for key, value in story.items()
                            if key != "sources_count"
                        }
                        for story in llm_stories
                    ]
                )

                # Шаг 3: Несгруппированные посты → группировка по каналам
                rest_ids = [pid for pid in by_id.keys() if pid not in used]
                log.info(
                    "digest.coverage",
                    run_id=run.id,
                    cards_total=len(by_id),
                    used_llm=len(used),
                    rest=len(rest_ids),
                )

                if rest_ids:
                    # Группируем по источникам
                    source_posts: dict[str, list[int]] = {}
                    
                    for pid in rest_ids:
                        c = by_id[pid]
                        src = normalize_source_name(c.get("source"))
                        if src:
                            if src not in source_posts:
                                source_posts[src] = []
                            source_posts[src].append(pid)
                    
                    # Разделяем: >3 постов → "Посты канала X", ≤3 → "Прочее"
                    channel_groups = [src for src, pids in source_posts.items() if len(pids) > 3]
                    misc_ids = []
                    
                    for pid in rest_ids:
                        c = by_id[pid]
                        src = normalize_source_name(c.get("source"))
                        if not src or src not in channel_groups:
                            misc_ids.append(pid)
                    
                    log.info(
                        "digest.rest_split",
                        run_id=run.id,
                        rest_total=len(rest_ids),
                        channel_groups=len(channel_groups),
                        misc=len(misc_ids),
                    )
                    
                    # Шаг 4: "Посты канала X" (>3 постов)
                    for src in sorted(channel_groups)[:10]:
                        pids = source_posts.get(src, [])
                        if not pids:
                            continue
                        
                        items_lines: list[str] = []
                        
                        for pid in pids[:15]:
                            c = by_id[pid]
                            u = (c.get("url") or "").strip()
                            t = (c.get("text") or "").strip()
                            
                            if not t:
                                continue
                            
                            # Встроенная ссылка в каждый пост
                            if u:
                                safe_url = html.escape(u)
                                items_lines.append(f'• {html.escape(t)} (<a href="{safe_url}">ссылка</a>)')
                            else:
                                items_lines.append(f'• {html.escape(t)}')
                        
                        if items_lines:
                            send_items.append({
                                "title": f"Посты канала {normalize_source_name(src)}",
                                "source": normalize_source_name(src),
                                "text": "\n\n".join(items_lines),
                                "url": None,
                                "links": [],  # ссылки встроены
                                "post_ids": [int(pid) for pid in pids[:15]],
                            })
                    
                    # Шаг 5: "Прочее" (≤3 постов от канала)
                    if misc_ids:
                        misc_ids = misc_ids[:25]
                        misc_items: list[str] = []
                        
                        for pid in misc_ids:
                            c = by_id[pid]
                            src = normalize_source_name(c.get("source"))
                            u = (c.get("url") or "").strip()
                            t = (c.get("text") or "").strip()
                            
                            if not t:
                                continue
                            
                            # Встроенная ссылка с названием источника
                            if u and src:
                                safe_url = html.escape(u)
                                safe_src = html.escape(src)
                                misc_items.append(f'• {html.escape(t)} (<a href="{safe_url}">{safe_src}</a>)')
                            elif u:
                                safe_url = html.escape(u)
                                misc_items.append(f'• {html.escape(t)} (<a href="{safe_url}">ссылка</a>)')
                            else:
                                misc_items.append(f'• {html.escape(t)}')
                        
                        if misc_items:
                            send_items.append({
                                "title": "Прочее",
                                "source": "Разные источники",
                                "text": "\n\n".join(misc_items),
                                "url": None,
                                "links": [],
                                "post_ids": [int(pid) for pid in misc_ids],
                            })

                # если вообще ничего не получилось — fallback на карточки
                if not send_items:

                    log.warning("digest.grouping.fallback_cards", run_id=run.id, cards_total=len(cards))
                    send_items = [
                        {
                            "source": c["source"],
                            "url": c.get("url"),
                            "text": c["text"],
                            "post_ids": [int(c["id"])],
                        }
                        for c in cards
                    ]
                    
                # Нормализуем source и чуть чистим текст от уродливых "Источники: @c"
                for item in send_items:
                    # 1) нормализуем source
                    item["source"] = normalize_source_name(item.get("source"))

                    # 2) грубый, но безопасный хак: заменяем "@c (" внутри текста на "Канал ("
                    txt = item.get("text") or ""
                    if "@c (" in txt:
                        item["text"] = txt.replace("@c (", "Канал (")    

                log.info("digest.send_items", run_id=run.id, send_items=len(send_items), cards_total=len(cards))

                # Note: status is already 'sending' from the atomic claim above.
                locale = get_user_locale(user)
                delivered_post_ids: list[int] = []
                delivered_post_ids_seen: set[int] = set()
                for item in send_items:
                    for raw_pid in item.get("post_ids") or []:
                        try:
                            pid = int(raw_pid)
                        except Exception:
                            continue
                        if pid in delivered_post_ids_seen:
                            continue
                        delivered_post_ids_seen.add(pid)
                        delivered_post_ids.append(pid)

                # Заголовок дня: одна фраза о главном событии из LLM-сюжетов
                digest_headline: str | None = None
                if llm_stories:
                    headline_input = [
                        {
                            "title": s["title"],
                            "summary": (s.get("text") or "")[:200],
                            "sources_count": int(s.get("sources_count") or 0),
                        }
                        for s in llm_stories[:8]
                    ]
                    try:
                        digest_headline = await asyncio.wait_for(
                            generate_digest_headline(headline_input),
                            timeout=30,
                        )
                    except Exception as e:
                        log.warning("digest.headline.failed", run_id=run.id, error=repr(e))
                        digest_headline = None

                pages = render_digest_pages(
                    run.period_start, run.period_end, send_items, locale=locale, headline=digest_headline
                )
                run.pages = pages
                await session.commit()

                route = await resolve_delivery_route(
                    session,
                    user_id=int(user.id),
                    delivery_kind="digest",
                )
                first_message_id = None
                send_text = pages[0] if pages else t(locale, "digest_empty_short")
                try:
                    first_message_id = await send_digest_first_page(
                        telegram_id=user.telegram_id,
                        run_id=run.id,
                        text=send_text,
                        page=0,
                        total=len(pages) if pages else 1,
                        locale=locale,
                        bot_token_override=route.token,
                    )
                except Exception:
                    if route.target == "plus" and MAIN_BOT_TOKEN:
                        first_message_id = await send_digest_first_page(
                            telegram_id=user.telegram_id,
                            run_id=run.id,
                            text=send_text,
                            page=0,
                            total=len(pages) if pages else 1,
                            locale=locale,
                            bot_token_override=MAIN_BOT_TOKEN,
                        )
                    else:
                        raise
                sent = 1
                from app.audit import audit
                for delivered_post_id in delivered_post_ids:
                    audit(
                        "send.ok",
                        post_id=delivered_post_id,
                        telegram_id=user.telegram_id,
                        user_id=user.id,
                        tg_username=getattr(user, "username", None),
                        message_id=first_message_id,
                        delivery_result="digest_message",
                        delivery_target=route.target,
                        delivery_route_reason=route.reason,
                    )

                run.status = "sent"
                run.sent_messages = int(sent or 0)
                run.error = None
                await session.commit()
                if route.target == "plus":
                    await _send_digest_plus_delivery_notice(int(user.telegram_id))
                return sent

            except Exception as e:
                # фиксируем failed
                try:
                    run2 = await session.get(DigestRun, run_id)
                    if run2 and run2.status not in {"sent", "skipped_empty"}:
                        run2.status = "failed"
                        run2.error = str(e)
                        await session.commit()
                except Exception:
                    pass
                raise

        return await _run_with_session(_work)

    try:
        return _run_async(_task())
    except SQLAlchemyError as e:
        raise self.retry(
            exc=e,
            countdown=min(60, 2 ** self.request.retries),
            max_retries=DB_MAX_RETRIES,
        )

@celery_app.task(name="app.tasks.preprocess_and_embed_task", bind=True)
def preprocess_and_embed_task(self, post_id: int):
    async def _task():
        # TODO[showcase]: stub after removing payments/promo/userbot/subscribe modules from public version
        # from app.userbot import preprocess_and_embed
        await preprocess_and_embed(post_id)

    try:
        return _run_async(_task(), timeout=120)
    except SQLAlchemyError as e:
        raise self.retry(
            exc=e,
            countdown=min(60, 2 ** self.request.retries),
            max_retries=DB_MAX_RETRIES,
        )


@celery_app.task(name="app.tasks.digest_scheduler_tick")
def digest_scheduler_tick():
    from app.digest_scheduler import scheduler_tick
    return _run_async(scheduler_tick())


@celery_app.task(
    name="app.tasks.handle_subscription_request_task",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": int(os.getenv("SUBSCRIBE_MAX_RETRIES", "10"))},
    queue=os.getenv("SUBSCRIBE_QUEUE", "subscribe_queue"),
)
def handle_subscription_request_task(self, request_id: int):
    async def _task():
        async for session in get_session():
            await ensure_subscription_request_operation(session, int(request_id))
            await session.commit()
            break

    return _run_async(_task())
