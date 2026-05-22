import os
import asyncio
import contextlib
import html
import re
import difflib
from uuid import uuid4
from dataclasses import dataclass
from functools import partial
from urllib.parse import urlparse

from aiogram.types import CallbackQuery
from app.notifications import build_digest_pager_kb, build_tracking_link, source_link_tracking_enabled

import structlog
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from aiogram import Bot, Dispatcher, types, F
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
try:
    from aiogram import BaseMiddleware
except ImportError:  # tests use lightweight aiogram stubs
    class BaseMiddleware:
        pass
try:
    from aiogram.exceptions import TelegramAPIError
except ImportError:  # tests use lightweight aiogram stubs
    TelegramAPIError = Exception
from aiogram.filters import Command
from sqlalchemy import select, update, or_, case, func, text
from app.celery_app import celery_app
from app.ai.deepseek import (
    arbitrate_storyline_follow_target,
    arbitrate_storyline_timeline,
    extract_storyline_anchor_profile,
    DeepseekAuthError,
    DeepseekRetryableError,
    generate_storyline_title,
    resolve_bot_router_intent,
    summarize_related_storyline_branches,
)
from app.audit import audit

from app.models import (
    User,
    Community,
    UserCommunity,
    Reaction,
    Post,
    PostInteraction,
    UserKeywordStat,
    SubscriptionRequest,
    get_session,
    UserDigestSettings, 
    UserDigestSlot,
    DigestRun,
    UserDeliveryPreference,
    UserStorylineFollow,
    StorylineUpdateEvent,
    UserNewsTimeSurvey,
    UserCsiSurvey,
    DispatchDelivery,
    SupportTicket,
)
from app.delivery_prefs import MAIN_BOT_TOKEN, get_or_create_delivery_prefs
from app.csi_survey import (
    CSI_NEWS_TIME_COOLDOWN,
    CSI_SCORE_KEYS,
    build_csi_eligibility,
    is_csi_pending,
    parse_csi_score,
    should_abandon_csi,
    should_ask_csi,
)
from app.news_time_survey import (
    NEWS_TIME_FOLLOWUP_RETRY_DELAY,
    NEWS_TIME_OPTION_KEYS,
    followup_due_at,
    parse_news_time_answer,
    parse_time_saved_answer,
    should_abandon_followup,
    should_ask_followup,
    utcnow,
)
from app.storyline_mvp import (
    collect_related_storyline_branches,
    collect_storyline_follow_candidates,
    get_macro_topic_timeline_steps,
    get_storyline_cluster_timeline_steps,
    get_storyline_context_for_post,
    get_storyline_context_for_storyline_id,
    search_storyline_structural_candidates,
    search_storyline_candidates,
    search_storyline_recovery_candidates,
)
# TODO[showcase]: stub after removing payments/promo/userbot/subscribe modules from public version
# from app.billing import recalculate_user_billing_state
from app.instruction_filters import (
    InstructionPromptError,
    disable_community_instruction_rule,
    disable_global_instruction_rule,
    get_instruction_rules_for_user,
    get_rule_bound_communities,
    get_subscribed_instruction_targets,
    upsert_community_instruction_rule,
    upsert_global_instruction_rule,
    user_has_instruction_filter_access,
)
# from app.payments.service import (
#     PAYMENT_METHOD_TELEGRAM_STARS,
#     PremiumAlreadyPrepaidError,
#     create_payment_order_for_user,
#     get_current_payment_snapshot,
#     get_telegram_stars_invoice_data,
#     process_telegram_stars_successful_payment,
#     sync_pending_payment_for_user,
#     validate_telegram_stars_pre_checkout,
# )
# from app.promo import redeem_promo_code
from app.reco_runtime import reset_user_recommendations, update_user_model_from_feedback
# from app.userbot_slots import choose_userbot_slot_for_new_group, USERBOT_SLOT_PRIMARY, USERBOT_SLOT_SECONDARY
# TODO[showcase]: stub after removing payments/promo/userbot/subscribe modules from public version
# from app.subscription_operations import (
#     ensure_community_aliases,
#     ensure_resolve_join_operation,
#     ensure_user_community,
#     find_existing_active_community,
# )
from app.storytracking_rollout import (
    storytracking_allowed_for_user,
    storytracking_rollout_restricted,
)
from kombu import Connection, Exchange, Producer, Queue
from app.i18n import (
    all_button_variants,
    button_text,
    detect_button_locale,
    get_user_locale,
    locale_from_telegram_language_code,
    normalize_button_label,
    t,
)
from app.nlu_router import (
    ExecutorResult,
    NLUResult,
    NLUValidationError,
    PendingActionPayload,
    build_pending_action_payload,
    validate_nlu_result,
)
from app.pending_actions import get_pending_action_store, pending_action_ttl_seconds
from app.bot_assistant_monitoring import (
    create_request as create_assistant_request,
    mark_answered as mark_assistant_answered,
    mark_applied as mark_assistant_applied,
    mark_cancelled as mark_assistant_cancelled,
    mark_clarification as mark_assistant_clarification,
    mark_confirmed as mark_assistant_confirmed,
    mark_expired as mark_assistant_expired,
    mark_failed as mark_assistant_failed,
    mark_proposed as mark_assistant_proposed,
    mark_understood as mark_assistant_understood,
    safe_mark_failed as safe_mark_assistant_failed,
)

log = structlog.get_logger()

STORYLINE_TIMELINE_PAYLOAD_TIMEOUT = float(os.getenv("STORYLINE_TIMELINE_PAYLOAD_TIMEOUT", "90"))
STORYLINE_TIMELINE_ARBITER_TIMEOUT = float(os.getenv("STORYLINE_TIMELINE_ARBITER_TIMEOUT", "60"))
STORYLINE_FOLLOW_TARGET_TIMEOUT = float(os.getenv("STORYLINE_FOLLOW_TARGET_TIMEOUT", "12"))
STORYLINE_ANCHOR_PROFILE_TIMEOUT = float(os.getenv("STORYLINE_ANCHOR_PROFILE_TIMEOUT", "10"))
STORYLINE_TIMELINE_LOOKBACK_DAYS = int(os.getenv("STORYLINE_TIMELINE_LOOKBACK_DAYS", "90"))
STORYLINE_TIMELINE_HEAVY_SOURCE_THRESHOLD = int(os.getenv("STORYLINE_TIMELINE_HEAVY_SOURCE_THRESHOLD", "25"))
STORYLINE_TIMELINE_HEAVY_SOURCE_LIMIT = int(os.getenv("STORYLINE_TIMELINE_HEAVY_SOURCE_LIMIT", "25"))
STORYLINE_TIMELINE_FAST_GROUP_THRESHOLD = int(os.getenv("STORYLINE_TIMELINE_FAST_GROUP_THRESHOLD", "35"))
STORYLINE_TIMELINE_FAST_CARD_THRESHOLD = int(os.getenv("STORYLINE_TIMELINE_FAST_CARD_THRESHOLD", "45"))
STORYLINE_TIMELINE_FAST_ARB_GROUP_LIMIT = int(os.getenv("STORYLINE_TIMELINE_FAST_ARB_GROUP_LIMIT", "14"))
STORYLINE_TIMELINE_FAST_SOURCE_EXPANSION_LIMIT = int(os.getenv("STORYLINE_TIMELINE_FAST_SOURCE_EXPANSION_LIMIT", "8"))
STORYLINE_TIMELINE_FAST_SOURCE_WINDOW_HOURS = int(os.getenv("STORYLINE_TIMELINE_FAST_SOURCE_WINDOW_HOURS", "6"))
STORYLINE_TIMELINE_FAST_ARB_TIMEOUT = float(os.getenv("STORYLINE_TIMELINE_FAST_ARB_TIMEOUT", "25"))
STORYLINE_SIMILAR_ARBITER_TIMEOUT = float(os.getenv("STORYLINE_SIMILAR_ARBITER_TIMEOUT", "18"))
STORYLINE_FOLLOW_LIMIT = max(1, int(os.getenv("STORYLINE_FOLLOW_LIMIT", "5")))
ENABLE_STORYTRACKING = os.getenv("ENABLE_STORYTRACKING", "true").lower() == "true"
BOT_ROLE = (os.getenv("BOT_ROLE", "main") or "main").strip().lower()
BOT_PLUS_USERNAME = (os.getenv("BOT_PLUS_USERNAME", "") or "").strip().lstrip("@")

bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
if not bot_token:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

bot = Bot(token=bot_token)
dp = Dispatcher()

SERVICE_MESSAGE_CLEANUP_ENABLED = os.getenv("SERVICE_MESSAGE_CLEANUP_ENABLED", "true").lower() == "true"
SERVICE_MESSAGE_TTL_SECONDS = max(0, int(os.getenv("SERVICE_MESSAGE_TTL_SECONDS", "900")))
USER_SERVICE_MESSAGE_CLEANUP_ENABLED = os.getenv("USER_SERVICE_MESSAGE_CLEANUP_ENABLED", "true").lower() == "true"
USER_SERVICE_MESSAGE_TTL_SECONDS = max(
    0, int(os.getenv("USER_SERVICE_MESSAGE_TTL_SECONDS", str(SERVICE_MESSAGE_TTL_SECONDS)))
)
_ORIGINAL_MESSAGE_ANSWER = getattr(types.Message, "answer", None)
_SERVICE_CLEANUP_TASKS: set[asyncio.Task] = set()
_SKIP_SERVICE_CLEANUP_KWARG = "skip_service_cleanup"


def _build_bot_commands() -> list[types.BotCommand]:
    if BOT_ROLE == "plus":
        return [
            types.BotCommand(command="help", description="Инструкция"),
        ]
    commands = [
        types.BotCommand(command="settings", description="Настройки"),
        types.BotCommand(command="subscriptions", description="Подписки"),
        types.BotCommand(command="billing", description="Тариф и оплата"),
        types.BotCommand(command="forwarding", description="Пересылка новостей"),
        types.BotCommand(command="digest", description="Дайджест"),
        types.BotCommand(command="language", description="Язык"),
        types.BotCommand(command="support", description="Поддержка"),
        types.BotCommand(command="help", description="Инструкция"),
    ]
    if ENABLE_STORYTRACKING and not storytracking_rollout_restricted():
        commands.insert(5, types.BotCommand(command="storylines", description="Отслеживание сюжета"))
    return commands


async def configure_bot_menu(bot_instance: Bot = bot) -> None:
    try:
        await bot_instance.set_my_commands(_build_bot_commands())
        await bot_instance.set_chat_menu_button(menu_button=types.MenuButtonCommands())
        log.info("bot_menu.configured")
    except TelegramAPIError as exc:
        log.warning("bot_menu.configure_failed", error=repr(exc))


async def _delete_service_message_later(*, chat_id: int, message_id: int, delay_seconds: int) -> None:
    if delay_seconds <= 0:
        return
    await asyncio.sleep(delay_seconds)
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except (TelegramBadRequest, TelegramForbiddenError) as exc:
        log.debug(
            "service_message_cleanup.skipped",
            chat_id=chat_id,
            message_id=message_id,
            reason=str(exc),
        )
    except TelegramAPIError as exc:
        log.warning(
            "service_message_cleanup.failed",
            chat_id=chat_id,
            message_id=message_id,
            error=str(exc),
        )
    except Exception as exc:
        log.warning(
            "service_message_cleanup.failed",
            chat_id=chat_id,
            message_id=message_id,
            error=str(exc),
        )


def _track_cleanup_task(task: asyncio.Task) -> None:
    _SERVICE_CLEANUP_TASKS.add(task)
    task.add_done_callback(_SERVICE_CLEANUP_TASKS.discard)


async def _try_delete_message(message: types.Message | None, *, reason: str) -> None:
    if message is None:
        return
    delete = getattr(message, "delete", None)
    if not callable(delete):
        return
    try:
        await delete()
    except (TelegramBadRequest, TelegramForbiddenError) as exc:
        log.debug("message_delete.skipped", reason=reason, error=str(exc))
    except TelegramAPIError as exc:
        log.warning("message_delete.failed", reason=reason, error=str(exc))
    except Exception as exc:
        log.warning("message_delete.failed", reason=reason, error=str(exc))


async def _answer_with_service_cleanup(self: types.Message, *args, **kwargs):
    if _ORIGINAL_MESSAGE_ANSWER is None:
        raise RuntimeError("Message.answer is unavailable")
    skip_cleanup = bool(kwargs.pop(_SKIP_SERVICE_CLEANUP_KWARG, False))
    sent_message = await _ORIGINAL_MESSAGE_ANSWER(self, *args, **kwargs)
    if skip_cleanup:
        return sent_message
    if not SERVICE_MESSAGE_CLEANUP_ENABLED:
        return sent_message
    if SERVICE_MESSAGE_TTL_SECONDS <= 0:
        return sent_message
    if sent_message is None:
        return sent_message
    chat = getattr(sent_message, "chat", None)
    chat_id = getattr(chat, "id", None)
    message_id = getattr(sent_message, "message_id", None)
    if chat_id is None or message_id is None:
        return sent_message
    task = asyncio.create_task(
        _delete_service_message_later(
            chat_id=chat_id,
            message_id=message_id,
            delay_seconds=SERVICE_MESSAGE_TTL_SECONDS,
        )
    )
    _track_cleanup_task(task)
    return sent_message


if _ORIGINAL_MESSAGE_ANSWER is not None:
    types.Message.answer = _answer_with_service_cleanup

DIGEST_SETUP_WAIT_OFFSET: set[int] = set()
DIGEST_SETUP_WAIT_TIME: set[int] = set()
DIGEST_TIME_MENU_USERS: set[int] = set()
BILLING_MENU_USERS: set[int] = set()
DIGEST_MENU_USERS: set[int] = set()
PLUS_DELIVERY_MENU_USERS: set[int] = set()
LANGUAGE_MENU_USERS: set[int] = set()
SUPPORT_REQUEST_WAIT_USERS: set[int] = set()
FORWARDING_MENU_USERS: set[int] = set()
RECO_RESET_CONFIRM_USERS: set[int] = set()
CSI_SCORE_WAIT_USERS: set[int] = set()
NEWS_TIME_BASELINE_WAIT_USERS: set[int] = set()
INSTRUCTION_MENU_USERS: set[int] = set()
INSTRUCTION_GLOBAL_WAIT_USERS: set[int] = set()
INSTRUCTION_GROUP_SELECT_USERS: set[int] = set()
INSTRUCTION_GROUP_PROMPT_WAIT_USERS: set[int] = set()
INSTRUCTION_GROUP_DISABLE_SELECT_USERS: set[int] = set()
INSTRUCTION_GROUP_CHOICES: dict[int, list[int]] = {}
INSTRUCTION_GROUP_DISABLE_CHOICES: dict[int, list[int]] = {}
INSTRUCTION_GROUP_PROMPT_TARGETS: dict[int, int] = {}
ADMIN_PANEL_USERS: set[int] = set()
STORYLINE_MENU_USERS: set[int] = set()
STORYLINE_REMOVE_WAIT_USERS: set[int] = set()
STORYLINE_REMOVE_CHOICES: dict[int, list[int]] = {}
PROMO_CODE_WAIT: set[int] = set()
CRYPTO_PAYMENT_MENU_USERS: set[int] = set()
PAYMENT_TERM_MENU_USERS: set[int] = set()
PAYMENT_METHOD_SELECTIONS: dict[int, str] = {}
INSTRUCTION_GROUP_PAGE_SIZE = int(os.getenv("INSTRUCTION_GROUP_PAGE_SIZE", "8"))
SUBSCRIPTION_ACTIVE_STATUSES = ("pending", "queued", "retrying", "processing", "joining", "joined", "syncing")
SUBSCRIPTION_STALE_MINUTES = int(os.getenv("SUBSCRIPTION_STALE_MINUTES", "30"))
PREMIUM_SUPPORT_EMAIL = os.getenv("BILLING_SUPPORT_EMAIL", "volgaoavel@gmail.com")
ADMIN_USERNAMES = {
    value.strip().lstrip("@").casefold()
    for value in os.getenv("ADMIN_USERNAMES", "vendor62x,vendior62x,Medoedisrussia").split(",")
    if value.strip()
}

SUPPORT_ALERT_BOT_TOKEN = (os.getenv("SUPPORT_ALERT_BOT_TOKEN") or os.getenv("ALERT_BOT_TOKEN") or "").strip()
SUPPORT_ALERT_CHAT_ID = (os.getenv("SUPPORT_ALERT_CHAT_ID") or os.getenv("ALERT_CHAT_ID") or "").strip()
# Keep support tickets separate from monitor logs thread.
# Default is the dedicated tickets topic; can be overridden via env.
SUPPORT_ALERT_THREAD_ID = (
    os.getenv("SUPPORT_ALERT_THREAD_ID")
    or os.getenv("ALERT_TICKETS_THREAD_ID")
    or "21606"
).strip()
SUPPORT_RETRY_POLL_SECONDS = max(5, int(os.getenv("SUPPORT_RETRY_POLL_SECONDS", "20")))
SUPPORT_RETRY_BATCH_SIZE = max(1, int(os.getenv("SUPPORT_RETRY_BATCH_SIZE", "30")))
SUPPORT_RETRY_BACKOFF_SECONDS = (30, 120, 300, 900, 1800, 3600)

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")
TELETHON_QUEUE = os.getenv("TELETHON_QUEUE", "telethon_queue")
SUBSCRIBE_QUEUE = os.getenv("SUBSCRIBE_QUEUE", "subscription_queue")
SUBSCRIBE_QUEUE_PRIMARY = os.getenv("SUBSCRIBE_QUEUE_PRIMARY", "subscribe_queue")
SUBSCRIBE_QUEUE_SECONDARY = os.getenv("SUBSCRIBE_QUEUE_SECONDARY", "subscribe_queue_secondary")
SUBSCRIBE_EXCHANGE = os.getenv("SUBSCRIBE_EXCHANGE", "newshub.subscriptions")

TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
OFFSET_RE = re.compile(r"^(?:UTC)?\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)
DIGEST_MSK_OFFSET_MINUTES = 3 * 60
DIGEST_MSK_PRESET_TIMES = (
    "06:00",
    "08:00",
    "10:00",
    "12:00",
    "14:00",
    "16:00",
    "18:00",
    "20:00",
    "22:00",
)

CITY_TO_TZ = {
    "москва": "Europe/Moscow",
    "moscow": "Europe/Moscow",
    "moskva": "Europe/Moscow",
    "санктпетербург": "Europe/Moscow",
    "питер": "Europe/Moscow",
    "spb": "Europe/Moscow",
    "екатеринбург": "Asia/Yekaterinburg",
    "новосибирск": "Asia/Novosibirsk",
    "омск": "Asia/Omsk",
    "красноярск": "Asia/Krasnoyarsk",
    "иркутск": "Asia/Irkutsk",
    "якутск": "Asia/Yakutsk",
    "владивосток": "Asia/Vladivostok",
    "магадан": "Asia/Magadan",
    "петропавловсккамчатский": "Asia/Kamchatka",
    "калининград": "Europe/Kaliningrad",
    "минск": "Europe/Minsk",
    "киев": "Europe/Kyiv",
    "алматы": "Asia/Almaty",
    "астана": "Asia/Almaty",
    "ташкент": "Asia/Tashkent",
    "бишкек": "Asia/Bishkek",
    "ереван": "Asia/Yerevan",
    "баку": "Asia/Baku",
    "тбилиси": "Asia/Tbilisi",
    "дубай": "Asia/Dubai",
    "лондон": "Europe/London",
    "берлин": "Europe/Berlin",
    "париж": "Europe/Paris",
    "рим": "Europe/Rome",
    "мадрид": "Europe/Madrid",
    "ньюйорк": "America/New_York",
    "newyork": "America/New_York",
    "чикаго": "America/Chicago",
    "лосанджелес": "America/Los_Angeles",
    "токио": "Asia/Tokyo",
    "пекин": "Asia/Shanghai",
    "шанхай": "Asia/Shanghai",
    "дели": "Asia/Kolkata",
    "мумбаи": "Asia/Kolkata",
}


def _keyboard(*rows: list[str]) -> types.ReplyKeyboardMarkup:
    return types.ReplyKeyboardMarkup(
        keyboard=[[types.KeyboardButton(text=text) for text in row] for row in rows],
        resize_keyboard=True,
    )


def get_main_keyboard(locale: str, is_admin: bool = False) -> types.ReplyKeyboardMarkup:
    if BOT_ROLE == "plus":
        return _keyboard([button_text("help", locale, icon="📘")])
    rows: list[list[str]] = [
        [button_text("settings", locale, icon="⚙️")],
        [button_text("help", locale, icon="📘")],
        [button_text("support_request", locale, icon="🆘")],
    ]
    if is_admin:
        rows.append([button_text("admin_panel", locale, icon="🛠️")])
    return _keyboard(*rows)


def get_settings_keyboard(locale: str) -> types.ReplyKeyboardMarkup:
    rows: list[list[str]] = [
        [button_text("subscriptions", locale, icon="📡")],
        [button_text("billing", locale, icon="💳")],
        [button_text("forwarding", locale, icon="📰")],
        [button_text("digest", locale, icon="🗞️")],
        [button_text("language_menu", locale, icon="🌐")],
        [button_text("back", locale, icon="◀️")],
    ]
    if ENABLE_STORYTRACKING:
        rows.insert(4, [button_text("storyline_tracking", locale, icon="🧭")])
        rows.insert(5, [button_text("plus_delivery_menu", locale, icon="🧩")])
    return _keyboard(*rows)


def get_admin_keyboard(locale: str, debug_enabled: bool) -> types.ReplyKeyboardMarkup:
    return _keyboard(
        [button_text("storyline_debug_disable" if debug_enabled else "storyline_debug_enable", locale, icon="🧪")],
        [button_text("back", locale, icon="◀️")],
    )


def get_storyline_tracking_keyboard(locale: str) -> types.ReplyKeyboardMarkup:
    return _keyboard(
        [button_text("storyline_remove", locale, icon="🧭")],
        [button_text("back", locale, icon="◀️")],
    )


def get_plus_delivery_keyboard(
    locale: str,
    *,
    digest_to_plus_bot: bool,
    storyline_to_plus_bot: bool,
) -> types.ReplyKeyboardMarkup:
    return _keyboard(
        [button_text("plus_delivery_digest_off" if digest_to_plus_bot else "plus_delivery_digest_on", locale)],
        [button_text("plus_delivery_storyline_off" if storyline_to_plus_bot else "plus_delivery_storyline_on", locale)],
        [button_text("back", locale, icon="◀️")],
    )


def get_billing_keyboard(locale: str) -> types.ReplyKeyboardMarkup:
    return _keyboard(
        [button_text("pay_tbank", locale, icon="🏦")],
        [button_text("pay_crypto", locale, icon="💵")],
        [button_text("pay_stars", locale, icon="⭐")],
        [button_text("enter_promo", locale, icon="🎁")],
        [button_text("refresh_payment", locale, icon="🔄")],
        [button_text("back", locale, icon="◀️")],
    )


def get_payment_term_keyboard(locale: str) -> types.ReplyKeyboardMarkup:
    return _keyboard(
        [button_text("pay_term_30d", locale, icon="🗓️")],
        [button_text("pay_term_90d", locale, icon="🗓️")],
        [button_text("pay_term_180d", locale, icon="🗓️")],
        [button_text("pay_term_365d", locale, icon="🗓️")],
        [button_text("back", locale, icon="◀️")],
    )


def get_forwarding_keyboard(
    locale: str,
    *,
    live_enabled: bool,
    summary_enabled: bool,
    ai_filter_enabled: bool,
    current_filter: str,
) -> types.ReplyKeyboardMarkup:
    filter_key_to_mode = (
        ("filter_all", "all"),
        ("filter_not_interesting", "not_interesting"),
        ("filter_only_fire", "only_fire"),
    )
    rows: list[list[str]] = [
        [button_text("toggle_forwarding", locale, icon="📡", state=live_enabled)],
        [button_text("toggle_summary", locale, icon="📝", state=summary_enabled)],
        [button_text("instruction_filter", locale, icon="🤖")],
    ]
    for key, value in filter_key_to_mode:
        selected = current_filter == value
        rows.append(
            [
                button_text(
                    key,
                    locale,
                    icon="🎯" if selected else "▫️",
                )
            ]
        )
    rows.append([button_text("reco_reset_start", locale, icon="🔄")])
    rows.append([button_text("back", locale, icon="◀️")])
    return _keyboard(*rows)


def get_reco_reset_confirm_keyboard(locale: str) -> types.ReplyKeyboardMarkup:
    return _keyboard(
        [button_text("reco_reset_confirm", locale, icon="✅")],
        [button_text("reco_reset_cancel", locale, icon="↩️")],
    )


def get_news_time_survey_keyboard(locale: str) -> types.InlineKeyboardMarkup:
    return types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(text=button_text("news_time_option_under_hour", locale), callback_data="news_time:opt:news_time_option_under_hour"),
                types.InlineKeyboardButton(text=button_text("news_time_option_1h", locale), callback_data="news_time:opt:news_time_option_1h"),
                types.InlineKeyboardButton(text=button_text("news_time_option_2h", locale), callback_data="news_time:opt:news_time_option_2h"),
            ],
            [
                types.InlineKeyboardButton(text=button_text("news_time_option_3h", locale), callback_data="news_time:opt:news_time_option_3h"),
                types.InlineKeyboardButton(text=button_text("news_time_option_4h", locale), callback_data="news_time:opt:news_time_option_4h"),
                types.InlineKeyboardButton(text=button_text("news_time_option_5h", locale), callback_data="news_time:opt:news_time_option_5h"),
            ],
        ]
    )



def get_news_time_saved_keyboard(locale: str) -> types.InlineKeyboardMarkup:
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
            ],
        ]
    )


def get_csi_score_keyboard(locale: str) -> types.InlineKeyboardMarkup:
    return types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(text=button_text("csi_score_1", locale), callback_data="csi:score:1"),
                types.InlineKeyboardButton(text=button_text("csi_score_2", locale), callback_data="csi:score:2"),
                types.InlineKeyboardButton(text=button_text("csi_score_3", locale), callback_data="csi:score:3"),
                types.InlineKeyboardButton(text=button_text("csi_score_4", locale), callback_data="csi:score:4"),
                types.InlineKeyboardButton(text=button_text("csi_score_5", locale), callback_data="csi:score:5"),
            ],
            [
                types.InlineKeyboardButton(text=button_text("csi_score_6", locale), callback_data="csi:score:6"),
                types.InlineKeyboardButton(text=button_text("csi_score_7", locale), callback_data="csi:score:7"),
                types.InlineKeyboardButton(text=button_text("csi_score_8", locale), callback_data="csi:score:8"),
                types.InlineKeyboardButton(text=button_text("csi_score_9", locale), callback_data="csi:score:9"),
                types.InlineKeyboardButton(text=button_text("csi_score_10", locale), callback_data="csi:score:10"),
            ],
        ]
    )


def get_filter_clarification_keyboard(locale: str) -> types.ReplyKeyboardMarkup:
    return _keyboard(
        [button_text("filter_all", locale, icon="🎯")],
        [button_text("filter_not_interesting", locale, icon="🎯")],
        [button_text("filter_only_fire", locale, icon="🎯")],
        [button_text("filter_digest_only", locale, icon="🎯")],
        [button_text("back", locale, icon="◀️")],
    )


def get_importance_filter_clarification_keyboard(locale: str) -> types.ReplyKeyboardMarkup:
    return _keyboard(
        [button_text("filter_only_fire", locale, icon="🎯")],
        [button_text("instruction_filter", locale, icon="🤖")],
        [button_text("back", locale, icon="◀️")],
    )


def get_instruction_filter_keyboard(locale: str) -> types.ReplyKeyboardMarkup:
    return _keyboard(
        [button_text("instruction_set_global", locale, icon="🌍")],
        [button_text("instruction_set_group", locale, icon="🧩")],
        [button_text("instruction_show_rules", locale, icon="📋")],
        [button_text("instruction_disable_global", locale, icon="⛔")],
        [button_text("instruction_disable_group", locale, icon="🧹")],
        [button_text("back", locale, icon="◀️")],
    )


def _instruction_group_callback(action: str, mode: str, value: int) -> str:
    return f"instrgrp:{action}:{mode}:{value}"


def _instruction_group_button_label(item: Community, *, limit: int = 32) -> str:
    label = str(getattr(item, "link", None) or getattr(item, "name", None) or f"community:{item.id}").strip()
    if len(label) <= limit:
        return label
    return label[: limit - 1].rstrip() + "…"


def _build_instruction_group_picker_keyboard(
    items: list[Community],
    *,
    mode: str,
    page: int = 0,
    locale: str = "ru",
) -> types.InlineKeyboardMarkup:
    page_size = max(1, INSTRUCTION_GROUP_PAGE_SIZE)
    total_pages = max(1, (len(items) + page_size - 1) // page_size)
    current_page = max(0, min(page, total_pages - 1))
    start = current_page * page_size
    end = start + page_size
    page_items = items[start:end]

    rows: list[list[types.InlineKeyboardButton]] = []
    for item in page_items:
        rows.append(
            [
                types.InlineKeyboardButton(
                    text=_instruction_group_button_label(item),
                    callback_data=_instruction_group_callback("pick", mode, int(item.id)),
                )
            ]
        )

    if total_pages > 1:
        nav_row: list[types.InlineKeyboardButton] = []
        if current_page > 0:
            nav_row.append(
                types.InlineKeyboardButton(
                    text=t(locale, "instruction_picker_prev"),
                    callback_data=_instruction_group_callback("page", mode, current_page - 1),
                )
            )
        nav_row.append(
            types.InlineKeyboardButton(
                text=f"{current_page + 1}/{total_pages}",
                callback_data="noop",
            )
        )
        if current_page < total_pages - 1:
            nav_row.append(
                types.InlineKeyboardButton(
                    text=t(locale, "instruction_picker_next"),
                    callback_data=_instruction_group_callback("page", mode, current_page + 1),
                )
            )
        rows.append(nav_row)

    rows.append(
        [
            types.InlineKeyboardButton(
                text=t(locale, "back"),
                callback_data=_instruction_group_callback("back", mode, 0),
            )
        ]
    )
    return types.InlineKeyboardMarkup(inline_keyboard=rows)


def get_digest_keyboard(locale: str, *, enabled: bool = False, time_label: str | None = None) -> types.ReplyKeyboardMarkup:
    toggle_key = "digest_toggle_off" if enabled else "digest_toggle_on"
    digest_toggle_label = f"🗞️ {button_text('digest', locale)}: {button_text(toggle_key, locale).lower()}"
    return _keyboard(
        [digest_toggle_label],
        [button_text("digest_setup_time", locale, icon="⏰")],
        [button_text("back", locale, icon="◀️")],
    )


def get_digest_time_keyboard(locale: str) -> types.ReplyKeyboardMarkup:
    return _keyboard(
        [button_text("digest_daily", locale, icon="⏰")],
        [button_text("digest_send_now", locale, icon="🚀")],
        [button_text("back", locale, icon="◀️")],
    )


def get_digest_msk_presets_keyboard() -> types.InlineKeyboardMarkup:
    buttons = [
        types.InlineKeyboardButton(text=value, callback_data=f"daily_msk:{value}")
        for value in DIGEST_MSK_PRESET_TIMES
    ]
    rows = [buttons[idx : idx + 3] for idx in range(0, len(buttons), 3)]
    return types.InlineKeyboardMarkup(inline_keyboard=rows)


def get_language_keyboard(locale: str) -> types.ReplyKeyboardMarkup:
    return _keyboard(
        [button_text("language_ru", locale, icon="🇷🇺"), button_text("language_en", locale, icon="🇬🇧")],
        [button_text("back", locale, icon="◀️")],
    )


def _matches_text(value: str | None, *keys: str) -> bool:
    return normalize_button_label(value) in all_button_variants(keys)


def _message_matches(*keys: str):
    return lambda m: _matches_text(getattr(m, "text", None), *keys)


def _is_known_button_text(value: str | None) -> bool:
    return normalize_button_label(value) in all_button_variants()


def _should_cleanup_user_service_message(message: types.Message) -> bool:
    if not USER_SERVICE_MESSAGE_CLEANUP_ENABLED:
        return False
    text = getattr(message, "text", None)
    if not text or text.startswith("/"):
        return False
    if not _is_known_button_text(text):
        return False
    chat = getattr(message, "chat", None)
    chat_type = getattr(chat, "type", None)
    return chat_type in (None, "private")


def _is_plain_text_message(message: types.Message) -> bool:
    text = getattr(message, "text", None)
    return bool(text and not text.startswith("/"))


def _should_handle_promo_code_input(message: types.Message) -> bool:
    return bool(
        message.from_user
        and message.from_user.id in PROMO_CODE_WAIT
        and _is_plain_text_message(message)
        and not _is_known_button_text(getattr(message, "text", None))
    )


def _should_handle_instruction_global_input(message: types.Message) -> bool:
    return bool(
        message.from_user
        and message.from_user.id in INSTRUCTION_GLOBAL_WAIT_USERS
        and _is_plain_text_message(message)
        and not _is_known_button_text(getattr(message, "text", None))
    )


def _should_handle_instruction_group_choice(message: types.Message) -> bool:
    return bool(
        message.from_user
        and message.from_user.id in INSTRUCTION_GROUP_SELECT_USERS
        and _is_plain_text_message(message)
        and not _is_known_button_text(getattr(message, "text", None))
    )


def _should_handle_instruction_group_prompt(message: types.Message) -> bool:
    return bool(
        message.from_user
        and message.from_user.id in INSTRUCTION_GROUP_PROMPT_WAIT_USERS
        and _is_plain_text_message(message)
        and not _is_known_button_text(getattr(message, "text", None))
    )


def _should_handle_instruction_group_disable_choice(message: types.Message) -> bool:
    return bool(
        message.from_user
        and message.from_user.id in INSTRUCTION_GROUP_DISABLE_SELECT_USERS
        and _is_plain_text_message(message)
        and not _is_known_button_text(getattr(message, "text", None))
    )


def _should_handle_csi_score_input(message: types.Message) -> bool:
    return bool(
        message.from_user
        and message.from_user.id in CSI_SCORE_WAIT_USERS
        and parse_csi_score(getattr(message, "text", None)) is not None
    )


def _should_handle_support_request_input(message: types.Message) -> bool:
    return bool(
        message.from_user
        and message.from_user.id in SUPPORT_REQUEST_WAIT_USERS
        and _is_plain_text_message(message)
        and not _is_known_button_text(getattr(message, "text", None))
    )


def _button_locale_from_message(message: types.Message) -> str:
    detected = detect_button_locale(getattr(message, "text", None))
    if detected:
        return detected
    return locale_from_telegram_language_code(getattr(message.from_user, "language_code", None))


def _button_locale_from_callback(callback: types.CallbackQuery) -> str:
    return locale_from_telegram_language_code(getattr(callback.from_user, "language_code", None))


def _normalize_admin_username(value: str | None) -> str:
    return (value or "").strip().lstrip("@").casefold()


def _is_admin_username(value: str | None) -> bool:
    normalized = _normalize_admin_username(value)
    return bool(normalized) and normalized in ADMIN_USERNAMES


def _is_admin_user(user: User | None, telegram_username: str | None = None) -> bool:
    if _is_admin_username(telegram_username):
        return True
    return bool(user and _is_admin_username(getattr(user, "username", None)))


def _user_locale_or_message(user: User | None, message: types.Message) -> str:
    return get_user_locale(user) if user else _button_locale_from_message(message)


def _user_locale_or_callback(user: User | None, callback: types.CallbackQuery) -> str:
    return get_user_locale(user) if user else _button_locale_from_callback(callback)


async def _resolve_message_locale(message: types.Message) -> str:
    fallback = _button_locale_from_message(message)
    telegram_id = getattr(getattr(message, "from_user", None), "id", None)
    if telegram_id is None:
        return fallback

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        return get_user_locale(user) if user else fallback

    return fallback


def _main_keyboard_for_user(locale: str, user: User | None = None, telegram_username: str | None = None) -> types.ReplyKeyboardMarkup:
    return get_main_keyboard(locale, is_admin=_is_admin_user(user, telegram_username))


def _settings_keyboard_for_user(locale: str, user: User | None = None, telegram_username: str | None = None) -> types.ReplyKeyboardMarkup:
    rows: list[list[str]] = [
        [button_text("subscriptions", locale, icon="📡")],
        [button_text("billing", locale, icon="💳")],
        [button_text("forwarding", locale, icon="📰")],
        [button_text("digest", locale, icon="🗞️")],
        [button_text("language_menu", locale, icon="🌐")],
    ]
    if _storytracking_available(user, username=telegram_username):
        rows.insert(4, [button_text("storyline_tracking", locale, icon="🧭")])
        rows.insert(5, [button_text("plus_delivery_menu", locale, icon="🧩")])
    rows.append([button_text("back", locale, icon="◀️")])
    return _keyboard(*rows)


def _support_alert_is_configured() -> bool:
    return bool(SUPPORT_ALERT_BOT_TOKEN and SUPPORT_ALERT_CHAT_ID)


def _support_backoff_seconds(attempts: int) -> int:
    idx = min(max(0, attempts), len(SUPPORT_RETRY_BACKOFF_SECONDS) - 1)
    return int(SUPPORT_RETRY_BACKOFF_SECONDS[idx])


def _format_support_payload_text(payload: dict[str, object]) -> str:
    return (
        "🆘 Новое обращение в поддержку\n\n"
        f"request_id: {payload.get('request_id')}\n"
        f"user_registered_date: {payload.get('user_registered_date')}\n"
        f"date_utc: {payload.get('date_utc')}\n"
        f"telegram_id: {payload.get('telegram_id')}\n"
        f"username: @{payload.get('username')}\n"
        f"premium_active: {'yes' if payload.get('premium_active') else 'no'}\n\n"
        f"Текст:\n{str(payload.get('question_text') or '').strip()}"
    )


def _support_ticket_to_payload(ticket: SupportTicket) -> dict[str, object]:
    return {
        "request_id": str(ticket.request_id),
        "telegram_id": int(ticket.telegram_id),
        "username": str(ticket.username or "unknown"),
        "premium_active": bool(ticket.premium_active),
        "user_registered_date": str(ticket.user_registered_at) if getattr(ticket, "user_registered_at", None) else "unknown",
        "date_utc": str(ticket.created_at),
        "question_text": str(ticket.question_text or ""),
    }


async def _send_support_payload_to_alert_chat(payload: dict[str, object]) -> None:
    if not _support_alert_is_configured():
        raise RuntimeError("support alert bot is not configured")
    support_bot = Bot(token=SUPPORT_ALERT_BOT_TOKEN)
    kwargs: dict[str, object] = {
        "chat_id": int(SUPPORT_ALERT_CHAT_ID),
        "text": _format_support_payload_text(payload),
    }
    if SUPPORT_ALERT_THREAD_ID:
        kwargs["message_thread_id"] = int(SUPPORT_ALERT_THREAD_ID)
    try:
        await support_bot.send_message(**kwargs)
    finally:
        await support_bot.session.close()


async def _try_deliver_pending_support_tickets_once() -> tuple[int, int]:
    if not _support_alert_is_configured():
        return 0, 0

    now = utcnow()
    sent = 0
    failed = 0
    async for session in get_session():
        result = await session.execute(
            select(SupportTicket)
            .where(
                SupportTicket.status.in_(("accepted", "retrying")),
                SupportTicket.next_retry_at <= now,
            )
            .order_by(SupportTicket.id.asc())
            .limit(SUPPORT_RETRY_BATCH_SIZE)
        )
        tickets = list(result.scalars().all())
        if not tickets:
            return 0, 0

        for ticket in tickets:
            payload = _support_ticket_to_payload(ticket)
            ticket.attempts = int(ticket.attempts or 0) + 1
            ticket.updated_at = utcnow()
            try:
                await _send_support_payload_to_alert_chat(payload)
                ticket.status = "delivered"
                ticket.delivered_at = utcnow()
                ticket.last_error = None
                sent += 1
                audit("support_request.forwarded", **payload, attempts=ticket.attempts)
            except Exception as exc:
                ticket.status = "retrying"
                ticket.last_error = repr(exc)
                ticket.next_retry_at = utcnow() + timedelta(seconds=_support_backoff_seconds(ticket.attempts))
                failed += 1
                audit(
                    "support_request.forward_failed",
                    request_id=ticket.request_id,
                    telegram_id=ticket.telegram_id,
                    attempts=ticket.attempts,
                    next_retry_at=str(ticket.next_retry_at),
                    error=repr(exc),
                )
        await session.commit()
        return sent, failed
    return 0, 0


async def _support_retry_loop(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            sent, failed = await _try_deliver_pending_support_tickets_once()
            if sent or failed:
                log.info("support_retry.tick", sent=sent, failed=failed)
        except Exception as exc:
            log.warning("support_retry.loop_failed", error=repr(exc))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=SUPPORT_RETRY_POLL_SECONDS)
        except asyncio.TimeoutError:
            continue


async def _deliver_support_ticket_now_or_schedule_retry(session, ticket: SupportTicket) -> None:
    payload = _support_ticket_to_payload(ticket)
    ticket.attempts = int(ticket.attempts or 0) + 1
    ticket.updated_at = utcnow()
    try:
        await _send_support_payload_to_alert_chat(payload)
        ticket.status = "delivered"
        ticket.delivered_at = utcnow()
        ticket.last_error = None
        audit("support_request.forwarded", **payload, attempts=ticket.attempts)
    except Exception as exc:
        ticket.status = "retrying"
        ticket.last_error = repr(exc)
        ticket.next_retry_at = utcnow() + timedelta(seconds=_support_backoff_seconds(ticket.attempts))
        audit(
            "support_request.forward_failed",
            request_id=ticket.request_id,
            telegram_id=ticket.telegram_id,
            attempts=ticket.attempts,
            next_retry_at=str(ticket.next_retry_at),
            error=repr(exc),
        )


def _create_support_ticket(
    *,
    user: User,
    username: str,
    premium_active: bool,
    question_text: str,
    now: datetime,
) -> SupportTicket:
    return SupportTicket(
        request_id=str(uuid4()),
        user_id=int(user.id),
        telegram_id=int(user.telegram_id),
        username=username,
        premium_active=bool(premium_active),
        user_registered_at=getattr(user, "created_at", None),
        question_text=question_text,
        status="accepted",
        attempts=0,
        next_retry_at=now,
    )

async def _has_instruction_filter_rules(session, *, user_id: int) -> bool:
    rules = await get_instruction_rules_for_user(session, user_id=user_id)
    return bool(rules)


async def _forwarding_keyboard_for_user(session, *, user: User, locale: str) -> types.ReplyKeyboardMarkup:
    current_filter = (getattr(user, "feed_filter", "all") or "all").strip().lower()
    return get_forwarding_keyboard(
        locale,
        live_enabled=current_filter != "digest_only",
        summary_enabled=bool(getattr(user, "summary_enabled", False)),
        ai_filter_enabled=await _has_instruction_filter_rules(session, user_id=user.id),
        current_filter=current_filter,
    )


def _digest_time_label(slot: UserDigestSlot | None) -> str | None:
    if not slot:
        return None
    return f"{slot.hour:02d}:{slot.minute:02d}"


def _render_storyline_follow_list(follows: list[UserStorylineFollow]) -> str:
    lines: list[str] = []
    for idx, follow in enumerate(follows, start=1):
        title = _trim_debug_text(
            getattr(follow, "storyline_title", None)
            or getattr(follow, "family_root_storyline_id", None)
            or f"storyline_{follow.id}",
            limit=120,
        )
        lines.append(f"{idx}. {title}")
    return "\n".join(lines)


async def _resolve_callback_locale(callback: types.CallbackQuery) -> str:
    fallback = _button_locale_from_callback(callback)
    telegram_id = getattr(getattr(callback, "from_user", None), "id", None)
    if telegram_id is None:
        return fallback

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        return get_user_locale(user) if user else fallback

    return fallback


async def _get_news_time_survey(session, *, user_id: int) -> UserNewsTimeSurvey | None:
    result = await session.execute(select(UserNewsTimeSurvey).where(UserNewsTimeSurvey.user_id == user_id))
    return result.scalar_one_or_none()


async def _user_has_any_subscription(session, *, user) -> bool:
    # Test doubles (SimpleNamespace, etc.) do not represent persisted users.
    if user.__class__.__name__ != "User":
        return True
    total = await session.scalar(
        select(func.count(UserCommunity.id)).where(UserCommunity.user_id == user.id)
    )
    return int(total or 0) > 0


async def _get_csi_survey(session, *, user_id: int) -> UserCsiSurvey | None:
    result = await session.execute(select(UserCsiSurvey).where(UserCsiSurvey.user_id == user_id))
    return result.scalar_one_or_none()


async def _get_csi_eligibility(session, *, user_id: int):
    row = (
        await session.execute(
            select(
                func.min(DispatchDelivery.sent_at),
                func.count(DispatchDelivery.post_id),
            ).where(
                DispatchDelivery.user_id == user_id,
                DispatchDelivery.sent_at.is_not(None),
            )
        )
    ).one()
    return build_csi_eligibility(first_delivery_at=row[0], delivery_count=row[1])


def _resolve_survey_target(event):
    callback_message = getattr(event, "message", None)
    if callback_message is not None and hasattr(callback_message, "answer"):
        return callback_message
    if hasattr(event, "answer"):
        return event
    return None


async def _send_persistent_survey_message(target, text: str, *, reply_markup):
    answer = getattr(target, "answer", None)
    if not callable(answer):
        raise RuntimeError("survey target has no answer()")
    try:
        return await answer(
            text,
            reply_markup=reply_markup,
            skip_service_cleanup=True,
        )
    except TypeError:
        # Fallback for targets that do not support skip_service_cleanup.
        return await answer(text, reply_markup=reply_markup)


async def _maybe_prompt_news_time_followup(
    session,
    *,
    user: User,
    target,
    locale: str | None = None,
    now: datetime | None = None,
) -> bool:
    survey = await _get_news_time_survey(session, user_id=user.id)
    current_time = now or utcnow()
    if (
        survey
        and survey.baseline_answered_at is not None
        and getattr(user, "created_at", None) is not None
    ):
        expected_due = followup_due_at(user.created_at)
        if survey.followup_due_at != expected_due:
            survey.followup_due_at = expected_due
            try:
                await session.commit()
            except Exception as exc:
                log.warning(
                    "news_time_survey.followup_due_sync_failed",
                    telegram_id=user.telegram_id,
                    user_id=user.id,
                    error=repr(exc),
                )
    if should_abandon_followup(survey, now=current_time):
        survey.followup_abandoned_at = current_time
        await session.commit()
        audit("news_time_survey.followup_abandoned", telegram_id=user.telegram_id, user_id=user.id)
        return False
    if not should_ask_followup(survey, now=current_time):
        return False

    locale_resolved = locale or get_user_locale(user)
    is_retry = survey.followup_asked_at is not None
    audit(
        "news_time_survey.followup_send_attempt",
        telegram_id=user.telegram_id,
        user_id=user.id,
        is_retry=is_retry,
    )
    try:
        await _send_persistent_survey_message(
            target,
            t(locale_resolved, "news_time_saved_question"),
            reply_markup=get_news_time_saved_keyboard(locale_resolved),
        )
    except Exception as exc:
        audit(
            "news_time_survey.followup_send_failed",
            telegram_id=user.telegram_id,
            user_id=user.id,
            is_retry=is_retry,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        log.warning(
            "news_time_survey.followup_send_failed",
            telegram_id=user.telegram_id,
            user_id=user.id,
            is_retry=is_retry,
            error=repr(exc),
        )
        return False

    asked_at = current_time
    if survey.followup_asked_at is None:
        survey.followup_asked_at = asked_at
        audit("news_time_survey.followup_asked", telegram_id=user.telegram_id, user_id=user.id)
    else:
        survey.followup_retry_asked_at = asked_at
        audit("news_time_survey.followup_retry_asked", telegram_id=user.telegram_id, user_id=user.id)
    try:
        await session.commit()
    except Exception as exc:
        log.warning(
            "news_time_survey.followup_state_commit_failed",
            telegram_id=user.telegram_id,
            user_id=user.id,
            error=repr(exc),
        )
    audit(
        "news_time_survey.followup_send_success",
        telegram_id=user.telegram_id,
        user_id=user.id,
        is_retry=is_retry,
    )
    return True


async def _maybe_prompt_news_time_baseline(
    session,
    *,
    user: User,
    target,
    locale: str | None = None,
) -> bool:
    survey = await _get_news_time_survey(session, user_id=user.id)
    current_time = utcnow()
    if not survey:
        survey = UserNewsTimeSurvey(user_id=user.id)
        session.add(survey)
        await session.flush()

    if survey.baseline_answered_at is not None:
        NEWS_TIME_BASELINE_WAIT_USERS.discard(int(user.telegram_id))
        return False

    telegram_id = int(user.telegram_id)
    if telegram_id in NEWS_TIME_BASELINE_WAIT_USERS:
        return False
    asked_at = getattr(survey, "baseline_asked_at", None)
    retry_asked_at = getattr(survey, "baseline_retry_asked_at", None)
    if asked_at is not None:
        if retry_asked_at is None and current_time < asked_at + NEWS_TIME_FOLLOWUP_RETRY_DELAY:
            return False
        if retry_asked_at is not None and current_time < retry_asked_at + NEWS_TIME_FOLLOWUP_RETRY_DELAY:
            return False

    locale_resolved = locale or get_user_locale(user)
    try:
        await _send_persistent_survey_message(
            target,
            t(locale_resolved, "news_time_baseline_question"),
            reply_markup=get_news_time_survey_keyboard(locale_resolved),
        )
    except Exception as exc:
        log.warning(
            "news_time_survey.baseline_send_failed",
            telegram_id=user.telegram_id,
            user_id=user.id,
            error=repr(exc),
        )
        return False

    NEWS_TIME_BASELINE_WAIT_USERS.add(telegram_id)
    sent_at = current_time
    if getattr(survey, "baseline_asked_at", None) is None:
        survey.baseline_asked_at = sent_at
    else:
        survey.baseline_retry_asked_at = sent_at
    audit("news_time_survey.baseline_asked", telegram_id=user.telegram_id, user_id=user.id)
    await session.commit()
    return True


async def _maybe_prompt_csi(
    session,
    *,
    user: User,
    target,
    locale: str | None = None,
    now: datetime | None = None,
) -> bool:
    current_time = now or utcnow()
    survey = await _get_csi_survey(session, user_id=user.id)

    news_survey = await _get_news_time_survey(session, user_id=user.id)
    if news_survey:
        if news_survey.time_saved_answered_at is None:
            return False
        news_completed_at = news_survey.time_saved_answered_at
        if news_survey.time_saved_answer == "yes":
            if news_survey.current_answered_at is None:
                return False
            news_completed_at = news_survey.current_answered_at
        if news_completed_at and current_time < news_completed_at + CSI_NEWS_TIME_COOLDOWN:
            return False

    if should_abandon_csi(survey, now=current_time):
        survey.abandoned_at = current_time
        CSI_SCORE_WAIT_USERS.discard(int(user.telegram_id))
        await session.commit()
        audit("csi_survey.abandoned", telegram_id=user.telegram_id, user_id=user.id)
        return False

    eligibility = await _get_csi_eligibility(session, user_id=user.id)
    if not should_ask_csi(survey, now=current_time, eligibility=eligibility):
        return False
    if not survey:
        survey = UserCsiSurvey(user_id=user.id)
        session.add(survey)

    survey.first_delivery_at = eligibility.first_delivery_at
    survey.delivery_count_at_prompt = eligibility.delivery_count
    survey.due_at = eligibility.due_at
    locale_resolved = locale or get_user_locale(user)
    try:
        await _send_persistent_survey_message(
            target,
            t(locale_resolved, "csi_question"),
            reply_markup=get_csi_score_keyboard(locale_resolved),
        )
    except Exception as exc:
        log.warning(
            "csi_survey.send_failed",
            telegram_id=user.telegram_id,
            user_id=user.id,
            delivery_count=eligibility.delivery_count,
            error=repr(exc),
        )
        return False

    if getattr(survey, "asked_at", None) is None:
        survey.asked_at = current_time
        audit(
            "csi_survey.asked",
            telegram_id=user.telegram_id,
            user_id=user.id,
            delivery_count=eligibility.delivery_count,
        )
    else:
        survey.retry_asked_at = current_time
        audit(
            "csi_survey.retry_asked",
            telegram_id=user.telegram_id,
            user_id=user.id,
            delivery_count=eligibility.delivery_count,
        )
    CSI_SCORE_WAIT_USERS.add(int(user.telegram_id))
    try:
        await session.commit()
    except Exception as exc:
        log.warning(
            "csi_survey.state_commit_failed",
            telegram_id=user.telegram_id,
            user_id=user.id,
            error=repr(exc),
        )
    return True


async def _maybe_prompt_news_time_followup_from_event(event) -> None:
    from_user = getattr(event, "from_user", None)
    target = _resolve_survey_target(event)
    if not from_user or not target or not hasattr(target, "answer"):
        return

    telegram_id = getattr(from_user, "id", None)
    if telegram_id is None:
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        if not user:
            return
        if not await _user_has_any_subscription(session, user=user):
            return
        await _maybe_prompt_news_time_followup(
            session,
            user=user,
            target=target,
            locale=get_user_locale(user),
        )
        return


async def _maybe_prompt_surveys_from_event(event) -> None:
    from_user = getattr(event, "from_user", None)
    target = _resolve_survey_target(event)
    if not from_user or not target or not hasattr(target, "answer"):
        return

    telegram_id = getattr(from_user, "id", None)
    if telegram_id is None:
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        if not user:
            return
        if not await _user_has_any_subscription(session, user=user):
            return
        locale = get_user_locale(user)
        if await _maybe_prompt_news_time_baseline(session, user=user, target=target, locale=locale):
            return
        if await _maybe_prompt_news_time_followup(session, user=user, target=target, locale=locale):
            return
        await _maybe_prompt_csi(session, user=user, target=target, locale=locale)
        return


async def _try_handle_csi_score_answer(message: types.Message, *, require_pending: bool = True) -> bool:
    score = parse_csi_score(getattr(message, "text", None))
    if score is None:
        return False
    telegram_id = getattr(getattr(message, "from_user", None), "id", None)
    if telegram_id is None:
        return False

    locale = _button_locale_from_message(message)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return True
        locale = get_user_locale(user)
        survey = await _get_csi_survey(session, user_id=user.id)
        if require_pending and telegram_id not in CSI_SCORE_WAIT_USERS and not is_csi_pending(survey):
            return False
        if not survey or not is_csi_pending(survey):
            return False

        now = utcnow()
        survey.score = score
        survey.answered_at = now
        CSI_SCORE_WAIT_USERS.discard(int(telegram_id))
        await session.commit()
        audit(
            "csi_survey.answered",
            telegram_id=user.telegram_id,
            user_id=user.id,
            score=score,
        )
        await message.answer(
            t(locale, "csi_thanks"),
            reply_markup=_main_keyboard_for_user(locale, user, getattr(message.from_user, "username", None)),
        )
        setattr(message, "_newshub_skip_survey_prompt", True)
        return True
    return False


class NewsTimeSurveyTouchMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        result = await handler(event, data)
        if getattr(event, "_newshub_skip_survey_prompt", False):
            return result
        try:
            await _maybe_prompt_surveys_from_event(event)
        except Exception as exc:
            log.warning("survey.followup_prompt_failed", error=repr(exc))
        return result


class NewsTimeSurveyCallbackAuditMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        callback = getattr(event, "callback_query", None)
        callback_data = str(getattr(callback, "data", "") or "")
        if callback and callback_data.startswith("news_time:"):
            try:
                audit(
                    "news_time_survey.callback_update_received",
                    callback_data=callback_data,
                    telegram_id=getattr(getattr(callback, "from_user", None), "id", None),
                    message_id=getattr(getattr(callback, "message", None), "message_id", None),
                )
            except Exception as exc:
                log.warning("news_time_survey.callback_audit_failed", error=repr(exc))
        return await handler(event, data)


class UserServiceMessageCleanupMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        result = await handler(event, data)
        if not isinstance(event, types.Message):
            return result
        if not _should_cleanup_user_service_message(event):
            return result

        chat_id = getattr(getattr(event, "chat", None), "id", None)
        message_id = getattr(event, "message_id", None)
        if chat_id is None or message_id is None:
            return result
        task = asyncio.create_task(
            _delete_service_message_later(
                chat_id=chat_id,
                message_id=message_id,
                delay_seconds=USER_SERVICE_MESSAGE_TTL_SECONDS,
            )
        )
        _track_cleanup_task(task)
        return result


def _register_news_time_survey_middleware() -> None:
    middleware = NewsTimeSurveyTouchMiddleware()
    for observer_name in ("message", "callback_query"):
        observer = getattr(dp, observer_name, None)
        register = getattr(observer, "middleware", None)
        if callable(register):
            register(middleware)


_register_news_time_survey_middleware()


def _register_news_time_callback_audit_middleware() -> None:
    observer = getattr(dp, "update", None)
    register = getattr(getattr(observer, "outer_middleware", None), "__call__", None)
    if callable(register):
        observer.outer_middleware(NewsTimeSurveyCallbackAuditMiddleware())


_register_news_time_callback_audit_middleware()


async def _record_post_interaction(
    session,
    *,
    user: User,
    post: Post,
    action: str,
    source: str,
) -> None:
    try:
        await session.execute(
            text(
                """
                INSERT INTO post_interactions
                    (user_id, post_id, community_id, action, source, interaction_count, first_interacted_at, last_interacted_at)
                VALUES
                    (:user_id, :post_id, :community_id, :action, :source, 1, now(), now())
                ON CONFLICT (user_id, post_id, action, source)
                DO UPDATE SET
                    interaction_count = post_interactions.interaction_count + 1,
                    last_interacted_at = now()
                """
            ),
            {
                "user_id": int(user.id),
                "post_id": int(post.id),
                "community_id": int(post.community_id) if post.community_id is not None else None,
                "action": str(action or "").strip(),
                "source": str(source or "").strip(),
            },
        )
        user.engagement_score = float(getattr(user, "engagement_score", 0.0) or 0.0) + 0.1
        session.add(user)
        audit(
            "post_interaction.recorded",
            user_id=user.id,
            telegram_id=user.telegram_id,
            post_id=post.id,
            community_id=post.community_id,
            action=action,
            source=source,
        )
        await session.commit()
    except Exception as exc:
        await session.rollback()
        log.warning(
            "post_interaction.record_failed",
            user_id=getattr(user, "id", None),
            post_id=getattr(post, "id", None),
            action=action,
            source=source,
            error=repr(exc),
        )


def _register_user_service_message_cleanup_middleware() -> None:
    observer = getattr(dp, "message", None)
    register = getattr(observer, "middleware", None)
    if callable(register):
        register(UserServiceMessageCleanupMiddleware())


_register_user_service_message_cleanup_middleware()


PLUS_ALLOWED_CALLBACK_PREFIXES = (
    "react:",
    "storytimeline:",
    "storytimelinecard:",
    "storysimilar:",
    "storysimtimeline:",
    "storyfollow:",
    "storyunfollowroot:",
    "dig:",
)
PLUS_ALLOWED_CALLBACK_EXACT = {"noop"}
PLUS_ALLOWED_COMMANDS = {"/start", "/help"}
PLUS_ALLOWED_HELP_BUTTONS = all_button_variants({"help"})


async def _plus_get_user_and_prefs(telegram_id: int) -> tuple[User | None, UserDeliveryPreference | None]:
    if telegram_id <= 0:
        return None, None
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        if not user:
            return None, None
        prefs = await session.get(UserDeliveryPreference, int(user.id))
        return user, prefs
    return None, None


def _plus_is_connected(prefs: UserDeliveryPreference | None) -> bool:
    return bool(getattr(prefs, "plus_bot_connected_at", None))


class PlusBotGuardMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        if BOT_ROLE != "plus":
            return await handler(event, data)

        if isinstance(event, types.Message):
            chat_type = str(getattr(getattr(event, "chat", None), "type", "") or "").strip().lower()
            if chat_type != "private":
                return None

            raw_text = str(getattr(event, "text", "") or "").strip()
            command = raw_text.split()[0].lower() if raw_text.startswith("/") else ""
            if command in PLUS_ALLOWED_COMMANDS or raw_text in PLUS_ALLOWED_HELP_BUTTONS:
                return await handler(event, data)

            locale = _button_locale_from_message(event)
            user, prefs = await _plus_get_user_and_prefs(int(getattr(getattr(event, "from_user", None), "id", 0) or 0))
            locale = get_user_locale(user) if user else locale
            if not _plus_is_connected(prefs):
                hint = t(locale, "plus_delivery_connect_required")
                link = _plus_connect_link()
                if link:
                    hint = f"{hint}\n{link}"
                await event.answer(hint, reply_markup=_plus_connect_inline_keyboard(locale))
                return None

            await event.answer(
                "Этот бот работает только для доставки карточек. Все настройки доступны в основном боте.",
                reply_markup=_main_keyboard_for_user(locale, user, getattr(getattr(event, "from_user", None), "username", None)),
            )
            return None

        if isinstance(event, types.CallbackQuery):
            message = getattr(event, "message", None)
            chat_type = str(getattr(getattr(message, "chat", None), "type", "") or "").strip().lower()
            if chat_type and chat_type != "private":
                await _safe_answer_callback(event)
                return None

            callback_data = str(getattr(event, "data", "") or "")
            allowed = callback_data in PLUS_ALLOWED_CALLBACK_EXACT or any(
                callback_data.startswith(prefix) for prefix in PLUS_ALLOWED_CALLBACK_PREFIXES
            )
            if not allowed:
                await _safe_answer_callback(event)
                return None

            locale = _button_locale_from_callback(event)
            user, prefs = await _plus_get_user_and_prefs(int(getattr(getattr(event, "from_user", None), "id", 0) or 0))
            locale = get_user_locale(user) if user else locale
            if not user or not _plus_is_connected(prefs):
                await _safe_answer_callback(event)
                hint = t(locale, "plus_delivery_connect_required")
                link = _plus_connect_link()
                if link:
                    hint = f"{hint}\n{link}"
                if message:
                    await message.answer(hint, reply_markup=_plus_connect_inline_keyboard(locale))
                return None

            return await handler(event, data)

        return await handler(event, data)


def _register_plus_bot_guard_middleware() -> None:
    if BOT_ROLE != "plus":
        return
    guard = PlusBotGuardMiddleware()
    message_observer = getattr(dp, "message", None)
    callback_observer = getattr(dp, "callback_query", None)
    message_register = getattr(message_observer, "outer_middleware", None)
    callback_register = getattr(callback_observer, "outer_middleware", None)
    if callable(message_register):
        message_observer.outer_middleware(guard)
    if callable(callback_register):
        callback_observer.outer_middleware(guard)


_register_plus_bot_guard_middleware()


def _reserved_digest_variants() -> set[str]:
    return all_button_variants(
        {
            "back",
            "settings",
            "subscriptions",
            "digest",
            "digest_daily",
            "digest_disable",
            "digest_toggle_on",
            "digest_toggle_off",
            "digest_setup_time",
            "digest_send_now",
            "toggle_summary",
        }
    )

def parse_hhmm(value: str) -> tuple[int, int] | None:
    s = (value or "").strip()
    if not TIME_RE.match(s):
        return None
    h, m = s.split(":")
    return int(h), int(m)

def parse_utc_offset_to_minutes(value: str) -> int | None:
    s = (value or "").strip().replace(" ", "")
    m = OFFSET_RE.match(s)
    if not m:
        return None
    sign, hh, mm = m.group(1), int(m.group(2)), m.group(3)
    minutes = int(mm) if mm is not None else 0
    if hh > 14 or minutes > 59:   # разумная валидация
        return None
    total = hh * 60 + minutes
    return total if sign == "+" else -total


def normalize_city_key(value: str) -> str:
    v = (value or "").strip().casefold().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", "", v)


def resolve_timezone_from_city(value: str) -> str | None:
    return CITY_TO_TZ.get(normalize_city_key(value))


def current_offset_minutes_for_tz(tz_name: str) -> int:
    now_utc = datetime.now(timezone.utc)
    return int(now_utc.astimezone(ZoneInfo(tz_name)).utcoffset().total_seconds() // 60)


# def compute_next_run_at_utc(tz_name: str, hour: int, minute: int, now_utc: datetime | None = None) -> datetime:
#     now_utc = now_utc or datetime.now(timezone.utc)
#     tz = ZoneInfo(tz_name)
#     now_local = now_utc.astimezone(tz)

#     candidate_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
#     if candidate_local <= now_local:
#         candidate_local = candidate_local + timedelta(days=1)

#     return candidate_local.astimezone(timezone.utc)

def compute_next_run_at_utc_from_offset(offset_minutes: int, hour: int, minute: int, now_utc: datetime | None = None) -> datetime:
    now_utc = now_utc or datetime.now(timezone.utc)
    offset = timezone(timedelta(minutes=offset_minutes))  # fixed offset tz [web:120]
    now_local = now_utc.astimezone(offset)

    candidate_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate_local <= now_local:
        candidate_local = candidate_local + timedelta(days=1)

    return candidate_local.astimezone(timezone.utc)

def parse_group_input(text: str) -> tuple[str, str]:
    """
    Нормализуем вход: @username | https://t.me/username | t.me/username
    Возвращаем: (@handle, https://t.me/username)
    """
    raw = (text or "").strip()

    if not raw:
        return "@", "https://t.me/"

    if raw.startswith("@"):
        username = raw[1:]
    elif "t.me/" in raw:
        username = raw.split("t.me/", 1)[1].split("?", 1)[0].strip("/")
    else:
        username = raw.lstrip("@")

    handle = f"@{username}" if username else "@"
    link = f"https://t.me/{username}" if username else "https://t.me/"
    return handle, link

def parse_multiline_group_input(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    valid: list[tuple[str, str]] = []
    invalid: list[str] = []

    for line in lines:
        if not re.match(r"^(?:@|https?://t\.me/|t\.me/)", line):
            invalid.append(line)
            continue
        handle, link = parse_group_input(line)
        if handle == "@":
            invalid.append(line)
            continue
        valid.append((handle, link))

    return valid, invalid


def _dedupe_groups(groups: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen_links: set[str] = set()
    unique: list[tuple[str, str]] = []
    for group_handle, group_link in groups:
        if group_link in seen_links:
            continue
        seen_links.add(group_link)
        unique.append((group_handle, group_link))
    return unique


@dataclass(slots=True)
class SubscriptionChangeResult:
    subscribed: list[str]
    queued: list[str]
    unsubscribed: list[str]
    already_queued: list[str]
    already_subscribed: list[str]
    missing: list[str]
    invalid_lines: list[str]
    request_ids: list[tuple[int, str]]


def _message_is_private_chat(message: types.Message) -> bool:
    chat_type = str(getattr(getattr(message, "chat", None), "type", "") or "").strip().lower()
    return chat_type == "private"


def _build_nlu_confirm_keyboard(locale: str, request_id: str | None = None) -> types.InlineKeyboardMarkup:
    suffix = f":{request_id}" if request_id else ""
    return types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(text=t(locale, "nlu_confirm_button"), callback_data=f"nlu:confirm{suffix}"),
                types.InlineKeyboardButton(text=t(locale, "nlu_cancel_button"), callback_data=f"nlu:cancel{suffix}"),
            ]
        ]
    )


def _render_digest_state_lines(
    *,
    locale: str,
    settings: UserDigestSettings | None,
    slot: UserDigestSlot | None,
) -> list[str]:
    enabled = bool(getattr(settings, "enabled", False))
    offset_min = getattr(settings, "utc_offset_minutes", None)
    lines = [
        f"{t(locale, 'digest_settings_state')}: {t(locale, 'enabled') if enabled else t(locale, 'disabled')}",
        f"{t(locale, 'offset_set')}: UTC{offset_min / 60:+g}" if offset_min is not None else f"{t(locale, 'offset_set')}: {t(locale, 'offset_not_set')}",
        f"{t(locale, 'time')}: {slot.hour:02d}:{slot.minute:02d}" if slot else f"{t(locale, 'time')}: {t(locale, 'offset_not_set')}",
    ]
    if settings and getattr(settings, "timezone", None):
        lines.append(f"{t(locale, 'timezone')}: {settings.timezone}")
    return lines


def _billing_cta_suffix(snapshot: dict, locale: str) -> str:
    due_amount = max(
        0,
        int(snapshot.get("required_amount_rub") or 0) - int(snapshot.get("paid_amount_rub") or 0),
    )
    active = int(snapshot.get("active_subscriptions") or 0)
    allowed = int(snapshot.get("allowed_subscriptions") or snapshot.get("free_limit") or 7)
    reserve = max(0, allowed - active)
    if due_amount > 0:
        audit("billing.cta_shown", due_amount=due_amount, active_subscriptions=active, allowed_subscriptions=allowed)
        return "\n\n" + t(locale, "nlu_billing_cta_due", due_amount=due_amount)
    if reserve <= 1:
        audit("billing.cta_shown", due_amount=0, active_subscriptions=active, allowed_subscriptions=allowed)
        return "\n\n" + t(locale, "nlu_billing_cta_low_reserve", reserve=reserve)
    return ""


async def _build_user_digest_context(session, user: User) -> tuple[UserDigestSettings | None, UserDigestSlot | None]:
    settings = await session.get(UserDigestSettings, user.id)
    slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
    slot = slot_res.scalar_one_or_none()
    return settings, slot


async def _build_nlu_context(session, user: User) -> dict:
    settings, slot = await _build_user_digest_context(session, user)
    snapshot = await get_current_payment_snapshot(session, user)
    return {
        "language_code": get_user_locale(user),
        "feed_filter": getattr(user, "feed_filter", "all") or "all",
        "summary_enabled": bool(getattr(user, "summary_enabled", False)),
        "digest": {
            "enabled": bool(getattr(settings, "enabled", False)),
            "utc_offset": getattr(settings, "utc_offset_minutes", None),
            "time_hhmm": f"{slot.hour:02d}:{slot.minute:02d}" if slot else None,
        },
        "billing": {
            "active_subscriptions": int(snapshot.get("active_subscriptions") or 0),
            "allowed_subscriptions": int(snapshot.get("allowed_subscriptions") or snapshot.get("free_limit") or 7),
            "premium_active": bool(snapshot.get("premium_active")),
            "status": str(snapshot.get("status") or "").strip(),
        },
        "private_chat": True,
    }


def _normalize_router_text(text: str) -> str:
    normalized = str(text or "").strip().lower().replace("ё", "е")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _has_concrete_instruction_topic(normalized: str) -> bool:
    return any(
        term in normalized
        for term in (
            "эконом",
            "полит",
            "спорт",
            "культур",
            "технолог",
            "финанс",
            "бизнес",
            "крипт",
            "рынк",
            "наук",
            "медицин",
            "econom",
            "politic",
            "sport",
            "culture",
            "tech",
            "finance",
            "business",
            "crypto",
            "market",
            "science",
            "health",
        )
    )


def _looks_like_ambiguous_importance_filter_request(normalized: str) -> bool:
    if not normalized:
        return False
    has_importance_word = any(
        term in normalized
        for term in (
            "важн",
            "главн",
            "лучше",
            "топ",
            "important",
            "main",
            "top",
            "best",
        )
    )
    if not has_importance_word:
        return False

    has_action = any(
        term in normalized
        for term in (
            "показыв",
            "покаж",
            "присыл",
            "отправля",
            "остав",
            "только",
            "show",
            "send",
            "keep",
            "only",
        )
    )
    if not has_action:
        return False

    return not _has_concrete_instruction_topic(normalized)


def _looks_like_global_instruction_filter_request(normalized: str) -> bool:
    if not normalized:
        return False
    has_news_term = any(term in normalized for term in ("новост", "пост", "лент", "news", "posts", "feed"))
    if not has_news_term:
        return False

    has_action = any(
        term in normalized
        for term in (
            "показыв",
            "покаж",
            "присыл",
            "отправля",
            "остав",
            "скры",
            "исключ",
            "не показыв",
            "не присыл",
            "show",
            "send",
            "keep",
            "hide",
            "exclude",
            "block",
            "only",
            "только",
        )
    )
    if not has_action:
        return False

    has_topic_marker = any(
        marker in normalized
        for marker in (
            " по ",
            " про ",
            " об ",
            " о ",
            " на тему",
            "about ",
            "topic",
            "category",
            "категор",
            "тем",
        )
    )
    has_known_topic = _has_concrete_instruction_topic(normalized)
    return has_topic_marker or has_known_topic


def _resolve_builtin_router_intent(text: str) -> NLUResult | None:
    normalized = _normalize_router_text(text)
    if not normalized:
        return None

    asks_section_help = any(
        phrase in normalized
        for phrase in (
            "что такое",
            "что это",
            "что делает",
            "для чего",
            "зачем",
            "как работает",
            "как это работает",
            "как пользоваться",
            "как использовать",
            "объясни",
            "расскажи",
            "подскажи",
            "помощь",
            "help",
            "what is",
            "what does",
            "how does it work",
            "how to use",
            "tell me about",
        )
    )

    has_filter_word = any(
        token in normalized
        for token in (
            "фильтр",
            "огнен",
            "неинтерес",
            "дайджест",
            "live",
            "пересылк",
            "hot post",
            "digest only",
        )
    )
    asks_filter_difference = any(
        phrase in normalized
        for phrase in (
            "чем отлич",
            "в чем разниц",
            "какая разниц",
            "что значит",
            "какие есть фильтр",
            "это все фильтр",
        )
    )
    asks_filter_recommendation = any(
        phrase in normalized
        for phrase in (
            "какой фильтр лучше",
            "какой фильтр выбрать",
            "какой режим лучше",
            "что выбрать",
            "что лучше выбрать",
            "какой мне выбрать",
            "что мне выбрать",
            "посоветуй фильтр",
            "помоги выбрать фильтр",
        )
    ) or (
        "filter" in normalized
        and any(phrase in normalized for phrase in ("difference", "choose", "better", "recommend"))
    )

    if has_filter_word and (asks_filter_difference or asks_filter_recommendation):
        return NLUResult(
            intent="show_help_topic",
            slots={},
            needs_clarification=False,
            clarify_question=None,
            faq_topic="filters",
            proposed_user_message="",
        )

    if _looks_like_ambiguous_importance_filter_request(normalized):
        return NLUResult(
            intent="set_feed_filter",
            slots={"clarification_kind": "importance_filter"},
            needs_clarification=True,
            clarify_question=t("ru", "nlu_clarify_importance_filter"),
            faq_topic=None,
            proposed_user_message="",
        )

    if _looks_like_global_instruction_filter_request(normalized):
        return NLUResult(
            intent="set_global_instruction_filter",
            slots={"prompt_text": str(text or "").strip()},
            needs_clarification=False,
            clarify_question=None,
            faq_topic=None,
            proposed_user_message="",
        )

    has_summary_word = any(token in normalized for token in ("саммаризац", "summariz"))
    asks_summary_help = any(
        phrase in normalized
        for phrase in (
            "что такое",
            "что это",
            "для чего",
            "зачем",
            "как работает",
            "как это работает",
            "что делает",
            "что дает",
            "what is",
            "how does it work",
            "what does it do",
        )
    )
    if has_summary_word and asks_summary_help:
        return NLUResult(
            intent="show_help_topic",
            slots={},
            needs_clarification=False,
            clarify_question=None,
            faq_topic="summary",
            proposed_user_message="",
        )

    if asks_section_help:
        faq_topic_by_keywords = (
            ("ai_filter", ("ai-фильтр", "ai filter", "ai instruction", "ai-инструк", "инструкц", "промпт", "prompt")),
            ("storyline_tracking", ("отслеживан", "сюжет", "storyline")),
            ("subscriptions", ("подписк", "subscription")),
            ("billing", ("тариф", "оплат", "premium", "stars", "t-bank", "tbank", "crypto", "usdt")),
            ("filters", ("фильтр", "огнен", "неинтерес")),
            ("forwarding", ("пересылк", "live", "live feed")),
            ("digest", ("дайджест", "digest")),
            ("language", ("язык", "language")),
            ("settings", ("настройк", "settings")),
        )
        for faq_topic, keywords in faq_topic_by_keywords:
            if any(keyword in normalized for keyword in keywords):
                return NLUResult(
                    intent="show_help_topic",
                    slots={},
                    needs_clarification=False,
                    clarify_question=None,
                    faq_topic=faq_topic,
                    proposed_user_message="",
                )
    return None


def _render_settings_overview_text(
    *,
    user: User,
    locale: str,
    settings: UserDigestSettings | None,
    slot: UserDigestSlot | None,
    snapshot: dict,
) -> str:
    active = int(snapshot.get("active_subscriptions") or 0)
    allowed = int(snapshot.get("allowed_subscriptions") or snapshot.get("free_limit") or 7)
    premium_active = bool(snapshot.get("premium_active"))
    lines = [
        t(locale, "nlu_settings_overview_title"),
        "",
        f"{button_text('subscriptions', locale)}: {active}/{allowed}",
        _render_forwarding_menu_text(user, locale),
        "",
        t(locale, "digest_description"),
        *_render_digest_state_lines(locale=locale, settings=settings, slot=slot),
        "",
        f"{t(locale, 'billing_premium_status')}: {t(locale, 'billing_premium_active' if premium_active else 'billing_premium_inactive')}",
    ]
    return "\n".join(lines) + _billing_cta_suffix(snapshot, locale)


async def _render_help_topic_text(
    *,
    session,
    locale: str,
    topic: str,
    user: User,
    settings: UserDigestSettings | None,
    slot: UserDigestSlot | None,
    snapshot: dict,
    subscriptions: list[str] | None = None,
) -> ExecutorResult:
    if topic == "subscriptions":
        active = int(snapshot.get("active_subscriptions") or len(subscriptions or []))
        allowed = int(snapshot.get("allowed_subscriptions") or snapshot.get("free_limit") or 7)
        subs_text = "\n".join(subscriptions or []) if subscriptions else t(locale, "subscriptions_none")
        text = t(locale, "nlu_help_subscriptions", active=active, allowed=allowed, links=subs_text)
        return ExecutorResult(text=text, reply_markup=_settings_keyboard_for_user(locale, user))
    if topic == "settings":
        text = _render_settings_overview_text(user=user, locale=locale, settings=settings, slot=slot, snapshot=snapshot)
        return ExecutorResult(
            text=text + "\n\n" + t(locale, "nlu_help_settings"),
            reply_markup=_settings_keyboard_for_user(locale, user),
            parse_mode="HTML",
        )
    if topic == "billing":
        text = _render_billing_status(snapshot, locale) + "\n\n" + _render_billing_instruction(snapshot, locale)
        return ExecutorResult(
            text=text + _billing_cta_suffix(snapshot, locale),
            reply_markup=get_billing_keyboard(locale),
            parse_mode="HTML",
        )
    if topic == "filters":
        return ExecutorResult(
            text=_render_forwarding_menu_text(user, locale) + "\n\n" + t(locale, "nlu_help_filters"),
            reply_markup=get_filter_clarification_keyboard(locale),
            parse_mode="HTML",
        )
    if topic == "forwarding":
        return ExecutorResult(
            text=_render_forwarding_menu_text(user, locale) + "\n\n" + t(locale, "nlu_help_forwarding"),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
            parse_mode="HTML",
        )
    if topic == "summary":
        return ExecutorResult(
            text=t(
                locale,
                "nlu_help_summary",
                state=t(locale, "forwarding_summary_on") if bool(getattr(user, "summary_enabled", False)) else t(locale, "forwarding_summary_off"),
            ),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
        )
    if topic == "digest":
        text = t(locale, "digest_description") + "\n" + "\n".join(_render_digest_state_lines(locale=locale, settings=settings, slot=slot))
        return ExecutorResult(
            text=text + "\n\n" + t(locale, "nlu_help_digest"),
            reply_markup=get_digest_keyboard(
                locale,
                enabled=bool(settings.enabled) if settings else False,
                time_label=_digest_time_label(slot),
            ),
        )
    if topic == "language":
        return ExecutorResult(
            text=t(locale, "nlu_help_language", current_language=get_user_locale(user)),
            reply_markup=get_language_keyboard(locale),
        )
    if topic == "ai_filter":
        return ExecutorResult(
            text=t(locale, "nlu_help_ai_filter") + "\n\n" + await _render_instruction_menu_text(session, user=user, locale=locale),
            reply_markup=get_instruction_filter_keyboard(locale),
            parse_mode="HTML",
        )
    if topic == "storyline_tracking":
        if not _storytracking_available(user, username=getattr(user, "username", None)):
            return ExecutorResult(
                text=t(locale, "storytracking_disabled"),
                reply_markup=_settings_keyboard_for_user(locale, user),
            )
        return ExecutorResult(
            text=t(locale, "nlu_help_storyline_tracking"),
            reply_markup=get_storyline_tracking_keyboard(locale),
        )
    return ExecutorResult(
        text=t(locale, "help_message"),
        reply_markup=_main_keyboard_for_user(locale, user, getattr(user, "username", None)),
        parse_mode="Markdown",
    )


def _describe_pending_action(intent: str, slots: dict[str, object], locale: str) -> str:
    if intent == "add_subscriptions":
        links = "\n".join(slots.get("links") or [])
        return t(locale, "nlu_confirm_add_subscriptions", links=links)
    if intent == "remove_subscriptions":
        links = "\n".join(slots.get("links") or [])
        return t(locale, "nlu_confirm_remove_subscriptions", links=links)
    if intent == "set_feed_filter":
        return t(locale, "nlu_confirm_set_filter", filter_label=_feed_filter_label(str(slots.get("filter_mode") or "all"), locale))
    if intent == "toggle_forwarding":
        return t(locale, "nlu_confirm_toggle_forwarding", state=t(locale, "enabled") if bool(slots.get("value_bool")) else t(locale, "disabled"))
    if intent == "toggle_summary":
        return t(locale, "nlu_confirm_toggle_summary", state=t(locale, "enabled") if bool(slots.get("value_bool")) else t(locale, "disabled"))
    if intent == "set_global_instruction_filter":
        return t(locale, "nlu_confirm_set_global_instruction_filter", prompt_text=str(slots.get("prompt_text") or ""))
    if intent == "digest_enable":
        return t(locale, "nlu_confirm_digest_enable")
    if intent == "digest_disable":
        return t(locale, "nlu_confirm_digest_disable")
    if intent == "digest_send_now":
        return t(locale, "nlu_confirm_digest_send_now")
    if intent == "digest_set_time":
        return t(locale, "nlu_confirm_digest_set_time", time_hhmm=str(slots.get("time_hhmm") or ""))
    if intent == "digest_set_offset":
        return t(locale, "nlu_confirm_digest_set_offset", utc_offset=str(slots.get("utc_offset") or ""))
    if intent == "set_language":
        return t(locale, "nlu_confirm_set_language", language_code=str(slots.get("language_code") or ""))
    return t(locale, "nlu_confirm_generic")


def _build_nlu_clarification_response(nlu_result: NLUResult, locale: str) -> tuple[str, types.ReplyKeyboardMarkup | None]:
    if nlu_result.slots.get("clarification_kind") == "importance_filter":
        return t(locale, "nlu_clarify_importance_filter"), get_importance_filter_clarification_keyboard(locale)
    if nlu_result.intent == "set_feed_filter":
        return t(locale, "nlu_clarify_filter_mode"), get_filter_clarification_keyboard(locale)
    return nlu_result.clarify_question or t(locale, "nlu_clarify_generic"), None


async def _finalize_subscription_request_progress(
    message: types.Message,
    *,
    request_ids: list[tuple[int, str]],
    locale: str,
) -> None:
    for request_id, group_link in request_ids:
        msg = None
        try:
            msg = await message.answer(t(locale, "subscription_queued_initial", group_link=group_link))
        except Exception as exc:
            log.warning("subscription.queue_message_failed", request_id=request_id, group_link=group_link, error=repr(exc))
        async for session in get_session():
            req = await session.get(SubscriptionRequest, request_id)
            if req:
                req.notify_chat_id = getattr(getattr(message, "chat", None), "id", None) or getattr(getattr(message, "from_user", None), "id", None)
                req.notify_message_id = getattr(msg, "message_id", None) if msg else None
                await session.commit()
            break


async def _enqueue_subscription_request_jobs(request_ids: list[tuple[int, str]]) -> None:
    for request_id, group_link in request_ids:
        log.info(
            "subscription request uses telegram operation queue",
            request_id=request_id,
            group_link=group_link,
        )


def _render_subscription_change_result_text(
    result: SubscriptionChangeResult,
    locale: str,
    *,
    include_queued: bool = True,
) -> str:
    response_lines: list[str] = []
    if result.subscribed:
        response_lines.append(t(locale, "subscriptions_connected", links="\n".join(result.subscribed)))
    if include_queued and result.queued:
        response_lines.append(t(locale, "subscriptions_queued", links="\n".join(result.queued)))
    if result.unsubscribed:
        response_lines.append(t(locale, "unsubscribed_from", links="\n".join(result.unsubscribed)))
    if result.already_subscribed:
        response_lines.append(t(locale, "already_subscribed", links="\n".join(result.already_subscribed)))
    if result.missing:
        response_lines.append(t(locale, "subscriptions_not_found", links="\n".join(result.missing)))
    if result.already_queued:
        response_lines.append(t(locale, "already_queued", links="\n".join(result.already_queued)))
    if result.invalid_lines:
        response_lines.append(t(locale, "invalid_lines", lines="\n".join(result.invalid_lines)))
    return "\n\n".join(response_lines) if response_lines else t(locale, "subscriptions_nothing_changed")


async def _apply_subscription_links(
    session,
    *,
    user: User,
    links: list[str],
    locale: str,
    action_mode: str,
    invalid_lines: list[str] | None = None,
) -> SubscriptionChangeResult:
    groups = _dedupe_groups([parse_group_input(item) for item in links if item])
    valid_links = [group_link for _, group_link in groups]
    invalid = list(invalid_lines or [])

    existing_result = await session.execute(
        select(Community.link, UserCommunity)
        .join(UserCommunity, UserCommunity.community_id == Community.id)
        .where(UserCommunity.user_id == user.id, Community.link.in_(valid_links))
    )
    existing_rels_by_link = {link: rel for link, rel in existing_result.all()}
    existing_links = set(existing_rels_by_link)

    stale_before = datetime.now(timezone.utc) - timedelta(minutes=SUBSCRIPTION_STALE_MINUTES)
    await session.execute(
        update(SubscriptionRequest)
        .where(
            SubscriptionRequest.user_id == user.id,
            SubscriptionRequest.status.in_(SUBSCRIPTION_ACTIVE_STATUSES),
            SubscriptionRequest.updated_at < stale_before,
        )
        .values(status="failed", error="stale request expired")
    )
    pending_result = await session.execute(
        select(SubscriptionRequest.group_link)
        .where(
            SubscriptionRequest.user_id == user.id,
            SubscriptionRequest.group_link.in_(valid_links),
            SubscriptionRequest.status.in_(SUBSCRIPTION_ACTIVE_STATUSES),
        )
    )
    pending_links = {row[0] for row in pending_result.all()}

    if action_mode in {"toggle", "add"}:
        state, req, plan = await recalculate_user_billing_state(session, user.id)
        allowed_subscriptions = int(req.allowed_subscriptions)
        current_active = int(req.active_subscriptions)
        promo_extra_groups = int(getattr(req, "promo_extra_groups", 0) or 0)
        premium_limit = int(getattr(req, "premium_limit", getattr(plan, "pack_size", 100)) or 100)
        hard_limit = premium_limit + promo_extra_groups
        to_add_candidates = 0
        to_remove_candidates = 0
        for group_link in valid_links:
            if action_mode == "add":
                if group_link not in existing_links and group_link not in pending_links:
                    to_add_candidates += 1
            elif action_mode == "toggle":
                if group_link in existing_links:
                    to_remove_candidates += 1
                elif group_link not in pending_links:
                    to_add_candidates += 1
        projected_active = current_active - to_remove_candidates + to_add_candidates
        if projected_active > hard_limit:
            raise RuntimeError(t(locale, "subscription_hard_limit_reached", support_email=PREMIUM_SUPPORT_EMAIL))
        if projected_active > allowed_subscriptions:
            due_rub = max(0, int(state.required_amount_rub or 0) - int(state.paid_amount_rub or 0))
            raise RuntimeError(
                t(
                    locale,
                    "subscription_limit_exceeded",
                    current_active=current_active,
                    allowed_subscriptions=allowed_subscriptions,
                    due_rub=due_rub,
                )
            )

    subscribed: list[str] = []
    queued: list[str] = []
    unsubscribed: list[str] = []
    already_queued: list[str] = []
    already_subscribed: list[str] = []
    missing: list[str] = []
    request_ids: list[tuple[int, str]] = []

    for group_handle, group_link in groups:
        exists = group_link in existing_links
        pending = group_link in pending_links

        if action_mode == "remove":
            if exists:
                await session.delete(existing_rels_by_link[group_link])
                unsubscribed.append(group_link)
                audit("subscription.action_resolved", user_id=user.id, telegram_id=user.telegram_id, group_link=group_link, action="unsubscribe")
            elif pending:
                already_queued.append(group_link)
                audit("subscription.action_resolved", user_id=user.id, telegram_id=user.telegram_id, group_link=group_link, action="already_queued")
            else:
                missing.append(group_link)
            continue

        if exists:
            if action_mode == "toggle":
                await session.delete(existing_rels_by_link[group_link])
                unsubscribed.append(group_link)
                audit("subscription.action_resolved", user_id=user.id, telegram_id=user.telegram_id, group_link=group_link, action="unsubscribe")
            else:
                already_subscribed.append(group_link)
                audit("subscription.action_resolved", user_id=user.id, telegram_id=user.telegram_id, group_link=group_link, action="already_subscribed")
            continue

        if pending:
            already_queued.append(group_link)
            audit("subscription.action_resolved", user_id=user.id, telegram_id=user.telegram_id, group_link=group_link, action="already_queued")
            continue

        known_community = await find_existing_active_community(
            session,
            group_handle=group_handle,
            group_link=group_link,
        )
        if known_community:
            created = await ensure_user_community(
                session,
                user_id=user.id,
                community_id=known_community.id,
            )
            await ensure_community_aliases(
                session,
                community_id=known_community.id,
                group_handle=group_handle,
                group_link=group_link,
                username=getattr(known_community, "username", None),
            )
            if created:
                subscribed.append(group_link)
                audit(
                    "subscription.action_resolved",
                    user_id=user.id,
                    telegram_id=user.telegram_id,
                    group_link=group_link,
                    community_id=known_community.id,
                    action="subscribe_existing_community",
                )
            else:
                already_subscribed.append(group_link)
                audit(
                    "subscription.action_resolved",
                    user_id=user.id,
                    telegram_id=user.telegram_id,
                    group_link=group_link,
                    community_id=known_community.id,
                    action="already_subscribed",
                )
            continue

        req = SubscriptionRequest(
            user_id=user.id,
            group_link=group_link,
            group_handle=group_handle,
            status="queued",
            userbot_slot=await choose_userbot_slot_for_new_group(session),
        )
        session.add(req)
        queued.append(group_link)
        await session.flush()
        await ensure_resolve_join_operation(session, req)
        request_ids.append((req.id, group_link))
        audit("subscription.action_resolved", user_id=user.id, telegram_id=user.telegram_id, group_link=group_link, action="subscribe")

    await session.commit()

    return SubscriptionChangeResult(
        subscribed=subscribed,
        queued=queued,
        unsubscribed=unsubscribed,
        already_queued=already_queued,
        already_subscribed=already_subscribed,
        missing=missing,
        invalid_lines=invalid,
        request_ids=request_ids,
    )

def publish_telethon_event(payload: dict) -> None:
    # payload: только примитивы (str/int/bool), иначе словишь "not JSON serializable"
    with Connection(CELERY_BROKER_URL) as conn:
        ex = Exchange("newshub.events", type="direct", durable=True)
        q = Queue(
            name=TELETHON_QUEUE,
            exchange=ex,
            routing_key=TELETHON_QUEUE,
            durable=True,
        )
        Producer(conn).publish(
            payload,
            exchange=ex,
            routing_key=TELETHON_QUEUE,
            serializer="json",
            retry=True,
            declare=[ex, q],
        )

async def publish_telethon_event_async(payload: dict) -> None:
    await asyncio.to_thread(publish_telethon_event, payload)


async def publish_subscription_request_async(request_id: int) -> None:
    await asyncio.to_thread(
        celery_app.send_task,
        "app.tasks.handle_subscription_request_task",
        kwargs={"request_id": request_id},
    )
    
@dp.callback_query(F.data.startswith("dig:"))
async def on_digest_page(cb: CallbackQuery):
    try:
        _, run_id_s, page_s = (cb.data or "").split(":")
        run_id = int(run_id_s)
        page = int(page_s)
    except Exception:
        await cb.answer(t(await _resolve_callback_locale(cb), "invalid_digest_button"))
        return

    async for session in get_session():
        run = await session.get(DigestRun, run_id)
        break

    pages = (getattr(run, "pages", None) or []) if run else []
    if not pages:
        await cb.answer(t(await _resolve_callback_locale(cb), "digest_stale"))
        return

    page = max(0, min(page, len(pages) - 1))
    text = pages[page]
    locale = await _resolve_callback_locale(cb)
    kb = build_digest_pager_kb(run_id, page=page, total=len(pages), locale=locale)

    await cb.message.edit_text(
        text,
        parse_mode="HTML",
        disable_web_page_preview=True,
        reply_markup=kb,
    )
    await cb.answer()

@dp.callback_query(F.data == "noop")
async def on_noop(cb: CallbackQuery):
    await cb.answer()


@dp.callback_query(F.data.startswith("daily_msk:"))
async def on_daily_msk_preset(callback: types.CallbackQuery):
    value = (callback.data or "").split(":", 1)[1] if ":" in (callback.data or "") else ""
    parsed = parse_hhmm(value)
    if not parsed:
        await callback.answer(t(await _resolve_callback_locale(callback), "digest_time_invalid"), show_alert=True)
        return
    hour, minute = parsed

    telegram_id = getattr(getattr(callback, "from_user", None), "id", None)
    if telegram_id is None:
        await callback.answer()
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_callback(user, callback)
        if not user:
            await callback.answer(t(locale, "start_required_register"), show_alert=True)
            return

        settings = await session.get(UserDigestSettings, user.id)
        if not settings:
            settings = UserDigestSettings(user_id=user.id, enabled=True)
            session.add(settings)
        settings.enabled = True
        settings.utc_offset_minutes = DIGEST_MSK_OFFSET_MINUTES
        settings.timezone = "Europe/Moscow"

        slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
        slot = slot_res.scalar_one_or_none()
        if not slot:
            slot = UserDigestSlot(user_id=user.id, hour=hour, minute=minute, days_mask=127, is_active=True)
            session.add(slot)
        else:
            slot.hour = hour
            slot.minute = minute
            slot.is_active = True

        next_run_at = compute_next_run_at_utc_from_offset(DIGEST_MSK_OFFSET_MINUTES, hour, minute)
        slot.next_run_at = next_run_at
        await session.commit()

    offset = timezone(timedelta(minutes=DIGEST_MSK_OFFSET_MINUTES))
    local_dt = next_run_at.astimezone(offset)
    DIGEST_SETUP_WAIT_OFFSET.discard(telegram_id)
    DIGEST_SETUP_WAIT_TIME.discard(telegram_id)
    await callback.message.answer(
        t(
            locale,
            "digest_ready",
            offset=DIGEST_MSK_OFFSET_MINUTES / 60,
            hour=hour,
            minute=minute,
            local_dt=f"{local_dt:%Y-%m-%d %H:%M}",
        ),
        reply_markup=get_digest_time_keyboard(locale),
    )
    await callback.answer()


@dp.callback_query(F.data.startswith("instrgrp:"))
async def on_instruction_group_picker(callback: types.CallbackQuery):
    try:
        _, action, mode, value_raw = (callback.data or "").split(":")
        value = int(value_raw)
    except Exception:
        await callback.answer(t(await _resolve_callback_locale(callback), "instruction_picker_expired"), show_alert=True)
        return

    user_id = getattr(getattr(callback, "from_user", None), "id", None)
    if user_id is None:
        await callback.answer()
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == user_id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_callback(user, callback)
        if not user:
            await callback.answer(t(locale, "start_required_register"), show_alert=True)
            return

        if mode == "set":
            target_ids = INSTRUCTION_GROUP_CHOICES.get(user_id) or []
            items = await get_subscribed_instruction_targets(session, user_id=user.id)
            waiting_set = INSTRUCTION_GROUP_SELECT_USERS
        else:
            target_ids = INSTRUCTION_GROUP_DISABLE_CHOICES.get(user_id) or []
            items = await get_rule_bound_communities(session, user_id=user.id)
            waiting_set = INSTRUCTION_GROUP_DISABLE_SELECT_USERS

        if not target_ids:
            await callback.answer(t(locale, "instruction_picker_expired"), show_alert=True)
            return

        items = [item for item in items if int(getattr(item, "id", 0) or 0) in set(target_ids)]
        if not items:
            await callback.answer(t(locale, "instruction_picker_expired"), show_alert=True)
            return

        if action == "page":
            await callback.message.edit_text(
                t(locale, "instruction_group_prompt_choose" if mode == "set" else "instruction_group_disable_choose"),
                reply_markup=_build_instruction_group_picker_keyboard(items, mode=mode, page=value, locale=locale),
            )
            await callback.answer()
            return

        if action == "back":
            _clear_instruction_states(user_id)
            INSTRUCTION_MENU_USERS.add(user_id)
            await callback.message.edit_text(
                await _render_instruction_menu_text(session, user=user, locale=locale),
                parse_mode="HTML",
            )
            await callback.answer()
            return

        if action != "pick" or int(value) not in set(target_ids):
            await callback.answer(t(locale, "instruction_invalid_choice"), show_alert=True)
            return

        selected = next((item for item in items if int(getattr(item, "id", 0) or 0) == int(value)), None)
        if not selected:
            await callback.answer(t(locale, "instruction_invalid_choice"), show_alert=True)
            return

        if mode == "set":
            waiting_set.discard(user_id)
            INSTRUCTION_GROUP_PROMPT_WAIT_USERS.add(user_id)
            INSTRUCTION_GROUP_PROMPT_TARGETS[user_id] = int(value)
            label = getattr(selected, "link", None) or getattr(selected, "name", None) or f"community:{value}"
            await callback.message.edit_text(
                t(locale, "instruction_group_prompt_request_selected", group=label),
            )
            await callback.answer()
            return

        changed = await disable_community_instruction_rule(
            session,
            user_id=user.id,
            community_id=int(value),
        )
        await session.commit()
        _clear_instruction_states(user_id)
        INSTRUCTION_MENU_USERS.add(user_id)
        text = t(locale, "instruction_group_disabled" if changed else "instruction_group_rules_missing")
        text += "\n\n" + await _render_instruction_menu_text(session, user=user, locale=locale)
        await callback.message.edit_text(text, parse_mode="HTML")
        await callback.answer()
        return
    
@dp.message(Command("digest"))
@dp.message(_message_matches("digest"))
async def show_digest_menu(message: types.Message):
    _clear_settings_submenu_states(message.from_user.id)
    DIGEST_MENU_USERS.add(message.from_user.id)
    DIGEST_TIME_MENU_USERS.discard(message.from_user.id)
    DIGEST_SETUP_WAIT_OFFSET.discard(message.from_user.id)
    DIGEST_SETUP_WAIT_TIME.discard(message.from_user.id)

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required"))
            return
        locale = get_user_locale(user)

        settings = await session.get(UserDigestSettings, user.id)
        slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
        slot = slot_res.scalar_one_or_none()

        enabled = bool(settings.enabled) if settings else False
        offset_min = getattr(settings, "utc_offset_minutes", None) if settings else None

        lines = [
            t(locale, "digest_description"),
            "",
            f"{t(locale, 'digest_settings_state')}: {t(locale, 'enabled') if enabled else t(locale, 'disabled')}",
        ]
        if offset_min is not None:
            lines.append(f"{t(locale, 'offset_set')}: UTC{offset_min/60:+g}")
        else:
            lines.append(f"{t(locale, 'offset_set')}: {t(locale, 'offset_not_set')}")
        if settings and getattr(settings, "timezone", None):
            lines.append(f"{t(locale, 'timezone')}: {settings.timezone}")
        if slot:
            lines.append(f"{t(locale, 'time')}: {slot.hour:02d}:{slot.minute:02d}")
        else:
            lines.append(f"{t(locale, 'time')}: {t(locale, 'offset_not_set')}")

        await message.answer(
            "\n".join(lines),
            reply_markup=get_digest_keyboard(locale, enabled=enabled, time_label=_digest_time_label(slot)),
        )
        return

@dp.message(_message_matches("digest_daily"))
async def digest_daily_enable(message: types.Message):
    DIGEST_TIME_MENU_USERS.add(message.from_user.id)
    DIGEST_SETUP_WAIT_OFFSET.add(message.from_user.id)
    DIGEST_SETUP_WAIT_TIME.discard(message.from_user.id)

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required"))
            return

        settings = await session.get(UserDigestSettings, user.id)
        if not settings:
            settings = UserDigestSettings(user_id=user.id, enabled=True)
            session.add(settings)
        settings.enabled = True
        await session.commit()
        locale = get_user_locale(user)

        await message.answer(
            t(locale, "digest_daily_enabled"),
        )
        return

@dp.message(_message_matches("digest_disable", "digest_toggle_on", "digest_toggle_off"))
async def digest_disable_btn(message: types.Message):
    DIGEST_TIME_MENU_USERS.discard(message.from_user.id)
    DIGEST_SETUP_WAIT_OFFSET.discard(message.from_user.id)
    DIGEST_SETUP_WAIT_TIME.discard(message.from_user.id)

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required"))
            return

        settings = await session.get(UserDigestSettings, user.id)
        currently_enabled = bool(settings.enabled) if settings else False
        new_enabled = not currently_enabled
        if not settings:
            settings = UserDigestSettings(user_id=user.id, enabled=new_enabled)
            session.add(settings)
        else:
            settings.enabled = new_enabled
        await session.commit()
        locale = get_user_locale(user)

        if new_enabled:
            DIGEST_TIME_MENU_USERS.add(message.from_user.id)
            await message.answer(
                t(locale, "digest_setup_intro"),
                reply_markup=get_digest_time_keyboard(locale),
            )
            return

        await message.answer(
            t(locale, "digest_disabled"),
            reply_markup=get_digest_keyboard(locale, enabled=False, time_label=None),
        )
        return

@dp.message(_message_matches("digest_setup_time"))
async def digest_setup_start(message: types.Message):
    DIGEST_TIME_MENU_USERS.add(message.from_user.id)
    DIGEST_SETUP_WAIT_TIME.discard(message.from_user.id)
    await message.answer(
        t(_button_locale_from_message(message), "digest_setup_intro"),
        reply_markup=get_digest_time_keyboard(_button_locale_from_message(message)),
    )

@dp.message(
    lambda m: m.from_user and m.from_user.id in DIGEST_SETUP_WAIT_OFFSET 
        and (m.text and not m.text.startswith("/") 
            and m.text.casefold() not in _reserved_digest_variants())
    )
async def digest_setup_offset(message: types.Message):
    raw_value = (message.text or "").strip()
    offset_min = parse_utc_offset_to_minutes(raw_value)
    tz_name = None
    if offset_min is None:
        tz_name = resolve_timezone_from_city(raw_value)
        if tz_name:
            offset_min = current_offset_minutes_for_tz(tz_name)
    if offset_min is None:
        await message.answer(
            t(await _resolve_message_locale(message), "digest_offset_invalid")
        )
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required"))
            return

        settings = await session.get(UserDigestSettings, user.id)
        if not settings:
            settings = UserDigestSettings(user_id=user.id, enabled=True)
            session.add(settings)
        settings.utc_offset_minutes = offset_min
        if tz_name:
            settings.timezone = tz_name
        await session.commit()
        locale = get_user_locale(user)

    DIGEST_SETUP_WAIT_OFFSET.discard(message.from_user.id)
    DIGEST_SETUP_WAIT_TIME.add(message.from_user.id)
    if tz_name:
        await message.answer(
            t(locale, "digest_enter_time_city", raw_value=raw_value, tz_name=tz_name, offset=offset_min / 60)
        )
    else:
        await message.answer(t(locale, "digest_enter_time", offset=offset_min / 60))

    await message.answer(
        t(locale, "digest_pick_time"),
        reply_markup=get_digest_msk_presets_keyboard(),
    )

@dp.message(
    lambda m: m.from_user and m.from_user.id in DIGEST_SETUP_WAIT_TIME 
        and (m.text and not m.text.startswith("/") 
            and m.text.casefold() not in _reserved_digest_variants())
    )
async def digest_setup_time(message: types.Message):
    parsed = parse_hhmm(message.text)
    if not parsed:
        await message.answer(t(await _resolve_message_locale(message), "digest_time_invalid"))
        return
    hour, minute = parsed

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required"))
            return

        settings = await session.get(UserDigestSettings, user.id)
        if not settings or settings.utc_offset_minutes is None:
            await message.answer(t(get_user_locale(user), "digest_need_offset"))
            return

        next_run_at = compute_next_run_at_utc_from_offset(settings.utc_offset_minutes, hour, minute)

        slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
        slot = slot_res.scalar_one_or_none()
        if not slot:
            slot = UserDigestSlot(user_id=user.id, hour=hour, minute=minute, days_mask=127, is_active=True)
            session.add(slot)
        else:
            slot.hour = hour
            slot.minute = minute
            slot.is_active = True

        slot.next_run_at = next_run_at
        await session.commit()

        offset = timezone(timedelta(minutes=settings.utc_offset_minutes))
        local_dt = next_run_at.astimezone(offset)

    DIGEST_SETUP_WAIT_OFFSET.discard(message.from_user.id)
    DIGEST_SETUP_WAIT_TIME.discard(message.from_user.id)
    locale = get_user_locale(user)

    await message.answer(
        t(
            locale,
            "digest_ready",
            offset=settings.utc_offset_minutes / 60,
            hour=hour,
            minute=minute,
            local_dt=f"{local_dt:%Y-%m-%d %H:%M}",
        ),
        reply_markup=get_digest_time_keyboard(locale),
    )


@dp.message(_message_matches("digest_send_now"))
async def digest_send_now(message: types.Message):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required"))
            return

        slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
        slot = slot_res.scalar_one_or_none()
        if not slot:
            # Технический слот для ручного запуска (не влияет на расписание отправок).
            slot = UserDigestSlot(user_id=user.id, hour=0, minute=0, days_mask=127, is_active=False)
            session.add(slot)
            await session.flush()

        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(hours=24)

        run = DigestRun(
            slot_id=slot.id,
            user_id=user.id,
            period_start=period_start,
            period_end=period_end,
            status="queued",
        )
        session.add(run)
        await session.commit()

        await asyncio.to_thread(
            celery_app.send_task,
            "app.tasks.build_and_send_digest",
            args=[run.id],
            queue="digest_queue",
        )

        await message.answer(
            t(get_user_locale(user), "digest_collecting_now"),
            reply_markup=get_digest_time_keyboard(get_user_locale(user)),
        )
        return

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    setattr(message, "_newshub_skip_survey_prompt", True)
    async for session in get_session():
        user_id = message.from_user.id
        result = await session.execute(select(User).where(User.telegram_id == user_id))
        user = result.scalar_one_or_none()
        created_new_user = False

        if not user:
            locale = locale_from_telegram_language_code(getattr(message.from_user, "language_code", None))
            new_user = User(
                telegram_id=user_id,
                first_name=message.from_user.first_name,
                last_name=message.from_user.last_name,
                username=message.from_user.username,
                email=f"{user_id}@telegram.com",
                language_code=locale,
                summary_enabled=True,
            )
            session.add(new_user)
            await session.commit()
            user = new_user
            session.add(UserNewsTimeSurvey(user_id=new_user.id))
            await session.commit()
            created_new_user = True
            log.info("Added new user", user_id=user_id)
        else:
            log.info("User already exists", user_id=user_id)
            locale = get_user_locale(user)
            user.first_name = message.from_user.first_name
            user.last_name = message.from_user.last_name
            user.username = message.from_user.username
            await session.commit()

        # In plus-bot runtime, /start confirms that user opened the second bot.
        if BOT_ROLE == "plus":
            prefs = await get_or_create_delivery_prefs(session, int(user.id))
            was_connected = bool(getattr(prefs, "plus_bot_connected_at", None))
            if not was_connected:
                prefs.plus_bot_connected_at = utcnow()
                await session.commit()
                if MAIN_BOT_TOKEN:
                    main_bot = Bot(token=MAIN_BOT_TOKEN)
                    try:
                        await main_bot.send_message(
                            chat_id=message.from_user.id,
                            text=(
                                "Подключение второго бота подтверждено.\n"
                                "Теперь можно включить доставку дайджестов и/или "
                                "отслеживания сюжетов в отдельный диалог."
                            ),
                        )
                    except Exception as exc:
                        log.warning(
                            "plus_connect.main_bot_confirmation_failed",
                            telegram_id=message.from_user.id,
                            error=repr(exc),
                        )
                    finally:
                        await main_bot.session.close()

        welcome_key = "plus_welcome_message" if BOT_ROLE == "plus" else "welcome_message"
        await message.answer(
            t(locale, welcome_key),
            parse_mode="Markdown",
            reply_markup=_main_keyboard_for_user(locale, user, message.from_user.username),
        )
        if created_new_user and await _user_has_any_subscription(session, user=user):
            await _send_persistent_survey_message(
                message,
                t(locale, "news_time_baseline_question"),
                reply_markup=get_news_time_survey_keyboard(locale),
            )
            survey = await _get_news_time_survey(session, user_id=user.id)
            if survey and getattr(survey, "baseline_asked_at", None) is None:
                survey.baseline_asked_at = utcnow()
                await session.commit()
            NEWS_TIME_BASELINE_WAIT_USERS.add(int(user.telegram_id))
        break


@dp.message(_message_matches(*NEWS_TIME_OPTION_KEYS))
async def handle_news_time_answer(message: types.Message):
    setattr(message, "_newshub_skip_survey_prompt", True)
    answer = parse_news_time_answer(getattr(message, "text", None))
    locale = _button_locale_from_message(message)
    if answer is None:
        await message.answer(
            t(locale, "main_menu_title"),
            reply_markup=_main_keyboard_for_user(locale, telegram_username=getattr(message.from_user, "username", None)),
        )
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return
        locale = get_user_locale(user)
        survey = await _get_news_time_survey(session, user_id=user.id)
        if not survey:
            await message.answer(
                t(locale, "main_menu_title"),
                reply_markup=_main_keyboard_for_user(locale, user, getattr(message.from_user, "username", None)),
            )
            return

        now = utcnow()
        if survey.baseline_answered_at is None:
            survey.baseline_answer = answer.label
            survey.baseline_minutes = answer.minutes
            survey.baseline_answered_at = now
            base_ts = getattr(user, "created_at", None) or now
            survey.followup_due_at = followup_due_at(base_ts)
            NEWS_TIME_BASELINE_WAIT_USERS.discard(int(user.telegram_id))
            await session.commit()
            audit(
                "news_time_survey.baseline_answered",
                telegram_id=user.telegram_id,
                user_id=user.id,
                answer=answer.label,
                minutes=answer.minutes,
            )
            await message.answer(
                t(locale, "news_time_thanks"),
                reply_markup=_main_keyboard_for_user(locale, user, getattr(message.from_user, "username", None)),
            )
            return

        if survey.time_saved_answer == "yes" and survey.current_answered_at is None:
            survey.current_answer = answer.label
            survey.current_minutes = answer.minutes
            survey.current_answered_at = now
            await session.commit()
            audit(
                "news_time_survey.current_answered",
                telegram_id=user.telegram_id,
                user_id=user.id,
                answer=answer.label,
                minutes=answer.minutes,
            )
            await message.answer(
                t(locale, "news_time_thanks"),
                reply_markup=_main_keyboard_for_user(locale, user, getattr(message.from_user, "username", None)),
            )
            return

        await message.answer(
            t(locale, "main_menu_title"),
            reply_markup=_main_keyboard_for_user(locale, user, getattr(message.from_user, "username", None)),
        )
        return


@dp.callback_query(F.data.startswith("news_time:opt:"))
async def handle_news_time_answer_callback(callback: types.CallbackQuery):
    if not callback.message:
        await callback.answer()
        return
    setattr(callback.message, "_newshub_skip_survey_prompt", True)

    option_key = str(callback.data or "").split(":", 2)[-1]
    locale_hint = _button_locale_from_callback(callback)
    if option_key not in NEWS_TIME_OPTION_KEYS:
        await callback.answer()
        return

    answer = parse_news_time_answer(button_text(option_key, locale_hint))
    if answer is None:
        await callback.answer()
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == callback.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await callback.answer()
            await callback.message.answer(t(locale_hint, "start_required_register"))
            return

        locale = get_user_locale(user)
        survey = await _get_news_time_survey(session, user_id=user.id)
        if not survey:
            survey = UserNewsTimeSurvey(user_id=user.id)
            session.add(survey)
            await session.flush()

        now = utcnow()
        if survey.baseline_answered_at is None:
            survey.baseline_answer = answer.label
            survey.baseline_minutes = answer.minutes
            survey.baseline_answered_at = now
            base_ts = getattr(user, "created_at", None) or now
            survey.followup_due_at = followup_due_at(base_ts)
            NEWS_TIME_BASELINE_WAIT_USERS.discard(int(user.telegram_id))
            await session.commit()
            audit(
                "news_time_survey.baseline_answered",
                telegram_id=user.telegram_id,
                user_id=user.id,
                answer=answer.label,
                minutes=answer.minutes,
            )
            await callback.answer()
            await callback.message.answer(
                t(locale, "news_time_thanks"),
                reply_markup=_main_keyboard_for_user(locale, user, getattr(callback.from_user, "username", None)),
            )
            return

        if survey.time_saved_answer == "yes" and survey.current_answered_at is None:
            survey.current_answer = answer.label
            survey.current_minutes = answer.minutes
            survey.current_answered_at = now
            await session.commit()
            audit(
                "news_time_survey.current_answered",
                telegram_id=user.telegram_id,
                user_id=user.id,
                answer=answer.label,
                minutes=answer.minutes,
            )
            await callback.answer()
            await callback.message.answer(
                t(locale, "news_time_thanks"),
                reply_markup=_main_keyboard_for_user(locale, user, getattr(callback.from_user, "username", None)),
            )
            return

        await callback.answer()
        await callback.message.answer(
            t(locale, "news_time_thanks"),
            reply_markup=_main_keyboard_for_user(locale, user, getattr(callback.from_user, "username", None)),
        )
        return

@dp.callback_query(F.data.startswith("news_time:saved:"))
async def handle_news_time_saved_answer_callback(callback: types.CallbackQuery):
    if not callback.message:
        await callback.answer()
        return
    audit(
        "news_time_survey.saved_callback_received",
        callback_data=str(callback.data or ""),
        telegram_id=getattr(getattr(callback, "from_user", None), "id", None),
        message_id=getattr(getattr(callback, "message", None), "message_id", None),
    )
    setattr(callback.message, "_newshub_skip_survey_prompt", True)

    saved_answer = str(callback.data or "").split(":", 2)[-1]
    if saved_answer not in {"yes", "no"}:
        await callback.answer()
        return

    locale_hint = _button_locale_from_callback(callback)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == callback.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await callback.answer()
            await callback.message.answer(t(locale_hint, "start_required_register"))
            return

        locale = get_user_locale(user)
        survey = await _get_news_time_survey(session, user_id=user.id)
        if not survey or survey.baseline_answered_at is None or survey.time_saved_answered_at is not None:
            await callback.answer()
            await callback.message.answer(
                t(locale, "main_menu_title"),
                reply_markup=_main_keyboard_for_user(locale, user, getattr(callback.from_user, "username", None)),
            )
            return

        now = utcnow()
        survey.time_saved_answer = saved_answer
        survey.time_saved_answered_at = now
        await session.commit()
        audit(
            "news_time_survey.time_saved_answered",
            telegram_id=user.telegram_id,
            user_id=user.id,
            answer=saved_answer,
        )

        await callback.answer()
        if saved_answer == "yes":
            await _send_persistent_survey_message(
                callback.message,
                t(locale, "news_time_current_question"),
                reply_markup=get_news_time_survey_keyboard(locale),
            )
            asked_at = utcnow()
            if survey.current_asked_at is None:
                survey.current_asked_at = asked_at
                audit("news_time_survey.current_asked", telegram_id=user.telegram_id, user_id=user.id)
            else:
                survey.current_retry_asked_at = asked_at
                audit("news_time_survey.current_retry_asked", telegram_id=user.telegram_id, user_id=user.id)
            await session.commit()
        else:
            await callback.message.answer(
                t(locale, "news_time_thanks"),
                reply_markup=_main_keyboard_for_user(locale, user, getattr(callback.from_user, "username", None)),
            )
        return


@dp.message(_message_matches("news_time_saved_yes", "news_time_saved_no"))
async def handle_news_time_saved_answer(message: types.Message):
    setattr(message, "_newshub_skip_survey_prompt", True)
    saved_answer = parse_time_saved_answer(getattr(message, "text", None))
    locale = _button_locale_from_message(message)
    if saved_answer is None:
        await message.answer(
            t(locale, "main_menu_title"),
            reply_markup=_main_keyboard_for_user(locale, telegram_username=getattr(message.from_user, "username", None)),
        )
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return
        locale = get_user_locale(user)
        survey = await _get_news_time_survey(session, user_id=user.id)
        if not survey or survey.baseline_answered_at is None or survey.time_saved_answered_at is not None:
            await message.answer(
                t(locale, "main_menu_title"),
                reply_markup=_main_keyboard_for_user(locale, user, getattr(message.from_user, "username", None)),
            )
            return

        now = utcnow()
        survey.time_saved_answer = saved_answer
        survey.time_saved_answered_at = now
        await session.commit()
        audit(
            "news_time_survey.time_saved_answered",
            telegram_id=user.telegram_id,
            user_id=user.id,
            answer=saved_answer,
        )
        if saved_answer == "yes":
            await _send_persistent_survey_message(
                message,
                t(locale, "news_time_current_question"),
                reply_markup=get_news_time_survey_keyboard(locale),
            )
            asked_at = utcnow()
            if survey.current_asked_at is None:
                survey.current_asked_at = asked_at
                audit("news_time_survey.current_asked", telegram_id=user.telegram_id, user_id=user.id)
            else:
                survey.current_retry_asked_at = asked_at
                audit("news_time_survey.current_retry_asked", telegram_id=user.telegram_id, user_id=user.id)
            await session.commit()
        else:
            await message.answer(
                t(locale, "news_time_thanks"),
                reply_markup=_main_keyboard_for_user(locale, user, getattr(message.from_user, "username", None)),
            )
        return


@dp.callback_query(F.data.startswith("csi:score:"))
async def handle_csi_score_callback(callback: types.CallbackQuery):
    if not callback.message:
        await callback.answer()
        return
    setattr(callback, "_newshub_skip_survey_prompt", True)
    setattr(callback.message, "_newshub_skip_survey_prompt", True)

    raw = str(callback.data or "").split(":", 2)[-1]
    try:
        score = int(raw)
    except ValueError:
        await callback.answer()
        return
    if score < 1 or score > 10:
        await callback.answer()
        return

    locale_hint = _button_locale_from_callback(callback)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == callback.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await callback.answer()
            await callback.message.answer(t(locale_hint, "start_required_register"))
            return

        locale = get_user_locale(user)
        survey = await _get_csi_survey(session, user_id=user.id)
        if not survey or not is_csi_pending(survey):
            await callback.answer()
            await callback.message.answer(
                t(locale, "main_menu_title"),
                reply_markup=_main_keyboard_for_user(locale, user, getattr(callback.from_user, "username", None)),
            )
            return

        now = utcnow()
        survey.score = score
        survey.answered_at = now
        CSI_SCORE_WAIT_USERS.discard(int(user.telegram_id))
        await session.commit()
        audit(
            "csi_survey.answered",
            telegram_id=user.telegram_id,
            user_id=user.id,
            score=score,
        )

        await callback.answer()
        await callback.message.answer(
            t(locale, "csi_thanks"),
            reply_markup=_main_keyboard_for_user(locale, user, getattr(callback.from_user, "username", None)),
        )
        return


@dp.message(_should_handle_csi_score_input)
async def handle_csi_score_answer(message: types.Message):
    setattr(message, "_newshub_skip_survey_prompt", True)
    handled = await _try_handle_csi_score_answer(message)
    if handled:
        return
    locale = _button_locale_from_message(message)
    await message.answer(
        t(locale, "main_menu_title"),
        reply_markup=_main_keyboard_for_user(locale, telegram_username=getattr(message.from_user, "username", None)),
    )


@dp.message(Command("help"))
@dp.message(_message_matches("help"))
async def show_help(message: types.Message):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        help_key = "plus_help_message" if BOT_ROLE == "plus" else "help_message"
        await message.answer(
            t(locale, help_key),
            parse_mode="Markdown",
            reply_markup=_main_keyboard_for_user(locale, user, message.from_user.username),
        )
        return


@dp.message(Command("support"))
@dp.message(_message_matches("support_request"))
async def start_support_request(message: types.Message):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return
        _clear_settings_submenu_states(message.from_user.id)
        SUPPORT_REQUEST_WAIT_USERS.add(message.from_user.id)
        await message.answer(
            t(locale, "support_request_prompt"),
            reply_markup=_main_keyboard_for_user(locale, user, message.from_user.username),
        )
        return


@dp.message(Command("settings"))
@dp.message(_message_matches("settings"))
async def show_settings(message: types.Message):
    _clear_settings_submenu_states(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        await message.answer(
            t(locale, "settings_title"),
            reply_markup=_settings_keyboard_for_user(locale, user, message.from_user.username),
        )
        return


@dp.message(_message_matches("admin_panel"))
async def show_admin_panel(message: types.Message):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not _is_admin_user(user, message.from_user.username):
            await message.answer(t(locale, "admin_access_denied"))
            return
        ADMIN_PANEL_USERS.add(message.from_user.id)
        await message.answer(
            t(
                locale,
                "admin_panel_title",
                status=t(locale, "enabled") if bool(getattr(user, "storyline_debug_enabled", False)) else t(locale, "disabled"),
            ),
            reply_markup=get_admin_keyboard(locale, bool(getattr(user, "storyline_debug_enabled", False))),
        )
        return


@dp.message(Command("storylines"))
@dp.message(_message_matches("storyline_tracking"))
async def show_storyline_tracking_menu(message: types.Message):
    if not _storytracking_available(
        telegram_id=message.from_user.id,
        username=message.from_user.username,
    ):
        locale = await _resolve_message_locale(message)
        _clear_storyline_states(message.from_user.id)
        await message.answer(
            t(locale, "storytracking_disabled"),
            reply_markup=_settings_keyboard_for_user(locale, telegram_username=message.from_user.username),
        )
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return
        STORYLINE_MENU_USERS.add(message.from_user.id)
        STORYLINE_REMOVE_WAIT_USERS.discard(message.from_user.id)
        STORYLINE_REMOVE_CHOICES.pop(message.from_user.id, None)
        await message.answer(
            t(locale, "storyline_tracking_title"),
            reply_markup=get_storyline_tracking_keyboard(locale),
        )
        return


def _render_plus_delivery_menu_text(
    *,
    locale: str,
    prefs: UserDeliveryPreference | None,
) -> str:
    lines = [t(locale, "plus_delivery_menu_title")]
    digest_target = "второй бот" if bool(getattr(prefs, "digest_to_plus_bot", False)) else "основной бот"
    storyline_target = "второй бот" if bool(getattr(prefs, "storyline_to_plus_bot", False)) else "основной бот"
    lines.append(f"Дайджесты: {digest_target}")
    lines.append(f"Отслеживание сюжетов: {storyline_target}")
    return "\n".join(lines)


@dp.message(_message_matches("plus_delivery_menu"))
async def show_plus_delivery_menu(message: types.Message):
    _clear_settings_submenu_states(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return
        prefs = await session.get(UserDeliveryPreference, int(user.id))
        PLUS_DELIVERY_MENU_USERS.add(message.from_user.id)
        await message.answer(
            _render_plus_delivery_menu_text(locale=locale, prefs=prefs),
            reply_markup=get_plus_delivery_keyboard(
                locale,
                digest_to_plus_bot=bool(getattr(prefs, "digest_to_plus_bot", False)),
                storyline_to_plus_bot=bool(getattr(prefs, "storyline_to_plus_bot", False)),
            ),
        )
        return


@dp.message(_message_matches("storyline_remove"))
async def start_storyline_remove(message: types.Message):
    if not _storytracking_available(
        telegram_id=message.from_user.id,
        username=message.from_user.username,
    ):
        locale = await _resolve_message_locale(message)
        _clear_storyline_states(message.from_user.id)
        await message.answer(
            t(locale, "storytracking_disabled"),
            reply_markup=_settings_keyboard_for_user(locale, telegram_username=message.from_user.username),
        )
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return

        follows_result = await session.execute(
            select(UserStorylineFollow)
            .where(
                UserStorylineFollow.user_id == user.id,
                UserStorylineFollow.is_active.is_(True),
            )
            .order_by(UserStorylineFollow.created_at.asc(), UserStorylineFollow.id.asc())
        )
        follows = follows_result.scalars().all()
        if not follows:
            STORYLINE_MENU_USERS.add(message.from_user.id)
            STORYLINE_REMOVE_WAIT_USERS.discard(message.from_user.id)
            STORYLINE_REMOVE_CHOICES.pop(message.from_user.id, None)
            await message.answer(
                t(locale, "storyline_tracking_empty"),
                reply_markup=get_storyline_tracking_keyboard(locale),
            )
            return

        STORYLINE_MENU_USERS.add(message.from_user.id)
        STORYLINE_REMOVE_WAIT_USERS.add(message.from_user.id)
        STORYLINE_REMOVE_CHOICES[message.from_user.id] = [int(follow.id) for follow in follows]
        await message.answer(
            t(locale, "storyline_tracking_list", items=_render_storyline_follow_list(follows))
            + "\n\n"
            + t(locale, "storyline_tracking_remove_prompt"),
            reply_markup=_keyboard([button_text("storyline_clear_all", locale), button_text("back", locale)]),
        )
        return


@dp.message(lambda m: m.from_user and m.from_user.id in STORYLINE_REMOVE_WAIT_USERS and _is_plain_text_message(m) and (m.text or "").strip().isdigit())
async def handle_storyline_remove_number(message: types.Message):
    if not _storytracking_available(
        telegram_id=message.from_user.id,
        username=message.from_user.username,
    ):
        locale = await _resolve_message_locale(message)
        _clear_storyline_states(message.from_user.id)
        await message.answer(
            t(locale, "storytracking_disabled"),
            reply_markup=_settings_keyboard_for_user(locale, telegram_username=message.from_user.username),
        )
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return

        raw = (message.text or "").strip()
        index = int(raw)
        choices = STORYLINE_REMOVE_CHOICES.get(message.from_user.id) or []
        if index < 1 or index > len(choices):
            await message.answer(
                t(locale, "storyline_tracking_remove_invalid"),
                reply_markup=_keyboard([button_text("storyline_clear_all", locale), button_text("back", locale)]),
            )
            return

        follow_id = int(choices[index - 1])
        follow = await session.get(UserStorylineFollow, follow_id)
        if follow and follow.user_id == user.id:
            follow.is_active = False
            await session.commit()

        follows_result = await session.execute(
            select(UserStorylineFollow)
            .where(
                UserStorylineFollow.user_id == user.id,
                UserStorylineFollow.is_active.is_(True),
            )
            .order_by(UserStorylineFollow.created_at.asc(), UserStorylineFollow.id.asc())
        )
        follows = follows_result.scalars().all()

        STORYLINE_MENU_USERS.add(message.from_user.id)
        if not follows:
            STORYLINE_REMOVE_WAIT_USERS.discard(message.from_user.id)
            STORYLINE_REMOVE_CHOICES.pop(message.from_user.id, None)
            await message.answer(
                t(locale, "storyline_tracking_remove_success") + "\n\n" + t(locale, "storyline_tracking_empty"),
                reply_markup=get_storyline_tracking_keyboard(locale),
            )
            return

        STORYLINE_REMOVE_WAIT_USERS.add(message.from_user.id)
        STORYLINE_REMOVE_CHOICES[message.from_user.id] = [int(item.id) for item in follows]
        await message.answer(
            t(locale, "storyline_tracking_remove_success")
            + "\n\n"
            + t(locale, "storyline_tracking_list", items=_render_storyline_follow_list(follows))
            + "\n\n"
            + t(locale, "storyline_tracking_remove_prompt"),
            reply_markup=_keyboard([button_text("storyline_clear_all", locale), button_text("back", locale)]),
        )
        return


@dp.message(
    lambda m: (
        m.from_user
        and m.from_user.id in STORYLINE_REMOVE_WAIT_USERS
        and _matches_text(getattr(m, "text", None), "storyline_clear_all")
    )
)
async def handle_storyline_remove_clear_all(message: types.Message):
    if not _storytracking_available(
        telegram_id=message.from_user.id,
        username=message.from_user.username,
    ):
        locale = await _resolve_message_locale(message)
        _clear_storyline_states(message.from_user.id)
        await message.answer(
            t(locale, "storytracking_disabled"),
            reply_markup=_settings_keyboard_for_user(locale, telegram_username=message.from_user.username),
        )
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return

        await session.execute(
            update(UserStorylineFollow)
            .where(UserStorylineFollow.user_id == user.id)
            .where(UserStorylineFollow.is_active.is_(True))
            .values(is_active=False)
        )
        await session.commit()

        STORYLINE_MENU_USERS.add(message.from_user.id)
        STORYLINE_REMOVE_WAIT_USERS.discard(message.from_user.id)
        STORYLINE_REMOVE_CHOICES.pop(message.from_user.id, None)
        await message.answer(
            t(locale, "storyline_tracking_clear_all_success") + "\n\n" + t(locale, "storyline_tracking_empty"),
            reply_markup=get_storyline_tracking_keyboard(locale),
        )
        return


@dp.message(
    lambda m: (
        m.from_user
        and m.from_user.id in STORYLINE_REMOVE_WAIT_USERS
        and _is_plain_text_message(m)
        and not _matches_text(getattr(m, "text", None), "back")
    )
)
async def handle_storyline_remove_invalid(message: types.Message):
    if not _storytracking_available(
        telegram_id=message.from_user.id,
        username=message.from_user.username,
    ):
        locale = await _resolve_message_locale(message)
        _clear_storyline_states(message.from_user.id)
        await message.answer(
            t(locale, "storytracking_disabled"),
            reply_markup=_settings_keyboard_for_user(locale, telegram_username=message.from_user.username),
        )
        return

    locale = await _resolve_message_locale(message)
    await message.answer(
        t(locale, "storyline_tracking_remove_invalid"),
        reply_markup=_keyboard([button_text("back", locale)]),
    )


@dp.message(_message_matches("storyline_debug_enable", "storyline_debug_disable"))
async def toggle_storyline_debug(message: types.Message):
    if not _storytracking_available(
        telegram_id=message.from_user.id,
        username=message.from_user.username,
    ):
        locale = await _resolve_message_locale(message)
        _clear_storyline_states(message.from_user.id)
        await message.answer(t(locale, "storytracking_disabled"))
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not _is_admin_user(user, message.from_user.username):
            await message.answer(t(locale, "admin_access_denied"))
            return
        enabled = _matches_text(message.text, "storyline_debug_enable")
        user.storyline_debug_enabled = enabled
        await session.commit()
        ADMIN_PANEL_USERS.add(message.from_user.id)
        await message.answer(
            t(locale, "storyline_debug_enabled_text" if enabled else "storyline_debug_disabled_text"),
            reply_markup=get_admin_keyboard(locale, enabled),
        )
        return


@dp.message(Command("language"))
@dp.message(_message_matches("language_menu"))
async def show_language_menu(message: types.Message):
    _clear_settings_submenu_states(message.from_user.id)
    LANGUAGE_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return
        await message.answer(t(locale, "language_menu_title"), reply_markup=get_language_keyboard(locale))
        return


@dp.message(_message_matches("language_ru", "language_en"))
async def set_language(message: types.Message):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return

        new_locale = "en" if _matches_text(message.text, "language_en") else "ru"
        user.language_code = new_locale
        await session.commit()
        await message.answer(
            t(new_locale, "language_changed"),
            reply_markup=_settings_keyboard_for_user(new_locale, user, message.from_user.username),
        )
        return


def _feed_filter_label(value: str | None, locale: str = "ru") -> str:
    current = (value or "all").strip().lower()
    labels = {
        "all": t(locale, "feed_filter_all"),
        "not_interesting": t(locale, "feed_filter_not_interesting"),
        "only_fire": t(locale, "feed_filter_only_fire"),
        "digest_only": t(locale, "feed_filter_digest_only"),
    }
    return labels.get(current, t(locale, "feed_filter_all"))


def _normalize_live_feed_filter(value: str | None) -> str:
    current = (value or "all").strip().lower()
    return current if current in {"all", "not_interesting", "only_fire"} else "all"


def _disable_live_forwarding(user: User) -> str:
    previous_filter = (getattr(user, "feed_filter", "all") or "all").strip().lower()
    if previous_filter != "digest_only":
        user.last_live_feed_filter = _normalize_live_feed_filter(previous_filter)
    user.feed_filter = "digest_only"
    return _normalize_live_feed_filter(getattr(user, "last_live_feed_filter", "all"))


def _enable_live_forwarding(user: User) -> str:
    restored_filter = _normalize_live_feed_filter(getattr(user, "last_live_feed_filter", "all"))
    user.last_live_feed_filter = restored_filter
    user.feed_filter = restored_filter
    return restored_filter


def _set_live_feed_filter(user: User, value: str) -> str:
    normalized = _normalize_live_feed_filter(value)
    user.feed_filter = normalized
    user.last_live_feed_filter = normalized
    return normalized


def _render_forwarding_menu_text(
    user: User,
    locale: str | None = None,
    *,
    summary_effect_notice: bool = False,
) -> str:
    current_locale = locale or get_user_locale(user)
    current = (getattr(user, "feed_filter", "all") or "all").strip().lower()
    live_enabled = current != "digest_only"
    summary_enabled = bool(getattr(user, "summary_enabled", False))
    text = (
        f"{t(current_locale, 'forwarding_intro')}\n"
        f"- {t(current_locale, 'forwarding_live_label')}: {t(current_locale, 'forwarding_live_on') if live_enabled else t(current_locale, 'forwarding_live_off')}\n"
        f"- {t(current_locale, 'forwarding_filter_label')}: {_feed_filter_label(current, current_locale)}\n"
        f"- {t(current_locale, 'forwarding_summary_label')}: {t(current_locale, 'forwarding_summary_on') if summary_enabled else t(current_locale, 'forwarding_summary_off')}"
    )
    if summary_effect_notice:
        text += "\n\n" + t(
            current_locale,
            "forwarding_summary_effect_on" if summary_enabled else "forwarding_summary_effect_off",
        )
    return text + f"\n\n{t(current_locale, 'forwarding_menu_footer')}"


def _clear_instruction_states(user_id: int) -> None:
    INSTRUCTION_GLOBAL_WAIT_USERS.discard(user_id)
    INSTRUCTION_GROUP_SELECT_USERS.discard(user_id)
    INSTRUCTION_GROUP_PROMPT_WAIT_USERS.discard(user_id)
    INSTRUCTION_GROUP_DISABLE_SELECT_USERS.discard(user_id)
    INSTRUCTION_GROUP_CHOICES.pop(user_id, None)
    INSTRUCTION_GROUP_DISABLE_CHOICES.pop(user_id, None)
    INSTRUCTION_GROUP_PROMPT_TARGETS.pop(user_id, None)


def _clear_payment_states(user_id: int) -> None:
    CRYPTO_PAYMENT_MENU_USERS.discard(user_id)
    PAYMENT_TERM_MENU_USERS.discard(user_id)
    PAYMENT_METHOD_SELECTIONS.pop(user_id, None)


def _clear_settings_submenu_states(user_id: int) -> None:
    BILLING_MENU_USERS.discard(user_id)
    DIGEST_MENU_USERS.discard(user_id)
    PLUS_DELIVERY_MENU_USERS.discard(user_id)
    LANGUAGE_MENU_USERS.discard(user_id)
    SUPPORT_REQUEST_WAIT_USERS.discard(user_id)
    NEWS_TIME_BASELINE_WAIT_USERS.discard(user_id)
    RECO_RESET_CONFIRM_USERS.discard(user_id)


def _clear_storyline_states(user_id: int) -> None:
    STORYLINE_MENU_USERS.discard(user_id)
    STORYLINE_REMOVE_WAIT_USERS.discard(user_id)
    STORYLINE_REMOVE_CHOICES.pop(user_id, None)


def _storytracking_available(
    user: User | None = None,
    *,
    telegram_id: int | None = None,
    username: str | None = None,
) -> bool:
    return ENABLE_STORYTRACKING and storytracking_allowed_for_user(
        user,
        telegram_id=telegram_id,
        username=username,
    )


def _payment_term_days_from_text(value: str | None) -> int | None:
    mapping = {
        "pay_term_30d": 30,
        "pay_term_90d": 90,
        "pay_term_180d": 180,
        "pay_term_365d": 365,
    }
    for key, days in mapping.items():
        if _matches_text(value, key):
            return days
    return None


def _instruction_rule_preview(text: str, *, limit: int = 160) -> str:
    compact = " ".join(str(text or "").strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


async def _ensure_instruction_filter_access(message: types.Message, session, user: User, locale: str) -> bool:
    if await user_has_instruction_filter_access(session, user.id, commit_on_sync=True):
        return True
    await message.answer(
        t(locale, "instruction_premium_required"),
        reply_markup=get_instruction_filter_keyboard(locale),
    )
    return False


def _render_instruction_intro(locale: str) -> str:
    return t(locale, "instruction_intro")


async def _render_active_instruction_rules(session, *, user: User, locale: str) -> str:
    rules = await get_instruction_rules_for_user(session, user_id=user.id)
    if not rules:
        return t(locale, "instruction_no_rules")

    community_targets = await get_rule_bound_communities(session, user_id=user.id)
    communities_by_id = {int(item.id): item for item in community_targets if getattr(item, "id", None) is not None}

    lines: list[str] = [t(locale, "instruction_rules_title")]
    global_rule = next((rule for rule in rules if str(rule.scope) == "global"), None)
    if global_rule:
        lines.append(
            t(
                locale,
                "instruction_global_rule_item",
                prompt=html.escape(_instruction_rule_preview(global_rule.prompt_text)),
            )
        )
    for rule in rules:
        if str(rule.scope) != "community":
            continue
        community = communities_by_id.get(int(rule.community_id or 0))
        label = getattr(community, "link", None) or getattr(community, "name", None) or f"community:{rule.community_id}"
        lines.append(
            t(
                locale,
                "instruction_group_rule_item",
                group=html.escape(str(label)),
                prompt=html.escape(_instruction_rule_preview(rule.prompt_text)),
            )
        )
    return "\n".join(lines)


async def _render_instruction_menu_text(session, *, user: User, locale: str) -> str:
    return f"{_render_instruction_intro(locale)}\n\n{await _render_active_instruction_rules(session, user=user, locale=locale)}"


async def _send_instruction_menu(message: types.Message, *, session, user: User, locale: str) -> None:
    INSTRUCTION_MENU_USERS.add(message.from_user.id)
    _clear_instruction_states(message.from_user.id)
    await message.answer(
        await _render_instruction_menu_text(session, user=user, locale=locale),
        reply_markup=get_instruction_filter_keyboard(locale),
        parse_mode="HTML",
    )


@dp.message(Command("forwarding"))
@dp.message(_message_matches("forwarding"))
async def show_forwarding_menu(message: types.Message):
    FORWARDING_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        await message.answer(
            _render_forwarding_menu_text(user, locale),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
            parse_mode="HTML",
        )
        return


@dp.message(_message_matches("instruction_filter"))
async def show_instruction_filter_menu(message: types.Message):
    FORWARDING_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        await _send_instruction_menu(message, session=session, user=user, locale=locale)
        return


@dp.message(_message_matches("instruction_show_rules"))
async def show_instruction_rules(message: types.Message):
    INSTRUCTION_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        text = f"{_render_instruction_intro(locale)}\n\n{await _render_active_instruction_rules(session, user=user, locale=locale)}"
        await message.answer(text, reply_markup=get_instruction_filter_keyboard(locale), parse_mode="HTML")
        return


@dp.message(_message_matches("instruction_set_global"))
async def start_instruction_global_prompt(message: types.Message):
    INSTRUCTION_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        if not await _ensure_instruction_filter_access(message, session, user, locale):
            return
        _clear_instruction_states(message.from_user.id)
        INSTRUCTION_MENU_USERS.add(message.from_user.id)
        INSTRUCTION_GLOBAL_WAIT_USERS.add(message.from_user.id)
        await message.answer(
            t(locale, "instruction_global_prompt_request"),
            reply_markup=get_instruction_filter_keyboard(locale),
        )
        return


@dp.message(_message_matches("instruction_set_group"))
async def start_instruction_group_prompt(message: types.Message):
    INSTRUCTION_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        if not await _ensure_instruction_filter_access(message, session, user, locale):
            return
        targets = await get_subscribed_instruction_targets(session, user_id=user.id)
        if not targets:
            await message.answer(
                t(locale, "instruction_group_no_subscriptions"),
                reply_markup=get_instruction_filter_keyboard(locale),
            )
            return
        _clear_instruction_states(message.from_user.id)
        INSTRUCTION_MENU_USERS.add(message.from_user.id)
        INSTRUCTION_GROUP_SELECT_USERS.add(message.from_user.id)
        INSTRUCTION_GROUP_CHOICES[message.from_user.id] = [int(item.id) for item in targets]
        await message.answer(
            t(locale, "instruction_group_prompt_choose"),
            reply_markup=_build_instruction_group_picker_keyboard(
                targets,
                mode="set",
                page=0,
                locale=locale,
            ),
        )
        return


@dp.message(_message_matches("instruction_disable_global"))
async def disable_instruction_global(message: types.Message):
    INSTRUCTION_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        changed = await disable_global_instruction_rule(session, user_id=user.id)
        await session.commit()
        await message.answer(
            t(locale, "instruction_global_disabled" if changed else "instruction_global_already_disabled"),
            reply_markup=get_instruction_filter_keyboard(locale),
        )
        return


@dp.message(_message_matches("instruction_disable_group"))
async def disable_instruction_group_start(message: types.Message):
    INSTRUCTION_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        communities = await get_rule_bound_communities(session, user_id=user.id)
        if not communities:
            await message.answer(
                t(locale, "instruction_group_rules_missing"),
                reply_markup=get_instruction_filter_keyboard(locale),
            )
            return
        _clear_instruction_states(message.from_user.id)
        INSTRUCTION_MENU_USERS.add(message.from_user.id)
        INSTRUCTION_GROUP_DISABLE_SELECT_USERS.add(message.from_user.id)
        INSTRUCTION_GROUP_DISABLE_CHOICES[message.from_user.id] = [int(item.id) for item in communities]
        await message.answer(
            t(locale, "instruction_group_disable_choose"),
            reply_markup=_build_instruction_group_picker_keyboard(
                communities,
                mode="disable",
                page=0,
                locale=locale,
            ),
        )
        return


def _render_billing_status(snapshot: dict, locale: str = "ru") -> str:
    paid_packs = int(snapshot.get("paid_packs") or 0)
    premium_active = bool(snapshot.get("premium_active") or paid_packs > 0)
    premium_expires_at = str(snapshot.get("premium_expires_at") or "").strip()
    lines = [
        f"<b>{t(locale, 'billing_title')}</b>",
        f"<b>{t(locale, 'billing_premium_status')}:</b> {t(locale, 'billing_premium_active' if premium_active else 'billing_premium_inactive')}",
    ]
    if premium_active:
        lines.append(t(locale, "billing_premium_unlocked"))
    if premium_expires_at:
        lines.append(f"<b>{t(locale, 'billing_premium_expires_at')}:</b> {premium_expires_at[:10]}")
    return "\n".join(lines)


def _render_billing_instruction(snapshot: dict, locale: str = "ru") -> str:
    free_limit = int(snapshot.get("free_limit") or 7)
    premium_price = int(snapshot.get("premium_price_rub") or snapshot.get("pack_price_rub") or 490)
    due_amount = max(
        0,
        int(snapshot.get("required_amount_rub") or 0) - int(snapshot.get("paid_amount_rub") or 0),
    )
    return t(
        locale,
        "billing_instruction",
        free_limit=free_limit,
        premium_price=premium_price,
        due_amount=due_amount,
    )


def _payment_provider_text(provider: str | None, locale: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "cryptopay":
        return t(locale, "payment_provider_cryptopay")
    if normalized == "telegram_stars":
        return t(locale, "payment_provider_stars")
    return t(locale, "payment_provider_tbank")


def _render_pending_payment(snapshot: dict, locale: str = "ru") -> str:
    pending = snapshot.get("pending_payment") or {}
    if not pending:
        return ""
    amount = pending.get("quote_amount") or "?"
    currency = pending.get("quote_currency") or "RUB"
    provider = _payment_provider_text(pending.get("provider"), locale)
    parts = [
        t(
            locale,
            "billing_pending_payment",
            provider=provider,
            amount=amount,
            currency=currency,
        )
    ]
    if pending.get("period_start_date") and pending.get("period_end_date"):
        parts.append(
            t(
                locale,
                "billing_pending_period",
                period_start=pending["period_start_date"],
                period_end=pending["period_end_date"],
                days=int(pending.get("payment_term_days") or 0),
            )
        )
    if pending.get("payment_url"):
        parts.append(f"{t(locale, 'billing_pending_link')}\n{pending['payment_url']}")
    if str(pending.get("provider") or "").strip().lower() == "cryptopay":
        parts.append(t(locale, "payment_crypto_polling_hint"))
    if str(pending.get("provider") or "").strip().lower() == "telegram_stars":
        parts.append(t(locale, "payment_stars_pending_hint"))
    return "\n\n".join(parts)


async def _send_billing_menu(message: types.Message, *, locale_hint: str | None = None):
    _clear_settings_submenu_states(message.from_user.id)
    BILLING_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(locale_hint or _button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        snapshot = await get_current_payment_snapshot(session, user)
        text = _render_billing_status(snapshot, locale) + "\n\n" + _render_billing_instruction(snapshot, locale)
        await message.answer(text, reply_markup=get_billing_keyboard(locale), parse_mode="HTML")
        return


@dp.message(Command("billing"))
@dp.message(_message_matches("billing"))
async def show_billing_menu(message: types.Message):
    await _send_billing_menu(message)


@dp.callback_query(F.data == "premium_week:billing")
async def open_billing_from_premium_week(callback: types.CallbackQuery):
    await callback.answer()
    if not callback.message:
        return
    _clear_settings_submenu_states(callback.from_user.id)
    BILLING_MENU_USERS.add(callback.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == callback.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await callback.message.answer(t(_button_locale_from_callback(callback), "start_required_register"))
            return
        locale = get_user_locale(user)
        snapshot = await get_current_payment_snapshot(session, user)
        text = _render_billing_status(snapshot, locale) + "\n\n" + _render_billing_instruction(snapshot, locale)
        await callback.message.answer(text, reply_markup=get_billing_keyboard(locale), parse_mode="HTML")
        return


@dp.message(_message_matches("refresh_payment"))
async def refresh_billing_status(message: types.Message):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        try:
            await sync_pending_payment_for_user(session, user)
        except Exception:
            # Do not fail status refresh if T-Bank poll is temporarily unavailable.
            await session.rollback()
        locale = get_user_locale(user)
        snapshot = await get_current_payment_snapshot(session, user)
        text = _render_billing_status(snapshot, locale) + "\n\n" + _render_billing_instruction(snapshot, locale)
        await message.answer(
            text,
            reply_markup=get_billing_keyboard(locale),
            parse_mode="HTML",
        )
        return


@dp.message(_message_matches("enter_promo"))
async def promo_code_start(message: types.Message):
    PROMO_CODE_WAIT.add(message.from_user.id)
    await message.answer(t(_button_locale_from_message(message), "promo_enter"))


@dp.message(_should_handle_promo_code_input)
async def promo_code_apply(message: types.Message):
    code = (message.text or "").strip()
    async for session in get_session():
        user = (
            await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        ).scalar_one_or_none()
        if not user:
            PROMO_CODE_WAIT.discard(message.from_user.id)
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        try:
            grant = await redeem_promo_code(session, user=user, code=code)
            snapshot = await get_current_payment_snapshot(session, user)
            await session.commit()
            PROMO_CODE_WAIT.discard(message.from_user.id)
            expires = grant.expires_at.strftime("%Y-%m-%d %H:%M UTC") if grant.expires_at else t(locale, "promo_no_expiry")
            if str(getattr(grant, "grant_type", "") or "") == "premium":
                await message.answer(
                    t(
                        locale,
                        "promo_premium_applied",
                        days=int(getattr(grant, "premium_days", None) or 30),
                        expires=expires,
                        status=_render_billing_status(snapshot, locale),
                    ),
                    reply_markup=get_billing_keyboard(locale),
                    parse_mode="HTML",
                )
                return
            await message.answer(
                t(
                    locale,
                    "promo_applied",
                    extra_groups=int(grant.extra_groups or 0),
                    expires=expires,
                    status=_render_billing_status(snapshot, locale),
                ),
                reply_markup=get_billing_keyboard(locale),
                parse_mode="HTML",
            )
            return
        except Exception as e:
            await session.rollback()
            PROMO_CODE_WAIT.discard(message.from_user.id)
            await message.answer(
                t(locale, "promo_apply_failed", error=e),
                reply_markup=get_billing_keyboard(locale),
            )
            return


async def _start_payment_term_selection(message: types.Message, *, payment_method: str, prompt_key: str) -> None:
    user_id = message.from_user.id
    _clear_payment_states(user_id)
    CRYPTO_PAYMENT_MENU_USERS.add(user_id)
    PAYMENT_TERM_MENU_USERS.add(user_id)
    PAYMENT_METHOD_SELECTIONS[user_id] = payment_method
    locale = _button_locale_from_message(message)
    await message.answer(
        t(locale, prompt_key),
        reply_markup=get_payment_term_keyboard(locale),
        parse_mode="HTML",
    )


@dp.message(_message_matches("pay_crypto"))
async def show_crypto_payment_options(message: types.Message):
    await _start_payment_term_selection(
        message,
        payment_method="cryptopay_usdt",
        prompt_key="payment_crypto_prompt",
    )


@dp.message(_message_matches("pay_stars"))
async def show_stars_payment_options(message: types.Message):
    await _start_payment_term_selection(
        message,
        payment_method=PAYMENT_METHOD_TELEGRAM_STARS,
        prompt_key="payment_stars_prompt",
    )


async def _create_payment_link(message: types.Message, payment_method: str, payment_term_days: int):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        try:
            order = await create_payment_order_for_user(
                session,
                user=user,
                extra_packs=0,
                payment_method=payment_method,
                payment_term_days=payment_term_days,
            )
            stars_invoice = None
            if payment_method == PAYMENT_METHOD_TELEGRAM_STARS:
                stars_invoice = get_telegram_stars_invoice_data(order)
                chat_id = getattr(getattr(message, "chat", None), "id", None) or message.from_user.id
                prices = [
                    types.LabeledPrice(
                        label=str(item.get("label") or stars_invoice["title"]),
                        amount=int(item.get("amount") or 0),
                    )
                    for item in (stars_invoice.get("prices") or [])
                ]
                await bot.send_invoice(
                    chat_id=chat_id,
                    title=str(stars_invoice.get("title") or t(locale, "payment_stars_invoice_title", days=payment_term_days)),
                    description=str(
                        stars_invoice.get("description")
                        or t(locale, "payment_stars_invoice_description", days=payment_term_days)
                    ),
                    payload=str(stars_invoice.get("payload") or ""),
                    currency="XTR",
                    prices=prices,
                )
            await session.commit()
            _clear_payment_states(message.from_user.id)
            text = t(
                locale,
                "payment_target_period",
                period_start=order.period_start.strftime("%Y-%m-%d"),
                period_end=order.period_end.strftime("%Y-%m-%d"),
                days=payment_term_days,
            )
            text += "\n" + t(
                locale,
                "payment_method_selected",
                provider=_payment_provider_text(order.provider, locale),
            )
            text += "\n" + t(locale, "payment_term_selected", days=payment_term_days)
            text += "\n" + t(
                locale,
                "payment_quote_selected",
                amount=str(order.quote_amount),
                currency=order.quote_currency,
            )
            if order.payment_url:
                text += f"\n\n{t(locale, 'payment_link_title')}\n{html.escape(order.payment_url)}"
            await message.answer(text, reply_markup=get_billing_keyboard(locale), parse_mode="HTML")
            return
        except PremiumAlreadyPrepaidError as e:
            await session.rollback()
            _clear_payment_states(message.from_user.id)
            await message.answer(
                t(locale, "payment_premium_already_prepaid", period=e.period_label),
                reply_markup=get_billing_keyboard(locale),
            )
            return
        except Exception as e:
            await session.rollback()
            _clear_payment_states(message.from_user.id)
            await message.answer(
                t(locale, "payment_create_failed", error=e),
                reply_markup=get_billing_keyboard(locale),
            )
            return


@dp.message(_message_matches("pay_current_limit", "pay_tbank"))
async def start_payment_link_tbank(message: types.Message):
    await _start_payment_term_selection(
        message,
        payment_method="tbank",
        prompt_key="payment_tbank_prompt",
    )


@dp.pre_checkout_query()
async def handle_pre_checkout_query(pre_checkout_query: types.PreCheckoutQuery):
    async for session in get_session():
        try:
            result = await validate_telegram_stars_pre_checkout(
                session,
                telegram_id=pre_checkout_query.from_user.id,
                invoice_payload=pre_checkout_query.invoice_payload,
                total_amount=int(getattr(pre_checkout_query, "total_amount", 0) or 0),
                currency=str(getattr(pre_checkout_query, "currency", "") or ""),
                pre_checkout_query_id=str(getattr(pre_checkout_query, "id", "") or ""),
            )
            if not result.get("ok"):
                await session.rollback()
                await bot.answer_pre_checkout_query(
                    pre_checkout_query.id,
                    ok=False,
                    error_message=t(await _resolve_callback_locale(pre_checkout_query), "payment_stars_precheckout_failed"),
                )
                return
            await session.commit()
            await bot.answer_pre_checkout_query(pre_checkout_query.id, ok=True)
            return
        except Exception:
            await session.rollback()
            await bot.answer_pre_checkout_query(
                pre_checkout_query.id,
                ok=False,
                error_message=t(_button_locale_from_callback(pre_checkout_query), "payment_stars_precheckout_failed"),
            )
            return


@dp.message(F.successful_payment)
async def handle_successful_payment(message: types.Message):
    payment = getattr(message, "successful_payment", None)
    if not payment:
        return
    payload = {
        "currency": getattr(payment, "currency", None),
        "invoice_payload": getattr(payment, "invoice_payload", None),
        "telegram_payment_charge_id": getattr(payment, "telegram_payment_charge_id", None),
        "provider_payment_charge_id": getattr(payment, "provider_payment_charge_id", None),
        "total_amount": getattr(payment, "total_amount", None),
    }
    async for session in get_session():
        user = (
            await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        ).scalar_one_or_none()
        locale = get_user_locale(user) if user else _button_locale_from_message(message)
        result = await process_telegram_stars_successful_payment(
            session,
            telegram_id=message.from_user.id,
            successful_payment=payload,
        )
        if not result.get("ok"):
            await session.rollback()
            await message.answer(
                t(locale, "payment_stars_precheckout_failed"),
                reply_markup=get_billing_keyboard(locale),
            )
            return
        await session.commit()
        snapshot = await get_current_payment_snapshot(session, user) if user else {}
        text = t(locale, "payment_stars_success")
        pending_text = _render_pending_payment(snapshot, locale) if snapshot else ""
        if pending_text:
            text += "\n\n" + pending_text
        await message.answer(text, reply_markup=get_billing_keyboard(locale))
        return


@dp.message(_message_matches("pay_term_30d", "pay_term_90d", "pay_term_180d", "pay_term_365d"))
async def create_payment_link_for_selected_term(message: types.Message):
    payment_method = PAYMENT_METHOD_SELECTIONS.get(message.from_user.id)
    payment_term_days = _payment_term_days_from_text(getattr(message, "text", None))
    locale = _button_locale_from_message(message)
    if not payment_method or payment_term_days is None:
        _clear_payment_states(message.from_user.id)
        await message.answer(
            t(locale, "payment_select_method_first"),
            reply_markup=get_billing_keyboard(locale),
        )
        return
    await _create_payment_link(message, payment_method=payment_method, payment_term_days=payment_term_days)


@dp.message(_message_matches("pay_plus_10"))
async def create_payment_link_plus_10(message: types.Message):
    await message.answer(
        t(_button_locale_from_message(message), "payment_option_disabled"),
        reply_markup=get_billing_keyboard(_button_locale_from_message(message)),
    )


@dp.message(F.text.casefold() == "оплатить +20 подписок")
async def create_payment_link_plus_20(message: types.Message):
    await message.answer(
        t(_button_locale_from_message(message), "payment_option_disabled"),
        reply_markup=get_billing_keyboard(_button_locale_from_message(message)),
    )


@dp.message(F.text.casefold() == "оплатить +30 подписок")
async def create_payment_link_plus_30(message: types.Message):
    await message.answer(
        t(_button_locale_from_message(message), "payment_option_disabled"),
        reply_markup=get_billing_keyboard(_button_locale_from_message(message)),
    )


@dp.message(_message_matches("back"))
async def go_back(message: types.Message):
    locale = _button_locale_from_message(message)
    if message.from_user.id in CRYPTO_PAYMENT_MENU_USERS or message.from_user.id in PAYMENT_TERM_MENU_USERS:
        _clear_payment_states(message.from_user.id)
        await _send_billing_menu(message, locale_hint=locale)
        return

    if message.from_user.id in DIGEST_TIME_MENU_USERS:
        DIGEST_TIME_MENU_USERS.discard(message.from_user.id)
        DIGEST_SETUP_WAIT_OFFSET.discard(message.from_user.id)
        DIGEST_SETUP_WAIT_TIME.discard(message.from_user.id)
        _clear_settings_submenu_states(message.from_user.id)
        DIGEST_MENU_USERS.add(message.from_user.id)
        async for session in get_session():
            result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
            user = result.scalar_one_or_none()
            if not user:
                await message.answer(t(locale, "start_required_register"))
                return
            settings = await session.get(UserDigestSettings, user.id)
            enabled = bool(settings.enabled) if settings else False
            slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
            slot = slot_res.scalar_one_or_none()
            await message.answer(
                t(locale, "digest_menu_title"),
                reply_markup=get_digest_keyboard(locale, enabled=enabled, time_label=_digest_time_label(slot)),
            )
            return
        return

    if (
        message.from_user.id in INSTRUCTION_GLOBAL_WAIT_USERS
        or message.from_user.id in INSTRUCTION_GROUP_SELECT_USERS
        or message.from_user.id in INSTRUCTION_GROUP_PROMPT_WAIT_USERS
        or message.from_user.id in INSTRUCTION_GROUP_DISABLE_SELECT_USERS
    ):
        _clear_instruction_states(message.from_user.id)
        INSTRUCTION_MENU_USERS.add(message.from_user.id)
        async for session in get_session():
            result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
            user = result.scalar_one_or_none()
            if not user:
                await message.answer(t(locale, "start_required_register"))
                return
            await message.answer(
                f"{_render_instruction_intro(locale)}\n\n{await _render_active_instruction_rules(session, user=user, locale=locale)}",
                reply_markup=get_instruction_filter_keyboard(locale),
                parse_mode="HTML",
            )
            return

    if message.from_user.id in INSTRUCTION_MENU_USERS:
        INSTRUCTION_MENU_USERS.discard(message.from_user.id)
        _clear_instruction_states(message.from_user.id)
        async for session in get_session():
            result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
            user = result.scalar_one_or_none()
            if not user:
                await message.answer(t(locale, "start_required_register"))
                return
            FORWARDING_MENU_USERS.add(message.from_user.id)
            await message.answer(
                _render_forwarding_menu_text(user, locale),
                reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
                parse_mode="HTML",
            )
            return

    if message.from_user.id in FORWARDING_MENU_USERS:
        FORWARDING_MENU_USERS.discard(message.from_user.id)
        async for session in get_session():
            result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
            user = result.scalar_one_or_none()
            await message.answer(
                t(locale, "settings_title"),
                reply_markup=_settings_keyboard_for_user(locale, user, message.from_user.username),
            )
            return

    if (
        message.from_user.id in BILLING_MENU_USERS
        or message.from_user.id in DIGEST_MENU_USERS
        or message.from_user.id in PLUS_DELIVERY_MENU_USERS
        or message.from_user.id in LANGUAGE_MENU_USERS
        or message.from_user.id in SUPPORT_REQUEST_WAIT_USERS
    ):
        _clear_settings_submenu_states(message.from_user.id)
        async for session in get_session():
            result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
            user = result.scalar_one_or_none()
            await message.answer(
                t(locale, "settings_title"),
                reply_markup=_settings_keyboard_for_user(locale, user, message.from_user.username),
            )
            return

    if message.from_user.id in STORYLINE_REMOVE_WAIT_USERS:
        _clear_storyline_states(message.from_user.id)
        if not _storytracking_available(
            telegram_id=message.from_user.id,
            username=message.from_user.username,
        ):
            async for session in get_session():
                result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
                user = result.scalar_one_or_none()
                await message.answer(
                    t(locale, "settings_title"),
                    reply_markup=_settings_keyboard_for_user(locale, user, message.from_user.username),
                )
                return
            return
        STORYLINE_MENU_USERS.add(message.from_user.id)
        await message.answer(
            t(locale, "storyline_tracking_title"),
            reply_markup=get_storyline_tracking_keyboard(locale),
        )
        return

    if message.from_user.id in STORYLINE_MENU_USERS:
        STORYLINE_MENU_USERS.discard(message.from_user.id)
        async for session in get_session():
            result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
            user = result.scalar_one_or_none()
            await message.answer(
                t(locale, "settings_title"),
                reply_markup=_settings_keyboard_for_user(locale, user, message.from_user.username),
            )
            return

    if message.from_user.id in ADMIN_PANEL_USERS:
        ADMIN_PANEL_USERS.discard(message.from_user.id)
        async for session in get_session():
            result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
            user = result.scalar_one_or_none()
            await message.answer(
                t(locale, "main_menu_title"),
                reply_markup=_main_keyboard_for_user(locale, user, message.from_user.username),
            )
            return
        return

    DIGEST_SETUP_WAIT_OFFSET.discard(message.from_user.id)
    DIGEST_SETUP_WAIT_TIME.discard(message.from_user.id)
    DIGEST_TIME_MENU_USERS.discard(message.from_user.id)
    FORWARDING_MENU_USERS.discard(message.from_user.id)
    INSTRUCTION_MENU_USERS.discard(message.from_user.id)
    _clear_instruction_states(message.from_user.id)
    STORYLINE_MENU_USERS.discard(message.from_user.id)
    STORYLINE_REMOVE_WAIT_USERS.discard(message.from_user.id)
    STORYLINE_REMOVE_CHOICES.pop(message.from_user.id, None)
    ADMIN_PANEL_USERS.discard(message.from_user.id)
    PROMO_CODE_WAIT.discard(message.from_user.id)
    SUPPORT_REQUEST_WAIT_USERS.discard(message.from_user.id)
    NEWS_TIME_BASELINE_WAIT_USERS.discard(message.from_user.id)
    _clear_payment_states(message.from_user.id)
    _clear_settings_submenu_states(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        await message.answer(
            t(locale, "main_menu_title"),
            reply_markup=_main_keyboard_for_user(locale, user, message.from_user.username),
        )
        return


@dp.message(Command("subscriptions"))
@dp.message(_message_matches("subscriptions"))
async def list_subscriptions(message: types.Message):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            break
        locale = get_user_locale(user)

        result = await session.execute(
            select(Community.link)
            .join(UserCommunity, UserCommunity.community_id == Community.id)
            .where(UserCommunity.user_id == user.id, Community.is_active.is_(True))
        )
        links = [row[0] for row in result.all()]

        if not links:
            await message.answer(
                t(locale, "subscriptions_none"),
                reply_markup=_settings_keyboard_for_user(
                    locale,
                    user,
                    getattr(message.from_user, "username", None),
                ),
            )
        else:
            await message.answer(
                t(locale, "subscriptions_list", links="\n".join(links)),
                reply_markup=_settings_keyboard_for_user(
                    locale,
                    user,
                    getattr(message.from_user, "username", None),
                ),
            )

        break


@dp.message(_message_matches("toggle_summary"))
async def toggle_summarization(message: types.Message):
    FORWARDING_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            break

        user.summary_enabled = not bool(user.summary_enabled)
        await session.commit()
        locale = get_user_locale(user)
        await message.answer(
            _render_forwarding_menu_text(user, locale, summary_effect_notice=True),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
            parse_mode="HTML",
        )
        break


@dp.message(_message_matches("toggle_forwarding", "forwarding_enable", "forwarding_disable"))
async def toggle_forwarding(message: types.Message):
    from app.audit import audit

    FORWARDING_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        previous = getattr(user, "feed_filter", "all") or "all"
        if previous == "digest_only":
            _enable_live_forwarding(user)
        else:
            _disable_live_forwarding(user)
        await session.commit()
        audit("filter.changed", user_id=user.id, telegram_id=user.telegram_id, previous_filter=previous, new_filter=user.feed_filter)
        locale = get_user_locale(user)
        await message.answer(
            _render_forwarding_menu_text(user, locale),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
            parse_mode="HTML",
        )
        return


@dp.message(_message_matches("filter_digest_only"))
async def set_filter_digest_only(message: types.Message):
    from app.audit import audit

    FORWARDING_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        previous = getattr(user, "feed_filter", "all") or "all"
        _disable_live_forwarding(user)
        await session.commit()
        audit("filter.changed", user_id=user.id, telegram_id=user.telegram_id, previous_filter=previous, new_filter=user.feed_filter)
        locale = get_user_locale(user)
        await message.answer(
            _render_forwarding_menu_text(user, locale),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
            parse_mode="HTML",
        )
        return


@dp.message(_message_matches("filter_all", "forwarding_filter_all"))
async def set_filter_all(message: types.Message):
    from app.audit import audit
    FORWARDING_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        previous = getattr(user, "feed_filter", "all") or "all"
        _set_live_feed_filter(user, "all")
        await session.commit()
        audit("filter.changed", user_id=user.id, telegram_id=user.telegram_id, previous_filter=previous, new_filter=user.feed_filter)
        locale = get_user_locale(user)
        await message.answer(
            _render_forwarding_menu_text(user, locale),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
            parse_mode="HTML",
        )
        return


@dp.message(_message_matches("filter_not_interesting", "forwarding_filter_not_interesting"))
async def set_filter_not_interesting(message: types.Message):
    from app.audit import audit
    FORWARDING_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        previous = getattr(user, "feed_filter", "all") or "all"
        _set_live_feed_filter(user, "not_interesting")
        await session.commit()
        audit("filter.changed", user_id=user.id, telegram_id=user.telegram_id, previous_filter=previous, new_filter=user.feed_filter)
        locale = get_user_locale(user)
        await message.answer(
            _render_forwarding_menu_text(user, locale),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
            parse_mode="HTML",
        )
        return


@dp.message(_message_matches("filter_only_fire", "forwarding_filter_only_fire"))
async def set_filter_only_fire(message: types.Message):
    from app.audit import audit
    FORWARDING_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        previous = getattr(user, "feed_filter", "all") or "all"
        _set_live_feed_filter(user, "only_fire")
        await session.commit()
        audit("filter.changed", user_id=user.id, telegram_id=user.telegram_id, previous_filter=previous, new_filter=user.feed_filter)
        locale = get_user_locale(user)
        await message.answer(
            _render_forwarding_menu_text(user, locale),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
            parse_mode="HTML",
        )
        return


@dp.message(_message_matches("reco_reset_start"))
async def start_reco_reset(message: types.Message):
    RECO_RESET_CONFIRM_USERS.add(message.from_user.id)
    await message.answer(
        t(_button_locale_from_message(message), "reco_reset_confirm_prompt"),
        reply_markup=get_reco_reset_confirm_keyboard(_button_locale_from_message(message)),
    )


@dp.message(_message_matches("reco_reset_cancel"))
async def cancel_reco_reset(message: types.Message):
    RECO_RESET_CONFIRM_USERS.discard(message.from_user.id)
    FORWARDING_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        await message.answer(
            _render_forwarding_menu_text(user, locale),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
            parse_mode="HTML",
        )
        return


@dp.message(_message_matches("reco_reset_confirm"))
async def confirm_reco_reset(message: types.Message):
    from app.audit import audit

    RECO_RESET_CONFIRM_USERS.discard(message.from_user.id)
    FORWARDING_MENU_USERS.add(message.from_user.id)
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        generation, deleted_keywords = await reset_user_recommendations(session, user_id=user.id)
        await session.commit()
        audit(
            "reco.reset",
            user_id=user.id,
            telegram_id=user.telegram_id,
            generation=generation,
            deleted_keywords=deleted_keywords,
        )
        locale = get_user_locale(user)
        await message.answer(
            t(locale, "reco_reset_done"),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
        )
        return


@dp.message(_should_handle_instruction_global_input)
async def save_instruction_global_prompt(message: types.Message):
    async for session in get_session():
        user = (
            await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        ).scalar_one_or_none()
        if not user:
            _clear_instruction_states(message.from_user.id)
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        if not await _ensure_instruction_filter_access(message, session, user, locale):
            _clear_instruction_states(message.from_user.id)
            return
        try:
            await upsert_global_instruction_rule(session, user_id=user.id, prompt_text=message.text or "")
            await session.commit()
            _clear_instruction_states(message.from_user.id)
            INSTRUCTION_MENU_USERS.add(message.from_user.id)
            await message.answer(
                t(locale, "instruction_global_saved"),
                reply_markup=get_instruction_filter_keyboard(locale),
            )
            return
        except InstructionPromptError as exc:
            await session.rollback()
            await message.answer(
                t(locale, str(exc)),
                reply_markup=get_instruction_filter_keyboard(locale),
            )
            return


@dp.message(_should_handle_instruction_group_choice)
async def choose_instruction_group_target(message: types.Message):
    async for session in get_session():
        user = (
            await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        ).scalar_one_or_none()
        if not user:
            _clear_instruction_states(message.from_user.id)
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        choices = INSTRUCTION_GROUP_CHOICES.get(message.from_user.id) or []
        try:
            selected_idx = int((message.text or "").strip()) - 1
        except ValueError:
            selected_idx = -1
        if selected_idx < 0 or selected_idx >= len(choices):
            await message.answer(
                t(locale, "instruction_invalid_choice"),
                reply_markup=get_instruction_filter_keyboard(locale),
            )
            return
        target_community_id = int(choices[selected_idx])
        INSTRUCTION_GROUP_SELECT_USERS.discard(message.from_user.id)
        INSTRUCTION_GROUP_PROMPT_WAIT_USERS.add(message.from_user.id)
        INSTRUCTION_GROUP_PROMPT_TARGETS[message.from_user.id] = target_community_id
        await message.answer(
            t(locale, "instruction_group_prompt_request"),
            reply_markup=get_instruction_filter_keyboard(locale),
        )
        return


@dp.message(_should_handle_instruction_group_prompt)
async def save_instruction_group_prompt(message: types.Message):
    async for session in get_session():
        user = (
            await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        ).scalar_one_or_none()
        if not user:
            _clear_instruction_states(message.from_user.id)
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        target_community_id = INSTRUCTION_GROUP_PROMPT_TARGETS.get(message.from_user.id)
        if not target_community_id:
            _clear_instruction_states(message.from_user.id)
            await message.answer(
                t(locale, "instruction_invalid_choice"),
                reply_markup=get_instruction_filter_keyboard(locale),
            )
            return
        if not await _ensure_instruction_filter_access(message, session, user, locale):
            _clear_instruction_states(message.from_user.id)
            return
        try:
            await upsert_community_instruction_rule(
                session,
                user_id=user.id,
                community_id=int(target_community_id),
                prompt_text=message.text or "",
            )
            await session.commit()
            _clear_instruction_states(message.from_user.id)
            INSTRUCTION_MENU_USERS.add(message.from_user.id)
            await message.answer(
                t(locale, "instruction_group_saved"),
                reply_markup=get_instruction_filter_keyboard(locale),
            )
            return
        except InstructionPromptError as exc:
            await session.rollback()
            await message.answer(
                t(locale, str(exc)),
                reply_markup=get_instruction_filter_keyboard(locale),
            )
            return


@dp.message(_should_handle_instruction_group_disable_choice)
async def disable_instruction_group_choice(message: types.Message):
    async for session in get_session():
        user = (
            await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        ).scalar_one_or_none()
        if not user:
            _clear_instruction_states(message.from_user.id)
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            return
        locale = get_user_locale(user)
        choices = INSTRUCTION_GROUP_DISABLE_CHOICES.get(message.from_user.id) or []
        try:
            selected_idx = int((message.text or "").strip()) - 1
        except ValueError:
            selected_idx = -1
        if selected_idx < 0 or selected_idx >= len(choices):
            await message.answer(
                t(locale, "instruction_invalid_choice"),
                reply_markup=get_instruction_filter_keyboard(locale),
            )
            return
        changed = await disable_community_instruction_rule(
            session,
            user_id=user.id,
            community_id=int(choices[selected_idx]),
        )
        await session.commit()
        _clear_instruction_states(message.from_user.id)
        INSTRUCTION_MENU_USERS.add(message.from_user.id)
        await message.answer(
            t(locale, "instruction_group_disabled" if changed else "instruction_group_rules_missing"),
            reply_markup=get_instruction_filter_keyboard(locale),
        )
        return


@dp.message(_should_handle_support_request_input)
async def submit_support_request(message: types.Message):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not user:
            SUPPORT_REQUEST_WAIT_USERS.discard(message.from_user.id)
            await message.answer(t(locale, "start_required_register"))
            return

        SUPPORT_REQUEST_WAIT_USERS.discard(message.from_user.id)
        state, req, _plan = await recalculate_user_billing_state(session, user.id)
        premium_active = bool(getattr(req, "premium_active", False) or int(getattr(state, "paid_packs", 0) or 0) > 0)
        now = utcnow()
        username = getattr(user, "username", None) or getattr(message.from_user, "username", None) or "unknown"
        ticket = _create_support_ticket(
            user=user,
            username=username,
            premium_active=premium_active,
            question_text=(message.text or "").strip(),
            now=now,
        )
        session.add(ticket)
        await session.commit()
        await session.refresh(ticket)
        payload = _support_ticket_to_payload(ticket)
        audit("support_request.accepted", **payload)

        if _support_alert_is_configured():
            await _deliver_support_ticket_now_or_schedule_retry(session, ticket)
            await session.commit()
        else:
            ticket.status = "retrying"
            ticket.attempts = int(ticket.attempts or 0) + 1
            ticket.last_error = "alert bot config is missing"
            ticket.next_retry_at = utcnow() + timedelta(seconds=_support_backoff_seconds(ticket.attempts))
            ticket.updated_at = utcnow()
            await session.commit()
            audit(
                "support_request.forward_failed",
                request_id=ticket.request_id,
                telegram_id=ticket.telegram_id,
                attempts=ticket.attempts,
                next_retry_at=str(ticket.next_retry_at),
                error=ticket.last_error,
            )

        await message.answer(
            t(locale, "support_request_received"),
            reply_markup=_main_keyboard_for_user(locale, user, message.from_user.username),
        )
        return


@dp.message(F.text.regexp(r"^(?:@|https?://t\.me/|t\.me/)"))
async def handle_group_subscription(message: types.Message):
    groups, invalid_lines = parse_multiline_group_input(message.text)
    groups = _dedupe_groups(groups)
    locale = await _resolve_message_locale(message)
    if not groups:
        details = ""
        if invalid_lines:
            details = "\n" + t(locale, "invalid_lines", lines="\n".join(invalid_lines))
        await message.answer(
            t(locale, "invalid_group_format", details=details)
        )
        return

    log.info(
        "Received group subscription request",
        user_id=message.from_user.id,
        groups_count=len(groups),
        invalid_count=len(invalid_lines),
    )

    async for session in get_session():
        user_result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = user_result.scalar_one_or_none()
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return
        locale = get_user_locale(user)
        try:
            result = await _apply_subscription_links(
                session,
                user=user,
                links=[link for _, link in groups],
                locale=locale,
                action_mode="toggle",
                invalid_lines=invalid_lines,
            )
        except RuntimeError as exc:
            await session.rollback()
            await message.answer(
                str(exc),
                reply_markup=_settings_keyboard_for_user(locale, user, getattr(message.from_user, "username", None)),
            )
            return
        break

    for group_link in result.unsubscribed:
        try:
            await publish_telethon_event_async({"event": "unsubscribe", "group": group_link})
        except Exception as exc:
            log.error("Failed to publish telethon unsubscribe event", error=str(exc), group_link=group_link, exc_info=True)

    if result.request_ids:
        await _finalize_subscription_request_progress(message, request_ids=result.request_ids, locale=locale)
        await _enqueue_subscription_request_jobs(result.request_ids)

    should_send_summary = bool(
        result.subscribed
        or result.unsubscribed
        or result.already_subscribed
        or result.missing
        or result.already_queued
        or result.invalid_lines
        or (result.queued and not result.request_ids)
    )
    if should_send_summary:
        summary_text = _render_subscription_change_result_text(
            result,
            locale,
            include_queued=not bool(result.request_ids),
        )
        if summary_text:
            await message.answer(summary_text)



@dp.message(Command("summary_on"))
async def summary_on(message: types.Message):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            break
        user.summary_enabled = True
        await session.commit()
        await message.answer(t(get_user_locale(user), "summary_enabled"))
        break


@dp.message(Command("summary_off"))
async def summary_off(message: types.Message):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required_register"))
            break
        user.summary_enabled = False
        await session.commit()
        await message.answer(t(get_user_locale(user), "summary_disabled"))
        break
    
@dp.message(Command("digest_on"))
async def digest_on(message: types.Message):
    async for session in get_session():
        user_id = message.from_user.id
        result = await session.execute(select(User).where(User.telegram_id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required"))
            return

        settings = await session.get(UserDigestSettings, user.id)
        if not settings:
            settings = UserDigestSettings(user_id=user.id, enabled=True)
            session.add(settings)
        settings.enabled = True
        await session.commit()
        await message.answer(t(get_user_locale(user), "digest_on_command"))
        return

@dp.message(Command("digest_off"))
async def digest_off(message: types.Message):
    async for session in get_session():
        user_id = message.from_user.id
        result = await session.execute(select(User).where(User.telegram_id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required"))
            return

        settings = await session.get(UserDigestSettings, user.id)
        if not settings:
            settings = UserDigestSettings(user_id=user.id, enabled=False)
            session.add(settings)
        settings.enabled = False
        await session.commit()
        await message.answer(t(get_user_locale(user), "digest_off_command"))
        return


def _plus_connect_link() -> str | None:
    if not BOT_PLUS_USERNAME:
        return None
    return f"https://t.me/{BOT_PLUS_USERNAME}?start=connect"


def _plus_connect_inline_keyboard(locale: str) -> types.InlineKeyboardMarkup | None:
    link = _plus_connect_link()
    if not link:
        return None
    label = "Открыть второго бота" if locale == "ru" else "Open second bot"
    return types.InlineKeyboardMarkup(
        inline_keyboard=[
            [types.InlineKeyboardButton(text=label, url=link)],
        ]
    )


def _render_plus_delivery_status(prefs: UserDeliveryPreference | None) -> str:
    connected = bool(getattr(prefs, "plus_bot_connected_at", None))
    digest_enabled = bool(getattr(prefs, "digest_to_plus_bot", False))
    storyline_enabled = bool(getattr(prefs, "storyline_to_plus_bot", False))
    return (
        "Настройки отдельного окна (plus bot):\n"
        f"- подключение: {'да' if connected else 'нет'}\n"
        f"- дайджест: {'в отдельный бот' if digest_enabled else 'в основной бот'}\n"
        f"- сюжеты: {'в отдельный бот' if storyline_enabled else 'в основной бот'}"
    )


@dp.message(Command("plus_status"))
async def plus_status(message: types.Message):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return
        prefs = await session.get(UserDeliveryPreference, int(user.id))
        await message.answer(_render_plus_delivery_status(prefs))
        return


@dp.message(Command("plus_connect"))
async def plus_connect(message: types.Message):
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return
        link = _plus_connect_link()
        if not link:
            await message.answer("PLUS bot пока не настроен: отсутствует BOT_PLUS_USERNAME.")
            return
        await message.answer(
            "Открой отдельного бота по ссылке и нажми Start, чтобы подтвердить подключение:\n"
            f"{link}"
        )
        return


async def _set_plus_delivery_flag(message: types.Message, *, kind: str, enabled: bool) -> None:
    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        locale = _user_locale_or_message(user, message)
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return
        prefs = await get_or_create_delivery_prefs(session, int(user.id))
        if kind == "digest":
            prefs.digest_to_plus_bot = bool(enabled)
        elif kind == "storyline":
            prefs.storyline_to_plus_bot = bool(enabled)
        await session.commit()
        connected = bool(getattr(prefs, "plus_bot_connected_at", None))
        if enabled and not connected:
            hint = t(locale, "plus_delivery_connect_required")
            link = _plus_connect_link()
            if link:
                hint = f"{hint}\n{link}"
            connect_kb = _plus_connect_inline_keyboard(locale)
            await message.answer(
                hint,
                reply_markup=connect_kb,
            )
        await message.answer(
            _render_plus_delivery_menu_text(locale=locale, prefs=prefs),
            reply_markup=get_plus_delivery_keyboard(
                locale,
                digest_to_plus_bot=bool(getattr(prefs, "digest_to_plus_bot", False)),
                storyline_to_plus_bot=bool(getattr(prefs, "storyline_to_plus_bot", False)),
            ),
        )
        return


@dp.message(Command("plus_digest_on"))
async def plus_digest_on(message: types.Message):
    await _set_plus_delivery_flag(message, kind="digest", enabled=True)


@dp.message(Command("plus_digest_off"))
async def plus_digest_off(message: types.Message):
    await _set_plus_delivery_flag(message, kind="digest", enabled=False)


@dp.message(Command("plus_story_on"))
async def plus_story_on(message: types.Message):
    await _set_plus_delivery_flag(message, kind="storyline", enabled=True)


@dp.message(Command("plus_story_off"))
async def plus_story_off(message: types.Message):
    await _set_plus_delivery_flag(message, kind="storyline", enabled=False)


@dp.message(_message_matches("plus_delivery_digest_on"))
async def plus_delivery_digest_on_button(message: types.Message):
    await _set_plus_delivery_flag(message, kind="digest", enabled=True)


@dp.message(_message_matches("plus_delivery_digest_off"))
async def plus_delivery_digest_off_button(message: types.Message):
    await _set_plus_delivery_flag(message, kind="digest", enabled=False)


@dp.message(_message_matches("plus_delivery_storyline_on"))
async def plus_delivery_storyline_on_button(message: types.Message):
    await _set_plus_delivery_flag(message, kind="storyline", enabled=True)


@dp.message(_message_matches("plus_delivery_storyline_off"))
async def plus_delivery_storyline_off_button(message: types.Message):
    await _set_plus_delivery_flag(message, kind="storyline", enabled=False)
    

def _strip_reaction_buttons_from_markup(
    reply_markup: types.InlineKeyboardMarkup | None,
) -> types.InlineKeyboardMarkup | None:
    if reply_markup is None:
        return None

    source_rows = getattr(reply_markup, "inline_keyboard", None)
    if source_rows is None:
        source_rows = getattr(reply_markup, "kwargs", {}).get("inline_keyboard")

    rows: list[list[types.InlineKeyboardButton]] = []
    for row in source_rows or []:
        filtered_row = [
            button
            for button in row
            if not str(getattr(button, "callback_data", "") or "").startswith("react:")
            and not str(getattr(button, "kwargs", {}).get("callback_data", "") or "").startswith("react:")
        ]
        if filtered_row:
            rows.append(filtered_row)

    if not rows:
        return None
    markup = types.InlineKeyboardMarkup(inline_keyboard=rows)
    if not hasattr(markup, "inline_keyboard"):
        try:
            setattr(markup, "inline_keyboard", rows)
        except Exception:
            pass
    return markup


@dp.callback_query(F.data.startswith("react:"))
async def handle_reaction(callback: types.CallbackQuery):
    try:
        _, post_id_str, reaction = callback.data.split(":", 2)
        post_id = int(post_id_str)
    except (ValueError, AttributeError, IndexError):
        await callback.answer(t(await _resolve_callback_locale(callback), "invalid_reaction"), show_alert=True)
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == callback.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await callback.answer(t(_button_locale_from_callback(callback), "start_required"), show_alert=True)
            break

        post_result = await session.execute(select(Post).where(Post.id == post_id))
        post = post_result.scalar_one_or_none()
        if not post:
            await callback.answer(t(get_user_locale(user), "post_not_found"), show_alert=True)
            break

        existing = await session.execute(
            select(Reaction).where(Reaction.user_id == user.id, Reaction.post_id == post_id)
        )
        reaction_row = existing.scalar_one_or_none()

        if reaction_row:
            reaction_row.reaction = reaction
        else:
            reaction_row = Reaction(user_id=user.id, post_id=post_id, reaction=reaction)
            session.add(reaction_row)

        if reaction == "fire":
            user.engagement_score += 2.0
        elif reaction == "up":
            user.engagement_score += 1.0
        elif reaction == "down":
            user.engagement_score -= 0.5

        if post.keywords:
            for keyword in post.keywords:
                kw_result = await session.execute(
                    select(UserKeywordStat).where(
                        UserKeywordStat.user_id == user.id,
                        UserKeywordStat.keyword == keyword,
                    )
                )
                kw_stat = kw_result.scalar_one_or_none()
                if not kw_stat:
                    kw_stat = UserKeywordStat(
                        user_id=user.id,
                        keyword=keyword,
                        clicks_count=0,
                        reactions_count=1,
                        score=1.0,
                    )
                    session.add(kw_stat)
                else:
                    kw_stat.reactions_count += 1
                    kw_stat.score += 1.0

        label = 1
        if reaction == "down":
            label = 0
        await update_user_model_from_feedback(
            session,
            user_id=user.id,
            post_id=post.id,
            label=label,
        )

        await session.commit()
        try:
            preserved_markup = _strip_reaction_buttons_from_markup(getattr(callback.message, "reply_markup", None))
            await callback.message.edit_reply_markup(reply_markup=preserved_markup)
        except Exception:
            pass
        await callback.answer(t(get_user_locale(user), "saved"))
        break


def _render_storyline_branch_preview(branches: list[dict]) -> str:
    lines: list[str] = []
    for branch in branches[:5]:
        title = str(branch.get("title") or "").strip() or str(branch.get("storyline_id") or "").strip()
        posts_count = int(branch.get("posts_count") or 0)
        lines.append(f"• {title} ({posts_count})")
    return "\n".join(lines)


def _trim_debug_text(value: str | None, limit: int = 180) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return "—"
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _render_storyline_follow_success_text(locale: str, title: str | None) -> str:
    cleaned_title = _trim_debug_text(title or "Сюжет", limit=160)
    return t(locale, "storyline_follow_success_card", title=html.escape(cleaned_title))


async def _resolve_storyline_follow_limit(session, user: User) -> int:
    return STORYLINE_FOLLOW_LIMIT


def _split_telegram_text(value: str, limit: int = 3500) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph.strip()
        if not candidate:
            continue
        if not current:
            if len(candidate) <= limit:
                current = candidate
                continue
        else:
            combined = current + "\n\n" + candidate
            if len(combined) <= limit:
                current = combined
                continue
            chunks.append(current)
            current = ""

        while len(candidate) > limit:
            chunks.append(candidate[:limit].rstrip())
            candidate = candidate[limit:].lstrip()
        current = candidate
    if current:
        chunks.append(current)
    return chunks


def _storyline_timeline_plain_text(value: str | None) -> str:
    raw = html.unescape(str(value or ""))
    raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
    raw = re.sub(r"</p\s*>", "\n\n", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _storyline_related_callback_data(prefix: str, post_id: int, storyline_id: str | None = None) -> str:
    base = f"{prefix}:{int(post_id)}"
    sid = str(storyline_id or "").strip()
    if not sid:
        return base
    candidate = f"{base}:{sid}"
    return candidate if len(candidate.encode("utf-8")) <= 64 else base


def _storyline_similar_markup(locale: str, post_id: int, storyline_id: str | None = None):
    return types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text=t(locale, "storyline_similar_button"),
                    callback_data=_storyline_related_callback_data("storysimilar", post_id, storyline_id),
                )
            ]
        ]
    )


def _storyline_related_branches_markup(locale: str, post_id: int, branches: list[dict]):
    rows: list[list[types.InlineKeyboardButton]] = []
    for idx, branch in enumerate(branches[:5], start=1):
        storyline_id = str(branch.get("storyline_id") or "").strip()
        if not storyline_id:
            continue
        rows.append(
            [
                types.InlineKeyboardButton(
                    text=t(locale, "storyline_similar_open_timeline", index=idx),
                    callback_data=_storyline_related_callback_data("storysimtimeline", post_id, storyline_id),
                )
            ]
        )
    return types.InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


def _storyline_related_relation_label(locale: str, relation_type: str | None) -> str:
    value = str(relation_type or "").strip()
    labels = {
        "same_family": {"ru": "та же сюжетная семья", "en": "same storyline family"},
        "same_macro_topic": {"ru": "общая большая тема", "en": "same broader topic"},
        "parallel_development": {"ru": "параллельная ветка", "en": "parallel branch"},
        "shared_entities": {"ru": "общие участники", "en": "shared actors"},
        "background_context": {"ru": "контекст", "en": "background context"},
    }
    return labels.get(value, {}).get(locale, labels.get(value, {}).get("en", "связанный сюжет" if locale == "ru" else "related storyline"))


def _clean_related_storyline_fallback_title(value: str | None, storyline_id: str) -> str:
    title = str(value or "").strip()
    title = re.sub(r"\s*\[(general|story|topic|unknown)\]\s*$", "", title, flags=re.IGNORECASE).strip()
    title = re.sub(r"\s+", " ", title).strip(" -–—:;")
    if not title or title.lower() in {"general", "unknown", "storyline"}:
        return storyline_id
    return title


def _fallback_related_storyline_branches(candidates: list[dict], *, limit: int = 5) -> list[dict]:
    branches: list[dict] = []
    seen_ids: set[str] = set()
    for item in candidates:
        storyline_id = str(item.get("storyline_id") or "").strip()
        if not storyline_id or storyline_id in seen_ids:
            continue
        seen_ids.add(storyline_id)
        raw_title = str(item.get("title") or item.get("storyline_title") or "").strip()
        title = _clean_related_storyline_fallback_title(raw_title, storyline_id)
        relation_type = "same_macro_topic" if int(item.get("same_macro_topic") or 0) > 0 else "shared_entities"
        if int(item.get("same_episode") or 0) > 0:
            relation_type = "parallel_development"
        if int(item.get("family_match") or 0) > 0:
            relation_type = "same_family"
        description = (
            "Граф нашел эту ветку рядом с текущим сюжетом. Редакторское описание не удалось подготовить автоматически."
        )
        branches.append(
            {
                "storyline_id": storyline_id,
                "family_root_storyline_id": str(item.get("family_root_storyline_id") or storyline_id).strip(),
                "title": title,
                "description": description,
                "relation_type": relation_type,
                "why_related": "Показан fallback без LLM-оформления.",
                "confidence": 0.0,
            }
        )
        if len(branches) >= limit:
            break
    return branches


async def _enrich_related_storyline_candidates_with_examples(session, candidates: list[dict]) -> list[dict]:
    post_ids = sorted(
        {
            int(post_id)
            for item in candidates
            for post_id in (item.get("source_post_ids") or [])
            if int(post_id or 0) > 0
        }
    )
    if not post_ids:
        return candidates
    try:
        result = await session.execute(select(Post).where(Post.id.in_(post_ids)))
        posts = result.scalars().all()
    except Exception:
        log.exception("storyline_similar_example_posts_lookup_failed", post_count=len(post_ids))
        return candidates

    text_by_id = {
        int(getattr(post, "id", 0) or 0): _trim_debug_text(_storyline_post_text(post), limit=520)
        for post in posts
        if int(getattr(post, "id", 0) or 0) > 0
    }
    enriched: list[dict] = []
    for item in candidates:
        local = dict(item)
        local["example_posts"] = [
            text_by_id[int(post_id)]
            for post_id in (local.get("source_post_ids") or [])
            if int(post_id or 0) in text_by_id and text_by_id[int(post_id)]
        ][:3]
        enriched.append(local)
    return enriched


_RELATED_DB_FALLBACK_STOP_TOKENS = {
    "адвокат",
    "банки",
    "банк",
    "глава",
    "главы",
    "долларов",
    "залог",
    "миллиона",
    "мониторинг",
    "необходимых",
    "подзащитного",
    "президента",
    "финансового",
    "украины",
}

_RELATED_DB_FALLBACK_PROCEDURAL_TERMS = (
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
)

_RELATED_DB_FALLBACK_CONTEXT_TERMS = (
    "гадал",
    "зеленск",
    "миндич",
    "переговор",
    "полит",
    "раскол",
    "умеров",
)

_RELATED_DB_FALLBACK_EXTENSION_TERMS = (
    "гадал",
    "миндич",
    "переговор",
    "полит",
    "раскол",
    "умеров",
    "влияни",
)


def _storyline_related_db_fallback_terms(post_text: str, focus_tokens: list[str]) -> list[str]:
    normalized_text = _normalize_storyline_text(post_text)
    priority_terms = ("ермак", "зеленск", "трамп", "иран", "китай", "миндич", "умеров")
    terms: list[str] = []
    seen: set[str] = set()
    for token in priority_terms:
        if token in normalized_text and token not in seen:
            seen.add(token)
            terms.append(token)
    if terms:
        return terms[:1]
    for raw_token in focus_tokens:
        token = str(raw_token or "").strip().lower()
        if len(token) < 5 or token in seen:
            continue
        if token in _STORYLINE_GENERIC_QUERY_TERMS or token in _RELATED_DB_FALLBACK_STOP_TOKENS:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= 3:
            break
    return terms[:3]


async def _recover_related_storyline_candidates_from_db(
    session,
    *,
    post_id: int,
    post_text: str,
    focus_tokens: list[str],
    payload: dict | None,
    limit: int = 12,
) -> tuple[dict | None, list[dict]]:
    search_terms = _storyline_related_db_fallback_terms(post_text, focus_tokens)
    if not search_terms:
        return payload, []

    text_expr = func.lower(
        func.concat(
            func.coalesce(Post.summary, ""),
            " ",
            func.coalesce(Post.processed_content, ""),
            " ",
            func.coalesce(Post.content, ""),
            " ",
            func.coalesce(Post.title, ""),
        )
    )
    try:
        result = await session.execute(
            select(Post)
            .where(Post.id != int(post_id))
            .where(or_(*[text_expr.contains(term) for term in search_terms]))
            .order_by(Post.id.desc())
            .limit(24)
        )
        related_posts = result.scalars().all()
    except Exception:
        log.exception("storyline_similar_db_fallback_lookup_failed", post_id=post_id, search_terms=search_terms)
        return payload, []

    current_anchor = dict((payload or {}).get("anchor") or {})
    current_storyline_id = str(current_anchor.get("storyline_id") or "").strip()
    current_macro_topic_id = str(current_anchor.get("macro_topic_id") or "").strip()
    seen_storyline_ids: set[str] = {current_storyline_id} if current_storyline_id else set()
    recovered_candidates: list[dict] = []
    recovered_payload: dict | None = payload

    for related_post in related_posts:
        related_post_id = int(getattr(related_post, "id", 0) or 0)
        if related_post_id <= 0:
            continue
        related_text = _storyline_post_text(related_post)
        related_norm = _normalize_storyline_text(related_text)
        if (
            any(term in related_norm for term in _RELATED_DB_FALLBACK_PROCEDURAL_TERMS)
            and not any(term in related_norm for term in _RELATED_DB_FALLBACK_CONTEXT_TERMS)
        ):
            continue
        context = await get_storyline_context_for_post(related_post_id)
        if not context:
            continue
        primary_norm = _normalize_storyline_text(
            " ".join(
                [
                    str(context.get("storyline_title") or ""),
                    str(context.get("storyline_seed_preview") or ""),
                    str(context.get("macro_topic_title") or ""),
                ]
            )
        )
        if not any(term in primary_norm for term in search_terms):
            continue
        if (
            any(term in primary_norm for term in _RELATED_DB_FALLBACK_PROCEDURAL_TERMS)
            and not any(term in primary_norm for term in _RELATED_DB_FALLBACK_EXTENSION_TERMS)
        ):
            continue
        storyline_id = str(context.get("storyline_id") or "").strip()
        if not storyline_id or storyline_id in seen_storyline_ids:
            continue
        seen_storyline_ids.add(storyline_id)
        macro_topic_id = str(context.get("macro_topic_id") or "").strip()
        if current_macro_topic_id and macro_topic_id == current_macro_topic_id:
            continue
        if recovered_payload is None:
            recovered_payload = {"anchor": context, "candidate_branches": [], "query_tokens": list(focus_tokens or [])}
        recovered_candidates.append(
            {
                "storyline_id": storyline_id,
                "family_root_storyline_id": str(context.get("family_root_storyline_id") or storyline_id).strip(),
                "story_family_id": str(context.get("story_family_id") or "").strip(),
                "macro_topic_id": macro_topic_id,
                "macro_topic_title": str(context.get("macro_topic_title") or "").strip(),
                "title": str(context.get("storyline_title") or "").strip(),
                "seed_preview": str(context.get("storyline_seed_preview") or related_text or "").strip()[:600],
                "posts_count": int(context.get("posts_count") or 1),
                "source_post_ids": [related_post_id],
                "focus_token_hits": sum(1 for token in focus_tokens if str(token or "").lower() in related_norm),
                "context_token_hits": sum(1 for token in _RELATED_DB_FALLBACK_CONTEXT_TERMS if token in related_norm),
                "related_score": 20.0,
                "_source": "db_context_fallback",
            }
        )
        if len(recovered_candidates) >= max(2, min(4, limit)):
            break

    log.info(
        "storyline_similar_db_fallback_done",
        post_id=post_id,
        search_terms=search_terms,
        recovered_count=len(recovered_candidates),
        recovered_ids=[str(item.get("storyline_id") or "") for item in recovered_candidates[:6]],
    )
    return recovered_payload, recovered_candidates[:limit]


def _render_related_storyline_branches_text(locale: str, branches: list[dict]) -> str | None:
    if not branches:
        return None
    blocks = [f"<b>{html.escape(t(locale, 'storyline_similar_title'))}</b>"]
    for idx, branch in enumerate(branches[:5], start=1):
        title = _trim_debug_text(branch.get("title") or branch.get("storyline_id") or "", limit=120)
        description = _trim_debug_text(branch.get("description"), limit=460)
        block_lines = [f"<b>{idx}. {html.escape(title)}</b>"]
        if description and description != "—":
            block_lines.append(html.escape(description))
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks).strip()


def _render_storyline_branch_debug_preview(branches: list[dict]) -> str:
    lines: list[str] = []
    for branch in branches[:5]:
        title = _trim_debug_text(branch.get("title") or branch.get("storyline_id") or "—", limit=110)
        storyline_id = str(branch.get("storyline_id") or "—").strip()
        posts_count = int(branch.get("posts_count") or 0)
        seed_preview = _trim_debug_text(branch.get("seed_preview"), limit=140)
        lines.append(
            "\n".join(
                [
                    f"• {title}",
                    f"  id: {storyline_id}",
                    f"  posts: {posts_count}",
                    f"  preview: {seed_preview}",
                ]
            )
        )
    return "\n\n".join(lines) if lines else "—"


def _render_storyline_recovery_candidates_debug_preview(candidates: list[dict]) -> str:
    if not candidates:
        return "—"
    lines: list[str] = []
    for item in candidates[:5]:
        title = _trim_debug_text(
            item.get("title") or item.get("storyline_title") or item.get("storyline_id") or "—",
            limit=110,
        )
        storyline_id = str(item.get("storyline_id") or "—").strip()
        trust_score = float(item.get("anchor_trust_score") or 0.0)
        recovery_score = int(item.get("recovery_score") or 0)
        overlap_count = int(item.get("overlap_count") or 0)
        lines.append(
            f"• {title}\n"
            f"  id: {storyline_id}\n"
            f"  trust: {trust_score:.2f} | recovery_score: {recovery_score} | overlap: {overlap_count}"
        )
    return "\n\n".join(lines)


def _render_storyline_debug_text(locale: str, context: dict) -> str:
    branches = list(context.get("sibling_candidates") or [])
    branch_preview = _render_storyline_branch_debug_preview(branches)
    title = _trim_debug_text(context.get("storyline_title") or context.get("storyline_id") or "—", limit=140)
    seed_preview = _trim_debug_text(context.get("storyline_seed_preview"), limit=260)
    trust_score = float(context.get("anchor_trust_score") or 0.0)
    overlap_count = int(context.get("overlap_count") or 0)
    term_hits = int(context.get("term_hits") or 0)
    jaccard = float(context.get("anchor_jaccard") or 0.0)
    recovery_mode = bool(context.get("recovery_mode"))
    selected_via = str(context.get("selected_via") or "assigned").strip()
    recovery_reason = str(context.get("recovery_reason") or "—").strip()
    assigned_storyline_id = str(context.get("assigned_storyline_id") or "").strip()
    assigned_storyline_title = _trim_debug_text(context.get("assigned_storyline_title"), limit=120)
    recovery_candidates = list(context.get("recovery_candidates") or [])
    recovery_preview = _render_storyline_recovery_candidates_debug_preview(recovery_candidates)

    lines = [
        "Слежение за сюжетом включено." if locale == "ru" else "Storyline follow is enabled.",
        "",
        "Debug storytracking" if locale == "ru" else "Storytracking debug",
        "",
        "Текущий storyline:" if locale == "ru" else "Current storyline:",
        f"- title: {title}",
        f"- storyline_id: {str(context.get('storyline_id') or '—').strip()}",
        f"- story_family_id: {str(context.get('story_family_id') or '—').strip()}",
        f"- family_root_storyline_id: {str(context.get('family_root_storyline_id') or '—').strip()}",
        f"- macro_topic_id: {str(context.get('macro_topic_id') or '—').strip()}",
        f"- macro_topic_title: {_trim_debug_text(context.get('macro_topic_title') or '—', limit=140)}",
        "",
        "Anchor validation:" if locale == "ru" else "Anchor validation:",
        f"- trust_score: {trust_score:.2f}",
        f"- overlap_count: {overlap_count}",
        f"- term_hits: {term_hits}",
        f"- jaccard: {jaccard:.3f}",
        f"- recovery_mode: {'yes' if recovery_mode else 'no'}",
        f"- selected_via: {selected_via}",
        f"- recovery_reason: {recovery_reason}",
    ]
    if assigned_storyline_id and assigned_storyline_id != str(context.get("storyline_id") or "").strip():
        lines.extend(
            [
                "",
                "Исходный assigned storyline:" if locale == "ru" else "Assigned storyline from graph:",
                f"- title: {assigned_storyline_title or '—'}",
                f"- storyline_id: {assigned_storyline_id}",
            ]
        )
    lines.extend(
        [
            "",
            "Seed preview:" if locale == "ru" else "Seed preview:",
            seed_preview,
            "",
            f"Nearby branches ({len(branches)}):",
            branch_preview,
            "",
            f"Recovery candidates ({len(recovery_candidates)}):" if locale == "ru" else f"Recovery candidates ({len(recovery_candidates)}):",
            recovery_preview,
        ]
    )
    return html.escape("\n".join(lines))


def _extract_post_url(post: Post) -> str | None:
    raw = getattr(post, "content_link", None) or {}
    if isinstance(raw, dict):
        value = str(raw.get("url") or "").strip()
        return value or None
    return None


def _format_storyline_post_time(post: Post) -> str:
    ts = getattr(post, "timestamp", None)
    if not ts:
        return "—"
    try:
        return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)


def _storyline_source_name(post: Post, community: Community | None) -> str:
    if community and getattr(community, "name", None):
        return str(community.name)
    raw = getattr(post, "content_link", None) or {}
    if isinstance(raw, dict):
        channel = str(raw.get("channel") or "").strip()
        if channel:
            return channel
    return "unknown"


def _looks_like_url(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith(("http://", "https://", "t.me/"))


def _storyline_source_display_text(raw_label: str | None, raw_url: str | None) -> str:
    label = str(raw_label or "").strip()
    url = str(raw_url or "").strip()
    normalized_label = label.casefold()
    if label and normalized_label != "unknown" and not _looks_like_url(label):
        return label
    if url:
        try:
            parsed = urlparse(url if "://" in url else f"https://{url}")
        except Exception:
            parsed = None
        if parsed is not None:
            host = (parsed.netloc or "").strip().lower()
            path = (parsed.path or "").strip("/")
            if host.endswith("t.me") and path:
                return f"t.me/{path}"
            if host and path:
                return f"{host}/{path}"
            if host:
                return host
    return label or url or "unknown"


def _storyline_source_link_url(
    raw_url: str | None,
    raw_post_id: int,
    telegram_id: int | None,
) -> str:
    source_url = str(raw_url or "").strip()
    if source_url:
        return source_url
    if telegram_id and raw_post_id > 0 and source_link_tracking_enabled():
        return build_tracking_link(raw_post_id, int(telegram_id), source="storyline_timeline")
    return ""


_STORYLINE_STOPWORDS = {
    "что", "это", "как", "или", "для", "при", "над", "под", "после", "перед", "если", "только",
    "когда", "пока", "очень", "снова", "также", "между", "через", "этого", "этой", "этом", "этот",
    "который", "которая", "которые", "уже", "были", "было", "будет", "сегодня", "завтра", "вчера",
    "there", "their", "about", "after", "before", "while", "from", "with", "into", "been", "have",
    "has", "had", "that", "this", "those", "these", "over", "under", "than", "then", "they",
    "them", "were", "will", "would", "could", "should",
    "россия", "россии", "россию", "сша", "израиль", "иран", "дагестан",
}

_STORYLINE_GENERIC_QUERY_TERMS = {
    "война", "войны", "военный", "военные", "удар", "удары", "конфликт", "переговоры",
    "переговоров", "прекращение", "огонь", "огня", "мирные", "мирный", "заявил", "заявила",
    "заявили", "сообщил", "сообщила", "сообщили", "должны", "сделать", "получить", "через",
    "продолжение", "начала", "самый", "самая", "самое", "страны", "страна", "армия",
}

_STORYLINE_SOURCE_EXPANSION_GENERIC_TERMS = {
    "украина", "украины", "украине", "украину", "россия", "россии", "россию", "сша",
    "глава", "главы", "бывший", "бывшая", "после", "перед", "президент", "президента",
    "администрации", "офиса", "должности", "заявил", "заявила", "заявили", "сообщил",
    "сообщила", "сообщили", "подробности", "визитом", "находится",
}

_STORYLINE_BROAD_TOPIC_TERMS = {
    "война", "войны", "конфликт", "операция", "операции", "эскалация", "эскалации",
    "военный", "военные", "специальная", "мобилизация", "мобилизации",
}

_STORYLINE_BROAD_TOPIC_MODIFIERS = {
    "российско", "российский", "российская", "российские", "российских",
    "украинский", "украинская", "украинские", "украинских", "украины",
    "россии", "international", "global", "regional",
}


def _storyline_post_text(post: Post) -> str:
    return (
        str(getattr(post, "summary", None) or "").strip()
        or str(getattr(post, "processed_content", None) or "").strip()
        or str(getattr(post, "content", None) or "").strip()
        or str(getattr(post, "title", None) or "").strip()
    )


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
        if len(tkn) < 4 or tkn in _STORYLINE_STOPWORDS or tkn in seen:
            continue
        seen.add(tkn)
        out.append(tkn)
    return out


def _storyline_jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / union if union else 0.0


def _storyline_term_match_count(value: str | None, anchor_terms: list[str]) -> int:
    hay = _normalize_storyline_text(value)
    if not hay or not anchor_terms:
        return 0
    return sum(1 for term in anchor_terms if term and term in hay)


def _storyline_step_terms(texts: list[str], fallback_terms: list[str] | None = None, limit: int = 10) -> list[str]:
    weighted_tokens: list[str] = []
    for text in texts:
        weighted_tokens.extend(_storyline_tokens(text))
    if len(weighted_tokens) < 5:
        weighted_tokens.extend([str(term or "").strip().lower() for term in (fallback_terms or [])])
    weighted_tokens.sort(key=lambda item: (-len(item), item))
    seen: set[str] = set()
    result: list[str] = []
    for token in weighted_tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def _storyline_source_expansion_terms(step_terms: list[str], fallback_terms: list[str] | None = None, *, limit: int = 8) -> list[str]:
    terms: list[str] = []
    for raw_term in [*list(step_terms or []), *list(fallback_terms or [])]:
        term = _normalize_storyline_text(str(raw_term or "")).strip()
        if len(term) < 4:
            continue
        if term in _STORYLINE_SOURCE_EXPANSION_GENERIC_TERMS:
            continue
        if term in _STORYLINE_GENERIC_QUERY_TERMS or term in _STORYLINE_BROAD_TOPIC_TERMS:
            continue
        if term in terms:
            continue
        terms.append(term)
        if len(terms) >= max(1, int(limit)):
            break
    return terms


def _storyline_best_anchor_excerpt(
    text: str | None,
    *,
    anchor_profile: dict | None,
    anchor_terms: list[str],
    anchor_markers: list[str] | None = None,
    focus_phrases: list[str] | None = None,
    limit: int = 360,
) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    candidates = [
        chunk.strip(" -\t\r\n")
        for chunk in re.split(r"(?<=[.!?])\s+|\n+|[•▪✹]\s+", raw)
        if str(chunk or "").strip()
    ]
    if not candidates:
        return _trim_debug_text(raw, limit=limit)

    best_text = _trim_debug_text(raw, limit=limit)
    best_score = -1.0
    for chunk in candidates[:24]:
        overlap = _storyline_profile_overlap({"seed_preview": chunk}, anchor_profile)
        marker_hits = _storyline_text_marker_hits(chunk, list(anchor_markers or []))
        term_hits = _storyline_term_match_count(chunk, anchor_terms[:10])
        phrase_hits = _storyline_phrase_hits(chunk, list(focus_phrases or []))
        score = (
            marker_hits * 5
            + int(overlap.get("must_phrase_overlap") or 0) * 6
            + int(overlap.get("must_overlap") or 0) * 4
            + phrase_hits * 4
            + term_hits
        )
        if score > best_score or (score == best_score and len(chunk) > len(best_text)):
            best_score = score
            best_text = _trim_debug_text(chunk, limit=limit)
    return best_text


def _storyline_prepare_supplemental_card_text(
    text: str | None,
    *,
    anchor_profile: dict | None,
    anchor_terms: list[str],
    anchor_markers: list[str] | None = None,
    focus_phrases: list[str] | None = None,
) -> str | None:
    excerpt = _storyline_best_anchor_excerpt(
        text,
        anchor_profile=anchor_profile,
        anchor_terms=anchor_terms,
        anchor_markers=anchor_markers,
        focus_phrases=focus_phrases,
    )
    if not excerpt:
        return None

    overlap = _storyline_profile_overlap({"seed_preview": excerpt}, anchor_profile)
    marker_hits = _storyline_text_marker_hits(excerpt, list(anchor_markers or []))
    term_hits = _storyline_term_match_count(excerpt, anchor_terms[:10])
    phrase_hits = _storyline_phrase_hits(excerpt, list(focus_phrases or []))

    if _storyline_profile_must_terms(anchor_profile):
        if (
            int(overlap.get("must_overlap") or 0) == 0
            and int(overlap.get("must_phrase_overlap") or 0) == 0
            and marker_hits < 2
            and phrase_hits == 0
        ):
            return None

    if _is_storyline_summary_garbage(excerpt) and marker_hits < 2 and phrase_hits == 0:
        return None

    if not (
        marker_hits >= 2
        or int(overlap.get("must_phrase_overlap") or 0) >= 1
        or phrase_hits >= 1
        or (int(overlap.get("must_overlap") or 0) >= 2 and term_hits >= 2)
        or (marker_hits >= 1 and term_hits >= 3)
    ):
        return None

    return excerpt


def _storyline_anchor_terms(anchor: dict, anchor_text: str) -> list[str]:
    weighted_tokens: list[str] = []
    primary_tokens = _storyline_tokens(anchor_text)
    weighted_tokens.extend(primary_tokens)

    if len(primary_tokens) < 8:
        for part in [
            str(anchor.get("storyline_title") or "").strip(),
            str(anchor.get("storyline_seed_preview") or "").strip(),
        ]:
            weighted_tokens.extend(_storyline_tokens(part))

    weighted_tokens.sort(key=lambda item: (-len(item), item))
    seen: set[str] = set()
    result: list[str] = []
    for token in weighted_tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= 12:
            break
    return result


def _storyline_named_anchor_terms(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9-]{2,}", raw):
        token = match.group(0).strip().lower()
        if len(token) < 4 or token in _STORYLINE_GENERIC_QUERY_TERMS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _storyline_deterministic_branch_terms(anchor: dict, anchor_text: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for part in [
        str(anchor.get("storyline_title") or "").strip(),
        str(anchor.get("storyline_seed_preview") or "").strip(),
        str(anchor_text or "").strip(),
        str(anchor.get("macro_topic_title") or "").strip(),
    ]:
        if len(result) >= 6:
            break
        for token in _storyline_named_anchor_terms(part):
            if token in seen:
                continue
            seen.add(token)
            result.append(token)
            if len(result) >= 6:
                break

    if len(result) < 4:
        for token in _storyline_anchor_terms(anchor, anchor_text):
            if token in seen or token in _STORYLINE_GENERIC_QUERY_TERMS:
                continue
            seen.add(token)
            result.append(token)
            if len(result) >= 6:
                break
    return result


def _storyline_anchor_markers(anchor: dict, anchor_text: str, *, limit: int = 5) -> list[str]:
    markers: list[str] = []
    seen: set[str] = set()
    for part in [
        str(anchor.get("storyline_title") or "").strip(),
        str(anchor.get("storyline_seed_preview") or "").strip(),
        str(anchor_text or "").strip(),
    ]:
        for token in _storyline_named_anchor_terms(part):
            if token in seen:
                continue
            seen.add(token)
            markers.append(token)
            if len(markers) >= limit:
                return markers
    return markers


def _storyline_candidate_marker_hits(candidate: dict[str, object], markers: list[str]) -> int:
    if not markers:
        return 0
    hay = _normalize_storyline_text(
        " ".join(
            part
            for part in [
                str(candidate.get("title") or "").strip(),
                str(candidate.get("seed_preview") or "").strip(),
                str(candidate.get("macro_topic_title") or "").strip(),
            ]
            if part
        )
    )
    if not hay:
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


def _storyline_text_marker_hits(text: str | None, markers: list[str]) -> int:
    return _storyline_candidate_marker_hits(
        {
            "title": "",
            "seed_preview": str(text or ""),
            "macro_topic_title": "",
        },
        markers,
    )


def _storyline_candidate_phrases(text: str, *, limit: int = 6) -> list[str]:
    tokens = [token for token in _storyline_tokens(text) if token not in _STORYLINE_GENERIC_QUERY_TERMS]
    phrases: list[str] = []
    seen: set[str] = set()
    for size in (3, 2):
        for idx in range(0, max(0, len(tokens) - size + 1)):
            chunk = tokens[idx : idx + size]
            if len(chunk) < size:
                continue
            if any(len(token) < 4 for token in chunk):
                continue
            phrase = " ".join(chunk)
            if phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
            if len(phrases) >= limit:
                return phrases
    return phrases


def _build_storyline_anchor_profile_fallback(post_text: str) -> dict | None:
    text = str(post_text or "").strip()
    if not text:
        return None
    anchor_terms = _storyline_anchor_terms({}, text)
    phrases = _storyline_candidate_phrases(text, limit=6)
    supporting_terms = [term for term in anchor_terms if term not in _STORYLINE_GENERIC_QUERY_TERMS][:8]
    must_terms = phrases[:4] or supporting_terms[:4]
    search_phrases = phrases[:3]
    downweight_terms = [term for term in anchor_terms if term in _STORYLINE_BROAD_TOPIC_TERMS][:6]
    return {
        "story_title": text[:160],
        "core_summary": text[:500],
        "macro_topic": "",
        "topic_terms": supporting_terms[:6],
        "topic_aliases": [],
        "must_terms": must_terms,
        "supporting_terms": supporting_terms[:8],
        "downweight_terms": downweight_terms,
        "search_phrases": search_phrases,
    }


def _storyline_anchor_profile_terms(profile: dict | None, fallback_terms: list[str]) -> list[str]:
    if not isinstance(profile, dict):
        return fallback_terms
    weighted_tokens: list[str] = []
    for field in ("topic_terms", "topic_aliases", "must_terms", "supporting_terms", "search_phrases"):
        raw_items = profile.get(field) or []
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            text = str(item or "").strip()
            if not text:
                continue
            weighted_tokens.extend(_storyline_tokens(text))

    must_terms = {
        token
        for item in (profile.get("must_terms") or [])
        for token in _storyline_tokens(str(item or "").strip())
    }
    topic_terms = {
        token
        for item in (profile.get("topic_terms") or [])
        for token in _storyline_tokens(str(item or "").strip())
    }
    search_phrase_tokens = {
        token
        for item in (profile.get("search_phrases") or [])
        for token in _storyline_tokens(str(item or "").strip())
    }

    weighted_tokens.sort(
        key=lambda item: (
            0 if item in must_terms else 1 if item in topic_terms else 2 if item in search_phrase_tokens else 3,
            0 if item not in _STORYLINE_GENERIC_QUERY_TERMS else 1,
            -len(item),
            item,
        )
    )

    stable_fallback_terms = [
        token
        for token in fallback_terms
        if token and token not in _STORYLINE_GENERIC_QUERY_TERMS
    ][:6] or list(fallback_terms[:6])

    result: list[str] = []
    seen: set[str] = set()
    for token in weighted_tokens[:4]:
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)

    for token in stable_fallback_terms:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)

    for token in weighted_tokens[4:]:
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= 12:
            break

    if len(result) < 8:
        for token in fallback_terms:
            if token in seen:
                continue
            seen.add(token)
            result.append(token)
            if len(result) >= 12:
                break
    return result or fallback_terms


def _storyline_anchor_profile_negative_terms(profile: dict | None) -> list[str]:
    if not isinstance(profile, dict):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in (profile.get("downweight_terms") or []):
        for token in _storyline_tokens(str(item or "").strip()):
            if token in seen:
                continue
            seen.add(token)
            result.append(token)
            if len(result) >= 10:
                return result
    return result


def _storyline_anchor_profile_topic_terms(profile: dict | None) -> list[str]:
    if not isinstance(profile, dict):
        return []
    weighted_tokens: list[str] = []
    for field in ("topic_terms", "topic_aliases"):
        raw_items = profile.get(field) or []
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            weighted_tokens.extend(_storyline_tokens(str(item or "").strip()))
    if profile.get("macro_topic"):
        weighted_tokens.extend(_storyline_tokens(str(profile.get("macro_topic") or "").strip()))
    result: list[str] = []
    seen: set[str] = set()
    for token in weighted_tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= 10:
            break
    return result


def _storyline_anchor_profile_historical_terms(profile: dict | None, fallback_terms: list[str]) -> list[str]:
    if not isinstance(profile, dict):
        return fallback_terms[:8]

    weighted_tokens: list[str] = []
    for field in ("macro_topic",):
        text = str(profile.get(field) or "").strip()
        if text:
            weighted_tokens.extend(_storyline_tokens(text))

    for field in ("topic_aliases", "must_terms", "topic_terms"):
        raw_items = profile.get(field) or []
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            text = str(item or "").strip()
            if text:
                weighted_tokens.extend(_storyline_tokens(text))

    weighted_tokens.sort(
        key=lambda item: (
            0 if item not in _STORYLINE_GENERIC_QUERY_TERMS else 1,
            -len(item),
            item,
        )
    )

    result: list[str] = []
    seen: set[str] = set()
    for token in weighted_tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= 8:
            break

    if len(result) < 3:
        for token in fallback_terms:
            token = str(token or "").strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            result.append(token)
            if len(result) >= 8:
                break
    return result


def _storyline_focus_phrases(profile: dict | None, anchor: dict | None = None, anchor_text: str | None = None) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()

    def _add_phrase(raw: str | None) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        tokens = [token for token in _storyline_tokens(text) if token not in _STORYLINE_GENERIC_QUERY_TERMS]
        if len(tokens) < 2:
            return
        if all(token in _STORYLINE_BROAD_TOPIC_TERMS for token in tokens):
            return
        normalized = " ".join(tokens)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        phrases.append(text)

    current = dict(profile or {})
    for field in ("search_phrases", "must_terms", "topic_aliases"):
        for item in (current.get(field) or []):
            _add_phrase(item)

    if anchor is not None:
        for item in _storyline_graph_entity_phrases(anchor):
            _add_phrase(item)

    if anchor_text:
        for item in _storyline_candidate_phrases(anchor_text, limit=6):
            _add_phrase(item)

    phrases.sort(key=lambda item: (-len(_storyline_tokens(item)), -len(item), item.casefold()))
    return phrases[:8]


def _storyline_phrase_hits(text: str | None, phrases: list[str]) -> int:
    hay = _normalize_storyline_text(text)
    if not hay or not phrases:
        return 0
    padded = f" {hay} "
    hits = 0
    for phrase in phrases:
        tokens = [token for token in _storyline_tokens(phrase) if token not in _STORYLINE_GENERIC_QUERY_TERMS]
        if len(tokens) < 2:
            continue
        normalized = " ".join(tokens)
        if f" {normalized} " in padded:
            hits += 1
    return hits


def _storyline_focus_title(anchor_profile: dict | None, default_title: str) -> str:
    phrases = _storyline_focus_phrases(anchor_profile)
    if not phrases:
        return default_title

    primary = phrases[0]
    secondary = ""
    for phrase in phrases[1:]:
        norm = _normalize_storyline_text(phrase)
        if "ядер" in norm or "сделк" in norm or "меморанд" in norm:
            secondary = phrase
            break

    primary_norm = _normalize_storyline_text(primary)
    if "пролив" in primary_norm:
        if secondary:
            if "ядер" in _normalize_storyline_text(secondary) or "сделк" in _normalize_storyline_text(secondary):
                return "США и Иран: переговоры об открытии Ормузского пролива и ядерной сделке"
        return "США и Иран: переговоры об открытии Ормузского пролива"
    return default_title


def _storyline_focus_overview(
    selected_items: list[dict],
    group_map: dict[str, dict],
    fallback_overview: str,
    anchor_profile: dict | None,
) -> str:
    focus_phrases = _storyline_focus_phrases(anchor_profile)
    relevant_summaries: list[str] = []
    for item in selected_items[:5]:
        item_group_ids = [str(group_id or "").strip() for group_id in item.get("group_ids") or [] if str(group_id or "").strip()]
        item_groups = [group_map[group_id] for group_id in item_group_ids if group_id in group_map]
        summary = str(item.get("_resolved_summary") or "").strip() or _storyline_best_group_summary(item, item_groups)
        if focus_phrases and _storyline_phrase_hits(summary, focus_phrases) == 0:
            continue
        if summary and not _is_storyline_summary_garbage(summary):
            relevant_summaries.append(_storyline_clean_summary_text(summary, limit=220))
        if len(relevant_summaries) >= 3:
            break

    if len(relevant_summaries) >= 2:
        return " ".join(relevant_summaries[:3]).strip()
    return fallback_overview


def _storyline_graph_entity_phrases(anchor: dict | None) -> list[str]:
    current = dict(anchor or {})
    phrases: list[str] = []
    seen: set[str] = set()

    def _add_phrase(raw: str | None, *, allow_single: bool = False) -> None:
        text = str(raw or "").strip().replace("_", " ").replace("-", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return
        tokens = _storyline_tokens(text)
        if len(tokens) < 2 and not allow_single:
            return
        normalized = " ".join(tokens) if tokens else text.casefold()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        phrases.append(text)

    for key in current.get("entity_keys") or []:
        entity_key = str(key or "").strip()
        if not entity_key or ":" not in entity_key:
            continue
        entity_type, entity_value = entity_key.split(":", 1)
        entity_type = entity_type.strip().lower()
        allow_single = entity_type in {"organization", "product", "law"}
        _add_phrase(entity_value, allow_single=allow_single)

    for signature in [
        *(current.get("family_topic_signatures") or []),
        *(current.get("macro_topic_signatures") or []),
        *(current.get("topic_signatures") or []),
    ]:
        value = str(signature or "").strip()
        if not value.startswith("topic_entity:"):
            continue
        _, _, entity_sig = value.partition("topic_entity:")
        if ":" not in entity_sig:
            continue
        entity_type, entity_value = entity_sig.split(":", 1)
        entity_type = entity_type.strip().lower()
        allow_single = entity_type in {"organization", "product", "law"}
        _add_phrase(entity_value, allow_single=allow_single)

    _add_phrase(current.get("macro_topic_title"), allow_single=False)
    return phrases[:10]


def _storyline_enrich_anchor_profile_with_graph_context(profile: dict | None, anchor: dict | None) -> dict | None:
    current = dict(profile or {})
    anchor_current = dict(anchor or {})
    if not current and not anchor_current:
        return profile

    graph_phrases = _storyline_graph_entity_phrases(anchor_current)
    if not graph_phrases:
        return current or profile

    enriched = dict(current)

    def _merge_list(field: str, values: list[str], *, limit: int) -> None:
        merged: list[str] = []
        seen: set[str] = set()
        for item in [*(enriched.get(field) or []), *values]:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(text)
            if len(merged) >= limit:
                break
        enriched[field] = merged

    phrase_values = [phrase for phrase in graph_phrases if len(_storyline_tokens(phrase)) >= 2]
    token_values = []
    seen_tokens: set[str] = set()
    for phrase in graph_phrases:
        for token in _storyline_tokens(phrase):
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            token_values.append(token)

    _merge_list("topic_terms", token_values[:6], limit=8)
    _merge_list("supporting_terms", token_values[:8], limit=10)
    if phrase_values:
        _merge_list("must_terms", phrase_values[:4], limit=8)
        _merge_list("search_phrases", phrase_values[:4], limit=6)
    elif graph_phrases:
        _merge_list("must_terms", graph_phrases[:3], limit=8)

    if not str(enriched.get("macro_topic") or "").strip():
        macro_topic_title = str(anchor_current.get("macro_topic_title") or "").strip()
        if macro_topic_title:
            enriched["macro_topic"] = macro_topic_title

    return enriched


async def _extract_storyline_anchor_profile_safe(post_text: str, *, post_id: int | None = None) -> dict | None:
    try:
        profile = await asyncio.wait_for(
            extract_storyline_anchor_profile(post_text=post_text),
            timeout=STORYLINE_ANCHOR_PROFILE_TIMEOUT,
        )
        normalized = _normalize_storyline_anchor_profile(profile)
        return normalized or _build_storyline_anchor_profile_fallback(post_text)
    except asyncio.TimeoutError:
        return _build_storyline_anchor_profile_fallback(post_text)
    except (DeepseekAuthError, DeepseekRetryableError):
        return _build_storyline_anchor_profile_fallback(post_text)
    except Exception:
        log.exception("storyline_anchor_profile_failed", post_id=post_id)
        return _build_storyline_anchor_profile_fallback(post_text)


def _normalize_storyline_anchor_profile(profile: dict | None) -> dict | None:
    if not isinstance(profile, dict):
        return profile

    normalized = dict(profile)
    must_phrase_terms = _storyline_profile_phrase_terms(normalized, "must_terms")
    search_phrase_terms = _storyline_profile_phrase_terms(normalized, "search_phrases")
    broad_topic_tokens = set(_storyline_tokens(str(normalized.get("macro_topic") or "").strip()))

    has_branch_phrases = bool(must_phrase_terms or search_phrase_terms)
    broad_core_tokens = {token for token in broad_topic_tokens if token in _STORYLINE_BROAD_TOPIC_TERMS}
    broad_modifier_tokens = {token for token in broad_topic_tokens if token in _STORYLINE_BROAD_TOPIC_MODIFIERS}
    broad_macro_topic = bool(broad_core_tokens) and (
        broad_topic_tokens == broad_core_tokens
        or broad_topic_tokens == (broad_core_tokens | broad_modifier_tokens)
    )

    if broad_macro_topic and has_branch_phrases:
        normalized["macro_topic"] = ""

        filtered_topic_terms: list[str] = []
        for item in (normalized.get("topic_terms") or []):
            text = str(item or "").strip()
            if not text:
                continue
            tokens = set(_storyline_tokens(text))
            if tokens and all(token in _STORYLINE_BROAD_TOPIC_TERMS for token in tokens):
                continue
            filtered_topic_terms.append(text)
        normalized["topic_terms"] = filtered_topic_terms[:6]

        filtered_aliases: list[str] = []
        for item in (normalized.get("topic_aliases") or []):
            text = str(item or "").strip()
            if not text:
                continue
            tokens = set(_storyline_tokens(text))
            if tokens and all(token in _STORYLINE_BROAD_TOPIC_TERMS for token in tokens):
                continue
            filtered_aliases.append(text)
        normalized["topic_aliases"] = filtered_aliases[:6]

        downweight_terms = [str(item or "").strip() for item in (normalized.get("downweight_terms") or []) if str(item or "").strip()]
        for broad_term in sorted(broad_topic_tokens):
            if broad_term not in downweight_terms:
                downweight_terms.append(broad_term)
        normalized["downweight_terms"] = downweight_terms[:10]

    return normalized


def _storyline_profile_must_terms(profile: dict | None) -> set[str]:
    if not isinstance(profile, dict):
        return set()
    out: set[str] = set()
    for item in (profile.get("must_terms") or []):
        out.update(_storyline_tokens(str(item or "").strip()))
    return out


def _storyline_profile_phrase_terms(profile: dict | None, field: str) -> list[set[str]]:
    if not isinstance(profile, dict):
        return []
    out: list[set[str]] = []
    for item in (profile.get(field) or []):
        tokens = set(_storyline_tokens(str(item or "").strip()))
        if len(tokens) >= 2:
            out.append(tokens)
    return out


def _storyline_profile_overlap(candidate: dict | None, profile: dict | None) -> dict[str, int]:
    current = dict(candidate or {})
    text = " ".join(
        part
        for part in [
            str(current.get("storyline_title") or current.get("title") or "").strip(),
            str(current.get("storyline_seed_preview") or current.get("seed_preview") or "").strip(),
        ]
        if part
    )
    candidate_tokens = set(_storyline_tokens(text))
    must_terms = _storyline_profile_must_terms(profile)
    negative_terms = set(_storyline_anchor_profile_negative_terms(profile))
    must_phrase_terms = _storyline_profile_phrase_terms(profile, "must_terms")
    negative_phrase_terms = _storyline_profile_phrase_terms(profile, "downweight_terms")
    return {
        "must_overlap": len(candidate_tokens & must_terms) if must_terms else 0,
        "negative_overlap": len(candidate_tokens & negative_terms) if negative_terms else 0,
        "must_phrase_overlap": sum(1 for tokens in must_phrase_terms if tokens <= candidate_tokens),
        "negative_phrase_overlap": sum(1 for tokens in negative_phrase_terms if tokens <= candidate_tokens),
    }


def _assess_storyline_anchor(post_text: str, context: dict | None) -> dict[str, float | int | bool]:
    current = dict(context or {})
    post_terms = _storyline_anchor_terms({}, post_text)
    post_tokens = set(post_terms or _storyline_tokens(post_text))
    storyline_text = " ".join(
        part
        for part in [
            str(current.get("storyline_title") or current.get("title") or "").strip(),
            str(current.get("storyline_seed_preview") or current.get("seed_preview") or "").strip(),
        ]
        if part
    )
    storyline_tokens = set(_storyline_tokens(storyline_text))
    overlap_count = len(post_tokens & storyline_tokens)
    term_hits = _storyline_term_match_count(storyline_text, post_terms[:10])
    anchor_jaccard = _storyline_jaccard(post_tokens, storyline_tokens)
    trust_score = overlap_count * 3.0 + term_hits * 2.0 + anchor_jaccard * 10.0
    trusted = overlap_count >= 2 or term_hits >= 2 or anchor_jaccard >= 0.16 or trust_score >= 6.0
    clearly_bad = overlap_count == 0 and term_hits == 0 and anchor_jaccard < 0.05
    return {
        "overlap_count": overlap_count,
        "term_hits": term_hits,
        "anchor_jaccard": round(anchor_jaccard, 4),
        "anchor_trust_score": round(trust_score, 3),
        "anchor_trusted": trusted,
        "anchor_clearly_bad": clearly_bad,
    }


def _should_recover_storyline_anchor(metrics: dict | None) -> bool:
    current = dict(metrics or {})
    if bool(current.get("anchor_clearly_bad")):
        return True
    return not bool(current.get("anchor_trusted"))


def _storyline_rescue_anchor_markers(post_text: str, anchor_profile: dict | None, *, limit: int = 6) -> list[str]:
    markers: list[str] = []
    seen: set[str] = set()
    for marker in _storyline_anchor_markers({}, post_text, limit=limit):
        text = str(marker or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        markers.append(text)
        if len(markers) >= limit:
            return markers

    for term in _storyline_anchor_profile_terms(anchor_profile, _storyline_anchor_terms({}, post_text)):
        text = str(term or "").strip().lower()
        if len(text) < 6 or text in seen or text in _STORYLINE_GENERIC_QUERY_TERMS:
            continue
        seen.add(text)
        markers.append(text)
        if len(markers) >= limit:
            break
    return markers


def _score_storyline_anchor_candidate(
    post_text: str,
    context: dict | None,
    *,
    anchor_profile: dict | None,
    anchor_markers: list[str] | None = None,
    support_count: int = 1,
) -> dict[str, object]:
    scored = dict(context or {})
    scored.update(_assess_storyline_anchor(post_text, scored))
    scored.update(_storyline_profile_overlap(scored, anchor_profile))
    scored["anchor_marker_hits"] = _storyline_candidate_marker_hits(scored, list(anchor_markers or []))
    scored["support_count"] = max(1, int(support_count or 1))
    adjusted_score = (
        float(scored.get("anchor_trust_score") or 0.0)
        + int(scored.get("must_overlap") or 0) * 6
        + int(scored.get("must_phrase_overlap") or 0) * 8
        + int(scored.get("anchor_marker_hits") or 0) * 5
        + int(scored.get("support_count") or 0) * 4
        - int(scored.get("negative_overlap") or 0) * 3
        - int(scored.get("negative_phrase_overlap") or 0) * 4
    )
    scored["adjusted_recovery_score"] = round(adjusted_score, 3)
    return scored


def _storyline_anchor_candidate_is_plausible(candidate: dict | None, anchor_profile: dict | None) -> bool:
    current = dict(candidate or {})
    must_terms = _storyline_profile_must_terms(anchor_profile)
    if must_terms:
        if (
            int(current.get("must_overlap") or 0) == 0
            and int(current.get("must_phrase_overlap") or 0) == 0
            and int(current.get("anchor_marker_hits") or 0) == 0
        ):
            return False
    return (
        bool(current.get("anchor_trusted"))
        or int(current.get("must_overlap") or 0) >= 1
        or int(current.get("must_phrase_overlap") or 0) >= 1
        or int(current.get("anchor_marker_hits") or 0) >= 2
        or float(current.get("adjusted_recovery_score") or 0.0) >= 10.0
    )


def _storyline_assigned_anchor_has_profile_support(candidate: dict | None) -> bool:
    """Keep a multi-post assigned anchor when it matches the click post's core profile."""
    current = dict(candidate or {})
    if int(current.get("posts_count") or 0) < 2:
        return False

    must_overlap = int(current.get("must_overlap") or 0)
    must_phrase_overlap = int(current.get("must_phrase_overlap") or 0)
    marker_hits = int(current.get("anchor_marker_hits") or 0)
    adjusted_score = float(current.get("adjusted_recovery_score") or current.get("anchor_trust_score") or 0.0)

    if must_phrase_overlap >= 1 and must_overlap >= 2 and (marker_hits >= 1 or adjusted_score >= 20.0):
        return True
    if marker_hits >= 2 and must_overlap >= 1:
        return True
    return False


async def _recover_storyline_context_from_neighbor_posts(
    session,
    post_id: int,
    post_text: str,
    *,
    anchor_profile: dict | None = None,
    anchor_post_ts: datetime | None = None,
    assigned_context: dict | None = None,
) -> dict | None:
    if session is None or anchor_post_ts is None:
        return None

    rescue_markers = _storyline_rescue_anchor_markers(post_text, anchor_profile)
    if len(rescue_markers) < 2:
        return None

    anchor_terms = _storyline_anchor_profile_terms(anchor_profile, _storyline_anchor_terms({}, post_text))
    negative_terms = _storyline_anchor_profile_negative_terms(anchor_profile)
    candidate_post_ids = await _expand_storyline_exact_marker_post_ids(
        session,
        anchor_markers=rescue_markers,
        exclude_post_ids={int(post_id)},
        anchor_post_ts=anchor_post_ts,
    )
    if len(candidate_post_ids) < 3:
        lexical_post_ids = await _expand_storyline_lexical_post_ids(
            session,
            anchor_terms=anchor_terms,
            exclude_post_ids={int(post_id), *candidate_post_ids},
            anchor_post_ts=anchor_post_ts,
            negative_terms=negative_terms,
        )
        candidate_post_ids |= lexical_post_ids

    candidate_contexts: dict[str, dict[str, object]] = {}
    support_posts: dict[str, set[int]] = {}
    assigned_storyline_id = str((assigned_context or {}).get("storyline_id") or "").strip()
    for candidate_post_id in sorted(candidate_post_ids):
        try:
            pid = int(candidate_post_id)
        except Exception:
            continue
        context = await get_storyline_context_for_post(pid)
        current = dict(context or {})
        storyline_id = str(current.get("storyline_id") or "").strip()
        if not storyline_id or storyline_id == assigned_storyline_id:
            continue
        candidate_contexts.setdefault(storyline_id, current)
        support_posts.setdefault(storyline_id, set()).add(pid)

    if not candidate_contexts:
        return None
    if _storyline_assigned_anchor_has_profile_support(assigned_context):
        return None

    scored_candidates: list[dict[str, object]] = []
    for storyline_id, current in candidate_contexts.items():
        support_count = len(support_posts.get(storyline_id) or ())
        scored = _score_storyline_anchor_candidate(
            post_text,
            current,
            anchor_profile=anchor_profile,
            anchor_markers=rescue_markers,
            support_count=support_count,
        )
        scored["support_post_ids"] = sorted(support_posts.get(storyline_id) or ())
        if not _storyline_anchor_candidate_is_plausible(scored, anchor_profile):
            continue
        scored_candidates.append(scored)

    if not scored_candidates:
        return None

    scored_candidates.sort(
        key=lambda item: (
            float(item.get("adjusted_recovery_score") or 0.0),
            int(item.get("support_count") or 0),
            int(item.get("must_phrase_overlap") or 0),
            int(item.get("must_overlap") or 0),
            int(item.get("anchor_marker_hits") or 0),
            float(item.get("anchor_trust_score") or 0.0),
        ),
        reverse=True,
    )

    top_candidate = scored_candidates[0]
    assigned_score = float((assigned_context or {}).get("adjusted_recovery_score") or (assigned_context or {}).get("anchor_trust_score") or 0.0)
    top_score = float(top_candidate.get("adjusted_recovery_score") or top_candidate.get("anchor_trust_score") or 0.0)
    if top_score < max(7.0, assigned_score + 2.0):
        if not (
            int(top_candidate.get("support_count") or 0) >= 2
            and top_score >= assigned_score + 1.0
        ):
            return None

    top_candidate["assigned_storyline_id"] = assigned_storyline_id
    top_candidate["assigned_storyline_title"] = str((assigned_context or {}).get("storyline_title") or "").strip()
    top_candidate["assigned_storyline_seed_preview"] = str((assigned_context or {}).get("storyline_seed_preview") or "").strip()
    top_candidate["recovery_mode"] = True
    top_candidate["selected_via"] = "neighbor_exact_marker_recovery"
    top_candidate["recovery_reason"] = "neighbor_storyline_has_stronger_text_match"
    top_candidate["recovery_candidates"] = scored_candidates[:5]
    return top_candidate


async def _resolve_storyline_follow_context(
    post_id: int,
    post_text: str,
    *,
    anchor_profile: dict | None = None,
    session=None,
    anchor_post_ts: datetime | None = None,
) -> dict | None:
    assigned_context = await get_storyline_context_for_post(post_id)
    if not assigned_context:
        return None

    assigned_context = _score_storyline_anchor_candidate(
        post_text,
        assigned_context,
        anchor_profile=anchor_profile,
        anchor_markers=_storyline_rescue_anchor_markers(post_text, anchor_profile),
    )
    assigned_context["selected_via"] = "assigned"
    assigned_context["recovery_mode"] = False
    assigned_context["recovery_reason"] = "assigned_anchor_trusted"
    assigned_context["recovery_candidates"] = []

    if _storyline_assigned_anchor_has_profile_support(assigned_context):
        assigned_context["recovery_reason"] = "assigned_anchor_profile_supported"
        return assigned_context
    if int(assigned_context.get("must_overlap") or 0) >= 1 and float(assigned_context.get("anchor_trust_score") or 0.0) >= 6.0:
        return assigned_context
    if not _should_recover_storyline_anchor(assigned_context):
        return assigned_context

    query_tokens = _storyline_anchor_profile_terms(anchor_profile, _storyline_anchor_terms({}, post_text))
    recovery_candidates = await search_storyline_recovery_candidates(
        post_text,
        query_tokens=query_tokens,
        limit=8,
        exclude_storyline_id=str(assigned_context.get("storyline_id") or "").strip(),
    )

    scored_candidates: list[dict] = []
    seen_storyline_ids: set[str] = set()
    for item in recovery_candidates:
        scored = dict(item)
        storyline_id = str(scored.get("storyline_id") or "").strip()
        if not storyline_id or storyline_id in seen_storyline_ids:
            continue
        seen_storyline_ids.add(storyline_id)
        scored["storyline_title"] = str(item.get("title") or "").strip()
        scored["storyline_seed_preview"] = str(item.get("seed_preview") or "").strip()
        scored = _score_storyline_anchor_candidate(
            post_text,
            scored,
            anchor_profile=anchor_profile,
            anchor_markers=_storyline_rescue_anchor_markers(post_text, anchor_profile),
        )
        scored["adjusted_recovery_score"] = round(
            float(scored.get("adjusted_recovery_score") or 0.0)
            + int(scored.get("recovery_score") or 0),
            3,
        )
        if _storyline_profile_must_terms(anchor_profile) and int(scored.get("must_overlap") or 0) == 0:
            continue
        scored_candidates.append(scored)

    scored_candidates.sort(
        key=lambda item: (
            float(item.get("adjusted_recovery_score") or 0.0),
            int(item.get("must_overlap") or 0),
            float(item.get("anchor_trust_score") or 0.0),
            int(item.get("recovery_score") or 0),
            int(item.get("overlap_count") or 0),
            int(item.get("term_hits") or 0),
        ),
        reverse=True,
    )
    assigned_context["recovery_candidates"] = scored_candidates[:5]
    if not scored_candidates:
        neighbor_recovery = await _recover_storyline_context_from_neighbor_posts(
            session,
            post_id,
            post_text,
            anchor_profile=anchor_profile,
            anchor_post_ts=anchor_post_ts,
            assigned_context=assigned_context,
        )
        if neighbor_recovery:
            return neighbor_recovery
        assigned_context["recovery_reason"] = "assigned_anchor_low_trust_no_candidates"
        return assigned_context

    top_candidate = scored_candidates[0]
    selected_storyline_id = ""
    selected_via = "assigned"
    recovery_reason = "assigned_anchor_low_trust"

    try:
        llm_result = await asyncio.wait_for(
            arbitrate_storyline_follow_target(
                post_text=post_text,
                assigned_context=assigned_context,
                recovery_candidates=scored_candidates[:6],
            ),
            timeout=STORYLINE_FOLLOW_TARGET_TIMEOUT,
        )
    except asyncio.TimeoutError:
        llm_result = None
        recovery_reason = "follow_target_arbiter_timeout"
    except (DeepseekAuthError, DeepseekRetryableError):
        llm_result = None
        recovery_reason = "follow_target_arbiter_unavailable"
    except Exception:
        log.exception("storyline_follow_target_arbiter_failed", post_id=post_id)
        llm_result = None
        recovery_reason = "follow_target_arbiter_failed"

    if isinstance(llm_result, dict):
        if bool(llm_result.get("assigned_is_valid")):
            assigned_context["recovery_reason"] = str(llm_result.get("reason") or "llm_validated_assigned")
            return assigned_context
        candidate_ids = {str(item.get("storyline_id") or "").strip() for item in scored_candidates}
        llm_selected = str(llm_result.get("selected_storyline_id") or "").strip()
        if llm_selected and llm_selected in candidate_ids:
            selected_storyline_id = llm_selected
            selected_via = "llm_recovery"
            recovery_reason = str(llm_result.get("reason") or "llm_selected_recovery_candidate")

    if not selected_storyline_id:
        top_trust = float(top_candidate.get("adjusted_recovery_score") or top_candidate.get("anchor_trust_score") or 0.0)
        assigned_trust = float(assigned_context.get("anchor_trust_score") or 0.0)
        if top_trust >= max(5.0, assigned_trust + 2.0):
            selected_storyline_id = str(top_candidate.get("storyline_id") or "").strip()
            selected_via = "score_recovery"
            recovery_reason = "recovery_candidate_has_stronger_text_match"

    if not selected_storyline_id:
        assigned_context["recovery_reason"] = recovery_reason
        return assigned_context

    recovered_context = await get_storyline_context_for_storyline_id(selected_storyline_id)
    if not recovered_context:
        assigned_context["recovery_reason"] = "recovery_selected_context_missing"
        return assigned_context

    recovered_context = dict(recovered_context)
    recovered_context.update(_assess_storyline_anchor(post_text, recovered_context))
    recovered_context.update(_storyline_profile_overlap(recovered_context, anchor_profile))
    recovered_context["assigned_storyline_id"] = str(assigned_context.get("storyline_id") or "").strip()
    recovered_context["assigned_storyline_title"] = str(assigned_context.get("storyline_title") or "").strip()
    recovered_context["assigned_storyline_seed_preview"] = str(assigned_context.get("storyline_seed_preview") or "").strip()
    recovered_context["recovery_mode"] = True
    recovered_context["selected_via"] = selected_via
    recovered_context["recovery_reason"] = recovery_reason
    recovered_context["recovery_candidates"] = scored_candidates[:5]
    return recovered_context


def _is_storyline_noise_text(value: str | None) -> bool:
    text = _normalize_storyline_text(value)
    if len(text) < 20:
        return True
    return len(_storyline_tokens(text)) < 3


def _is_storyline_summary_garbage(value: str | None) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    norm = _normalize_storyline_text(text)
    if len(norm) < 30:
        return True
    tokens = _storyline_tokens(text)
    if len(tokens) < 4:
        return True
    punct_count = sum(text.count(ch) for ch in ".!?")
    if punct_count == 0 and len(tokens) >= 12:
        return True
    lower = text.lower()
    if lower == norm and punct_count == 0:
        return True
    return False


def _storyline_clean_summary_text(value: str | None, *, limit: int = 280) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    text = re.sub(r"^[\W_]+", "", text).strip()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    split_chunks = re.split(r"(?:\s+[|•]\s+|\s+👉|👉\s*|\s+—\s+)", text)
    candidates = [chunk.strip(" -\u2014") for chunk in split_chunks if str(chunk).strip(" -\u2014")]
    sentence_candidates: list[str] = []
    for chunk in candidates or [text]:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk):
            cleaned = sentence.strip(" -\u2014")
            if cleaned:
                sentence_candidates.append(cleaned)

    def _score(candidate: str) -> tuple[int, int, int]:
        garbage_penalty = 0 if _is_storyline_summary_garbage(candidate) else 1
        token_count = len(_storyline_tokens(candidate))
        return (garbage_penalty, token_count, len(candidate))

    best = max(sentence_candidates or [text], key=_score)
    return _trim_debug_text(best, limit=limit)


def _storyline_best_group_summary(item: dict, item_groups: list[dict]) -> str:
    candidate = str(item.get("summary") or "").strip()
    if candidate and not _is_storyline_summary_garbage(candidate):
        return _storyline_clean_summary_text(candidate, limit=280)

    source_texts: list[str] = []
    for group in item_groups[:2]:
        rep = str(group.get("representative_text") or "").strip()
        if rep:
            source_texts.append(rep)
    for text in source_texts:
        if not _is_storyline_summary_garbage(text):
            return _storyline_clean_summary_text(text, limit=280)

    summary_fallback = " ".join(
        _trim_debug_text(str(group.get("representative_text") or "").strip(), limit=180)
        for group in item_groups[:2]
    ).strip()
    return _storyline_clean_summary_text(summary_fallback, limit=280)


def _storyline_source_key(src: dict) -> tuple[int, str]:
    return (int(src.get("post_id") or 0), str(src.get("url") or "").strip())


def _storyline_summary_quality(summary: str | None) -> tuple[int, int, int]:
    text = str(summary or "").strip()
    if not text:
        return (0, 0, 0)
    garbage_penalty = 0 if _is_storyline_summary_garbage(text) else 1
    return (garbage_penalty, len(_storyline_tokens(text)), len(text))


def _storyline_pick_better_summary(*candidates: str | None) -> str:
    cleaned = [str(candidate or "").strip() for candidate in candidates if str(candidate or "").strip()]
    if not cleaned:
        return ""
    return max(cleaned, key=_storyline_summary_quality)


def _storyline_item_time_bounds(item: dict, group_map: dict[str, dict]) -> tuple[datetime | None, datetime | None]:
    timestamps: list[datetime] = []
    for src in item.get("expanded_sources") or []:
        try:
            value = str(src.get("timestamp") or "").strip()
            if value:
                timestamps.append(datetime.fromisoformat(value))
        except ValueError:
            continue
    if not timestamps:
        for group_id in item.get("group_ids") or []:
            group = group_map.get(str(group_id or "").strip())
            if not group:
                continue
            for field in ("start_timestamp", "end_timestamp"):
                try:
                    value = str(group.get(field) or "").strip()
                    if value:
                        timestamps.append(datetime.fromisoformat(value))
                except ValueError:
                    continue
    if not timestamps:
        return (None, None)
    return (min(timestamps), max(timestamps))


def _storyline_item_macro_topic_ids(item: dict, group_map: dict[str, dict]) -> set[str]:
    macro_topic_ids: set[str] = set()
    for group_id in item.get("group_ids") or []:
        group = group_map.get(str(group_id or "").strip())
        if not group:
            continue
        macro_topic_id = str(group.get("macro_topic_id") or "").strip()
        if macro_topic_id:
            macro_topic_ids.add(macro_topic_id)
    return macro_topic_ids


def _storyline_item_post_ids(item: dict, group_map: dict[str, dict]) -> set[int]:
    post_ids: set[int] = set()
    for src in item.get("expanded_sources") or []:
        try:
            pid = int(src.get("post_id") or 0)
        except Exception:
            pid = 0
        if pid > 0:
            post_ids.add(pid)
    if post_ids:
        return post_ids
    for group_id in item.get("group_ids") or []:
        group = group_map.get(str(group_id or "").strip())
        if not group:
            continue
        for raw_pid in group.get("post_ids") or []:
            try:
                pid = int(raw_pid or 0)
            except Exception:
                pid = 0
            if pid > 0:
                post_ids.add(pid)
    return post_ids


def _storyline_group_time(group: dict) -> datetime | None:
    for field in ("start_timestamp", "end_timestamp"):
        try:
            value = str(group.get(field) or "").strip()
            if value:
                return datetime.fromisoformat(value)
        except ValueError:
            continue
    return None


def _storyline_group_in_time_window(
    group: dict,
    *,
    center_ts: datetime | None,
    days_before: int = 21,
    days_after: int = 14,
) -> bool:
    if center_ts is None:
        return True
    group_ts = _storyline_group_time(group)
    if group_ts is None:
        return True
    return center_ts - timedelta(days=days_before) <= group_ts <= center_ts + timedelta(days=days_after)


def _storyline_enrich_group_anchor_relevance(
    raw_group: dict[str, object],
    *,
    anchor_profile: dict | None,
    anchor_markers: list[str] | None = None,
) -> dict[str, object]:
    group = dict(raw_group)
    group_text = _storyline_group_anchor_text(group)
    overlap = _storyline_profile_overlap(
        {
            "title": group_text,
            "seed_preview": group_text,
        },
        anchor_profile,
    )
    normalized_markers = [str(item or "").strip().lower() for item in (anchor_markers or []) if str(item or "").strip()]
    focus_phrases = _storyline_focus_phrases(anchor_profile)
    marker_hits = _storyline_text_marker_hits(group_text, normalized_markers)
    focus_phrase_hits = _storyline_phrase_hits(group_text, focus_phrases)
    must_overlap = int(overlap.get("must_overlap") or 0)
    must_phrase_overlap = int(overlap.get("must_phrase_overlap") or 0)
    negative_overlap = int(overlap.get("negative_overlap") or 0)
    negative_phrase_overlap = int(overlap.get("negative_phrase_overlap") or 0)
    source_count = int(group.get("source_count") or 0)
    post_count = int(group.get("post_count") or 0)

    relevance_score = (
        focus_phrase_hits * 15
        + marker_hits * 12
        + must_phrase_overlap * 10
        + must_overlap * 7
        + min(source_count, 4)
        + min(post_count, 4)
        - negative_overlap * 4
        - negative_phrase_overlap * 7
    )
    group["focus_phrase_hits"] = focus_phrase_hits
    group["anchor_marker_hits"] = marker_hits
    group["must_overlap"] = must_overlap
    group["must_phrase_overlap"] = must_phrase_overlap
    group["negative_overlap"] = negative_overlap
    group["negative_phrase_overlap"] = negative_phrase_overlap
    group["anchor_relevance_score"] = relevance_score
    return group


def _storyline_item_anchor_relevance(item: dict, group_map: dict[str, dict]) -> int:
    scores: list[int] = []
    for group_id in item.get("group_ids") or []:
        group = group_map.get(str(group_id or "").strip())
        if not group:
            continue
        scores.append(int(group.get("anchor_relevance_score") or 0))
    return max(scores) if scores else 0


def _storyline_group_has_strong_anchor_signal(
    *,
    relevance: int,
    marker_hits: int,
    focus_phrase_hits: int,
    must_overlap: int,
    must_phrase_overlap: int,
    summary_garbage: bool,
) -> bool:
    if summary_garbage:
        return False
    if focus_phrase_hits >= 1:
        return True
    if must_phrase_overlap >= 1:
        return True
    if marker_hits >= 2:
        return True
    if marker_hits >= 1 and must_overlap >= 1:
        return True
    if must_overlap >= 2 and relevance >= 10:
        return True
    if must_overlap >= 1 and relevance >= 14 and marker_hits >= 1:
        return True
    return False


def _storyline_filter_selected_items_for_anchor(selected_items: list[dict], group_map: dict[str, dict]) -> list[dict]:
    if not selected_items:
        return []

    filtered: list[dict] = []
    for item in selected_items:
        relevance = _storyline_item_anchor_relevance(item, group_map)
        item_group_ids = [str(group_id or "").strip() for group_id in item.get("group_ids") or [] if str(group_id or "").strip()]
        item_groups = [group_map[group_id] for group_id in item_group_ids if group_id in group_map]
        marker_hits = max((int(group.get("anchor_marker_hits") or 0) for group in item_groups), default=0)
        focus_phrase_hits = max((int(group.get("focus_phrase_hits") or 0) for group in item_groups), default=0)
        must_overlap = max((int(group.get("must_overlap") or 0) for group in item_groups), default=0)
        must_phrase_overlap = max((int(group.get("must_phrase_overlap") or 0) for group in item_groups), default=0)
        summary_text = str(item.get("_resolved_summary") or "").strip() or _storyline_best_group_summary(item, item_groups)
        summary_garbage = _is_storyline_summary_garbage(summary_text)
        if _storyline_group_has_strong_anchor_signal(
            relevance=relevance,
            marker_hits=marker_hits,
            focus_phrase_hits=focus_phrase_hits,
            must_overlap=must_overlap,
            must_phrase_overlap=must_phrase_overlap,
            summary_garbage=summary_garbage,
        ):
            filtered.append(item)

    return filtered


def _storyline_backfill_selected_items(
    selected_items: list[dict],
    groups: list[dict[str, object]],
    group_map: dict[str, dict],
    *,
    min_items: int = 3,
) -> list[dict]:
    if len(selected_items) >= max(1, int(min_items)):
        return selected_items

    selected_group_ids = {
        str(group_id or "").strip()
        for item in selected_items
        for group_id in (item.get("group_ids") or [])
        if str(group_id or "").strip()
    }
    latest_selected_ts: datetime | None = None
    for item in selected_items:
        _start_ts, end_ts = _storyline_item_time_bounds(item, group_map)
        current_ts = end_ts or _start_ts
        if current_ts is None:
            continue
        if latest_selected_ts is None or current_ts > latest_selected_ts:
            latest_selected_ts = current_ts

    candidates: list[dict[str, object]] = []
    for group in groups:
        group_id = str(group.get("group_id") or "").strip()
        if not group_id or group_id in selected_group_ids:
            continue
        group_time = _storyline_group_time(group)
        if latest_selected_ts is not None and group_time is not None:
            if group_time < latest_selected_ts - timedelta(days=7):
                continue
        relevance = int(group.get("anchor_relevance_score") or 0)
        marker_hits = int(group.get("anchor_marker_hits") or 0)
        focus_phrase_hits = int(group.get("focus_phrase_hits") or 0)
        must_overlap = int(group.get("must_overlap") or 0)
        must_phrase_overlap = int(group.get("must_phrase_overlap") or 0)
        summary_text = _storyline_clean_summary_text(str(group.get("representative_text") or "").strip(), limit=280)
        summary_garbage = _is_storyline_summary_garbage(summary_text)
        if not _storyline_group_has_strong_anchor_signal(
            relevance=relevance,
            marker_hits=marker_hits,
            focus_phrase_hits=focus_phrase_hits,
            must_overlap=must_overlap,
            must_phrase_overlap=must_phrase_overlap,
            summary_garbage=summary_garbage,
        ):
            continue
        candidates.append(group)

    candidates = sorted(
        candidates,
        key=lambda group: (
            -int(group.get("anchor_relevance_score") or 0),
            -int(group.get("source_count") or 0),
            str(group.get("end_timestamp") or group.get("start_timestamp") or ""),
        ),
    )

    enriched = list(selected_items)
    for group in candidates:
        group_id = str(group.get("group_id") or "").strip()
        if not group_id or group_id in selected_group_ids:
            continue
        enriched.append(
            {
                "group_ids": [group_id],
                "summary": _storyline_clean_summary_text(str(group.get("representative_text") or "").strip(), limit=280),
            }
        )
        selected_group_ids.add(group_id)
        if len(enriched) >= max(1, int(min_items)):
            break

    return enriched


def _storyline_backfill_distinct_timeline_items_after_merge(
    merged_items: list[dict],
    groups: list[dict[str, object]],
    group_map: dict[str, dict],
    *,
    min_items: int,
) -> list[dict]:
    if len(merged_items) >= max(1, int(min_items)):
        return merged_items

    selected_group_ids = {
        str(group_id or "").strip()
        for item in merged_items
        for group_id in (item.get("group_ids") or [])
        if str(group_id or "").strip()
    }

    candidates: list[dict[str, object]] = []
    for group in groups:
        group_id = str(group.get("group_id") or "").strip()
        if not group_id or group_id in selected_group_ids:
            continue
        summary_text = _storyline_clean_summary_text(str(group.get("representative_text") or "").strip(), limit=280)
        summary_garbage = _is_storyline_summary_garbage(summary_text)
        is_strong = _storyline_group_has_strong_anchor_signal(
            relevance=int(group.get("anchor_relevance_score") or 0),
            marker_hits=int(group.get("anchor_marker_hits") or 0),
            focus_phrase_hits=int(group.get("focus_phrase_hits") or 0),
            must_overlap=int(group.get("must_overlap") or 0),
            must_phrase_overlap=int(group.get("must_phrase_overlap") or 0),
            summary_garbage=summary_garbage,
        )
        if not is_strong and not bool(group.get("explicit_timeline_update_post")):
            continue
        candidates.append(group)

    candidates.sort(
        key=lambda group: (
            -int(group.get("anchor_relevance_score") or 0),
            -int(group.get("source_count") or 0),
            str(group.get("start_timestamp") or group.get("end_timestamp") or ""),
        )
    )

    enriched = list(merged_items)
    for group in candidates:
        group_id = str(group.get("group_id") or "").strip()
        if not group_id or group_id in selected_group_ids:
            continue
        candidate_item = {
            "group_ids": [group_id],
            "summary": _storyline_clean_summary_text(str(group.get("representative_text") or "").strip(), limit=280),
            "expanded_sources": [dict(src) for src in (group.get("sources") or [])],
        }
        candidate_item["_resolved_summary"] = _storyline_best_group_summary(candidate_item, [group])
        if any(_storyline_should_merge_timeline_items(existing, candidate_item, group_map) for existing in enriched):
            continue
        enriched.append(candidate_item)
        selected_group_ids.add(group_id)
        if len(enriched) >= max(1, int(min_items)):
            break

    def _group_sort_key(item: dict) -> tuple[str, int]:
        ids = [str(group_id or "").strip() for group_id in item.get("group_ids") or [] if str(group_id or "").strip()]
        selected_groups = [group_map[group_id] for group_id in ids if group_id in group_map]
        if not selected_groups:
            return ("", 0)
        min_ts = min(str(group.get("start_timestamp") or "") for group in selected_groups)
        min_post_id = min(int(pid) for group in selected_groups for pid in group.get("post_ids", []) or [0])
        return (min_ts, min_post_id)

    enriched.sort(key=_group_sort_key)
    return enriched


def _storyline_timeline_min_items(groups: list[dict[str, object]]) -> int:
    strong_groups = 0
    for group in groups:
        if _storyline_group_has_strong_anchor_signal(
            relevance=int(group.get("anchor_relevance_score") or 0),
            marker_hits=int(group.get("anchor_marker_hits") or 0),
            focus_phrase_hits=int(group.get("focus_phrase_hits") or 0),
            must_overlap=int(group.get("must_overlap") or 0),
            must_phrase_overlap=int(group.get("must_phrase_overlap") or 0),
            summary_garbage=_is_storyline_summary_garbage(
                _storyline_clean_summary_text(str(group.get("representative_text") or "").strip(), limit=280)
            ),
        ):
            strong_groups += 1
    if strong_groups >= 5:
        return 5
    if strong_groups >= 4:
        return 4
    return 3


def _storyline_ensure_anchor_post_item(
    selected_items: list[dict],
    groups: list[dict[str, object]],
    *,
    anchor_post_id: int,
) -> list[dict]:
    if anchor_post_id <= 0:
        return selected_items

    selected_group_ids = {
        str(group_id or "").strip()
        for item in selected_items
        for group_id in (item.get("group_ids") or [])
        if str(group_id or "").strip()
    }
    for group in groups:
        group_id = str(group.get("group_id") or "").strip()
        if not group_id or group_id in selected_group_ids:
            continue
        if int(anchor_post_id) not in {int(pid or 0) for pid in (group.get("post_ids") or [])}:
            continue
        if not _storyline_group_has_strong_anchor_signal(
            relevance=int(group.get("anchor_relevance_score") or 0),
            marker_hits=int(group.get("anchor_marker_hits") or 0),
            focus_phrase_hits=int(group.get("focus_phrase_hits") or 0),
            must_overlap=int(group.get("must_overlap") or 0),
            must_phrase_overlap=int(group.get("must_phrase_overlap") or 0),
            summary_garbage=_is_storyline_summary_garbage(
                _storyline_clean_summary_text(str(group.get("representative_text") or "").strip(), limit=280)
            ),
        ):
            return selected_items
        return [
            *selected_items,
            {
                "group_ids": [group_id],
                "summary": str(group.get("representative_text") or "").strip(),
            },
        ]
    return selected_items


def _storyline_filter_items_for_anchor_time_window(
    selected_items: list[dict],
    group_map: dict[str, dict],
    *,
    anchor_post_ts: datetime | None,
) -> list[dict]:
    if not selected_items or anchor_post_ts is None:
        return selected_items
    min_allowed_ts = anchor_post_ts - timedelta(days=STORYLINE_TIMELINE_LOOKBACK_DAYS)
    filtered: list[dict] = []
    for item in selected_items:
        start_ts, end_ts = _storyline_item_time_bounds(item, group_map)
        current_ts = end_ts or start_ts
        if current_ts is not None and current_ts < min_allowed_ts:
            continue
        filtered.append(item)
    return filtered


def _storyline_should_merge_timeline_items(left: dict, right: dict, group_map: dict[str, dict]) -> bool:
    left_summary = str(left.get("_resolved_summary") or "").strip()
    right_summary = str(right.get("_resolved_summary") or "").strip()
    left_tokens = set(_storyline_tokens(left_summary))
    right_tokens = set(_storyline_tokens(right_summary))
    summary_jaccard = _storyline_jaccard(left_tokens, right_tokens)
    summary_ratio = difflib.SequenceMatcher(
        None,
        _normalize_storyline_text(left_summary),
        _normalize_storyline_text(right_summary),
    ).ratio()

    left_sources = {_storyline_source_key(src) for src in (left.get("expanded_sources") or [])}
    right_sources = {_storyline_source_key(src) for src in (right.get("expanded_sources") or [])}
    source_jaccard = _storyline_jaccard(left_sources, right_sources)
    shared_sources = len(left_sources & right_sources)
    subset_overlap = bool(left_sources and right_sources and (left_sources <= right_sources or right_sources <= left_sources))

    left_start, left_end = _storyline_item_time_bounds(left, group_map)
    right_start, right_end = _storyline_item_time_bounds(right, group_map)
    left_macro_topics = _storyline_item_macro_topic_ids(left, group_map)
    right_macro_topics = _storyline_item_macro_topic_ids(right, group_map)
    left_post_ids = _storyline_item_post_ids(left, group_map)
    right_post_ids = _storyline_item_post_ids(right, group_map)
    same_macro_topic = bool(left_macro_topics & right_macro_topics)
    disjoint_posts = bool(left_post_ids and right_post_ids and not (left_post_ids & right_post_ids))
    if left_end and right_start:
        gap_seconds = abs((right_start - left_end).total_seconds())
    elif left_start and right_start:
        gap_seconds = abs((right_start - left_start).total_seconds())
    else:
        gap_seconds = 0.0

    if gap_seconds > 36 * 3600:
        return False
    if (
        same_macro_topic
        and disjoint_posts
        and shared_sources == 0
        and not subset_overlap
        and summary_ratio < 0.96
        and summary_jaccard < 0.8
    ):
        return False
    if same_macro_topic and gap_seconds <= 48 * 3600:
        if summary_ratio >= 0.9 or summary_jaccard >= 0.55:
            return True
        if left_summary and right_summary:
            left_norm = _normalize_storyline_text(left_summary)
            right_norm = _normalize_storyline_text(right_summary)
            if left_norm and right_norm and (left_norm in right_norm or right_norm in left_norm):
                return True
    if shared_sources >= 3 and gap_seconds <= 18 * 3600:
        return True
    if shared_sources >= 2 and source_jaccard >= 0.9 and gap_seconds <= 18 * 3600:
        return True
    if source_jaccard >= 0.45 and (summary_jaccard >= 0.18 or summary_ratio >= 0.62):
        return True
    if subset_overlap and gap_seconds <= 18 * 3600 and (summary_jaccard >= 0.15 or summary_ratio >= 0.58):
        return True
    if (summary_jaccard >= 0.62 or summary_ratio >= 0.84) and gap_seconds <= 24 * 3600:
        return True
    if same_macro_topic and gap_seconds <= 24 * 3600 and (summary_jaccard >= 0.35 or summary_ratio >= 0.72):
        return True
    return False


def _merge_storyline_timeline_items(selected_items: list[dict], group_map: dict[str, dict]) -> list[dict]:
    merged: list[dict] = []
    for raw_item in selected_items:
        item = dict(raw_item)
        item["group_ids"] = [
            str(group_id or "").strip()
            for group_id in item.get("group_ids") or []
            if str(group_id or "").strip()
        ]
        item["expanded_sources"] = sorted(
            [dict(src) for src in item.get("expanded_sources") or []],
            key=lambda row: (str(row.get("timestamp") or ""), int(row.get("post_id") or 0)),
        )
        if not item.get("_resolved_summary"):
            item_groups = [group_map[group_id] for group_id in item["group_ids"] if group_id in group_map]
            item["_resolved_summary"] = _storyline_best_group_summary(item, item_groups)

        if not merged:
            merged.append(item)
            continue

        prev = merged[-1]
        if not _storyline_should_merge_timeline_items(prev, item, group_map):
            merged.append(item)
            continue

        merged_group_ids = list(dict.fromkeys([*(prev.get("group_ids") or []), *(item.get("group_ids") or [])]))
        merged_sources = sorted(
            [*(prev.get("expanded_sources") or []), *(item.get("expanded_sources") or [])],
            key=lambda row: (str(row.get("timestamp") or ""), int(row.get("post_id") or 0)),
        )
        deduped_sources: list[dict] = []
        seen_source_keys: set[tuple[int, str]] = set()
        for src in merged_sources:
            key = _storyline_source_key(src)
            if key in seen_source_keys:
                continue
            seen_source_keys.add(key)
            deduped_sources.append(src)

        prev["_resolved_summary"] = _storyline_pick_better_summary(
            prev.get("_resolved_summary"),
            item.get("_resolved_summary"),
            prev.get("summary"),
            item.get("summary"),
        )
        prev["summary"] = prev["_resolved_summary"]
        prev["group_ids"] = merged_group_ids
        prev["expanded_sources"] = deduped_sources
        prev["step_terms"] = list(
            dict.fromkeys([*(prev.get("step_terms") or []), *(item.get("step_terms") or [])])
        )

    return merged


def _storyline_fast_mode_enabled(payload: dict[str, object] | None) -> bool:
    current = dict(payload or {})
    groups = list(current.get("groups") or [])
    cards = list(current.get("cards") or [])
    return len(groups) >= STORYLINE_TIMELINE_FAST_GROUP_THRESHOLD or len(cards) >= STORYLINE_TIMELINE_FAST_CARD_THRESHOLD


def _storyline_group_foundational_score(group: dict[str, object]) -> int:
    text = " ".join(
        part
        for part in [
            str(group.get("representative_text") or "").strip(),
            str(group.get("episode_title") or "").strip(),
            str(group.get("macro_topic_title") or "").strip(),
        ]
        if part
    )
    return _storyline_foundational_text_score(text)


def _storyline_foundational_text_score(text: str | None) -> int:
    normalized = _normalize_storyline_text(text)
    score = 0
    patterns = [
        r"\bстарт\w*",
        r"\bзапуск\w*",
        r"\bстартовал\w*",
        r"\bстартовала\w*",
        r"\bзапустил\w*",
        r"\bзапустила\w*",
        r"\bначал\w*",
        r"\bначалась\b",
        r"\bобратн\w* отсчет",
        r"\bcountdown\b",
        r"\blaunch\w*",
        r"\blaunched\b",
        r"\bcrew(ed)? mission\b",
        r"\bcrew\b",
        r"\bэкипаж\w*",
        r"\bподготов\w*",
        r"\bfirst crewed\b",
        r"\bв ночь на\b",
    ]
    for pattern in patterns:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            score += 1
    if re.search(r"\bмисси\w*\b", normalized, flags=re.IGNORECASE):
        score += 1
    return score


def _storyline_select_cluster_storylines(
    anchor: dict[str, object],
    candidate_storylines: list[dict[str, object]],
    *,
    anchor_profile: dict | None = None,
    anchor_markers: list[str] | None = None,
    limit: int = 16,
) -> list[dict[str, object]]:
    anchor_storyline_id = str(anchor.get("storyline_id") or "").strip()
    anchor_macro_topic_id = str(anchor.get("macro_topic_id") or "").strip()
    anchor_family_root_id = str(anchor.get("family_root_storyline_id") or anchor_storyline_id).strip()

    shortlisted: list[dict[str, object]] = []
    seen_storyline_ids: set[str] = set()
    for raw_item in candidate_storylines:
        item = dict(raw_item)
        storyline_id = str(item.get("storyline_id") or "").strip()
        if not storyline_id or storyline_id in seen_storyline_ids:
            continue
        seen_storyline_ids.add(storyline_id)

        family_root_id = str(item.get("family_root_storyline_id") or storyline_id).strip()
        macro_topic_id = str(item.get("macro_topic_id") or "").strip()
        family_match = int(item.get("family_match") or 0)
        same_macro_topic = int(item.get("same_macro_topic") or 0)
        shared_entity_count = int(item.get("shared_entity_count") or 0)
        shared_topic_count = int(item.get("shared_topic_count") or 0)
        shared_signature_count = int(item.get("shared_signature_count") or 0)
        token_overlap = int(item.get("token_overlap") or 0)
        topic_token_overlap = int(item.get("topic_token_overlap") or 0)
        retrieval_score = float(item.get("retrieval_score") or 0.0)
        overlap = _storyline_profile_overlap(item, anchor_profile)
        must_overlap = int(overlap.get("must_overlap") or 0)
        must_phrase_overlap = int(overlap.get("must_phrase_overlap") or 0)
        negative_overlap = int(overlap.get("negative_overlap") or 0)
        negative_phrase_overlap = int(overlap.get("negative_phrase_overlap") or 0)
        marker_hits = _storyline_candidate_marker_hits(item, list(anchor_markers or []))

        include = False
        if storyline_id == anchor_storyline_id:
            include = True
        elif family_root_id and family_root_id == anchor_family_root_id and (
            marker_hits >= 1 or must_overlap >= 1 or must_phrase_overlap >= 1 or shared_entity_count >= 1
        ):
            include = True
        elif family_match > 0 and (
            marker_hits >= 1 or must_overlap >= 1 or must_phrase_overlap >= 1 or shared_entity_count >= 1
        ):
            include = True
        elif same_macro_topic > 0 and (
            marker_hits >= 1
            or must_phrase_overlap >= 1
            or must_overlap >= 2
            or (must_overlap >= 1 and shared_entity_count >= 1)
            or (shared_entity_count >= 2 and shared_topic_count >= 1)
        ):
            include = True
        elif macro_topic_id and anchor_macro_topic_id and macro_topic_id == anchor_macro_topic_id and (
            marker_hits >= 1 or must_overlap >= 1 or must_phrase_overlap >= 1
        ):
            include = True
        elif marker_hits >= 2 and (shared_entity_count >= 1 or shared_topic_count >= 1 or token_overlap >= 1):
            include = True
        elif must_phrase_overlap >= 1 and (shared_entity_count >= 1 or topic_token_overlap >= 1):
            include = True
        elif must_overlap >= 2 and (shared_entity_count >= 1 or shared_topic_count >= 1):
            include = True
        elif retrieval_score >= 18.0 and marker_hits >= 1 and (shared_entity_count >= 1 or topic_token_overlap >= 1):
            include = True

        if not include:
            continue
        if negative_phrase_overlap > 0 and must_phrase_overlap == 0:
            continue
        if negative_overlap > 0 and must_overlap == 0 and marker_hits == 0:
            continue

        cluster_score = (
            (100 if storyline_id == anchor_storyline_id else 0)
            + (24 if family_root_id and family_root_id == anchor_family_root_id else 0)
            + family_match * 14
            + same_macro_topic * 12
            + marker_hits * 11
            + must_phrase_overlap * 10
            + must_overlap * 8
            + shared_entity_count * 7
            + shared_topic_count * 6
            + shared_signature_count * 3
            + token_overlap * 2
            + topic_token_overlap * 2
            - negative_overlap * 4
            - negative_phrase_overlap * 6
            + min(int(retrieval_score), 20)
        )
        item["cluster_score"] = cluster_score
        item["anchor_marker_hits"] = marker_hits
        item["must_overlap"] = must_overlap
        item["must_phrase_overlap"] = must_phrase_overlap
        item["negative_overlap"] = negative_overlap
        item["negative_phrase_overlap"] = negative_phrase_overlap
        shortlisted.append(item)

    shortlisted.sort(
        key=lambda item: (
            -int(item.get("cluster_score") or 0),
            -int(item.get("same_macro_topic") or 0),
            -int(item.get("family_match") or 0),
            -int(item.get("shared_entity_count") or 0),
            -int(item.get("shared_topic_count") or 0),
            -float(item.get("retrieval_score") or 0.0),
            str(item.get("storyline_id") or ""),
        )
    )
    return shortlisted[: max(1, int(limit))]


def _storyline_group_anchor_text(group: dict[str, object]) -> str:
    parts: list[str] = []
    for field in ("representative_text", "episode_title", "macro_topic_title", "event_signature"):
        value = str(group.get(field) or "").strip()
        if value:
            parts.append(value)
    for item in (group.get("step_anchor_entities") or [])[:8]:
        value = str(item or "").strip()
        if value:
            parts.append(value)
    return " ".join(parts).strip()


def _storyline_filter_groups_for_anchor(
    groups: list[dict[str, object]],
    *,
    anchor_profile: dict | None,
    anchor_markers: list[str] | None,
) -> list[dict[str, object]]:
    if not groups:
        return []

    filtered: list[dict[str, object]] = []
    has_profile_must_terms = bool(_storyline_profile_must_terms(anchor_profile))

    for raw_group in groups:
        group = _storyline_enrich_group_anchor_relevance(
            raw_group,
            anchor_profile=anchor_profile,
            anchor_markers=anchor_markers,
        )
        marker_hits = int(group.get("anchor_marker_hits") or 0)
        focus_phrase_hits = int(group.get("focus_phrase_hits") or 0)
        must_overlap = int(group.get("must_overlap") or 0)
        must_phrase_overlap = int(group.get("must_phrase_overlap") or 0)
        negative_overlap = int(group.get("negative_overlap") or 0)
        negative_phrase_overlap = int(group.get("negative_phrase_overlap") or 0)
        relevance_score = int(group.get("anchor_relevance_score") or 0)
        summary_garbage = _is_storyline_summary_garbage(str(group.get("representative_text") or "").strip())

        keep = False
        if focus_phrase_hits >= 1:
            keep = True
        elif marker_hits >= 2:
            keep = True
        elif must_phrase_overlap >= 1:
            keep = True
        elif marker_hits >= 1 and must_overlap >= 1 and not summary_garbage:
            keep = True
        elif must_overlap >= 2 and relevance_score >= 10 and not summary_garbage:
            keep = True
        elif not has_profile_must_terms and marker_hits >= 1 and relevance_score >= 8:
            keep = True

        if negative_phrase_overlap > 0 and must_phrase_overlap == 0:
            keep = False
        if negative_overlap > 0 and marker_hits == 0 and focus_phrase_hits == 0 and must_overlap == 0:
            keep = False

        if keep:
            filtered.append(group)

    if filtered:
        filtered.sort(
            key=lambda group: (
                str(group.get("start_timestamp") or ""),
                -int(group.get("anchor_relevance_score") or 0),
                str(group.get("group_id") or ""),
            )
        )
        return filtered
    return groups


def _storyline_fast_mode_groups(payload: dict[str, object], *, limit: int = STORYLINE_TIMELINE_FAST_ARB_GROUP_LIMIT) -> list[dict]:
    retrieval_debug = dict(payload.get("retrieval_debug") or {})
    anchor_profile = dict(retrieval_debug.get("anchor_profile") or {})
    anchor_markers = [str(item or "").strip() for item in (retrieval_debug.get("anchor_markers") or []) if str(item or "").strip()]
    anchor_post_id = int(retrieval_debug.get("anchor_post_id") or 0)
    anchor_post_ts = None
    anchor_post_timestamp = str(retrieval_debug.get("anchor_post_timestamp") or "").strip()
    if anchor_post_timestamp:
        try:
            anchor_post_ts = datetime.fromisoformat(anchor_post_timestamp)
        except Exception:
            anchor_post_ts = None

    groups = [
        _storyline_enrich_group_anchor_relevance(
            dict(item),
            anchor_profile=anchor_profile,
            anchor_markers=anchor_markers,
        )
        for item in (payload.get("groups") or [])
    ]
    if len(groups) <= max(1, int(limit)):
        return groups

    def _is_explicit(group: dict) -> bool:
        if bool(group.get("explicit_timeline_update_post")):
            return True
        if anchor_post_id <= 0:
            return False
        return anchor_post_id in {int(pid or 0) for pid in (group.get("post_ids") or [])}

    def _is_graph_step(group: dict) -> bool:
        return str(group.get("group_id") or "").startswith("update_step_")

    def _is_strong(group: dict) -> bool:
        if (
            bool(group.get("supplemental_exact_marker_match"))
            and not _is_explicit(group)
            and not _storyline_group_in_time_window(group, center_ts=anchor_post_ts)
        ):
            return False
        summary = _storyline_clean_summary_text(str(group.get("representative_text") or "").strip(), limit=280)
        return _storyline_group_has_strong_anchor_signal(
            relevance=int(group.get("anchor_relevance_score") or 0),
            marker_hits=int(group.get("anchor_marker_hits") or 0),
            focus_phrase_hits=int(group.get("focus_phrase_hits") or 0),
            must_overlap=int(group.get("must_overlap") or 0),
            must_phrase_overlap=int(group.get("must_phrase_overlap") or 0),
            summary_garbage=_is_storyline_summary_garbage(summary),
        )

    def _relevant_enough(group: dict) -> bool:
        if _is_explicit(group) or _is_strong(group):
            return True
        if not _storyline_group_in_time_window(group, center_ts=anchor_post_ts):
            return False
        return (
            int(group.get("anchor_relevance_score") or 0) >= 12
            or int(group.get("focus_phrase_hits") or 0) >= 1
            or int(group.get("must_phrase_overlap") or 0) >= 1
            or int(group.get("must_overlap") or 0) >= 2
            or int(group.get("anchor_marker_hits") or 0) >= 2
        )

    def _chronological_key(group: dict) -> tuple[str, int]:
        return (
            str(group.get("start_timestamp") or group.get("end_timestamp") or ""),
            int((group.get("post_ids") or [0])[0] or 0),
        )

    def _ranked_key(group: dict) -> tuple[int, int, int, int, str]:
        return (
            -int(group.get("anchor_relevance_score") or 0),
            -int(group.get("source_count") or 0),
            -int(group.get("post_count") or 0),
            0 if _storyline_group_in_time_window(group, center_ts=anchor_post_ts) else 1,
            str(group.get("start_timestamp") or group.get("end_timestamp") or ""),
        )

    target = max(1, int(limit))
    chronological = sorted(
        [group for group in groups if _relevant_enough(group)],
        key=_chronological_key,
    )
    recent = sorted(
        [group for group in groups if _relevant_enough(group)],
        key=lambda group: (
            str(group.get("end_timestamp") or group.get("start_timestamp") or ""),
            int(group.get("source_count") or 0),
            int(group.get("post_count") or 0),
        ),
        reverse=True,
    )
    strong_chronological = sorted([group for group in groups if _is_strong(group)], key=_chronological_key)
    graph_strong_chronological = sorted(
        [group for group in groups if _is_graph_step(group) and _is_strong(group)],
        key=_chronological_key,
    )
    strong_ranked = sorted([group for group in groups if _is_strong(group)], key=_ranked_key)
    foundational = sorted(
        [group for group in groups if _relevant_enough(group)],
        key=lambda group: (
            -_storyline_group_foundational_score(group),
            str(group.get("start_timestamp") or group.get("end_timestamp") or ""),
        ),
    )
    dense = sorted(
        [group for group in groups if _relevant_enough(group)],
        key=_ranked_key,
    )

    selected: list[dict] = []
    seen: set[str] = set()

    def _take(pool: list[dict], count: int) -> None:
        taken = 0
        for group in pool:
            group_id = str(group.get("group_id") or "").strip()
            if not group_id or group_id in seen:
                continue
            seen.add(group_id)
            selected.append(group)
            taken += 1
            if taken >= max(0, int(count)) or len(selected) >= target:
                return

    _take([group for group in groups if _is_explicit(group)], min(3, target))
    if len(selected) < target:
        _take(graph_strong_chronological, min(4, target - len(selected)))
    if len(selected) < target:
        _take(recent, min(5, target - len(selected)))
    if len(selected) < target:
        _take(foundational, min(2, target - len(selected)))
    if len(selected) < target:
        _take(chronological, min(4, target - len(selected)))
    if len(selected) < target:
        _take(strong_chronological, min(4, target - len(selected)))
    if len(selected) < target:
        _take(strong_ranked, min(4, target - len(selected)))
    if len(selected) < target:
        _take(dense, target - len(selected))
    if len(selected) < target:
        _take(sorted(groups, key=_ranked_key), target - len(selected))

    selected.sort(key=_chronological_key)
    return selected[:target]


async def _build_storyline_graph_step_payload(
    session,
    *,
    anchor: dict[str, object],
    candidate_storylines: list[dict[str, object]],
    macro_topic_id: str,
    anchor_profile: dict | None,
    anchor_terms: list[str],
    topic_terms: list[str],
    deterministic_branch_terms: list[str] | None = None,
    deterministic_branch_candidate_count: int = 0,
    anchor_markers: list[str] | None = None,
    anchor_post_ts: datetime | None = None,
    anchor_post_id: int | None = None,
    include_post_id: int | None = None,
    explicit_post_ids: list[int] | None = None,
    follow_storyline_title: str | None = None,
    strict_anchor_groups: bool = False,
) -> dict[str, object] | None:
    macro_topic_titles: dict[str, str] = {
        str(macro_topic_id): str(anchor.get("macro_topic_title") or "").strip(),
    }
    focus_phrases = _storyline_focus_phrases(
        anchor_profile,
        anchor,
        str(anchor.get("storyline_seed_preview") or ""),
    )

    def _candidate_supports_sibling_topic(item: dict[str, object]) -> bool:
        overlap = _storyline_profile_overlap(item, anchor_profile)
        same_macro_topic = int(item.get("same_macro_topic") or 0)
        shared_entity_count = int(item.get("shared_entity_count") or 0)
        shared_topic_count = int(item.get("shared_topic_count") or 0)
        token_overlap = int(item.get("token_overlap") or 0)
        topic_token_overlap = int(item.get("topic_token_overlap") or 0)
        retrieval_score = float(item.get("retrieval_score") or 0.0)
        must_overlap = int(overlap.get("must_overlap") or 0)
        negative_overlap = int(overlap.get("negative_overlap") or 0)
        must_phrase_overlap = int(overlap.get("must_phrase_overlap") or 0)
        negative_phrase_overlap = int(overlap.get("negative_phrase_overlap") or 0)
        has_profile_must_terms = bool(_storyline_profile_must_terms(anchor_profile))
        strong_branch_signal = (
            must_phrase_overlap >= 1
            or must_overlap >= 2
            or (must_overlap >= 1 and shared_entity_count >= 2)
            or (must_overlap >= 1 and token_overlap >= 2)
        )
        if same_macro_topic > 0:
            return True
        if has_profile_must_terms and not strong_branch_signal:
            if shared_entity_count < 3 and token_overlap < 3 and topic_token_overlap < 2:
                return False
        if negative_phrase_overlap > 0 and must_phrase_overlap == 0:
            if shared_entity_count < 2 and token_overlap < 3:
                return False
        if negative_overlap > 0 and must_overlap == 0 and must_phrase_overlap == 0:
            if shared_entity_count < 2 and token_overlap < 3:
                return False
        if strong_branch_signal and (shared_entity_count >= 1 or shared_topic_count >= 1 or token_overlap >= 1):
            return True
        if shared_entity_count >= 2:
            return True
        if must_phrase_overlap >= 1 and (token_overlap >= 1 or topic_token_overlap >= 1 or shared_topic_count >= 1):
            return True
        if shared_entity_count >= 1 and (
            token_overlap >= 2
            or topic_token_overlap >= 2
            or (shared_topic_count >= 2 and token_overlap >= 2)
        ):
            return True
        if must_overlap >= 1 and (token_overlap >= 2 or topic_token_overlap >= 1 or shared_topic_count >= 1):
            return True
        if token_overlap >= 3 and topic_token_overlap >= 1:
            return True
        if retrieval_score >= 18.0 and (
            shared_entity_count >= 1
            or token_overlap >= 3
        ):
            return True
        return False

    for item in candidate_storylines:
        candidate_macro_topic_id = str(item.get("macro_topic_id") or "").strip()
        candidate_macro_topic_title = str(item.get("macro_topic_title") or "").strip()
        if candidate_macro_topic_id and candidate_macro_topic_title:
            macro_topic_titles[candidate_macro_topic_id] = candidate_macro_topic_title
    candidate_macro_topics_map: dict[str, dict[str, object]] = {}
    for item in candidate_storylines:
        candidate_macro_topic_id = str(item.get("macro_topic_id") or "").strip()
        if not candidate_macro_topic_id:
            continue
        overlap = _storyline_profile_overlap(item, anchor_profile)
        entry = candidate_macro_topics_map.setdefault(
            candidate_macro_topic_id,
            {
                "macro_topic_id": candidate_macro_topic_id,
                "macro_topic_title": str(item.get("macro_topic_title") or "").strip(),
                "best_retrieval_score": float(item.get("retrieval_score") or 0.0),
                "storyline_ids": [],
                "storyline_titles": [],
                "foundational_score": _storyline_foundational_text_score(
                    " ".join(
                        part
                        for part in [
                            str(item.get("title") or "").strip(),
                            str(item.get("seed_preview") or "").strip(),
                        ]
                        if part
                    )
                ),
                "shared_entity_count": int(item.get("shared_entity_count") or 0),
                "shared_signature_count": int(item.get("shared_signature_count") or 0),
                "shared_topic_count": int(item.get("shared_topic_count") or 0),
                "token_overlap": int(item.get("token_overlap") or 0),
                "topic_token_overlap": int(item.get("topic_token_overlap") or 0),
                "same_macro_topic": int(item.get("same_macro_topic") or 0),
                "must_overlap": int(overlap.get("must_overlap") or 0),
                "negative_overlap": int(overlap.get("negative_overlap") or 0),
                "must_phrase_overlap": int(overlap.get("must_phrase_overlap") or 0),
                "negative_phrase_overlap": int(overlap.get("negative_phrase_overlap") or 0),
            },
        )
        entry["best_retrieval_score"] = max(entry["best_retrieval_score"], float(item.get("retrieval_score") or 0.0))
        entry["foundational_score"] = max(
            int(entry.get("foundational_score") or 0),
            _storyline_foundational_text_score(
                " ".join(
                    part
                    for part in [
                        str(item.get("title") or "").strip(),
                        str(item.get("seed_preview") or "").strip(),
                    ]
                    if part
                )
            ),
        )
        entry["shared_entity_count"] = max(entry["shared_entity_count"], int(item.get("shared_entity_count") or 0))
        entry["shared_signature_count"] = max(entry["shared_signature_count"], int(item.get("shared_signature_count") or 0))
        entry["shared_topic_count"] = max(entry["shared_topic_count"], int(item.get("shared_topic_count") or 0))
        entry["token_overlap"] = max(entry["token_overlap"], int(item.get("token_overlap") or 0))
        entry["topic_token_overlap"] = max(entry["topic_token_overlap"], int(item.get("topic_token_overlap") or 0))
        entry["same_macro_topic"] = max(entry["same_macro_topic"], int(item.get("same_macro_topic") or 0))
        entry["must_overlap"] = max(entry["must_overlap"], int(overlap.get("must_overlap") or 0))
        entry["negative_overlap"] = max(entry["negative_overlap"], int(overlap.get("negative_overlap") or 0))
        entry["must_phrase_overlap"] = max(entry["must_phrase_overlap"], int(overlap.get("must_phrase_overlap") or 0))
        entry["negative_phrase_overlap"] = max(entry["negative_phrase_overlap"], int(overlap.get("negative_phrase_overlap") or 0))
        storyline_id = str(item.get("storyline_id") or "").strip()
        storyline_title = str(item.get("title") or "").strip()
        if storyline_id and storyline_id not in entry["storyline_ids"]:
            entry["storyline_ids"].append(storyline_id)
        if storyline_title and storyline_title not in entry["storyline_titles"]:
            entry["storyline_titles"].append(storyline_title)
    if macro_topic_id and macro_topic_id not in candidate_macro_topics_map:
        candidate_macro_topics_map[macro_topic_id] = {
            "macro_topic_id": macro_topic_id,
            "macro_topic_title": macro_topic_titles.get(macro_topic_id, ""),
            "best_retrieval_score": 0.0,
            "storyline_ids": [str(anchor.get("storyline_id") or "").strip()] if str(anchor.get("storyline_id") or "").strip() else [],
            "storyline_titles": [str(anchor.get("storyline_title") or "").strip()] if str(anchor.get("storyline_title") or "").strip() else [],
            "foundational_score": _storyline_foundational_text_score(
                " ".join(
                    part
                    for part in [
                        str(anchor.get("storyline_title") or "").strip(),
                        str(anchor.get("storyline_seed_preview") or "").strip(),
                    ]
                    if part
                )
            ),
            "shared_entity_count": 0,
            "shared_signature_count": 0,
            "shared_topic_count": 0,
            "token_overlap": 0,
            "topic_token_overlap": 0,
            "same_macro_topic": 1,
            "must_overlap": 0,
            "negative_overlap": 0,
            "must_phrase_overlap": 0,
            "negative_phrase_overlap": 0,
        }
    candidate_macro_topics = sorted(
        [
            item
            for item in candidate_macro_topics_map.values()
            if int(item.get("same_macro_topic") or 0) > 0 or _candidate_supports_sibling_topic(item)
        ],
        key=lambda item: (
            -int(item.get("same_macro_topic") or 0),
            -int(item.get("must_phrase_overlap") or 0),
            -int(item.get("must_overlap") or 0),
            -min(int(item.get("shared_entity_count") or 0), 3),
            int(item.get("negative_phrase_overlap") or 0),
            int(item.get("negative_overlap") or 0),
            -float(item.get("best_retrieval_score") or 0.0),
            -int(item.get("shared_topic_count") or 0),
            str(item.get("macro_topic_title") or ""),
        ),
    )[:6]

    if len(candidate_macro_topics) < 6:
        existing_macro_topic_ids = {
            str(item.get("macro_topic_id") or "").strip()
            for item in candidate_macro_topics
            if str(item.get("macro_topic_id") or "").strip()
        }
        widened_candidates = sorted(
            [
                item
                for item in candidate_macro_topics_map.values()
                if str(item.get("macro_topic_id") or "").strip() not in existing_macro_topic_ids
                and float(item.get("best_retrieval_score") or 0.0) >= 15.0
                and (
                    int(item.get("shared_entity_count") or 0) >= 1
                    or int(item.get("shared_signature_count") or 0) >= 1
                    or int(item.get("foundational_score") or 0) >= 2
                )
                and (
                    int(item.get("shared_topic_count") or 0) >= 2
                    or int(item.get("token_overlap") or 0) >= 1
                    or int(item.get("topic_token_overlap") or 0) >= 1
                    or int(item.get("must_overlap") or 0) >= 1
                    or int(item.get("foundational_score") or 0) >= 3
                )
            ],
            key=lambda item: (
                -int(item.get("foundational_score") or 0),
                -int(item.get("must_phrase_overlap") or 0),
                -int(item.get("must_overlap") or 0),
                -min(int(item.get("shared_entity_count") or 0), 3),
                -min(int(item.get("shared_signature_count") or 0), 3),
                -int(item.get("shared_topic_count") or 0),
                -float(item.get("best_retrieval_score") or 0.0),
                str(item.get("macro_topic_title") or ""),
            ),
        )
        for item in widened_candidates:
            candidate_macro_topics.append(item)
            existing_macro_topic_ids.add(str(item.get("macro_topic_id") or "").strip())
            if len(candidate_macro_topics) >= 6:
                break

    primary_macro_topic_id = str(macro_topic_id or "").strip()
    if not primary_macro_topic_id:
        primary_macro_topic_id = str((candidate_macro_topics[0] or {}).get("macro_topic_id") or "").strip() if candidate_macro_topics else ""
    primary_macro_topic_title = str(
        anchor.get("macro_topic_title")
        or macro_topic_titles.get(primary_macro_topic_id)
        or ""
    ).strip()

    sibling_macro_topic_ids = [
        str(item.get("macro_topic_id") or "").strip()
        for item in candidate_macro_topics
        if str(item.get("macro_topic_id") or "").strip() and str(item.get("macro_topic_id") or "").strip() != primary_macro_topic_id
    ][:3]

    cluster_storylines = _storyline_select_cluster_storylines(
        anchor,
        candidate_storylines,
        anchor_profile=anchor_profile,
        anchor_markers=list(anchor_markers or []),
        limit=16,
    )
    cluster_storyline_ids = [
        str(item.get("storyline_id") or "").strip()
        for item in cluster_storylines
        if str(item.get("storyline_id") or "").strip()
    ]

    step_rows: list[dict[str, object]] = []
    payload_mode = "macro_topic_graph_steps"
    if cluster_storyline_ids:
        cluster_rows = await get_storyline_cluster_timeline_steps(cluster_storyline_ids, limit=240)
        if cluster_rows:
            payload_mode = "story_cluster_graph_steps"
            for row in cluster_rows:
                enriched = dict(row)
                origin_macro_topic_id = str(
                    enriched.get("macro_topic_id")
                    or primary_macro_topic_id
                    or ""
                ).strip()
                enriched["_origin_macro_topic_id"] = origin_macro_topic_id
                enriched["_origin_macro_topic_title"] = str(
                    enriched.get("macro_topic_title")
                    or macro_topic_titles.get(origin_macro_topic_id)
                    or primary_macro_topic_title
                    or ""
                ).strip()
                step_rows.append(enriched)

    if not step_rows:
        step_rows = await get_macro_topic_timeline_steps(primary_macro_topic_id, limit=160)
        for sibling_macro_topic_id in sibling_macro_topic_ids:
            sibling_rows = await get_macro_topic_timeline_steps(sibling_macro_topic_id, limit=80)
            if sibling_rows:
                for row in sibling_rows:
                    enriched = dict(row)
                    enriched["_origin_macro_topic_id"] = sibling_macro_topic_id
                    enriched["_origin_macro_topic_title"] = macro_topic_titles.get(sibling_macro_topic_id, "")
                    step_rows.append(enriched)
        if sibling_macro_topic_ids:
            payload_mode = "macro_topic_graph_steps_with_siblings"

    if not step_rows:
        return None

    deduped_step_rows: list[dict[str, object]] = []
    seen_step_ids: set[str] = set()
    for row in step_rows:
        if "_origin_macro_topic_id" not in row:
            row = {
                **dict(row),
                "_origin_macro_topic_id": primary_macro_topic_id,
                "_origin_macro_topic_title": macro_topic_titles.get(primary_macro_topic_id, primary_macro_topic_title),
            }
        step_id = str(row.get("step_id") or "").strip()
        if step_id and step_id in seen_step_ids:
            continue
        if step_id:
            seen_step_ids.add(step_id)
        deduped_step_rows.append(dict(row))
    step_rows = deduped_step_rows

    post_ids = sorted(
        {
            int(post_id)
            for row in step_rows
            for post_id in (row.get("source_post_ids") or [])
            if int(post_id or 0) > 0
        }
    )
    timeline_explicit_post_ids = {
        int(post_id)
        for post_id in (explicit_post_ids or [])
        if int(post_id or 0) > 0
    }
    if include_post_id and int(include_post_id) > 0:
        timeline_explicit_post_ids.add(int(include_post_id))
    if timeline_explicit_post_ids:
        post_ids = sorted({*post_ids, *timeline_explicit_post_ids})
    exact_marker_post_ids: set[int] = set()
    if anchor_markers:
        exact_marker_post_ids = await _expand_storyline_exact_marker_post_ids(
            session,
            anchor_markers=list(anchor_markers),
            exclude_post_ids=set(post_ids),
            anchor_post_ts=anchor_post_ts,
        )
        if exact_marker_post_ids:
            post_ids = sorted({*post_ids, *exact_marker_post_ids})
    if not post_ids:
        return None

    result = await session.execute(
        select(Post, Community)
        .outerjoin(Community, Post.community_id == Community.id)
        .where(Post.id.in_(post_ids))
    )
    rows = result.all()

    post_cards: dict[int, dict[str, object]] = {}
    for post_obj, community in rows:
        text = _storyline_post_text(post_obj)
        post_cards[int(post_obj.id)] = {
            "post_id": int(post_obj.id),
            "timestamp": getattr(post_obj, "timestamp", None).isoformat() if getattr(post_obj, "timestamp", None) else "",
            "source": _storyline_source_name(post_obj, community),
            "url": _extract_post_url(post_obj) or "",
            "text": _trim_debug_text(text, limit=700),
            "retrieval_origin": "graph_step",
            "term_match_count": _storyline_term_match_count(text, anchor_terms),
        }

    groups: list[dict[str, object]] = []
    cards: list[dict[str, object]] = []
    episode_ids: set[str] = set()
    for row in step_rows:
        source_ids = [
            int(post_id)
            for post_id in (row.get("source_post_ids") or [])
            if int(post_id or 0) in post_cards
        ]
        if not source_ids:
            continue
        sources = [post_cards[post_id] for post_id in source_ids]
        episode_id = str(row.get("episode_id") or "").strip()
        origin_macro_topic_id = str(row.get("_origin_macro_topic_id") or macro_topic_id or "").strip()
        origin_macro_topic_title = str(
            row.get("_origin_macro_topic_title")
            or macro_topic_titles.get(origin_macro_topic_id)
            or ""
        ).strip()
        if episode_id:
            episode_ids.add(episode_id)
        groups.append(
            {
                "group_id": str(row.get("step_id") or "").strip(),
                "macro_topic_id": origin_macro_topic_id,
                "macro_topic_title": origin_macro_topic_title,
                "episode_id": episode_id,
                "episode_title": str(row.get("episode_title") or "").strip(),
                "episode_phase_type": str(row.get("episode_phase_type") or "").strip(),
                "representative_text": str(row.get("canonical_summary") or "").strip(),
                "start_timestamp": str(row.get("first_seen_at") or "").strip(),
                "end_timestamp": str(row.get("last_seen_at") or row.get("first_seen_at") or "").strip(),
                "post_ids": source_ids,
                "sources": [
                    {
                        "post_id": int(src.get("post_id") or 0),
                        "timestamp": str(src.get("timestamp") or ""),
                        "source": str(src.get("source") or ""),
                        "url": str(src.get("url") or ""),
                    }
                    for src in sources
                ],
                "post_count": len(source_ids),
                "source_count": len(source_ids),
                "event_signature": str(row.get("event_signature") or "").strip(),
                "step_anchor_entities": [str(x or "").strip() for x in (row.get("anchor_entities") or []) if str(x or "").strip()],
            }
        )
        cards.extend(sources)

    groups = _storyline_filter_groups_for_anchor(
        groups,
        anchor_profile=anchor_profile,
        anchor_markers=list(anchor_markers or []),
    )

    if not groups:
        return None

    supplemental_post_ids = [post_id for post_id in sorted(exact_marker_post_ids) if post_id in post_cards]
    for supplemental_post_id in supplemental_post_ids:
        card = post_cards[supplemental_post_id]
        if any(supplemental_post_id in group.get("post_ids", []) for group in groups):
            continue
        supplemental_text = _storyline_prepare_supplemental_card_text(
            str(card.get("text") or "").strip(),
            anchor_profile=anchor_profile,
            anchor_terms=anchor_terms,
            anchor_markers=list(anchor_markers or []),
            focus_phrases=focus_phrases,
        )
        if not supplemental_text:
            continue
        ts = str(card.get("timestamp") or "")
        groups.append(
            _storyline_enrich_group_anchor_relevance(
                {
                    "group_id": f"supplemental_post_{supplemental_post_id}",
                    "macro_topic_id": primary_macro_topic_id,
                    "macro_topic_title": primary_macro_topic_title,
                    "episode_id": "",
                    "episode_title": "",
                    "episode_phase_type": "",
                    "representative_text": supplemental_text,
                    "start_timestamp": ts,
                    "end_timestamp": ts,
                    "post_ids": [supplemental_post_id],
                    "sources": [
                        {
                            "post_id": supplemental_post_id,
                            "timestamp": ts,
                            "source": str(card.get("source") or ""),
                            "url": str(card.get("url") or ""),
                        }
                    ],
                    "post_count": 1,
                    "source_count": 1,
                    "event_signature": "",
                    "step_anchor_entities": [],
                    "supplemental_exact_marker_match": True,
                },
                anchor_profile=anchor_profile,
                anchor_markers=list(anchor_markers or []),
            )
        )

    for explicit_post_id in sorted(post_id for post_id in timeline_explicit_post_ids if post_id in post_cards):
        card = post_cards[explicit_post_id]
        if any(explicit_post_id in group.get("post_ids", []) for group in groups):
            continue
        include_text = _storyline_prepare_supplemental_card_text(
            str(card.get("text") or "").strip(),
            anchor_profile=anchor_profile,
            anchor_terms=anchor_terms,
            anchor_markers=list(anchor_markers or []),
            focus_phrases=focus_phrases,
        ) or _storyline_clean_summary_text(str(card.get("text") or "").strip(), limit=280)
        if not include_text:
            continue
        ts = str(card.get("timestamp") or "")
        group_prefix = "timeline_card_update_post" if explicit_post_id == int(include_post_id or 0) else "timeline_follow_event_post"
        include_group = _storyline_enrich_group_anchor_relevance(
            {
                "group_id": f"{group_prefix}_{explicit_post_id}",
                "macro_topic_id": primary_macro_topic_id,
                "macro_topic_title": primary_macro_topic_title,
                "episode_id": "",
                "episode_title": "",
                "episode_phase_type": "",
                "representative_text": include_text,
                "start_timestamp": ts,
                "end_timestamp": ts,
                "post_ids": [explicit_post_id],
                "sources": [
                    {
                        "post_id": explicit_post_id,
                        "timestamp": ts,
                        "source": str(card.get("source") or ""),
                        "url": str(card.get("url") or ""),
                    }
                ],
                "post_count": 1,
                "source_count": 1,
                "event_signature": "",
                "step_anchor_entities": [],
                "explicit_timeline_update_post": True,
            },
            anchor_profile=anchor_profile,
            anchor_markers=list(anchor_markers or []),
        )
        include_group["focus_phrase_hits"] = max(1, int(include_group.get("focus_phrase_hits") or 0))
        include_group["anchor_marker_hits"] = max(1, int(include_group.get("anchor_marker_hits") or 0))
        include_group["must_overlap"] = max(1, int(include_group.get("must_overlap") or 0))
        include_group["must_phrase_overlap"] = max(1, int(include_group.get("must_phrase_overlap") or 0))
        include_group["negative_overlap"] = 0
        include_group["negative_phrase_overlap"] = 0
        include_group["anchor_relevance_score"] = max(99, int(include_group.get("anchor_relevance_score") or 0))
        groups.append(include_group)
        cards.append(card)

    if strict_anchor_groups:
        strict_groups: list[dict[str, object]] = []
        for group in groups:
            if bool(group.get("explicit_timeline_update_post")):
                strict_groups.append(group)
                continue
            if _storyline_group_has_strong_anchor_signal(
                relevance=int(group.get("anchor_relevance_score") or 0),
                marker_hits=int(group.get("anchor_marker_hits") or 0),
                focus_phrase_hits=int(group.get("focus_phrase_hits") or 0),
                must_overlap=int(group.get("must_overlap") or 0),
                must_phrase_overlap=int(group.get("must_phrase_overlap") or 0),
                summary_garbage=_is_storyline_summary_garbage(
                    _storyline_clean_summary_text(str(group.get("representative_text") or "").strip(), limit=280)
                ),
            ):
                strict_groups.append(group)
        if strict_groups:
            groups = strict_groups

    cards.sort(key=lambda item: (str(item.get("timestamp") or ""), int(item.get("post_id") or 0)))
    groups.sort(key=lambda item: (str(item.get("start_timestamp") or ""), str(item.get("group_id") or "")))
    group_map = {str(item.get("group_id") or ""): item for item in groups}
    group_counts_by_macro_topic: dict[str, int] = {}
    for group in groups:
        key = str(group.get("macro_topic_id") or "").strip()
        if not key:
            continue
        group_counts_by_macro_topic[key] = group_counts_by_macro_topic.get(key, 0) + 1

    mixed_topic_mode = len(
        [
            item
            for item in candidate_macro_topics
            if int(item.get("same_macro_topic") or 0) > 0
            or int(item.get("must_phrase_overlap") or 0) >= 1
            or int(item.get("must_overlap") or 0) >= 1
            or float(item.get("best_retrieval_score") or 0.0) >= 8.0
            or int(item.get("shared_topic_count") or 0) >= 1
            or int(item.get("shared_entity_count") or 0) >= 2
        ]
    ) >= 2

    return {
        "anchor": {
            **anchor,
            "macro_topic_id": primary_macro_topic_id,
            "macro_topic_title": primary_macro_topic_title,
        },
        "candidate_storylines": candidate_storylines,
        "candidate_macro_topics": candidate_macro_topics,
        "cards": cards,
        "groups": groups,
        "group_map": group_map,
        "post_map": {},
        "retrieval_debug": {
            "anchor_terms": anchor_terms,
            "anchor_markers": list(anchor_markers or []),
            "anchor_post_id": int(anchor_post_id or 0),
            "anchor_post_timestamp": anchor_post_ts.isoformat() if anchor_post_ts else "",
            "deterministic_branch_terms": list(deterministic_branch_terms or []),
            "topic_terms": topic_terms,
            "follow_storyline_title": str(follow_storyline_title or "").strip(),
            "strict_anchor_groups": bool(strict_anchor_groups),
            "explicit_post_count": len(timeline_explicit_post_ids),
            "anchor_profile": anchor_profile or {},
            "deterministic_branch_candidate_count": int(deterministic_branch_candidate_count or 0),
            "graph_step_count": len(groups),
            "graph_episode_count": len(episode_ids),
            "graph_step_post_count": len(cards),
            "sibling_macro_topic_count": len(sibling_macro_topic_ids),
            "sibling_macro_topic_ids": sibling_macro_topic_ids,
            "candidate_macro_topic_count": len(candidate_macro_topics),
            "candidate_macro_topic_ids": [str(item.get("macro_topic_id") or "") for item in candidate_macro_topics],
            "mixed_topic_mode": mixed_topic_mode,
            "cluster_storyline_count": len(cluster_storyline_ids),
            "cluster_storyline_ids": cluster_storyline_ids,
            "filtered_group_count": len(groups),
            "exact_marker_post_count": len(exact_marker_post_ids),
            "effective_macro_topic_id": primary_macro_topic_id,
            "effective_macro_topic_title": primary_macro_topic_title,
            "group_count": len(groups),
            "total_post_count": len(cards),
            "min_timestamp": groups[0]["start_timestamp"] if groups else "",
            "max_timestamp": groups[-1]["end_timestamp"] if groups else "",
            "payload_mode": payload_mode,
            "macro_topic_id": primary_macro_topic_id,
            "macro_topic_title": primary_macro_topic_title,
        },
    }


def _collapse_storyline_groups(cards: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for card in sorted(cards, key=lambda item: (str(item.get("timestamp") or ""), int(item.get("post_id") or 0))):
        card_text = str(card.get("text") or "").strip()
        if _is_storyline_noise_text(card_text):
            continue
        norm = _normalize_storyline_text(card_text)
        tokens = set(_storyline_tokens(card_text))
        ts = str(card.get("timestamp") or "")
        best_group: dict[str, object] | None = None
        best_score = 0.0
        for group in groups:
            group_tokens = set(group.get("tokens") or set())
            token_score = _storyline_jaccard(tokens, group_tokens)
            if token_score <= 0:
                continue
            ratio = difflib.SequenceMatcher(None, norm, str(group.get("norm_text") or "")).ratio()
            score = max(token_score, ratio)
            if token_score >= 0.78 or ratio >= 0.9 or (token_score >= 0.65 and ratio >= 0.78):
                if score > best_score:
                    best_group = group
                    best_score = score
        if best_group is None:
            group_id = f"group_{len(groups) + 1:04d}"
            groups.append(
                {
                    "group_id": group_id,
                    "norm_text": norm,
                    "tokens": tokens,
                    "representative_text": card_text,
                    "start_timestamp": ts,
                    "end_timestamp": ts,
                    "post_ids": [int(card.get("post_id") or 0)],
                    "storyline_ids": [str(card.get("storyline_id") or "").strip()] if str(card.get("storyline_id") or "").strip() else [],
                    "sources": [
                        {
                            "post_id": int(card.get("post_id") or 0),
                            "timestamp": ts,
                            "source": str(card.get("source") or "").strip(),
                            "url": str(card.get("url") or "").strip(),
                        }
                    ],
                    "retrieval_origins": [str(card.get("retrieval_origin") or "").strip()],
                }
            )
            continue

        best_group["end_timestamp"] = ts or best_group.get("end_timestamp")
        best_group["post_ids"] = list(dict.fromkeys([*best_group.get("post_ids", []), int(card.get("post_id") or 0)]))
        if str(card.get("storyline_id") or "").strip():
            best_group["storyline_ids"] = list(
                dict.fromkeys([*best_group.get("storyline_ids", []), str(card.get("storyline_id") or "").strip()])
            )
        best_group["sources"] = [
            *best_group.get("sources", []),
            {
                "post_id": int(card.get("post_id") or 0),
                "timestamp": ts,
                "source": str(card.get("source") or "").strip(),
                "url": str(card.get("url") or "").strip(),
            },
        ]
        best_group["retrieval_origins"] = list(
            dict.fromkeys([*best_group.get("retrieval_origins", []), str(card.get("retrieval_origin") or "").strip()])
        )
        if len(card_text) > len(str(best_group.get("representative_text") or "")):
            best_group["representative_text"] = card_text
            best_group["norm_text"] = norm
            best_group["tokens"] = tokens

    for group in groups:
        group["sources"] = sorted(
            group.get("sources", []),
            key=lambda item: (str(item.get("timestamp") or ""), int(item.get("post_id") or 0)),
        )
        group["post_count"] = len(group.get("post_ids", []))
        group["source_count"] = len(group.get("sources", []))
    return groups


async def _expand_storyline_lexical_post_ids(
    session,
    *,
    anchor_terms: list[str],
    exclude_post_ids: set[int],
    anchor_post_ts: datetime | None = None,
    negative_terms: list[str] | None = None,
) -> set[int]:
    preferred_terms = [
        term
        for term in anchor_terms
        if len(term) >= 5 and term not in _STORYLINE_GENERIC_QUERY_TERMS
    ]
    fallback_terms = [term for term in anchor_terms if len(term) >= 6]
    terms = (preferred_terms or fallback_terms or [term for term in anchor_terms if len(term) >= 4])[:8]
    if not terms:
        return set()

    haystack = func.lower(
        func.concat_ws(
            " ",
            func.coalesce(Post.summary, ""),
            func.coalesce(Post.processed_content, ""),
            func.coalesce(Post.content, ""),
            func.coalesce(Post.title, ""),
        )
    )
    hit_expr = None
    for term in terms:
        clause = case((haystack.like(f"%{term}%"), 1), else_=0)
        hit_expr = clause if hit_expr is None else hit_expr + clause
    if hit_expr is None:
        return set()

    strong_terms = [term for term in terms if len(term) >= 8 and term not in _STORYLINE_GENERIC_QUERY_TERMS]
    min_hits = 2 if len(terms) >= 2 else 1
    if len(terms) >= 4:
        min_hits = 3
    query = select(Post.id).where(hit_expr >= min_hits)
    if strong_terms:
        strong_hit_expr = None
        for term in strong_terms[:4]:
            clause = case((haystack.like(f"%{term}%"), 1), else_=0)
            strong_hit_expr = clause if strong_hit_expr is None else strong_hit_expr + clause
        if strong_hit_expr is not None:
            query = query.where(strong_hit_expr >= 1)
    if anchor_post_ts is not None:
        query = query.where(Post.timestamp >= anchor_post_ts - timedelta(days=STORYLINE_TIMELINE_LOOKBACK_DAYS))
    negative_terms = [term for term in (negative_terms or []) if term and term not in terms][:4]
    if negative_terms and terms:
        neg_hit_expr = None
        for term in negative_terms:
            clause = case((haystack.like(f"%{term}%"), 1), else_=0)
            neg_hit_expr = clause if neg_hit_expr is None else neg_hit_expr + clause
        if neg_hit_expr is not None:
            query = query.where((hit_expr - neg_hit_expr) >= 1)
    if exclude_post_ids:
        query = query.where(Post.id.not_in(sorted(exclude_post_ids)))
    result = await session.execute(query)
    return {int(row[0]) for row in result.all()}


async def _expand_storyline_historical_topic_post_ids(
    session,
    *,
    historical_terms: list[str],
    exclude_post_ids: set[int],
    anchor_post_ts: datetime | None,
    earliest_seen_ts: datetime | None,
    negative_terms: list[str] | None = None,
) -> set[int]:
    if anchor_post_ts is None:
        return set()

    terms = [
        term
        for term in historical_terms
        if len(term) >= 4 and term not in _STORYLINE_GENERIC_QUERY_TERMS
    ][:8]
    if not terms:
        return set()

    haystack = func.lower(
        func.concat_ws(
            " ",
            func.coalesce(Post.summary, ""),
            func.coalesce(Post.processed_content, ""),
            func.coalesce(Post.content, ""),
            func.coalesce(Post.title, ""),
        )
    )

    hit_expr = None
    for term in terms:
        clause = case((haystack.like(f"%{term}%"), 1), else_=0)
        hit_expr = clause if hit_expr is None else hit_expr + clause
    if hit_expr is None:
        return set()

    strong_terms = [term for term in terms if len(term) >= 7]
    min_hits = 2 if len(terms) >= 2 else 1
    if len(terms) >= 4:
        min_hits = 3

    query = (
        select(Post.id)
        .where(Post.timestamp <= anchor_post_ts)
        .where(Post.timestamp >= anchor_post_ts - timedelta(days=STORYLINE_TIMELINE_LOOKBACK_DAYS))
    )
    if earliest_seen_ts is not None:
        query = query.where(Post.timestamp < earliest_seen_ts)

    query = query.where(hit_expr >= min_hits)

    if strong_terms:
        strong_hit_expr = None
        for term in strong_terms[:4]:
            clause = case((haystack.like(f"%{term}%"), 1), else_=0)
            strong_hit_expr = clause if strong_hit_expr is None else strong_hit_expr + clause
        if strong_hit_expr is not None:
            query = query.where(strong_hit_expr >= 1)

    negative_terms = [term for term in (negative_terms or []) if term and term not in terms][:4]
    if negative_terms:
        neg_hit_expr = None
        for term in negative_terms:
            clause = case((haystack.like(f"%{term}%"), 1), else_=0)
            neg_hit_expr = clause if neg_hit_expr is None else neg_hit_expr + clause
        if neg_hit_expr is not None:
            query = query.where((hit_expr - neg_hit_expr) >= 1)

    if exclude_post_ids:
        query = query.where(Post.id.not_in(sorted(exclude_post_ids)))

    result = await session.execute(query)
    return {int(row[0]) for row in result.all()}


async def _expand_storyline_exact_marker_post_ids(
    session,
    *,
    anchor_markers: list[str],
    exclude_post_ids: set[int],
    anchor_post_ts: datetime | None,
) -> set[int]:
    terms = [
        str(term or "").strip().lower()
        for term in anchor_markers
        if len(str(term or "").strip()) >= 4
    ][:5]
    if len(terms) < 2 or anchor_post_ts is None:
        return set()

    haystack = func.lower(
        func.concat_ws(
            " ",
            func.coalesce(Post.summary, ""),
            func.coalesce(Post.processed_content, ""),
            func.coalesce(Post.content, ""),
            func.coalesce(Post.title, ""),
        )
    )

    hit_expr = None
    for term in terms:
        clause = case((haystack.like(f"%{term}%"), 1), else_=0)
        hit_expr = clause if hit_expr is None else hit_expr + clause
    if hit_expr is None:
        return set()

    query = (
        select(Post.id, Post.summary, Post.processed_content, Post.content, Post.title)
        .where(Post.timestamp <= anchor_post_ts + timedelta(days=3))
        .where(Post.timestamp >= anchor_post_ts - timedelta(days=STORYLINE_TIMELINE_LOOKBACK_DAYS))
        .where(hit_expr >= 2)
    )
    if exclude_post_ids:
        query = query.where(Post.id.not_in(sorted(exclude_post_ids)))

    result = await session.execute(query)
    matched: set[int] = set()
    for row in result.all():
        try:
            pid = int(row[0])
        except Exception:
            continue
        text = (
            str(row[1] or "").strip()
            or str(row[2] or "").strip()
            or str(row[3] or "").strip()
            or str(row[4] or "").strip()
        )
        if _storyline_text_marker_hits(text, terms) >= 2:
            matched.add(pid)
    return matched


async def _build_storyline_timeline_payload(
    session,
    post_id: int,
    *,
    anchor_storyline_id: str | None = None,
    include_post_id: int | None = None,
    anchor_title_override: str | None = None,
    explicit_post_ids: list[int] | None = None,
) -> dict[str, object] | None:
    result = await session.execute(
        select(Post, Community)
        .outerjoin(Community, Post.community_id == Community.id)
        .where(Post.id == int(post_id))
    )
    anchor_rows = result.all()
    anchor_post_text = ""
    anchor_post_ts: datetime | None = None
    for post_obj, _community in anchor_rows:
        if int(post_obj.id) == int(post_id):
            anchor_post_text = _storyline_post_text(post_obj)
            anchor_post_ts = getattr(post_obj, "timestamp", None)
            break

    anchor_title_override = str(anchor_title_override or "").strip()
    explicit_post_ids = sorted({int(pid) for pid in (explicit_post_ids or []) if int(pid or 0) > 0})
    profile_anchor_text = "\n".join(
        part for part in [anchor_title_override, anchor_post_text] if str(part or "").strip()
    ).strip() or anchor_post_text
    anchor_profile = await _extract_storyline_anchor_profile_safe(profile_anchor_text, post_id=post_id)
    initial_query_tokens = _storyline_anchor_profile_terms(anchor_profile, _storyline_anchor_terms({}, profile_anchor_text))

    graph_payload = await collect_storyline_follow_candidates(
        post_id,
        anchor_storyline_id=str(anchor_storyline_id or "").strip() or None,
        query_tokens=initial_query_tokens,
    )
    if not graph_payload:
        return None

    anchor = dict(graph_payload.get("anchor") or {})
    if anchor_title_override:
        anchor["storyline_title"] = anchor_title_override
        anchor["storyline_seed_preview"] = anchor_post_text or str(anchor.get("storyline_seed_preview") or "").strip()
    else:
        anchor_profile = _storyline_enrich_anchor_profile_with_graph_context(anchor_profile, anchor)
    fallback_anchor_terms = _storyline_anchor_terms(anchor, profile_anchor_text)
    deterministic_branch_terms = _storyline_deterministic_branch_terms(anchor, profile_anchor_text)
    anchor_terms = _storyline_anchor_profile_terms(anchor_profile, fallback_anchor_terms)
    topic_terms = _storyline_anchor_profile_topic_terms(anchor_profile)
    negative_terms = _storyline_anchor_profile_negative_terms(anchor_profile)
    focus_phrases = _storyline_focus_phrases(anchor_profile, anchor, profile_anchor_text)
    candidate_storylines = [dict(item) for item in graph_payload.get("candidate_storylines") or []]
    post_refs = [dict(item) for item in graph_payload.get("post_refs") or []]
    deterministic_branch_storylines: list[dict[str, object]] = []
    anchor_markers = _storyline_anchor_markers(anchor, profile_anchor_text, limit=5)
    anchor_storyline_id_value = str(anchor.get("storyline_id") or "").strip()
    if anchor_storyline_id_value:
        structural_candidates = await search_storyline_structural_candidates(
            anchor_storyline_id=anchor_storyline_id_value,
            limit=12,
        )
        for row in structural_candidates:
            candidate = dict(row)
            marker_hits = _storyline_candidate_marker_hits(candidate, anchor_markers)
            if marker_hits >= 2 or (
                marker_hits >= 1
                and (
                    int(candidate.get("same_macro_topic") or 0) > 0
                    or int(candidate.get("shared_topic_count") or 0) >= 1
                    or int(candidate.get("shared_signature_count") or 0) >= 3
                )
            ):
                candidate["anchor_marker_hits"] = marker_hits
                deterministic_branch_storylines.append(candidate)
        if deterministic_branch_storylines:
            merged_candidate_storylines: list[dict[str, object]] = []
            seen_storyline_ids: set[str] = set()
            for item in [*candidate_storylines, *deterministic_branch_storylines]:
                storyline_id = str(item.get("storyline_id") or "").strip()
                if not storyline_id or storyline_id in seen_storyline_ids:
                    continue
                seen_storyline_ids.add(storyline_id)
                merged_candidate_storylines.append(item)
            candidate_storylines = merged_candidate_storylines
    macro_topic_id = str(anchor.get("macro_topic_id") or "").strip()

    if macro_topic_id:
        graph_step_payload = await _build_storyline_graph_step_payload(
            session,
            anchor=anchor,
            candidate_storylines=candidate_storylines,
            macro_topic_id=macro_topic_id,
            anchor_profile=anchor_profile,
            anchor_terms=anchor_terms,
            topic_terms=topic_terms,
            deterministic_branch_terms=deterministic_branch_terms,
            deterministic_branch_candidate_count=len(deterministic_branch_storylines),
            anchor_markers=anchor_markers,
            anchor_post_ts=anchor_post_ts,
            anchor_post_id=int(post_id),
            include_post_id=include_post_id,
            explicit_post_ids=explicit_post_ids,
            follow_storyline_title=anchor_title_override,
            strict_anchor_groups=bool(anchor_title_override and explicit_post_ids),
        )
        if graph_step_payload:
            return graph_step_payload

    storyline_meta = {
        str(item.get("storyline_id") or "").strip(): item
        for item in candidate_storylines
        if str(item.get("storyline_id") or "").strip()
    }
    storyline_by_post_id: dict[int, dict] = {}
    graph_post_ids: set[int] = set()
    for ref in post_refs:
        try:
            pid = int(ref.get("post_id"))
        except Exception:
            continue
        storyline_id = str(ref.get("storyline_id") or "").strip()
        graph_post_ids.add(pid)
        if storyline_id and storyline_id in storyline_meta:
            storyline_by_post_id[pid] = storyline_meta[storyline_id]

    graph_post_ids.add(int(post_id))
    for explicit_post_id in explicit_post_ids:
        graph_post_ids.add(int(explicit_post_id))
    if include_post_id and int(include_post_id) > 0:
        graph_post_ids.add(int(include_post_id))
    lexical_post_ids = await _expand_storyline_lexical_post_ids(
        session,
        anchor_terms=(anchor_terms + [term for term in topic_terms if term not in anchor_terms])[:14],
        exclude_post_ids=graph_post_ids,
        anchor_post_ts=anchor_post_ts,
        negative_terms=negative_terms,
    )
    historical_terms = _storyline_anchor_profile_historical_terms(anchor_profile, topic_terms or anchor_terms)
    historical_post_ids: set[int] = set()
    if anchor_post_ts is not None:
        earliest_seen_ts = None
        for ref in post_refs:
            try:
                pid = int(ref.get("post_id") or 0)
            except Exception:
                continue
            if pid == int(post_id):
                continue
            info = storyline_by_post_id.get(pid, {})
            for text_candidate in (
                str(info.get("seed_preview") or "").strip(),
                str(info.get("title") or "").strip(),
            ):
                if not text_candidate:
                    continue
            # earliest current timeline candidate is determined later from loaded cards,
            # so for topic-backfill we initially search the whole allowed pre-anchor window.
        historical_post_ids = await _expand_storyline_historical_topic_post_ids(
            session,
            historical_terms=historical_terms,
            exclude_post_ids=graph_post_ids | lexical_post_ids,
            anchor_post_ts=anchor_post_ts,
            earliest_seen_ts=None,
            negative_terms=negative_terms,
        )

    all_post_ids = sorted(graph_post_ids | lexical_post_ids | historical_post_ids)
    if not all_post_ids:
        return None

    result = await session.execute(
        select(Post, Community)
        .outerjoin(Community, Post.community_id == Community.id)
        .where(Post.id.in_(all_post_ids))
    )
    rows = result.all()

    post_map: dict[int, dict] = {}
    cards: list[dict[str, object]] = []
    min_allowed_ts: datetime | None = None
    if anchor_post_ts is not None:
        min_allowed_ts = anchor_post_ts - timedelta(days=STORYLINE_TIMELINE_LOOKBACK_DAYS)

    for post_obj, community in rows:
        storyline_info = storyline_by_post_id.get(int(post_obj.id), {})
        source_name = _storyline_source_name(post_obj, community)
        url = _extract_post_url(post_obj)
        text = _storyline_post_text(post_obj)
        term_match_count = _storyline_term_match_count(text, anchor_terms)
        post_ts = getattr(post_obj, "timestamp", None)
        retrieval_origin = "graph"
        if int(post_obj.id) == int(post_id):
            retrieval_origin = "anchor"
        elif int(post_obj.id) in lexical_post_ids:
            retrieval_origin = "lexical"
        elif int(post_obj.id) in historical_post_ids:
            retrieval_origin = "historical_topic"
        if int(post_obj.id) != int(post_id):
            if min_allowed_ts is not None and post_ts is not None and post_ts < min_allowed_ts and term_match_count < 2:
                continue
            if retrieval_origin == "graph" and term_match_count == 0:
                continue
        card = {
            "post_id": int(post_obj.id),
            "storyline_id": str(storyline_info.get("storyline_id") or ""),
            "storyline_title": str(storyline_info.get("title") or ""),
            "timestamp": post_ts.isoformat() if post_ts else "",
            "source": source_name,
            "url": url or "",
            "retrieval_origin": retrieval_origin,
            "term_match_count": term_match_count,
            "text": _trim_debug_text(text, limit=700),
        }
        cards.append(card)
        post_map[int(post_obj.id)] = {
            "post": post_obj,
            "community": community,
            "card": card,
        }

    cards.sort(key=lambda item: (item.get("timestamp") or "", int(item.get("post_id") or 0)))
    groups = _collapse_storyline_groups(cards)
    group_map = {str(item.get("group_id") or ""): item for item in groups}

    return {
        "anchor": anchor,
        "candidate_storylines": candidate_storylines,
        "candidate_macro_topics": [],
        "cards": cards,
        "groups": groups,
        "group_map": group_map,
        "post_map": post_map,
        "retrieval_debug": {
            "anchor_terms": anchor_terms,
            "anchor_markers": anchor_markers,
            "anchor_post_id": int(post_id),
            "anchor_post_timestamp": anchor_post_ts.isoformat() if anchor_post_ts else "",
            "deterministic_branch_terms": deterministic_branch_terms,
            "topic_terms": topic_terms,
            "follow_storyline_title": anchor_title_override,
            "explicit_post_count": len(explicit_post_ids),
            "historical_terms": historical_terms,
            "anchor_profile": anchor_profile or {},
            "negative_terms": negative_terms,
            "focus_phrases": focus_phrases,
            "deterministic_branch_candidate_count": len(deterministic_branch_storylines),
            "graph_post_count": len(graph_post_ids),
            "lexical_post_count": len(lexical_post_ids),
            "historical_post_count": len(historical_post_ids),
            "total_post_count": len(cards),
            "group_count": len(groups),
            "min_timestamp": cards[0]["timestamp"] if cards else "",
            "max_timestamp": cards[-1]["timestamp"] if cards else "",
        },
    }


def _extract_storyline_timeline_items(arbiter: dict | None, payload: dict[str, object]) -> tuple[str, str, list[dict], dict[str, dict]]:
    anchor = dict(payload.get("anchor") or {})
    groups = [dict(item) for item in payload.get("groups") or []]
    group_map = dict(payload.get("group_map") or {})
    retrieval_debug = dict(payload.get("retrieval_debug") or {})
    anchor_profile = dict(retrieval_debug.get("anchor_profile") or {})
    focus_phrases = [str(item or "").strip() for item in (retrieval_debug.get("focus_phrases") or []) if str(item or "").strip()]
    anchor_post_id = int(retrieval_debug.get("anchor_post_id") or 0)
    anchor_post_ts = None
    anchor_post_timestamp = str(retrieval_debug.get("anchor_post_timestamp") or "").strip()
    if anchor_post_timestamp:
        try:
            anchor_post_ts = datetime.fromisoformat(anchor_post_timestamp)
        except Exception:
            anchor_post_ts = None

    selected_items: list[dict] = []
    overview = ""
    story_title = str(
        retrieval_debug.get("follow_storyline_title")
        or retrieval_debug.get("macro_topic_title")
        or anchor.get("macro_topic_title")
        or anchor.get("storyline_title")
        or anchor.get("storyline_id")
        or "Сюжет"
    ).strip()

    if arbiter and isinstance(arbiter, dict):
        overview = str(arbiter.get("overview") or "").strip()
        story_title = str(arbiter.get("story_title") or story_title).strip() or story_title
        for item in arbiter.get("items") or []:
            if not isinstance(item, dict):
                continue
            group_ids = [
                str(group_id or "").strip()
                for group_id in (item.get("group_ids") or [])
                if str(group_id or "").strip()
            ]
            if not group_ids and item.get("group_id"):
                group_ids = [str(item.get("group_id") or "").strip()]
            if not group_ids:
                continue
            selected_groups = [group_map[group_id] for group_id in group_ids if group_id in group_map]
            if not selected_groups:
                continue
            selected_items.append({"group_ids": group_ids, "summary": str(item.get("summary") or "").strip()})

    if not selected_items:
        fallback_groups = list(groups)
        if _storyline_fast_mode_enabled(payload):
            fallback_groups = sorted(
                groups,
                key=lambda group: (str(group.get("end_timestamp") or group.get("start_timestamp") or ""), int((group.get("post_ids") or [0])[-1] or 0)),
                reverse=True,
            )[:10]
        for group in fallback_groups[:10]:
            group_id = str(group.get("group_id") or "").strip()
            if not group_id:
                continue
            selected_items.append({"group_ids": [group_id], "summary": str(group.get("representative_text") or "").strip()})
        if not overview:
            overview = str(anchor_profile.get("core_summary") or "").strip()
        if not overview:
            overview = "Связный сюжет собран по похожим постам и отсортирован по времени."
        story_title = str(anchor_profile.get("story_title") or story_title).strip() or story_title

    def _group_sort_key(item: dict) -> tuple[str, int]:
        ids = [str(group_id or "").strip() for group_id in item.get("group_ids") or [] if str(group_id or "").strip()]
        selected_groups = [group_map[group_id] for group_id in ids if group_id in group_map]
        if not selected_groups:
            return ("", 0)
        min_ts = min(str(group.get("start_timestamp") or "") for group in selected_groups)
        min_post_id = min(int(pid) for group in selected_groups for pid in group.get("post_ids", []) or [0])
        return (min_ts, min_post_id)

    selected_items.sort(key=_group_sort_key)

    payload_mode = str(retrieval_debug.get("payload_mode") or "").strip()
    if payload_mode.startswith("macro_topic_graph_steps") and 0 < len(groups) <= 6:
        selected_group_ids = {
            str(group_id or "").strip()
            for item in selected_items
            for group_id in (item.get("group_ids") or [])
            if str(group_id or "").strip()
        }
        for group in groups:
            group_id = str(group.get("group_id") or "").strip()
            if not group_id or group_id in selected_group_ids:
                continue
            selected_items.append(
                {
                    "group_ids": [group_id],
                    "summary": str(group.get("representative_text") or "").strip(),
                }
            )
        selected_items.sort(key=_group_sort_key)

    selected_group_ids = {
        str(group_id or "").strip()
        for item in selected_items
        for group_id in (item.get("group_ids") or [])
        if str(group_id or "").strip()
    }
    selected_has_foundational = any(
        _storyline_group_foundational_score(group_map[group_id]) >= 2
        for group_id in selected_group_ids
        if group_id in group_map
    )
    if not selected_has_foundational:
        earliest_selected = None
        for item in selected_items:
            start_ts, _end_ts = _storyline_item_time_bounds(item, group_map)
            if start_ts is not None:
                earliest_selected = start_ts
                break

        foundational_candidates = [
            group
            for group in groups
            if str(group.get("group_id") or "").strip() not in selected_group_ids
            and _storyline_group_foundational_score(group) >= 2
            and _storyline_group_has_strong_anchor_signal(
                relevance=int(group.get("anchor_relevance_score") or 0),
                marker_hits=int(group.get("anchor_marker_hits") or 0),
                focus_phrase_hits=int(group.get("focus_phrase_hits") or 0),
                must_overlap=int(group.get("must_overlap") or 0),
                must_phrase_overlap=int(group.get("must_phrase_overlap") or 0),
                summary_garbage=_is_storyline_summary_garbage(
                    _storyline_clean_summary_text(str(group.get("representative_text") or "").strip(), limit=280)
                ),
            )
        ]
        foundational_candidates.sort(
            key=lambda group: (
                _storyline_group_time(group) or datetime.max.replace(tzinfo=timezone.utc),
                -_storyline_group_foundational_score(group),
            )
        )
        if foundational_candidates:
            best_group = foundational_candidates[0]
            best_group_time = _storyline_group_time(best_group)
            if earliest_selected is None or best_group_time is None or best_group_time <= earliest_selected:
                selected_items.insert(
                    0,
                    {
                        "group_ids": [str(best_group.get("group_id") or "").strip()],
                        "summary": str(best_group.get("representative_text") or "").strip(),
                    },
                )

    selected_items = _storyline_ensure_anchor_post_item(selected_items, groups, anchor_post_id=anchor_post_id)
    selected_items = _storyline_filter_items_for_anchor_time_window(
        selected_items,
        group_map,
        anchor_post_ts=anchor_post_ts,
    )
    selected_items = _storyline_filter_selected_items_for_anchor(selected_items, group_map)
    selected_items = _storyline_backfill_selected_items(
        selected_items,
        groups,
        group_map,
        min_items=_storyline_timeline_min_items(groups),
    )
    selected_items.sort(key=_group_sort_key)
    if focus_phrases and _storyline_phrase_hits(story_title, focus_phrases) == 0:
        story_title = _storyline_focus_title(anchor_profile, story_title)
    if focus_phrases and _storyline_phrase_hits(overview, focus_phrases) == 0:
        overview = _storyline_focus_overview(selected_items, group_map, overview, anchor_profile)
    return story_title, overview, selected_items, group_map


async def _expand_storyline_timeline_items(session, payload: dict[str, object], arbiter: dict | None) -> tuple[str, str, list[dict], dict[str, dict]]:
    story_title, overview, selected_items, group_map = _extract_storyline_timeline_items(arbiter, payload)
    fallback_terms = [str(term or "").strip().lower() for term in ((payload.get("retrieval_debug") or {}).get("anchor_terms") or [])]
    fast_mode = _storyline_fast_mode_enabled(payload)

    haystack = func.lower(
        func.concat_ws(
            " ",
            func.coalesce(Post.summary, ""),
            func.coalesce(Post.processed_content, ""),
            func.coalesce(Post.content, ""),
            func.coalesce(Post.title, ""),
        )
    )

    for item in selected_items[:10]:
        item_group_ids = [str(group_id or "").strip() for group_id in item.get("group_ids") or [] if str(group_id or "").strip()]
        item_groups = [group_map[group_id] for group_id in item_group_ids if group_id in group_map]
        if not item_groups:
            item["expanded_sources"] = []
            continue

        existing_sources: list[dict] = []
        for group in item_groups:
            existing_sources.extend(group.get("sources", []))
        existing_sources.sort(key=lambda row: (str(row.get("timestamp") or ""), int(row.get("post_id") or 0)))

        step_texts = [str(item.get("summary") or "").strip()]
        step_texts.extend(str(group.get("representative_text") or "").strip() for group in item_groups)
        step_terms = _storyline_step_terms(step_texts, fallback_terms=fallback_terms)
        item["step_terms"] = step_terms

        timestamps = [
            datetime.fromisoformat(str(src.get("timestamp")))
            for src in existing_sources
            if str(src.get("timestamp") or "").strip()
        ]
        if timestamps:
            source_window_hours = STORYLINE_TIMELINE_FAST_SOURCE_WINDOW_HOURS if fast_mode else 18
            window_start = min(timestamps) - timedelta(hours=source_window_hours)
            window_end = max(timestamps) + timedelta(hours=source_window_hours)
        else:
            window_start = None
            window_end = None

        extra_sources: list[dict] = []
        expansion_terms = _storyline_source_expansion_terms(step_terms, fallback_terms=fallback_terms, limit=8)
        allow_extra_source_expansion = step_terms and window_start and window_end and (
            not fast_mode or len(selected_items) <= 6
        )
        if allow_extra_source_expansion and len(expansion_terms) < 2:
            allow_extra_source_expansion = False
        if allow_extra_source_expansion:
            hit_expr = None
            for term in expansion_terms[:8]:
                clause = case((haystack.like(f"%{term}%"), 1), else_=0)
                hit_expr = clause if hit_expr is None else hit_expr + clause

            min_hits = 3 if fast_mode and len(expansion_terms) >= 3 else 2
            query = (
                select(Post, Community)
                .outerjoin(Community, Post.community_id == Community.id)
                .where(Post.timestamp >= window_start)
                .where(Post.timestamp <= window_end)
                .where(hit_expr >= min_hits)
                .order_by(Post.timestamp.asc(), Post.id.asc())
            )
            if fast_mode:
                query = query.limit(max(1, int(STORYLINE_TIMELINE_FAST_SOURCE_EXPANSION_LIMIT)))
            result = await session.execute(query)
            for post_obj, community in result.all():
                text = _storyline_post_text(post_obj)
                if _storyline_term_match_count(text, expansion_terms) < min_hits:
                    continue
                extra_sources.append(
                    {
                        "post_id": int(post_obj.id),
                        "timestamp": getattr(post_obj, "timestamp", None).isoformat() if getattr(post_obj, "timestamp", None) else "",
                        "source": _storyline_source_name(post_obj, community),
                        "url": _extract_post_url(post_obj) or "",
                    }
                )

        merged_sources = [*existing_sources, *extra_sources]
        deduped_sources: list[dict] = []
        seen_post_ids: set[int] = set()
        seen_links: set[str] = set()
        for src in sorted(merged_sources, key=lambda row: (str(row.get("timestamp") or ""), int(row.get("post_id") or 0))):
            pid = int(src.get("post_id") or 0)
            link = str(src.get("url") or "").strip()
            if pid and pid in seen_post_ids:
                continue
            if link and link in seen_links:
                continue
            if pid:
                seen_post_ids.add(pid)
            if link:
                seen_links.add(link)
            deduped_sources.append(src)

        item["expanded_sources"] = deduped_sources
        item["_resolved_summary"] = _storyline_best_group_summary(item, item_groups)

    min_items = _storyline_timeline_min_items([dict(item) for item in (payload.get("groups") or [])])
    selected_items_before_merge_count = len(selected_items)
    selected_items = _merge_storyline_timeline_items(selected_items, group_map)
    if len(selected_items) < min_items:
        selected_items = _storyline_backfill_distinct_timeline_items_after_merge(
            selected_items,
            [dict(item) for item in (payload.get("groups") or [])],
            group_map,
            min_items=min_items,
        )
    if fast_mode and selected_items_before_merge_count != len(selected_items):
        log.info(
            "storyline_timeline_fast_items_post_merge",
            before_merge=selected_items_before_merge_count,
            after_merge=len(selected_items),
            min_items=min_items,
        )
    return story_title, overview, selected_items, group_map


def _build_storyline_timeline_message(
    locale: str,
    arbiter: dict | None,
    payload: dict[str, object],
    *,
    telegram_id: int | None = None,
    expanded_items: list[dict] | None = None,
) -> str | None:
    story_title, overview, selected_items, group_map = _extract_storyline_timeline_items(arbiter, payload)
    if expanded_items is not None:
        selected_items = expanded_items

    collected_sources: list[dict] = []
    item_sources_map: dict[int, list[dict]] = {}
    for idx, item in enumerate(selected_items[:10], start=1):
        item_group_ids = [str(group_id or "").strip() for group_id in item.get("group_ids") or [] if str(group_id or "").strip()]
        item_groups = [group_map[group_id] for group_id in item_group_ids if group_id in group_map]
        if not item_groups:
            continue
        all_sources: list[dict] = list(item.get("expanded_sources") or [])
        if not all_sources:
            for group in item_groups:
                all_sources.extend(group.get("sources", []))
        all_sources.sort(key=lambda row: (str(row.get("timestamp") or ""), int(row.get("post_id") or 0)))
        item_sources_map[idx] = all_sources
        collected_sources.extend(all_sources)

    unique_sources: list[dict] = []
    seen_unique: set[tuple[int, str]] = set()
    for src in sorted(collected_sources, key=lambda row: (str(row.get("timestamp") or ""), int(row.get("post_id") or 0))):
        key = (int(src.get("post_id") or 0), str(src.get("url") or "").strip())
        if key in seen_unique:
            continue
        seen_unique.add(key)
        unique_sources.append(src)

    heavy_story_mode = len(unique_sources) > STORYLINE_TIMELINE_HEAVY_SOURCE_THRESHOLD

    lines: list[str] = [f"<b>{html.escape('Краткая хронология сюжета:' if locale == 'ru' else 'Storyline timeline:')}</b>", "", f"<b><i>{html.escape(story_title)}</i></b>"]
    if overview:
        lines.extend(["", html.escape(overview)])
    if heavy_story_mode:
        lines.extend(
            [
                "",
                html.escape(
                    f"Сюжет очень активный: найдено {len(unique_sources)} релевантных источников. "
                    f"Ниже показаны {min(STORYLINE_TIMELINE_HEAVY_SOURCE_LIMIT, len(unique_sources))} самых новых ссылок."
                    if locale == "ru"
                    else f"This is a heavy storyline: {len(unique_sources)} relevant sources were found. "
                    f"Below are the {min(STORYLINE_TIMELINE_HEAVY_SOURCE_LIMIT, len(unique_sources))} newest links."
                ),
            ]
        )

    visible_items: list[tuple[dict, list[dict], list[dict]]] = []
    for idx, item in enumerate(selected_items[:10], start=1):
        item_group_ids = [str(group_id or "").strip() for group_id in item.get("group_ids") or [] if str(group_id or "").strip()]
        item_groups = [group_map[group_id] for group_id in item_group_ids if group_id in group_map]
        if not item_groups:
            continue
        all_sources = list(item_sources_map.get(idx) or [])
        all_sources.sort(key=lambda row: (str(row.get("timestamp") or ""), int(row.get("post_id") or 0)))
        if heavy_story_mode and len(all_sources) > STORYLINE_TIMELINE_HEAVY_SOURCE_LIMIT:
            head_count = max(1, STORYLINE_TIMELINE_HEAVY_SOURCE_LIMIT // 2)
            tail_count = max(1, STORYLINE_TIMELINE_HEAVY_SOURCE_LIMIT - head_count)
            sliced = [*all_sources[:head_count], *all_sources[-tail_count:]]
            deduped_sliced: list[dict] = []
            seen_sliced: set[tuple[int, str]] = set()
            for src in sliced:
                key = _storyline_source_key(src)
                if key in seen_sliced:
                    continue
                seen_sliced.add(key)
                deduped_sliced.append(src)
            all_sources = deduped_sliced
        if not all_sources:
            continue
        visible_items.append((item, item_groups, all_sources))

    for idx, (item, item_groups, all_sources) in enumerate(visible_items, start=1):
        start_ts, _end_ts = _storyline_item_time_bounds(item, group_map)
        if start_ts is not None:
            time_text = start_ts.astimezone(timezone.utc).isoformat(sep=" ", timespec="seconds").replace("+00:00", " UTC")
        else:
            first_source = all_sources[0] if all_sources else {}
            time_text = str(first_source.get("timestamp") or "—").replace("T", " ").replace("+00:00", " UTC")
        summary_text = str(item.get("_resolved_summary") or "").strip() or _storyline_best_group_summary(item, item_groups)
        source_lines: list[str] = []
        tracking_enabled = bool(telegram_id) and source_link_tracking_enabled()
        for src in all_sources:
            raw_label = str(src.get("source") or "unknown").strip()
            raw_url = str(src.get("url") or "").strip()
            raw_post_id = int(src.get("post_id") or 0)
            display_text = _storyline_source_display_text(raw_label, raw_url)
            tracked_url = _storyline_source_link_url(
                raw_url,
                raw_post_id,
                int(telegram_id) if tracking_enabled else telegram_id,
            )
            if tracked_url:
                source_lines.append(
                    f'- <a href="{html.escape(tracked_url)}">{html.escape(display_text)}</a>'
                )
            else:
                source_lines.append(f"- {html.escape(display_text)}")
        lines.extend(
            [
                "",
                f"<b>{html.escape(f'{idx}. {time_text}')}</b>",
                html.escape(summary_text),
                html.escape("Источники:" if locale == "ru" else "Sources:"),
                *source_lines,
            ]
        )

    lines.extend(["", "<i>Отслеживание сюжетов находится в фазе тестирования и активной доработки. Могут возникать ошибки.</i>"])
    return "\n".join(lines).strip() if lines else None


async def _send_storyline_timeline_for_post(
    *,
    session,
    message: types.Message | None,
    locale: str,
    post_id: int,
    user_id: int,
    telegram_id: int | None = None,
    target_chat_id: int | None = None,
    anchor_storyline_id: str | None = None,
    include_post_id: int | None = None,
    anchor_title_override: str | None = None,
    explicit_post_ids: list[int] | None = None,
    show_similar_button: bool = True,
    log_prefix: str = "storyline_timeline",
) -> None:
    if not message and target_chat_id is None:
        return

    log.info(f"{log_prefix}_start", post_id=post_id, user_id=user_id, anchor_storyline_id=anchor_storyline_id)
    send_chat_id = target_chat_id
    if send_chat_id is None and message is not None:
        send_chat_id = getattr(getattr(message, "chat", None), "id", None)
    progress_message: types.Message | None = None
    if message is not None:
        progress_message = await message.answer(
            t(locale, "storyline_timeline_building"),
            skip_service_cleanup=True,
        )
    elif send_chat_id is not None:
        progress_message = await bot.send_message(
            chat_id=int(send_chat_id),
            text=t(locale, "storyline_timeline_building"),
        )
    try:
        timeline_payload = await asyncio.wait_for(
            _build_storyline_timeline_payload(
                session,
                post_id,
                anchor_storyline_id=str(anchor_storyline_id or "").strip() or None,
                include_post_id=include_post_id,
                anchor_title_override=str(anchor_title_override or "").strip() or None,
                explicit_post_ids=explicit_post_ids,
            ),
            timeout=STORYLINE_TIMELINE_PAYLOAD_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning(
            f"{log_prefix}_payload_timeout",
            post_id=post_id,
            user_id=user_id,
            timeout=STORYLINE_TIMELINE_PAYLOAD_TIMEOUT,
        )
        timeline_payload = None
    except Exception:
        log.exception(f"{log_prefix}_payload_failed", post_id=post_id, user_id=user_id)
        timeline_payload = None

    timeline_text: str | None = None
    if timeline_payload:
        fast_mode = _storyline_fast_mode_enabled(timeline_payload)
        if fast_mode:
            log.info(
                f"{log_prefix}_fast_mode",
                post_id=post_id,
                user_id=user_id,
                cards=len(timeline_payload.get("cards") or []),
                groups=len(timeline_payload.get("groups") or []),
            )
        try:
            if fast_mode:
                arbiter_result = await asyncio.wait_for(
                    arbitrate_storyline_timeline(
                        anchor=dict(timeline_payload.get("anchor") or {}),
                        candidate_storylines=[dict(item) for item in timeline_payload.get("candidate_storylines") or []],
                        candidate_macro_topics=[dict(item) for item in timeline_payload.get("candidate_macro_topics") or []],
                        mixed_topic_mode=bool(((timeline_payload.get("retrieval_debug") or {}).get("mixed_topic_mode"))),
                        anchor_profile=dict(((timeline_payload.get("retrieval_debug") or {}).get("anchor_profile") or {})),
                        groups=_storyline_fast_mode_groups(timeline_payload),
                        max_items=8,
                    ),
                    timeout=STORYLINE_TIMELINE_FAST_ARB_TIMEOUT,
                )
            else:
                arbiter_result = await asyncio.wait_for(
                    arbitrate_storyline_timeline(
                        anchor=dict(timeline_payload.get("anchor") or {}),
                        candidate_storylines=[dict(item) for item in timeline_payload.get("candidate_storylines") or []],
                        candidate_macro_topics=[dict(item) for item in timeline_payload.get("candidate_macro_topics") or []],
                        mixed_topic_mode=bool(((timeline_payload.get("retrieval_debug") or {}).get("mixed_topic_mode"))),
                        anchor_profile=dict(((timeline_payload.get("retrieval_debug") or {}).get("anchor_profile") or {})),
                        groups=[dict(item) for item in timeline_payload.get("groups") or []],
                        max_items=10,
                    ),
                    timeout=STORYLINE_TIMELINE_ARBITER_TIMEOUT,
                )
        except asyncio.TimeoutError:
            log.warning(
                f"{log_prefix}_arbiter_timeout",
                post_id=post_id,
                user_id=user_id,
                timeout=STORYLINE_TIMELINE_FAST_ARB_TIMEOUT if fast_mode else STORYLINE_TIMELINE_ARBITER_TIMEOUT,
            )
            arbiter_result = None
        except (DeepseekAuthError, DeepseekRetryableError):
            arbiter_result = None
        except Exception:
            log.exception(f"{log_prefix}_arbiter_failed", post_id=post_id, user_id=user_id)
            arbiter_result = None
        try:
            _story_title, _overview, expanded_items, _group_map = await asyncio.wait_for(
                _expand_storyline_timeline_items(session, timeline_payload, arbiter_result),
                timeout=STORYLINE_TIMELINE_PAYLOAD_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning(
                f"{log_prefix}_source_expansion_timeout",
                post_id=post_id,
                user_id=user_id,
                timeout=STORYLINE_TIMELINE_PAYLOAD_TIMEOUT,
            )
            expanded_items = None
        except Exception:
            log.exception(f"{log_prefix}_source_expansion_failed", post_id=post_id, user_id=user_id)
            expanded_items = None
        timeline_text = _build_storyline_timeline_message(
            locale,
            arbiter_result,
            timeline_payload,
            telegram_id=telegram_id,
            expanded_items=expanded_items,
        )
    if not timeline_text:
        timeline_text = t(locale, "storyline_timeline_empty")
    timeline_parts = _split_telegram_text(timeline_text)
    if not timeline_parts:
        timeline_parts = [t(locale, "storyline_timeline_empty")]
    log.info(
        f"{log_prefix}_ready",
        post_id=post_id,
        user_id=user_id,
        parts=len(timeline_parts),
        first_part_len=len(timeline_parts[0]),
        payload_present=bool(timeline_payload),
    )
    similar_markup = (
        _storyline_similar_markup(
            locale,
            int(include_post_id or post_id),
            None if include_post_id else anchor_storyline_id,
        )
        if show_similar_button and timeline_payload
        else None
    )
    try:
        if progress_message is not None and message is not None:
            await progress_message.edit_text(
                timeline_parts[0],
                disable_web_page_preview=True,
                parse_mode="HTML",
                reply_markup=similar_markup,
            )
            log.info(f"{log_prefix}_sent_via_edit", post_id=post_id, user_id=user_id)
        elif send_chat_id is not None:
            await bot.send_message(
                chat_id=int(send_chat_id),
                text=timeline_parts[0],
                disable_web_page_preview=True,
                parse_mode="HTML",
                reply_markup=similar_markup,
            )
            log.info(f"{log_prefix}_sent_via_chat_send_html", post_id=post_id, user_id=user_id)
        else:
            raise RuntimeError("timeline target chat is unavailable")
    except Exception as exc:
        log.warning(f"{log_prefix}_edit_failed", post_id=post_id, user_id=user_id, error=repr(exc))
        await _try_delete_message(progress_message, reason=f"{log_prefix}_stale_progress")
        try:
            if message is not None:
                await message.answer(
                    timeline_parts[0],
                    disable_web_page_preview=True,
                    parse_mode="HTML",
                    skip_service_cleanup=True,
                    reply_markup=similar_markup,
                )
                log.info(f"{log_prefix}_sent_via_reply_html", post_id=post_id, user_id=user_id)
            elif send_chat_id is not None:
                await bot.send_message(
                    chat_id=int(send_chat_id),
                    text=timeline_parts[0],
                    disable_web_page_preview=True,
                    parse_mode="HTML",
                    reply_markup=similar_markup,
                )
                log.info(f"{log_prefix}_sent_via_chat_html_fallback", post_id=post_id, user_id=user_id)
        except Exception:
            log.exception(f"{log_prefix}_send_html_failed", post_id=post_id, user_id=user_id)
            plain_part = _storyline_timeline_plain_text(timeline_parts[0]) or t(locale, "storyline_timeline_empty")
            if message is not None:
                await message.answer(
                    plain_part,
                    disable_web_page_preview=True,
                    skip_service_cleanup=True,
                    reply_markup=similar_markup,
                )
                log.info(f"{log_prefix}_sent_via_reply_plain", post_id=post_id, user_id=user_id)
            elif send_chat_id is not None:
                await bot.send_message(
                    chat_id=int(send_chat_id),
                    text=plain_part,
                    disable_web_page_preview=True,
                    reply_markup=similar_markup,
                )
                log.info(f"{log_prefix}_sent_via_chat_plain_fallback", post_id=post_id, user_id=user_id)
    for extra_part in timeline_parts[1:]:
        try:
            if message is not None:
                await message.answer(
                    extra_part,
                    disable_web_page_preview=True,
                    parse_mode="HTML",
                    skip_service_cleanup=True,
                )
            elif send_chat_id is not None:
                await bot.send_message(
                    chat_id=int(send_chat_id),
                    text=extra_part,
                    disable_web_page_preview=True,
                    parse_mode="HTML",
                )
        except Exception:
            log.exception(f"{log_prefix}_extra_part_html_failed", post_id=post_id, user_id=user_id)
            plain_extra = _storyline_timeline_plain_text(extra_part)
            if plain_extra:
                if message is not None:
                    await message.answer(
                        plain_extra,
                        disable_web_page_preview=True,
                        skip_service_cleanup=True,
                    )
                elif send_chat_id is not None:
                    await bot.send_message(
                        chat_id=int(send_chat_id),
                        text=plain_extra,
                        disable_web_page_preview=True,
                    )
    log.info(f"{log_prefix}_done", post_id=post_id, user_id=user_id, parts=len(timeline_parts))


async def _safe_answer_callback(
    callback: types.CallbackQuery,
    *,
    text: str | None = None,
    show_alert: bool = False,
) -> bool:
    try:
        await callback.answer(text, show_alert=show_alert)
        return True
    except TelegramBadRequest as exc:
        message = str(exc).lower()
        if "query is too old" in message or "query id is invalid" in message:
            log.info(
                "callback_answer_expired_continue",
                callback_data=str(getattr(callback, "data", "") or ""),
                error=str(exc),
            )
            return False
        raise


async def _notify_callback_or_message(
    callback: types.CallbackQuery,
    *,
    text: str,
    show_alert: bool = True,
) -> None:
    answered = await _safe_answer_callback(callback, text=text, show_alert=show_alert)
    if answered:
        return
    if callback.message:
        await callback.message.answer(text)


def _storyline_event_source_post_ids(sources: object) -> set[int]:
    post_ids: set[int] = set()
    if not isinstance(sources, list):
        return post_ids
    for item in sources:
        if not isinstance(item, dict):
            continue
        try:
            post_id = int(item.get("post_id") or 0)
        except Exception:
            post_id = 0
        if post_id > 0:
            post_ids.add(post_id)
    return post_ids


async def _storyline_follow_timeline_post_ids(
    session,
    *,
    follow: UserStorylineFollow | None,
    include_post_id: int | None = None,
) -> list[int]:
    if follow is None:
        return [int(include_post_id)] if include_post_id and int(include_post_id) > 0 else []

    post_ids: set[int] = set()
    source_post_id = int(getattr(follow, "source_post_id", 0) or 0)
    if source_post_id > 0:
        post_ids.add(source_post_id)
    if include_post_id and int(include_post_id) > 0:
        post_ids.add(int(include_post_id))

    event_rows = await session.execute(
        select(StorylineUpdateEvent)
        .where(StorylineUpdateEvent.user_id == int(getattr(follow, "user_id", 0) or 0))
        .where(StorylineUpdateEvent.follow_id == int(getattr(follow, "id", 0) or 0))
        .order_by(StorylineUpdateEvent.first_seen_at.asc(), StorylineUpdateEvent.id.asc())
        .limit(120)
    )
    for event in event_rows.scalars().all():
        for raw_post_id in (
            getattr(event, "canonical_post_id", None),
            getattr(event, "last_post_id", None),
        ):
            try:
                post_id = int(raw_post_id or 0)
            except Exception:
                post_id = 0
            if post_id > 0:
                post_ids.add(post_id)
        post_ids.update(_storyline_event_source_post_ids(getattr(event, "sources", None)))

    return sorted(post_ids)


@dp.callback_query(F.data.startswith("storytimeline:") | F.data.startswith("storytimelinecard:"))
async def handle_storyline_timeline(callback: types.CallbackQuery):
    if not _storytracking_available(
        telegram_id=callback.from_user.id,
        username=callback.from_user.username,
    ):
        locale = await _resolve_callback_locale(callback)
        await _safe_answer_callback(callback, text=t(locale, "storytracking_disabled"), show_alert=True)
        return

    try:
        parts = str(callback.data or "").split(":")
        prefix = parts[0]
        post_id_str = parts[1]
        post_id = int(post_id_str)
        timeline_anchor_post_id = int(parts[2]) if prefix == "storytimelinecard" and len(parts) >= 3 and str(parts[2]).strip() else None
    except (ValueError, AttributeError):
        await _notify_callback_or_message(
            callback,
            text=t(await _resolve_callback_locale(callback), "storyline_follow_not_found"),
        )
        return

    await _safe_answer_callback(callback)

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == callback.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await _notify_callback_or_message(
                callback,
                text=t(_button_locale_from_callback(callback), "start_required"),
            )
            break

        locale = get_user_locale(user)
        post_result = await session.execute(select(Post).where(Post.id == post_id))
        post = post_result.scalar_one_or_none()
        if not post:
            await _notify_callback_or_message(callback, text=t(locale, "post_not_found"))
            break

        await _record_post_interaction(
            session,
            user=user,
            post=post,
            action="storytimeline_open",
            source="storyline_card" if prefix == "storytimelinecard" else "storyline_post",
        )

        timeline_follow: UserStorylineFollow | None = None
        if prefix == "storytimelinecard" and timeline_anchor_post_id:
            follow_result = await session.execute(
                select(UserStorylineFollow)
                .where(UserStorylineFollow.user_id == user.id)
                .where(UserStorylineFollow.is_active.is_(True))
                .where(UserStorylineFollow.source_post_id == int(timeline_anchor_post_id))
                .order_by(UserStorylineFollow.id.desc())
                .limit(1)
            )
            timeline_follow = follow_result.scalar_one_or_none()

        if prefix == "storytimelinecard" and not timeline_anchor_post_id:
            post_tokens = set(_storyline_tokens(_storyline_post_text(post)))
            follow_rows = await session.execute(
                select(UserStorylineFollow)
                .where(UserStorylineFollow.user_id == user.id)
                .where(UserStorylineFollow.is_active.is_(True))
            )
            best_follow_source_post_id = 0
            best_follow_score = 0
            best_follow: UserStorylineFollow | None = None
            for follow in follow_rows.scalars().all():
                source_post_id = int(getattr(follow, "source_post_id", 0) or 0)
                if source_post_id <= 0 or source_post_id == int(post_id):
                    continue
                follow_tokens = set(_storyline_tokens(str(getattr(follow, "storyline_title", "") or "")))
                exact_overlap = len(post_tokens & follow_tokens)
                loose_overlap = sum(
                    1
                    for post_token in post_tokens
                    for follow_token in follow_tokens
                    if min(len(post_token), len(follow_token)) >= 5
                    and (post_token in follow_token or follow_token in post_token)
                )
                overlap = exact_overlap + loose_overlap
                if overlap > best_follow_score:
                    best_follow_score = overlap
                    best_follow_source_post_id = source_post_id
                    best_follow = follow
            if best_follow_score >= 1 and best_follow_source_post_id > 0:
                timeline_anchor_post_id = best_follow_source_post_id
                timeline_follow = best_follow

        timeline_post_id = int(timeline_anchor_post_id or post_id)
        include_post_id = int(post_id) if prefix == "storytimelinecard" and timeline_post_id != int(post_id) else None
        timeline_post = post
        if timeline_post_id != int(post_id):
            timeline_post_result = await session.execute(select(Post).where(Post.id == timeline_post_id))
            timeline_post = timeline_post_result.scalar_one_or_none()
            if not timeline_post:
                timeline_post = post
                timeline_post_id = int(post_id)
                include_post_id = None

        post_text = _storyline_post_text(timeline_post)
        anchor_profile = await _extract_storyline_anchor_profile_safe(post_text, post_id=timeline_post_id)
        context = await _resolve_storyline_follow_context(
            timeline_post_id,
            post_text,
            anchor_profile=anchor_profile,
            session=session,
            anchor_post_ts=getattr(timeline_post, "timestamp", None),
        )
        if not context:
            log.info("storyline_timeline_missing_context", post_id=timeline_post_id, user_id=user.id)
            if callback.message:
                await callback.message.answer(t(locale, "storyline_follow_not_found"))
            else:
                await _notify_callback_or_message(callback, text=t(locale, "storyline_follow_not_found"))
            break

        stored_follow_storyline_id = str(getattr(timeline_follow, "storyline_id", "") or "").strip()
        stored_follow_root_id = str(getattr(timeline_follow, "family_root_storyline_id", "") or "").strip()
        anchor_storyline_id = stored_follow_storyline_id or str(context.get("storyline_id") or "").strip() or None
        log.info(
            "storyline_timeline_anchor_resolved",
            post_id=post_id,
            timeline_post_id=timeline_post_id,
            include_post_id=include_post_id,
            user_id=user.id,
            follow_id=int(getattr(timeline_follow, "id", 0) or 0),
            stored_follow_storyline_id=stored_follow_storyline_id,
            stored_follow_root_id=stored_follow_root_id,
            context_storyline_id=str(context.get("storyline_id") or "").strip(),
            anchor_storyline_id=anchor_storyline_id,
        )

        timeline_explicit_post_ids = await _storyline_follow_timeline_post_ids(
            session,
            follow=timeline_follow,
            include_post_id=include_post_id,
        )
        timeline_title_override = str(getattr(timeline_follow, "storyline_title", "") or "").strip()

        target_chat_id = getattr(getattr(callback, "message", None), "chat", None)
        target_chat_id_value = int(getattr(target_chat_id, "id", 0) or 0) or None

        await _send_storyline_timeline_for_post(
            session=session,
            message=callback.message,
            locale=locale,
            post_id=timeline_post_id,
            user_id=user.id,
            telegram_id=user.telegram_id,
            target_chat_id=target_chat_id_value,
            anchor_storyline_id=anchor_storyline_id,
            include_post_id=include_post_id,
            anchor_title_override=timeline_title_override,
            explicit_post_ids=timeline_explicit_post_ids,
            log_prefix="storyline_timeline",
        )
        break


@dp.callback_query(F.data.startswith("storysimilar:"))
async def handle_storyline_similar(callback: types.CallbackQuery):
    if not _storytracking_available(
        telegram_id=callback.from_user.id,
        username=callback.from_user.username,
    ):
        locale = await _resolve_callback_locale(callback)
        await _safe_answer_callback(callback, text=t(locale, "storytracking_disabled"), show_alert=True)
        return

    try:
        parts = str(callback.data or "").split(":", 2)
        post_id = int(parts[1])
        anchor_storyline_id = str(parts[2] or "").strip() if len(parts) >= 3 else ""
    except (ValueError, AttributeError, IndexError):
        await _notify_callback_or_message(
            callback,
            text=t(await _resolve_callback_locale(callback), "storyline_follow_not_found"),
        )
        return

    await _safe_answer_callback(callback)

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == callback.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await _notify_callback_or_message(
                callback,
                text=t(_button_locale_from_callback(callback), "start_required"),
            )
            break

        locale = get_user_locale(user)
        post_result = await session.execute(select(Post).where(Post.id == post_id))
        post = post_result.scalar_one_or_none()
        if not post:
            await _notify_callback_or_message(callback, text=t(locale, "post_not_found"))
            break

        await _record_post_interaction(
            session,
            user=user,
            post=post,
            action="storysimilar_open",
            source="storyline_timeline",
        )
        post_text = _storyline_post_text(post)
        anchor_profile = await _extract_storyline_anchor_profile_safe(post_text, post_id=post_id)
        focus_tokens = _storyline_anchor_profile_terms(anchor_profile, _storyline_anchor_terms({}, post_text))

        progress_message = None
        if callback.message:
            progress_message = await callback.message.answer(
                t(locale, "storyline_similar_building"),
                skip_service_cleanup=True,
            )

        log.info(
            "storyline_similar_lookup_start",
            post_id=post_id,
            user_id=user.id,
            anchor_storyline_id=anchor_storyline_id,
            focus_tokens=focus_tokens[:12],
        )
        payload = await collect_related_storyline_branches(
            post_id,
            anchor_storyline_id=anchor_storyline_id or None,
            query_tokens=focus_tokens,
            limit=12,
        )
        if payload:
            payload["anchor"] = {
                **dict(payload.get("anchor") or {}),
                "focus_post_text": _trim_debug_text(post_text, limit=900),
            }
        candidates = list((payload or {}).get("candidate_branches") or [])
        if payload and not candidates:
            payload, candidates = await _recover_related_storyline_candidates_from_db(
                session,
                post_id=post_id,
                post_text=post_text,
                focus_tokens=focus_tokens,
                payload=payload,
                limit=12,
            )
            if payload:
                payload["anchor"] = {
                    **dict(payload.get("anchor") or {}),
                    "focus_post_text": _trim_debug_text(post_text, limit=900),
                }
        candidates = await _enrich_related_storyline_candidates_with_examples(session, [dict(item) for item in candidates])
        if not payload or not candidates:
            text = t(locale, "storyline_similar_empty")
            if progress_message:
                with contextlib.suppress(Exception):
                    await progress_message.edit_text(text)
            elif callback.message:
                await callback.message.answer(text, skip_service_cleanup=True)
            else:
                await _notify_callback_or_message(callback, text=text, show_alert=True)
            log.info("storyline_similar_empty", post_id=post_id, user_id=user.id, payload_present=bool(payload))
            break

        branches: list[dict] = []
        try:
            llm_result = await asyncio.wait_for(
                summarize_related_storyline_branches(
                    anchor=dict(payload.get("anchor") or {}),
                    candidates=[dict(item) for item in candidates],
                    max_branches=5,
                ),
                timeout=STORYLINE_SIMILAR_ARBITER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning(
                "storyline_similar_arbiter_timeout",
                post_id=post_id,
                user_id=user.id,
                timeout=STORYLINE_SIMILAR_ARBITER_TIMEOUT,
            )
            llm_result = None
        except (DeepseekAuthError, DeepseekRetryableError):
            llm_result = None
        except Exception:
            log.exception("storyline_similar_arbiter_failed", post_id=post_id, user_id=user.id)
            llm_result = None

        candidates_by_storyline_id = {
            str(item.get("storyline_id") or "").strip(): dict(item)
            for item in candidates
            if str(item.get("storyline_id") or "").strip()
        }
        candidates_by_family_root_id = {
            str(item.get("family_root_storyline_id") or item.get("storyline_id") or "").strip(): dict(item)
            for item in candidates
            if str(item.get("family_root_storyline_id") or item.get("storyline_id") or "").strip()
        }
        if isinstance(llm_result, dict):
            for item in list(llm_result.get("branches") or []):
                if not isinstance(item, dict):
                    continue
                storyline_id = str(item.get("storyline_id") or "").strip()
                family_root_id = str(item.get("family_root_storyline_id") or "").strip()
                matched_candidate = candidates_by_storyline_id.get(storyline_id) or candidates_by_family_root_id.get(family_root_id)
                if matched_candidate:
                    branch = dict(item)
                    branch["storyline_id"] = str(matched_candidate.get("storyline_id") or storyline_id).strip()
                    branch["family_root_storyline_id"] = str(
                        matched_candidate.get("family_root_storyline_id")
                        or branch.get("family_root_storyline_id")
                        or branch.get("storyline_id")
                    ).strip()
                    branches.append(branch)
                if len(branches) >= 5:
                    break
        if not branches:
            branches = _fallback_related_storyline_branches(candidates, limit=5)

        text = _render_related_storyline_branches_text(locale, branches) or t(locale, "storyline_similar_empty")
        markup = _storyline_related_branches_markup(locale, post_id, branches)
        try:
            if progress_message:
                await progress_message.edit_text(
                    text,
                    disable_web_page_preview=True,
                    parse_mode="HTML",
                    reply_markup=markup,
                )
            elif callback.message:
                await callback.message.answer(
                    text,
                    disable_web_page_preview=True,
                    parse_mode="HTML",
                    skip_service_cleanup=True,
                    reply_markup=markup,
                )
            else:
                await _notify_callback_or_message(callback, text=_storyline_timeline_plain_text(text) or text, show_alert=True)
        except Exception:
            log.exception("storyline_similar_send_failed", post_id=post_id, user_id=user.id)
            plain = _storyline_timeline_plain_text(text) or t(locale, "storyline_similar_empty")
            if callback.message:
                await callback.message.answer(plain, disable_web_page_preview=True, skip_service_cleanup=True)
        log.info(
            "storyline_similar_sent",
            post_id=post_id,
            user_id=user.id,
            candidate_count=len(candidates),
            branch_count=len(branches),
            candidate_ids=[str(item.get("storyline_id") or "") for item in candidates[:8]],
            branch_titles=[str(item.get("title") or "")[:120] for item in branches[:5]],
        )
        break


@dp.callback_query(F.data.startswith("storysimtimeline:"))
async def handle_storyline_similar_timeline(callback: types.CallbackQuery):
    if not _storytracking_available(
        telegram_id=callback.from_user.id,
        username=callback.from_user.username,
    ):
        locale = await _resolve_callback_locale(callback)
        await _safe_answer_callback(callback, text=t(locale, "storytracking_disabled"), show_alert=True)
        return

    try:
        _, post_id_str, storyline_id = str(callback.data or "").split(":", 2)
        post_id = int(post_id_str)
        storyline_id = str(storyline_id or "").strip()
        if not storyline_id:
            raise ValueError("missing storyline_id")
    except (ValueError, AttributeError):
        await _notify_callback_or_message(
            callback,
            text=t(await _resolve_callback_locale(callback), "storyline_follow_not_found"),
        )
        return

    await _safe_answer_callback(callback)

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == callback.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await _notify_callback_or_message(
                callback,
                text=t(_button_locale_from_callback(callback), "start_required"),
            )
            break

        locale = get_user_locale(user)
        post_result = await session.execute(select(Post).where(Post.id == post_id))
        post = post_result.scalar_one_or_none()
        if not post:
            await _notify_callback_or_message(callback, text=t(locale, "post_not_found"))
            break

        await _record_post_interaction(
            session,
            user=user,
            post=post,
            action="storysimilar_timeline_open",
            source="storyline_similar",
        )

        target_chat = getattr(getattr(callback, "message", None), "chat", None)
        target_chat_id = int(getattr(target_chat, "id", 0) or 0) or None
        await _send_storyline_timeline_for_post(
            session=session,
            message=callback.message,
            locale=locale,
            post_id=post_id,
            user_id=user.id,
            telegram_id=user.telegram_id,
            target_chat_id=target_chat_id,
            anchor_storyline_id=storyline_id,
            log_prefix="storyline_similar_timeline",
        )
        break


@dp.callback_query(F.data.startswith("storyfollow:"))
async def handle_storyline_follow(callback: types.CallbackQuery):
    if not _storytracking_available(
        telegram_id=callback.from_user.id,
        username=callback.from_user.username,
    ):
        locale = await _resolve_callback_locale(callback)
        await callback.answer(t(locale, "storytracking_disabled"), show_alert=True)
        return

    try:
        _, post_id_str = callback.data.split(":", 1)
        post_id = int(post_id_str)
    except (ValueError, AttributeError):
        await callback.answer(t(await _resolve_callback_locale(callback), "storyline_follow_not_found"), show_alert=True)
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == callback.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await callback.answer(t(_button_locale_from_callback(callback), "start_required"), show_alert=True)
            break

        locale = get_user_locale(user)
        post_result = await session.execute(select(Post).where(Post.id == post_id))
        post = post_result.scalar_one_or_none()
        if not post:
            await callback.answer(t(locale, "post_not_found"), show_alert=True)
            break

        await _record_post_interaction(
            session,
            user=user,
            post=post,
            action="storyfollow_enable",
            source="storyline_post",
        )

        post_text = _storyline_post_text(post)
        anchor_profile = await _extract_storyline_anchor_profile_safe(post_text, post_id=post_id)
        context = await _resolve_storyline_follow_context(
            post_id,
            post_text,
            anchor_profile=anchor_profile,
            session=session,
            anchor_post_ts=getattr(post, "timestamp", None),
        )
        if not context:
            await callback.answer(t(locale, "storyline_follow_not_found"), show_alert=True)
            break

        family_root_storyline_id = str(context.get("family_root_storyline_id") or context.get("storyline_id") or "").strip()
        existing = await session.execute(
            select(UserStorylineFollow).where(
                UserStorylineFollow.user_id == user.id,
                UserStorylineFollow.family_root_storyline_id == family_root_storyline_id,
            )
        )
        follow = existing.scalar_one_or_none()
        debug_enabled = bool(getattr(user, "storyline_debug_enabled", False)) and _is_admin_user(user, callback.from_user.username)
        human_title = str(context.get("storyline_title") or "").strip()
        try:
            generated_title = await generate_storyline_title(
                seed_post_text=post_text,
                current_title=human_title,
            )
            if generated_title:
                human_title = generated_title
        except Exception as exc:
            log.warning(
                "storyline_follow.title_generation_failed",
                post_id=post_id,
                user_id=getattr(user, "id", None),
                error=str(exc),
            )

        if follow:
            if not bool(getattr(follow, "is_active", False)):
                active_follows_count = await session.scalar(
                    select(func.count())
                    .select_from(UserStorylineFollow)
                    .where(UserStorylineFollow.user_id == user.id)
                    .where(UserStorylineFollow.is_active.is_(True))
                )
                follow_limit = await _resolve_storyline_follow_limit(session, user)
                if int(active_follows_count or 0) >= int(follow_limit):
                    limit_text = t(locale, "storyline_follow_limit_reached", limit=follow_limit)
                    await callback.answer(limit_text, show_alert=True)
                    if callback.message:
                        await callback.message.answer(limit_text)
                    break
            try:
                follow.is_active = True
                follow.storyline_id = str(context.get("storyline_id") or "")
                follow.story_family_id = str(context.get("story_family_id") or "") or family_root_storyline_id
                follow.storyline_title = human_title[:500] or None
                follow.source_post_id = post.id
                await session.commit()
            except Exception:
                await session.rollback()
                log.exception(
                    "storyline_follow.save_failed",
                    post_id=post_id,
                    user_id=getattr(user, "id", None),
                    family_root_storyline_id=family_root_storyline_id,
                )
                await callback.answer(t(locale, "storyline_follow_failed"), show_alert=True)
                if callback.message:
                    await callback.message.answer(t(locale, "storyline_follow_failed"))
                break
            text = _render_storyline_debug_text(locale, context) if debug_enabled else _render_storyline_follow_success_text(locale, human_title)
        else:
            active_follows_count = await session.scalar(
                select(func.count())
                .select_from(UserStorylineFollow)
                .where(UserStorylineFollow.user_id == user.id)
                .where(UserStorylineFollow.is_active.is_(True))
            )
            follow_limit = await _resolve_storyline_follow_limit(session, user)
            if int(active_follows_count or 0) >= int(follow_limit):
                limit_text = t(locale, "storyline_follow_limit_reached", limit=follow_limit)
                await callback.answer(limit_text, show_alert=True)
                if callback.message:
                    await callback.message.answer(limit_text)
                break
            try:
                follow = UserStorylineFollow(
                    user_id=user.id,
                    source_post_id=post.id,
                    storyline_id=str(context.get("storyline_id") or ""),
                    story_family_id=str(context.get("story_family_id") or "") or family_root_storyline_id,
                    family_root_storyline_id=family_root_storyline_id,
                    storyline_title=(human_title[:500] or None),
                    branch_mode="root_only",
                    selected_branch_ids=[],
                    is_active=True,
                )
                session.add(follow)
                await session.commit()
            except Exception:
                await session.rollback()
                log.exception(
                    "storyline_follow.save_failed",
                    post_id=post_id,
                    user_id=getattr(user, "id", None),
                    family_root_storyline_id=family_root_storyline_id,
                )
                await callback.answer(t(locale, "storyline_follow_failed"), show_alert=True)
                if callback.message:
                    await callback.message.answer(t(locale, "storyline_follow_failed"))
                break
            if debug_enabled:
                text = _render_storyline_debug_text(locale, context)
            else:
                text = _render_storyline_follow_success_text(locale, human_title)

        await _safe_answer_callback(callback, text=t(locale, "saved"))
        if callback.message:
            await callback.message.answer(text, parse_mode="HTML")
        break


@dp.callback_query(F.data.startswith("storyunfollowroot:"))
async def handle_storyline_unfollow_root(callback: types.CallbackQuery):
    if not _storytracking_available(
        telegram_id=callback.from_user.id,
        username=callback.from_user.username,
    ):
        locale = await _resolve_callback_locale(callback)
        await callback.answer(t(locale, "storytracking_disabled"), show_alert=True)
        return

    try:
        _, post_id_str, family_root_storyline_id = callback.data.split(":", 2)
        post_id = int(post_id_str)
    except (ValueError, AttributeError):
        await callback.answer(t(await _resolve_callback_locale(callback), "storyline_unfollow_not_found"), show_alert=True)
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == callback.from_user.id))
        user = result.scalar_one_or_none()
        locale = get_user_locale(user) if user else _button_locale_from_callback(callback)
        if not user:
            await callback.answer(t(locale, "start_required"), show_alert=True)
            break

        follows_result = await session.execute(
            select(UserStorylineFollow).where(
                UserStorylineFollow.user_id == user.id,
                UserStorylineFollow.family_root_storyline_id == family_root_storyline_id,
                UserStorylineFollow.is_active.is_(True),
            )
        )
        follows = follows_result.scalars().all()
        if not follows:
            await callback.answer(t(locale, "storyline_unfollow_not_found"), show_alert=True)
            break

        post_result = await session.execute(select(Post).where(Post.id == post_id))
        post = post_result.scalar_one_or_none()
        if post:
            await _record_post_interaction(
                session,
                user=user,
                post=post,
                action="storyunfollow_disable",
                source="storyline_card",
            )

        for follow in follows:
            follow.is_active = False
        await session.commit()

        await callback.answer(t(locale, "storyline_unfollow_success"))
        if callback.message:
            with contextlib.suppress(Exception):
                await callback.message.edit_reply_markup(reply_markup=None)
            await callback.message.answer(t(locale, "storyline_unfollow_success"))
        break


@dp.message(Command("story_search"))
async def storyline_search(message: types.Message):
    if not _storytracking_available(
        telegram_id=message.from_user.id,
        username=message.from_user.username,
    ):
        locale = await _resolve_message_locale(message)
        await message.answer(t(locale, "storytracking_disabled"))
        return

    raw = (message.text or "").split(maxsplit=1)
    query = raw[1].strip() if len(raw) > 1 else ""
    locale = await _resolve_message_locale(message)
    if not query:
        await message.answer(t(locale, "storyline_search_empty"))
        return

    items = await search_storyline_candidates(query, limit=8)
    if not items:
        await message.answer(t(locale, "storyline_search_no_results"))
        return

    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        title = str(item.get("title") or "").strip() or str(item.get("storyline_id") or "").strip()
        posts_count = int(item.get("posts_count") or 0)
        lines.append(f"{idx}. {title} ({posts_count})")
    await message.answer(t(locale, "storyline_search_title", items="\n".join(lines)))
    
@dp.message(Command("digest_offset"))
async def digest_offset(message: types.Message):
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer(t(await _resolve_message_locale(message), "digest_offset_usage"))
        return

    raw_value = (parts[1] or "").strip()
    offset_min = parse_utc_offset_to_minutes(raw_value)
    tz_name = None
    if offset_min is None:
        tz_name = resolve_timezone_from_city(raw_value)
        if tz_name:
            offset_min = current_offset_minutes_for_tz(tz_name)
    if offset_min is None:
        await message.answer(t(await _resolve_message_locale(message), "digest_offset_unknown"))
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required"))
            return
        locale = get_user_locale(user)

        settings = await session.get(UserDigestSettings, user.id)
        if not settings:
            settings = UserDigestSettings(user_id=user.id, enabled=True)
            session.add(settings)

        settings.utc_offset_minutes = offset_min
        if tz_name:
            settings.timezone = tz_name

        # если слот уже есть — пересчитаем next_run_at
        slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
        slot = slot_res.scalar_one_or_none()
        if slot:
            slot.next_run_at = compute_next_run_at_utc_from_offset(settings.utc_offset_minutes, slot.hour, slot.minute)

        await session.commit()
        if tz_name:
            await message.answer(
                t(locale, "digest_offset_saved_city", raw_value=raw_value, tz_name=tz_name, offset=offset_min / 60)
            )
        else:
            await message.answer(t(locale, "digest_offset_saved", offset=offset_min / 60))
        return
    
@dp.message(Command("digest_time"))
async def digest_time(message: types.Message):
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer(t(await _resolve_message_locale(message), "digest_time_usage"))
        return

    parsed = parse_hhmm(parts[1])
    if not parsed:
        await message.answer(t(await _resolve_message_locale(message), "digest_time_invalid_cmd"))
        return
    hour, minute = parsed

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == message.from_user.id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(_button_locale_from_message(message), "start_required"))
            return
        locale = get_user_locale(user)

        settings = await session.get(UserDigestSettings, user.id)
        if not settings or settings.utc_offset_minutes is None:
            await message.answer(t(locale, "digest_set_offset_first"))
            return

        next_run_at = compute_next_run_at_utc_from_offset(settings.utc_offset_minutes, hour, minute)

        slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
        slot = slot_res.scalar_one_or_none()
        if not slot:
            slot = UserDigestSlot(user_id=user.id, hour=hour, minute=minute, days_mask=127, is_active=True)
            session.add(slot)
        else:
            slot.hour = hour
            slot.minute = minute
            slot.is_active = True

        slot.next_run_at = next_run_at
        await session.commit()

        offset = timezone(timedelta(minutes=settings.utc_offset_minutes))
        local_dt = next_run_at.astimezone(offset)

        await message.answer(
            t(
                locale,
                "digest_time_done",
                offset=settings.utc_offset_minutes / 60,
                hour=hour,
                minute=minute,
                local_dt=f"{local_dt:%Y-%m-%d %H:%M}",
            )
        )
        return


async def _execute_show_settings_overview(session, *, user: User, locale: str) -> ExecutorResult:
    settings, slot = await _build_user_digest_context(session, user)
    snapshot = await get_current_payment_snapshot(session, user)
    text = _render_settings_overview_text(user=user, locale=locale, settings=settings, slot=slot, snapshot=snapshot)
    return ExecutorResult(text=text, reply_markup=_settings_keyboard_for_user(locale, user, getattr(user, "username", None)))


async def _get_active_subscription_links(session, user: User) -> list[str]:
    result = await session.execute(
        select(Community.link)
        .join(UserCommunity, UserCommunity.community_id == Community.id)
        .where(UserCommunity.user_id == user.id, Community.is_active.is_(True))
    )
    return [row[0] for row in result.all()]


async def _execute_show_subscriptions(session, *, user: User, locale: str) -> ExecutorResult:
    links = await _get_active_subscription_links(session, user)
    snapshot = await get_current_payment_snapshot(session, user)
    if not links:
        text = t(locale, "subscriptions_none")
    else:
        text = t(locale, "subscriptions_list", links="\n".join(links))
    text += _billing_cta_suffix(snapshot, locale)
    return ExecutorResult(text=text, reply_markup=_settings_keyboard_for_user(locale, user, getattr(user, "username", None)))


async def _execute_show_billing(session, *, user: User, locale: str) -> ExecutorResult:
    snapshot = await get_current_payment_snapshot(session, user)
    text = _render_billing_status(snapshot, locale) + "\n\n" + _render_billing_instruction(snapshot, locale)
    text += _billing_cta_suffix(snapshot, locale)
    return ExecutorResult(text=text, reply_markup=get_billing_keyboard(locale), parse_mode="HTML")


async def _execute_show_help_topic(session, *, user: User, locale: str, nlu_result: NLUResult) -> ExecutorResult:
    settings, slot = await _build_user_digest_context(session, user)
    snapshot = await get_current_payment_snapshot(session, user)
    links = await _get_active_subscription_links(session, user)
    return await _render_help_topic_text(
        session=session,
        locale=locale,
        topic=nlu_result.faq_topic or "general",
        user=user,
        settings=settings,
        slot=slot,
        snapshot=snapshot,
        subscriptions=links,
    )


async def _execute_mutation_intent(message: types.Message, session, *, user: User, locale: str, payload: PendingActionPayload) -> ExecutorResult:
    intent = payload.intent
    slots = payload.slots

    if intent == "set_global_instruction_filter":
        if not await user_has_instruction_filter_access(session, user.id):
            return ExecutorResult(
                text=t(locale, "instruction_premium_required"),
                reply_markup=get_instruction_filter_keyboard(locale),
            )
        try:
            await upsert_global_instruction_rule(
                session,
                user_id=user.id,
                prompt_text=str(slots.get("prompt_text") or ""),
            )
            await session.commit()
        except InstructionPromptError as exc:
            await session.rollback()
            return ExecutorResult(
                text=t(locale, str(exc)),
                reply_markup=get_instruction_filter_keyboard(locale),
            )
        return ExecutorResult(
            text=t(locale, "instruction_global_saved") + "\n\n" + await _render_instruction_menu_text(session, user=user, locale=locale),
            reply_markup=get_instruction_filter_keyboard(locale),
            parse_mode="HTML",
        )

    if intent == "set_feed_filter":
        previous = getattr(user, "feed_filter", "all") or "all"
        if str(slots.get("filter_mode") or "") == "digest_only":
            _disable_live_forwarding(user)
        else:
            _set_live_feed_filter(user, str(slots.get("filter_mode") or "all"))
        await session.commit()
        audit("filter.changed", user_id=user.id, telegram_id=user.telegram_id, previous_filter=previous, new_filter=user.feed_filter)
        return ExecutorResult(
            text=_render_forwarding_menu_text(user, locale),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
            parse_mode="HTML",
        )

    if intent == "toggle_forwarding":
        previous = getattr(user, "feed_filter", "all") or "all"
        if bool(slots.get("value_bool")):
            _enable_live_forwarding(user)
        else:
            _disable_live_forwarding(user)
        await session.commit()
        audit("filter.changed", user_id=user.id, telegram_id=user.telegram_id, previous_filter=previous, new_filter=user.feed_filter)
        return ExecutorResult(
            text=_render_forwarding_menu_text(user, locale),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
            parse_mode="HTML",
        )

    if intent == "toggle_summary":
        user.summary_enabled = bool(slots.get("value_bool"))
        await session.commit()
        return ExecutorResult(
            text=_render_forwarding_menu_text(user, locale, summary_effect_notice=True),
            reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
            parse_mode="HTML",
        )

    if intent == "digest_enable":
        settings = await session.get(UserDigestSettings, user.id)
        if not settings:
            settings = UserDigestSettings(user_id=user.id, enabled=True)
            session.add(settings)
        settings.enabled = True
        await session.commit()
        slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
        slot = slot_res.scalar_one_or_none()
        text = t(locale, "digest_description") + "\n" + "\n".join(_render_digest_state_lines(locale=locale, settings=settings, slot=slot))
        return ExecutorResult(
            text=text,
            reply_markup=get_digest_keyboard(locale, enabled=True, time_label=_digest_time_label(slot)),
        )

    if intent == "digest_disable":
        settings = await session.get(UserDigestSettings, user.id)
        if not settings:
            settings = UserDigestSettings(user_id=user.id, enabled=False)
            session.add(settings)
        settings.enabled = False
        await session.commit()
        slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
        slot = slot_res.scalar_one_or_none()
        text = t(locale, "digest_description") + "\n" + "\n".join(_render_digest_state_lines(locale=locale, settings=settings, slot=slot))
        return ExecutorResult(
            text=text,
            reply_markup=get_digest_keyboard(locale, enabled=False, time_label=_digest_time_label(slot)),
        )

    if intent == "digest_send_now":
        slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
        slot = slot_res.scalar_one_or_none()
        if not slot:
            slot = UserDigestSlot(user_id=user.id, hour=0, minute=0, days_mask=127, is_active=False)
            session.add(slot)
            await session.flush()
        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(hours=24)
        run = DigestRun(slot_id=slot.id, user_id=user.id, period_start=period_start, period_end=period_end, status="queued")
        session.add(run)
        await session.commit()
        await asyncio.to_thread(
            celery_app.send_task,
            "app.tasks.build_and_send_digest",
            args=[run.id],
            queue="digest_queue",
        )
        return ExecutorResult(text=t(locale, "digest_collecting_now"), reply_markup=get_digest_time_keyboard(locale))

    if intent == "digest_set_offset":
        settings = await session.get(UserDigestSettings, user.id)
        if not settings:
            settings = UserDigestSettings(user_id=user.id, enabled=True)
            session.add(settings)
        offset_text = str(slots.get("utc_offset") or "")
        offset_min = parse_utc_offset_to_minutes(offset_text)
        if offset_min is None:
            return ExecutorResult(
                text=t(locale, "digest_offset_invalid"),
                reply_markup=get_digest_keyboard(
                    locale,
                    enabled=bool(settings.enabled) if settings else False,
                    time_label=None,
                ),
            )
        settings.utc_offset_minutes = offset_min
        await session.commit()
        slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
        slot = slot_res.scalar_one_or_none()
        text = t(locale, "digest_description") + "\n" + "\n".join(_render_digest_state_lines(locale=locale, settings=settings, slot=slot))
        return ExecutorResult(
            text=text,
            reply_markup=get_digest_keyboard(
                locale,
                enabled=bool(settings.enabled),
                time_label=_digest_time_label(slot),
            ),
        )

    if intent == "digest_set_time":
        parsed = parse_hhmm(str(slots.get("time_hhmm") or ""))
        if not parsed:
            return ExecutorResult(
                text=t(locale, "digest_time_invalid"),
                reply_markup=get_digest_keyboard(locale, enabled=False, time_label=None),
            )
        hour, minute = parsed
        settings = await session.get(UserDigestSettings, user.id)
        if not settings or settings.utc_offset_minutes is None:
            return ExecutorResult(
                text=t(locale, "digest_need_offset"),
                reply_markup=get_digest_keyboard(
                    locale,
                    enabled=bool(settings.enabled) if settings else False,
                    time_label=None,
                ),
            )
        next_run_at = compute_next_run_at_utc_from_offset(settings.utc_offset_minutes, hour, minute)
        slot_res = await session.execute(select(UserDigestSlot).where(UserDigestSlot.user_id == user.id))
        slot = slot_res.scalar_one_or_none()
        if not slot:
            slot = UserDigestSlot(user_id=user.id, hour=hour, minute=minute, days_mask=127, is_active=True)
            session.add(slot)
        else:
            slot.hour = hour
            slot.minute = minute
            slot.is_active = True
        slot.next_run_at = next_run_at
        await session.commit()
        offset = timezone(timedelta(minutes=settings.utc_offset_minutes))
        local_dt = next_run_at.astimezone(offset)
        return ExecutorResult(
            text=t(
                locale,
                "digest_ready",
                offset=settings.utc_offset_minutes / 60,
                hour=hour,
                minute=minute,
                local_dt=f"{local_dt:%Y-%m-%d %H:%M}",
            ),
            reply_markup=get_digest_keyboard(
                locale,
                enabled=bool(settings.enabled),
                time_label=_digest_time_label(slot),
            ),
        )

    if intent == "set_language":
        new_locale = str(slots.get("language_code") or "ru")
        user.language_code = new_locale
        await session.commit()
        return ExecutorResult(
            text=t(new_locale, "language_changed"),
            reply_markup=_settings_keyboard_for_user(new_locale, user, getattr(user, "username", None)),
        )

    if intent in {"add_subscriptions", "remove_subscriptions"}:
        try:
            result = await _apply_subscription_links(
                session,
                user=user,
                links=list(slots.get("links") or []),
                locale=locale,
                action_mode="add" if intent == "add_subscriptions" else "remove",
            )
        except RuntimeError as exc:
            await session.rollback()
            return ExecutorResult(
                text=str(exc),
                reply_markup=_settings_keyboard_for_user(locale, user, getattr(user, "username", None)),
            )

        for group_link in result.unsubscribed:
            try:
                await publish_telethon_event_async({"event": "unsubscribe", "group": group_link})
            except Exception as exc:
                log.error("Failed to publish telethon unsubscribe event", error=str(exc), group_link=group_link, exc_info=True)

        if result.request_ids:
            await _finalize_subscription_request_progress(message, request_ids=result.request_ids, locale=locale)
            await _enqueue_subscription_request_jobs(result.request_ids)

        should_send_summary = bool(
            result.subscribed
            or result.unsubscribed
            or result.already_subscribed
            or result.missing
            or result.already_queued
            or result.invalid_lines
            or (result.queued and not result.request_ids)
        )
        text = (
            _render_subscription_change_result_text(result, locale, include_queued=not bool(result.request_ids))
            if should_send_summary
            else ""
        )
        return ExecutorResult(
            text=text,
            reply_markup=_settings_keyboard_for_user(locale, user, getattr(user, "username", None)),
        )

    return ExecutorResult(text=t(locale, "nlu_action_not_supported"))


async def _execute_show_help_or_read_only(message: types.Message, session, *, user: User, locale: str, nlu_result: NLUResult) -> ExecutorResult:
    if nlu_result.intent == "show_settings_overview":
        return await _execute_show_settings_overview(session, user=user, locale=locale)
    if nlu_result.intent == "show_subscriptions":
        return await _execute_show_subscriptions(session, user=user, locale=locale)
    if nlu_result.intent == "show_billing":
        return await _execute_show_billing(session, user=user, locale=locale)
    return await _execute_show_help_topic(session, user=user, locale=locale, nlu_result=nlu_result)


@dp.callback_query(F.data.startswith("nlu:"))
async def handle_nlu_pending_action(callback: types.CallbackQuery):
    user_id = getattr(getattr(callback, "from_user", None), "id", None)
    locale = await _resolve_callback_locale(callback)
    if user_id is None:
        await callback.answer()
        return

    callback_parts = (callback.data or "").split(":")
    action = callback_parts[1] if len(callback_parts) > 1 else ""
    callback_request_id = callback_parts[2] if len(callback_parts) > 2 else None
    store = get_pending_action_store()
    payload = await store.load(user_id)
    if not payload:
        if callback_request_id:
            async for session in get_session():
                try:
                    await mark_assistant_expired(session, request_id=callback_request_id)
                    await session.commit()
                except Exception:
                    await session.rollback()
                break
        await callback.answer(t(locale, "nlu_action_expired"), show_alert=True)
        return
    request_id = payload.request_id or callback_request_id

    if action == "cancel":
        await store.delete(user_id)
        async for session in get_session():
            try:
                await mark_assistant_cancelled(session, request_id=request_id)
                await session.commit()
            except Exception:
                await session.rollback()
            break
        audit("nlu.action_denied", telegram_id=user_id, intent=payload.intent, reason="user_cancelled", request_id=request_id)
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        await callback.message.answer(t(locale, "nlu_action_cancelled"))
        await callback.answer()
        return

    if action != "confirm":
        await callback.answer()
        return

    await store.delete(user_id)

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            await callback.message.answer(t(locale, "start_required_register"))
            await callback.answer()
            return
        try:
            validated_payload = PendingActionPayload.from_dict(payload.to_dict())
            try:
                await mark_assistant_confirmed(session, request_id=validated_payload.request_id or request_id)
                await session.commit()
            except Exception:
                await session.rollback()
            executor_result = await _execute_mutation_intent(
                callback.message,
                session,
                user=user,
                locale=get_user_locale(user),
                payload=validated_payload,
            )
            try:
                await mark_assistant_applied(session, request_id=validated_payload.request_id or request_id)
                await session.commit()
            except Exception:
                await session.rollback()
            audit(
                "nlu.action_applied",
                telegram_id=user_id,
                user_id=user.id,
                intent=validated_payload.intent,
                slots=validated_payload.slots,
                request_id=validated_payload.request_id or request_id,
            )
        except Exception as exc:
            log.exception("nlu.action_failed", telegram_id=user_id, intent=payload.intent)
            await session.rollback()
            try:
                await mark_assistant_failed(
                    session,
                    request_id=request_id,
                    error_code=type(exc).__name__,
                    error_message=repr(exc),
                )
                await session.commit()
            except Exception:
                await session.rollback()
            audit(
                "nlu.action_denied",
                telegram_id=user_id,
                user_id=getattr(user, "id", None),
                intent=payload.intent,
                reason=repr(exc),
                request_id=request_id,
            )
            await callback.message.answer(t(locale, "nlu_action_failed"))
            await callback.answer()
            return
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        await callback.message.answer(
            executor_result.text,
            reply_markup=executor_result.reply_markup,
            parse_mode=executor_result.parse_mode,
        )
        await callback.answer(t(locale, "saved"))
        return


@dp.message(lambda m: _is_plain_text_message(m))
async def fallback_menu_router(message: types.Message):
    locale = await _resolve_message_locale(message)
    text = (message.text or "").strip()
    user_id = getattr(getattr(message, "from_user", None), "id", None)

    log.warning(
        "bot.unmatched_text",
        telegram_id=user_id,
        text=text,
        detected_locale=locale,
    )

    if await _try_handle_csi_score_answer(message):
        return

    if user_id is not None:
        DIGEST_SETUP_WAIT_OFFSET.discard(user_id)
        DIGEST_SETUP_WAIT_TIME.discard(user_id)
        DIGEST_TIME_MENU_USERS.discard(user_id)
        _clear_settings_submenu_states(user_id)
        FORWARDING_MENU_USERS.discard(user_id)
        STORYLINE_MENU_USERS.discard(user_id)
        STORYLINE_REMOVE_WAIT_USERS.discard(user_id)
        STORYLINE_REMOVE_CHOICES.pop(user_id, None)

    if _is_known_button_text(text):
        if user_id is not None:
            PROMO_CODE_WAIT.discard(user_id)
        await message.answer(
            t(locale, "main_menu_title"),
            reply_markup=_main_keyboard_for_user(locale, telegram_username=getattr(message.from_user, "username", None)),
        )
        return

    if not _message_is_private_chat(message):
        await message.answer(
            t(locale, "help_message"),
            parse_mode="Markdown",
            reply_markup=_main_keyboard_for_user(locale, telegram_username=getattr(message.from_user, "username", None)),
        )
        return

    async for session in get_session():
        result = await session.execute(select(User).where(User.telegram_id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            await message.answer(t(locale, "start_required_register"))
            return
        locale = get_user_locale(user)
        request_id = str(uuid4())
        try:
            await create_assistant_request(
                session,
                request_id=request_id,
                telegram_id=user_id,
                user_id=user.id,
                locale=locale,
                raw_text=text,
            )
            await session.commit()
        except Exception as exc:
            await session.rollback()
            log.warning("assistant_monitoring.create_failed", telegram_id=user_id, error=repr(exc))
            request_id = None
        try:
            context = await _build_nlu_context(session, user)
            builtin_nlu_result = _resolve_builtin_router_intent(text)
            if builtin_nlu_result is not None:
                nlu_result = builtin_nlu_result
            else:
                raw_result = await resolve_bot_router_intent(text=text, context=context)
                nlu_result = validate_nlu_result(raw_result or {})
            try:
                await mark_assistant_understood(session, request_id=request_id, nlu_result=nlu_result)
                await session.commit()
            except Exception as exc:
                await session.rollback()
                log.warning("assistant_monitoring.understood_failed", telegram_id=user_id, request_id=request_id, error=repr(exc))
            audit(
                "nlu.intent_detected",
                telegram_id=user_id,
                user_id=user.id,
                intent=nlu_result.intent,
                slots=nlu_result.slots,
                faq_topic=nlu_result.faq_topic,
                request_id=request_id,
            )
        except NLUValidationError as exc:
            await safe_mark_assistant_failed(
                session,
                request_id=request_id,
                error_code="NLUValidationError",
                error_message=exc,
            )
            audit("nlu.validation_failed", telegram_id=user_id, user_id=user.id, error=str(exc), raw_text=text, request_id=request_id)
            break
        except (DeepseekAuthError, DeepseekRetryableError) as exc:
            await safe_mark_assistant_failed(
                session,
                request_id=request_id,
                error_code=type(exc).__name__,
                error_message=exc,
            )
            log.warning("nlu.router_unavailable", telegram_id=user_id, error=repr(exc))
            break
        except Exception as exc:
            await safe_mark_assistant_failed(
                session,
                request_id=request_id,
                error_code=type(exc).__name__,
                error_message=exc,
            )
            log.exception("nlu.router_failed", telegram_id=user_id, error_repr=repr(exc))
            break

        if nlu_result.needs_clarification:
            try:
                await mark_assistant_clarification(session, request_id=request_id)
                await session.commit()
            except Exception:
                await session.rollback()
            audit(
                "nlu.clarification_requested",
                telegram_id=user_id,
                user_id=user.id,
                intent=nlu_result.intent,
                slots=nlu_result.slots,
                request_id=request_id,
            )
            clarify_text, clarify_markup = _build_nlu_clarification_response(nlu_result, locale)
            await message.answer(clarify_text, reply_markup=clarify_markup)
            return

        if nlu_result.intent in {"show_settings_overview", "show_subscriptions", "show_billing", "show_help_topic"}:
            if nlu_result.intent == "show_help_topic":
                audit("nlu.faq_answered", telegram_id=user_id, user_id=user.id, faq_topic=nlu_result.faq_topic or "general", request_id=request_id)
            executor_result = await _execute_show_help_or_read_only(message, session, user=user, locale=locale, nlu_result=nlu_result)
            await message.answer(
                executor_result.text,
                reply_markup=executor_result.reply_markup,
                parse_mode=executor_result.parse_mode,
            )
            try:
                await mark_assistant_answered(session, request_id=request_id)
                await session.commit()
            except Exception:
                await session.rollback()
            return

        if nlu_result.intent == "toggle_summary":
            requested_state = bool(nlu_result.slots.get("value_bool"))
            current_state = bool(getattr(user, "summary_enabled", False))
            if requested_state == current_state:
                await message.answer(
                    t(locale, "summary_enabled" if current_state else "summary_disabled"),
                    reply_markup=await _forwarding_keyboard_for_user(session, user=user, locale=locale),
                )
                try:
                    await mark_assistant_answered(session, request_id=request_id)
                    await session.commit()
                except Exception:
                    await session.rollback()
                return

        summary_text = _describe_pending_action(nlu_result.intent, nlu_result.slots, locale)
        payload = build_pending_action_payload(
            user_id=user.id,
            telegram_id=user.telegram_id,
            intent=nlu_result.intent,
            slots=nlu_result.slots,
            locale=locale,
            summary_text=summary_text,
            ttl_seconds=pending_action_ttl_seconds(),
            request_id=request_id,
        )
        stored = await get_pending_action_store().save(payload, ttl_seconds=pending_action_ttl_seconds())
        if not stored:
            try:
                await mark_assistant_failed(
                    session,
                    request_id=request_id,
                    error_code="pending_action_store_failed",
                    error_message="Could not store pending action",
                )
                await session.commit()
            except Exception:
                await session.rollback()
            await message.answer(t(locale, "nlu_action_store_failed"))
            return
        try:
            await mark_assistant_proposed(session, request_id=request_id)
            await session.commit()
        except Exception:
            await session.rollback()
        audit("nlu.action_proposed", telegram_id=user_id, user_id=user.id, intent=nlu_result.intent, slots=nlu_result.slots, request_id=request_id)
        await message.answer(summary_text, reply_markup=_build_nlu_confirm_keyboard(locale, request_id))
        return

    await message.answer(
        t(locale, "help_message"),
        parse_mode="Markdown",
        reply_markup=_main_keyboard_for_user(locale, telegram_username=getattr(message.from_user, "username", None)),
    )


async def start_bot():
    retry_stop_event = asyncio.Event()
    retry_task = asyncio.create_task(_support_retry_loop(retry_stop_event))
    await configure_bot_menu(bot)
    while True:
        try:
            await dp.start_polling(bot)

            # Если polling завершился штатно (например, stop/shutdown) — выходим,
            # иначе будет бесконечный рестарт и systemd поймает timeout.
            log.info("Bot polling finished, exiting")
            retry_stop_event.set()
            retry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await retry_task
            return

        except asyncio.CancelledError:
            log.info("Bot polling cancelled, exiting")
            retry_stop_event.set()
            retry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await retry_task
            raise

        except Exception as e:
            log.error("Bot polling error", error=str(e))
            await asyncio.sleep(10)
