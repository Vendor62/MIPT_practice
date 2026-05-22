import os
import html
import re
import mimetypes
import asyncio
import contextlib
from urllib.parse import urlencode, urlparse

import structlog
from aiogram import Bot, types

from app.ai.deepseek import summarize
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from app.i18n import normalize_locale, t
from app.storytracking_rollout import storytracking_allowed_for_user


log = structlog.get_logger()

bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
bot = Bot(token=bot_token) if bot_token else None

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/")
SOURCE_LINK_TRACKING_ENABLED = os.getenv("SOURCE_LINK_TRACKING_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
SUMMARY_MIN_CHARS = int(os.getenv("SUMMARY_MIN_CHARS", "200"))
MEDIA_MAX_BYTES = int(os.getenv("MEDIA_MAX_BYTES", str(10 * 1024 * 1024)))
TELEGRAM_SEND_TIMEOUT = int(os.getenv("TELEGRAM_SEND_TIMEOUT", "25"))
ENABLE_STORYTRACKING = os.getenv("ENABLE_STORYTRACKING", "true").lower() == "true"
TELEGRAM_MAX_LEN = 4096
TELEGRAM_SAFE_LEN = 3900
TELEGRAM_CAPTION_SAFE_LEN = 900


class NotificationsNotConfiguredError(RuntimeError):
    pass


def _require_bot() -> Bot:
    if not bot:
        raise NotificationsNotConfiguredError("TELEGRAM_BOT_TOKEN not set, cannot send notifications")
    return bot


def _build_temp_bot(token: str | None) -> Bot | None:
    token_norm = str(token or "").strip()
    if not token_norm:
        return None
    if token_norm == bot_token and bot:
        return bot
    return Bot(token=token_norm)


def _try_get_service_bot(*, action: str, **details) -> Bot | None:
    if bot:
        return bot
    log.warning("notifications.bot_not_configured", action=action, **details)
    return None


def _is_local_base_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return True
    return host in {"", "localhost", "127.0.0.1", "0.0.0.0"}


def build_tracking_link(post_id: int, telegram_id: int, source: str | None = None) -> str:
    params = {"tg": int(telegram_id)}
    if source:
        params["source"] = str(source)
    return f"{APP_BASE_URL}/r/{post_id}?{urlencode(params)}"


def source_link_tracking_enabled() -> bool:
    return SOURCE_LINK_TRACKING_ENABLED and not _is_local_base_url(APP_BASE_URL)


def build_reaction_keyboard(
    post_id: int,
    locale: str | None = None,
    *,
    storytracking_enabled: bool | None = None,
) -> types.InlineKeyboardMarkup:
    current_locale = normalize_locale(locale)
    rows: list[list[types.InlineKeyboardButton]] = [
        [
            types.InlineKeyboardButton(text="👎", callback_data=f"react:{post_id}:down"),
            types.InlineKeyboardButton(text="👍", callback_data=f"react:{post_id}:up"),
            types.InlineKeyboardButton(text="🔥", callback_data=f"react:{post_id}:fire"),
        ]
    ]
    storyline_visible = ENABLE_STORYTRACKING if storytracking_enabled is None else bool(storytracking_enabled)
    if storyline_visible:
        rows.append(
            [
                types.InlineKeyboardButton(
                    text=t(current_locale, "storyline_timeline_button"),
                    callback_data=f"storytimeline:{post_id}",
                ),
                types.InlineKeyboardButton(
                    text=t(current_locale, "follow_storyline_button"),
                    callback_data=f"storyfollow:{post_id}",
                ),
            ]
        )
    return types.InlineKeyboardMarkup(inline_keyboard=rows)


def extract_foreign_agent_notice(text: str) -> str | None:
    if not text:
        return None
    pattern = re.compile(
        r"(ДАННОЕ\s+СООБЩЕНИЕ.*?АГЕНТА)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _pick_mime(media_path: str | None, media_mime: str | None) -> str:
    if media_mime:
        return media_mime
    if not media_path:
        return ""
    mime, _ = mimetypes.guess_type(media_path)
    return mime or ""


def _resolve_media_size(media_path: str | None) -> int | None:
    if not media_path:
        return None
    try:
        return os.path.getsize(media_path)
    except OSError:
        return None


def _valid_photo_media_items(
    media_items: list[dict] | None,
    *,
    post_id: int,
    telegram_id: int,
) -> list[dict]:
    valid_items: list[dict] = []
    for item in media_items or []:
        if not isinstance(item, dict):
            continue
        media_path = item.get("path")
        media_mime = _pick_mime(media_path, item.get("mime"))
        if not media_path or not media_mime.startswith("image/"):
            continue

        media_size = item.get("size")
        if not isinstance(media_size, int):
            media_size = _resolve_media_size(media_path)
        if media_size is None:
            log.warning("Skip media item send: file unavailable", post_id=post_id, media_path=media_path)
            continue
        if media_size > MEDIA_MAX_BYTES:
            log.info(
                "Skip media item send: file too large",
                post_id=post_id,
                telegram_id=telegram_id,
                media_path=media_path,
                media_size=media_size,
                media_max_bytes=MEDIA_MAX_BYTES,
            )
            continue

        valid_items.append(
            {
                "path": media_path,
                "mime": media_mime,
                "size": media_size,
                "message_id": item.get("message_id"),
            }
        )
    return valid_items


def _split_text_block(text: str, limit: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest[:limit].rstrip()
        tail_start = max(0, len(cut) - 500)
        tail = cut[tail_start:]
        split_at = -1
        for separator in ("\n\n", "\n", ". ", "! ", "? ", " "):
            idx = tail.rfind(separator)
            if idx > 0:
                split_at = tail_start + idx + len(separator.rstrip())
                break
        if split_at <= 0:
            split_at = len(cut)
        chunk = rest[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        rest = rest[split_at:].strip()

    if rest:
        chunks.append(rest)
    return chunks


def _post_source_footer(
    *,
    original_url: str | None,
    source_title: str | None,
    tracking_url: str | None,
    current_locale: str,
) -> str:
    safe_title = html.escape(source_title or t(current_locale, "notif_source"))
    if original_url:
        safe_url = html.escape(original_url)
        return f'<a href="{safe_url}">{safe_title}</a>'
    if tracking_url:
        safe_tracking_url = html.escape(tracking_url)
        return f"{safe_title}: {safe_tracking_url}"
    return safe_title


def _render_post_message_parts(
    *,
    plain_body: str,
    original_url: str | None,
    source_title: str | None,
    tracking_url: str | None,
    current_locale: str,
) -> list[str]:
    footer = _post_source_footer(
        original_url=original_url,
        source_title=source_title,
        tracking_url=tracking_url,
        current_locale=current_locale,
    )
    footer_len = len(footer) + 2
    body_limit = max(500, TELEGRAM_SAFE_LEN - footer_len)
    chunks = _split_text_block(plain_body, body_limit) or [t(current_locale, "notif_news_without_text")]
    parts = [html.escape(chunk) for chunk in chunks]
    if len(parts[-1]) + footer_len <= TELEGRAM_SAFE_LEN:
        parts[-1] = f"{parts[-1]}\n\n{footer}"
    else:
        parts.append(footer)
    return parts


async def send_post_to_user(
    telegram_id: int,
    user_id: int,
    post_id: int,
    text: str,
    original_url: str | None,
    source_title: str | None,
    summary_enabled: bool,
    summary_text: str | None = None,
    media_path: str | None = None,
    media_mime: str | None = None,
    media_items: list[dict] | None = None,
    tg_username: str | None = None,
    locale: str | None = None,
    system_notice: str | None = None,
    storytracking_enabled: bool | None = None,
) -> dict[str, object]:
    async def _send_with_timeout(coro):
        return await asyncio.wait_for(coro, timeout=TELEGRAM_SEND_TIMEOUT)

    from app.audit import audit
    telegram_bot = _require_bot()
    current_locale = normalize_locale(locale)

    resolved_summary = summary_text
    if summary_enabled and text:
        content_text = text.strip()
        if len(content_text) < SUMMARY_MIN_CHARS:
            resolved_summary = None
        elif not resolved_summary:
            try:
                resolved_summary = await summarize(text)
            except Exception as e:
                # Do not break live delivery on summarization provider failures.
                log.warning(
                    "notifications.summary_failed_fallback_full_text",
                    post_id=post_id,
                    user_id=user_id,
                    telegram_id=telegram_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                resolved_summary = None

    plain_body = resolved_summary or text or t(current_locale, "notif_news_without_text")

    notice = extract_foreign_agent_notice(text)
    if notice:
        plain_body = f"{plain_body}\n\n{notice}"
    if system_notice:
        plain_body = f"{system_notice.strip()}\n\n{plain_body}"

    tracking_enabled = source_link_tracking_enabled()
    tracking_url = build_tracking_link(post_id, telegram_id, source="tg") if tracking_enabled else None
    storyline_visible = (
        ENABLE_STORYTRACKING
        and (
            storytracking_enabled
            if storytracking_enabled is not None
            else storytracking_allowed_for_user(telegram_id=telegram_id, username=tg_username)
        )
    )
    reply_markup = build_reaction_keyboard(
        post_id,
        locale=current_locale,
        storytracking_enabled=storyline_visible,
    )
    message_parts = _render_post_message_parts(
        plain_body=plain_body,
        original_url=original_url,
        source_title=source_title,
        tracking_url=tracking_url,
        current_locale=current_locale,
    )

    async def _send_message_parts(parts: list[str], *, delivery_result: str) -> dict[str, object]:
        sent_ids: list[int] = []
        last_msg = None
        for idx, part in enumerate(parts):
            log.info(
                "tg.send_message.call",
                telegram_id=telegram_id,
                post_id=post_id,
                chunk=idx + 1,
                chunks=len(parts),
            )
            last_msg = await _send_with_timeout(
                telegram_bot.send_message(
                    telegram_id,
                    part,
                    reply_markup=reply_markup if idx == len(parts) - 1 else None,
                    parse_mode="HTML",
                    disable_web_page_preview=False if idx == len(parts) - 1 else True,
                )
            )
            sent_ids.append(int(last_msg.message_id))
            log.info(
                "tg.send_message.ok",
                telegram_id=telegram_id,
                post_id=post_id,
                message_id=last_msg.message_id,
                chunk=idx + 1,
                chunks=len(parts),
            )
        return {
            "message_id": last_msg.message_id if last_msg else None,
            "message_ids": sent_ids,
            "delivery_result": delivery_result if len(parts) == 1 else f"{delivery_result}_split",
        }

    valid_photo_media_items = _valid_photo_media_items(
        media_items,
        post_id=post_id,
        telegram_id=telegram_id,
    )
    has_media = bool(media_path or valid_photo_media_items)
    media_allowed = True
    if media_path:
        media_size = _resolve_media_size(media_path)
        if media_size is None:
            media_allowed = False
            log.warning("Skip media send: file unavailable", post_id=post_id, media_path=media_path)
        elif media_size > MEDIA_MAX_BYTES:
            media_allowed = False
            log.info(
                "Skip media send: file too large",
                post_id=post_id,
                telegram_id=telegram_id,
                media_path=media_path,
                media_size=media_size,
                media_max_bytes=MEDIA_MAX_BYTES,
            )

    effective_media_path = media_path if media_allowed else None
    effective_media_mime = media_mime if media_allowed else None
    effective_mime = _pick_mime(effective_media_path, effective_media_mime)

    try:
        if len(valid_photo_media_items) >= 2 and (not effective_media_path or effective_mime.startswith("image/")):
            media_group = [
                types.InputMediaPhoto(media=types.FSInputFile(item["path"]))
                for item in valid_photo_media_items
            ]
            media_group_messages = await _send_with_timeout(
                telegram_bot.send_media_group(
                    telegram_id,
                    media=media_group,
                )
            )
            log.info(
                "tg.send_media_group.ok",
                telegram_id=telegram_id,
                post_id=post_id,
                media_group_count=len(valid_photo_media_items),
            )
            msg = await _send_message_parts(message_parts, delivery_result="photo_group")
            audit(
                "send.ok",
                post_id=post_id,
                telegram_id=telegram_id,
                user_id=user_id,
                tg_username=tg_username,
                has_summary=bool(resolved_summary),
                has_media=True,
                message_id=msg["message_id"],
                delivery_result=msg["delivery_result"],
                media_group_count=len(valid_photo_media_items),
                media_group_message_ids=[item.message_id for item in media_group_messages],
            )
            return msg

        if len(valid_photo_media_items) == 1 and (not effective_media_path or effective_mime.startswith("image/")):
            effective_media_path = valid_photo_media_items[0]["path"]
            effective_media_mime = valid_photo_media_items[0].get("mime")
            effective_mime = _pick_mime(effective_media_path, effective_media_mime)

        if not effective_media_path:
            delivery = await _send_message_parts(message_parts, delivery_result="message")

            audit(
                "send.ok",
                post_id=post_id,
                telegram_id=telegram_id,
                user_id=user_id,
                tg_username=tg_username,
                has_summary=bool(resolved_summary),
                has_media=has_media,
                message_id=delivery["message_id"],
                delivery_result=delivery["delivery_result"],
            )
            return delivery

        mime = _pick_mime(effective_media_path, effective_media_mime)
        file = types.FSInputFile(effective_media_path)
        can_use_caption = len(message_parts) == 1 and len(message_parts[0]) <= TELEGRAM_CAPTION_SAFE_LEN

        if mime.startswith("image/"):
            msg = await _send_with_timeout(
                telegram_bot.send_photo(
                    telegram_id,
                    file,
                    caption=message_parts[0] if can_use_caption else None,
                    reply_markup=reply_markup if can_use_caption else None,
                    parse_mode="HTML" if can_use_caption else None,
                )
            )
            delivery_result = "photo"
            message_id = msg.message_id
            if not can_use_caption:
                text_delivery = await _send_message_parts(message_parts, delivery_result="photo_text")
                delivery_result = text_delivery["delivery_result"]
                message_id = text_delivery["message_id"]
            
            audit(
                "send.ok",
                post_id=post_id,
                telegram_id=telegram_id,
                user_id=user_id,
                tg_username=tg_username,
                has_summary=bool(resolved_summary),
                has_media=has_media,
                message_id=message_id,
                delivery_result=delivery_result,
            )
            return {"message_id": message_id, "delivery_result": delivery_result}
        elif mime.startswith("video/"):
            msg = await _send_with_timeout(
                telegram_bot.send_video(
                    telegram_id,
                    file,
                    caption=message_parts[0] if can_use_caption else None,
                    reply_markup=reply_markup if can_use_caption else None,
                    parse_mode="HTML" if can_use_caption else None,
                )
            )
            delivery_result = "video"
            message_id = msg.message_id
            if not can_use_caption:
                text_delivery = await _send_message_parts(message_parts, delivery_result="video_text")
                delivery_result = text_delivery["delivery_result"]
                message_id = text_delivery["message_id"]
            
            audit(
                "send.ok",
                post_id=post_id,
                telegram_id=telegram_id,
                user_id=user_id,
                tg_username=tg_username,
                has_summary=bool(resolved_summary),
                has_media=has_media,
                message_id=message_id,
                delivery_result=delivery_result,
            )
            return {"message_id": message_id, "delivery_result": delivery_result}
        else:
            msg = await _send_with_timeout(
                telegram_bot.send_document(
                    telegram_id,
                    file,
                    caption=message_parts[0] if can_use_caption else None,
                    reply_markup=reply_markup if can_use_caption else None,
                    parse_mode="HTML" if can_use_caption else None,
                )
            )
            delivery_result = "document"
            message_id = msg.message_id
            if not can_use_caption:
                text_delivery = await _send_message_parts(message_parts, delivery_result="document_text")
                delivery_result = text_delivery["delivery_result"]
                message_id = text_delivery["message_id"]
            
            audit(
                "send.ok",
                post_id=post_id,
                telegram_id=telegram_id,
                user_id=user_id,
                tg_username=tg_username,
                has_summary=bool(resolved_summary),
                has_media=has_media,
                message_id=message_id,
                delivery_result=delivery_result,
            )
            return {"message_id": message_id, "delivery_result": delivery_result}
            
    except Exception as e:
        if isinstance(e, TelegramForbiddenError):
            audit(
                "send.blocked_by_user",
                post_id=post_id,
                telegram_id=telegram_id,
                user_id=user_id,
                tg_username=tg_username,
                error=str(e),
            )
            log.info(
                "Telegram user blocked bot, skip delivery",
                telegram_id=telegram_id,
                post_id=post_id,
                user_id=user_id,
            )
            return {"message_id": None, "delivery_result": "blocked_by_user"}
        log.error(
            "Failed to send media, fallback to text",
            error=str(e),
            telegram_id=telegram_id,
            post_id=post_id,
            media_path=effective_media_path,
            exc_info=True,
        )
        delivery = await _send_message_parts(message_parts, delivery_result="fallback_text")
        
        audit(
            "send.fallback_text_ok",
            post_id=post_id,
            telegram_id=telegram_id,
            user_id=user_id,
            tg_username=tg_username,
            error=str(e),
            message_id=delivery["message_id"],
            delivery_result=delivery["delivery_result"],
        )
        return delivery


async def send_subscription_status(
    telegram_id: int,
    group_link: str,
    ok: bool,
    error: str | None = None,
    locale: str | None = None,
) -> None:
    telegram_bot = _try_get_service_bot(
        action="send_subscription_status",
        telegram_id=telegram_id,
        group_link=group_link,
        ok=ok,
    )
    if telegram_bot is None:
        return
    current_locale = normalize_locale(locale)

    if ok:
        text = t(current_locale, "subscription_success", group_link=group_link)
    else:
        text = t(current_locale, "subscription_failed", group_link=group_link)
        if error:
            reason = _subscription_error_text(error, current_locale)
            text = f"{text}\n{t(current_locale, 'reason')}: {reason}"

    await telegram_bot.send_message(telegram_id, text)


def _subscription_error_text(error: str, locale: str | None = None) -> str:
    current_locale = normalize_locale(locale)
    prefix = "telegram_flood_wait:"
    if error.startswith(prefix):
        raw_seconds = error[len(prefix) :]
        try:
            seconds = max(0, int(raw_seconds))
        except ValueError:
            seconds = 0
        return t(current_locale, "subscription_error_telegram_flood_wait", seconds=seconds)
    return error


def _subscription_progress_text(
    group_link: str,
    stage: str,
    error: str | None = None,
    locale: str | None = None,
) -> str:
    current_locale = normalize_locale(locale)
    reason_suffix = ""
    if error:
        reason = _subscription_error_text(error, current_locale)
        reason_suffix = f"\n{t(current_locale, 'reason')}: {reason}"

    if stage == "queued":
        return t(current_locale, "subscription_stage_queued", group_link=group_link)
    if stage == "retrying":
        return t(
            current_locale,
            "subscription_stage_retrying",
            group_link=group_link,
            reason=reason_suffix,
        )
    if stage == "joining":
        return t(current_locale, "subscription_stage_joining", group_link=group_link)
    if stage == "joined":
        return t(current_locale, "subscription_stage_joined", group_link=group_link)
    if stage == "syncing":
        return t(current_locale, "subscription_stage_syncing", group_link=group_link)
    if stage == "completed":
        return t(current_locale, "subscription_stage_completed", group_link=group_link)
    if stage == "sync_failed":
        return t(current_locale, "subscription_stage_sync_failed", group_link=group_link, reason=reason_suffix)
    if stage == "failed":
        return t(current_locale, "subscription_stage_failed", group_link=group_link, reason=reason_suffix)
    return t(current_locale, "subscription_stage_updated", group_link=group_link)


async def update_subscription_progress(
    chat_id: int,
    message_id: int | None,
    group_link: str,
    stage: str,
    error: str | None = None,
    locale: str | None = None,
) -> None:
    text = _subscription_progress_text(group_link=group_link, stage=stage, error=error, locale=locale)
    telegram_bot = _try_get_service_bot(
        action="update_subscription_progress",
        chat_id=chat_id,
        message_id=message_id,
        group_link=group_link,
        stage=stage,
    )
    if telegram_bot is None:
        return

    if message_id:
        try:
            await telegram_bot.edit_message_text(text, chat_id=chat_id, message_id=message_id)
            return
        except Exception as e:
            log.warning(
                "subscription_progress_edit_failed",
                error=str(e),
                chat_id=chat_id,
                message_id=message_id,
                group_link=group_link,
                stage=stage,
            )

    try:
        await telegram_bot.send_message(chat_id, text)
    except Exception as e:
        log.error(
            "subscription_progress_send_failed",
            error=str(e),
            chat_id=chat_id,
            message_id=message_id,
            group_link=group_link,
            stage=stage,
        )


async def update_subscription_status(
    chat_id: int,
    message_id: int | None,
    group_link: str,
    ok: bool,
    error: str | None = None,
    locale: str | None = None,
) -> None:
    await update_subscription_progress(
        chat_id=chat_id,
        message_id=message_id,
        group_link=group_link,
        stage="completed" if ok else "failed",
        error=error,
        locale=locale,
    )

def _chunk_blocks(blocks: list[str], limit: int = TELEGRAM_SAFE_LEN) -> list[str]:
    chunks: list[str] = []
    buf = ""

    for b in blocks:
        b = (b or "").strip()
        if not b:
            continue

        candidate = (buf + "\n\n" + b).strip() if buf else b
        if len(candidate) > limit:
            if buf:
                chunks.append(buf)
                if len(b) > limit:
                    chunks.extend(_split_text_block(b, limit))
                    buf = ""
                else:
                    buf = b
            else:
                chunks.extend(_split_text_block(b, limit))
                buf = ""
        else:
            buf = candidate

    if buf:
        chunks.append(buf)

    return chunks

def render_digest_pages(
    period_start,
    period_end,
    items: list[dict],
    locale: str | None = None,
    headline: str | None = None,
) -> list[str]:
    current_locale = normalize_locale(locale)
    header = t(
        current_locale,
        "digest_header_24h",
        period_start=f"{period_start:%Y-%m-%d %H:%M}",
        period_end=f"{period_end:%Y-%m-%d %H:%M}",
    )

    if headline:
        header = header + "\n\n" + t(current_locale, "digest_headline_label", headline=html.escape(headline.strip()))

    blocks: list[str] = [header]

    for i, it in enumerate(items or [], 1):
        title_raw = str((it.get("title") or "")).strip()
        source_raw = (
            it.get("source")
            or it.get("channel")
            or it.get("community")
            or t(current_locale, "notif_source")
        )
        source_txt = str(source_raw).strip() or t(current_locale, "notif_source")

        title = html.escape(title_raw) if title_raw else ""
        source = html.escape(source_txt)
        headline = title or source

        text_raw = str((it.get("summary") or it.get("text") or "")).strip()
        if ("<a href=" in text_raw) or ("&lt;a href=" in text_raw):
            text_ = html.unescape(text_raw)
        else:
            text_ = html.escape(text_raw)

        url = str((it.get("url") or "")).strip()
        links = it.get("links") or []
        if not isinstance(links, list):
            links = []
        links = [str(u).strip() for u in links if str(u).strip()]
        link_labels = it.get("link_labels") if isinstance(it.get("link_labels"), dict) else {}

        if url and url not in links:
            links = [url] + links

        safe_url = html.escape(url) if url else ""
        if not text_ and not safe_url and not links:
            continue

        parts: list[str] = [f"{i}) <b>{headline}</b>"]
        if text_:
            parts.append(text_)

        if links:
            link_tags = []
            for u in links[:10]:
                su = html.escape(u)
                channel_name = ""
                if link_labels:
                    channel_name = str(
                        link_labels.get(u)
                        or link_labels.get(u.rstrip("/"))
                        or ""
                    ).strip()
                if "t.me/" in u:
                    try:
                        tg_name = u.split("t.me/", 1)[1].strip("/").split("?", 1)[0].split("/", 1)[0]
                        if tg_name and tg_name != "c" and not channel_name:
                            channel_name = "@" + tg_name
                        elif tg_name == "c" and not channel_name:
                            channel_name = t(current_locale, "notif_channel")
                    except Exception:
                        channel_name = channel_name or ""
                if not channel_name:
                    try:
                        parsed = urlparse(u)
                        channel_name = parsed.hostname or t(current_locale, "notif_link")
                    except Exception:
                        channel_name = t(current_locale, "notif_link")

                link_tags.append(f"<a href=\"{su}\">{html.escape(channel_name)}</a>")

            parts.append("")
            parts.append(t(current_locale, "digest_sources_label") + ", ".join(link_tags))

        blocks.append("\n".join(parts).strip())

    if len(blocks) == 1:
        blocks.append(t(current_locale, "digest_empty"))

    return _chunk_blocks(blocks, TELEGRAM_SAFE_LEN)


def build_digest_pager_kb(run_id: int, page: int, total: int, locale: str | None = None) -> types.InlineKeyboardMarkup:
    _ = normalize_locale(locale)
    prev_page = max(0, page - 1)
    next_page = min(total - 1, page + 1)

    return types.InlineKeyboardMarkup(inline_keyboard=[[
        types.InlineKeyboardButton(text="⬅️", callback_data=f"dig:{run_id}:{prev_page}"),
        types.InlineKeyboardButton(text=f"{page+1}/{total}", callback_data="noop"),
        types.InlineKeyboardButton(text="➡️", callback_data=f"dig:{run_id}:{next_page}"),
    ]])


async def send_digest_first_page(
    telegram_id: int,
    run_id: int,
    text: str,
    page: int,
    total: int,
    locale: str | None = None,
    bot_token_override: str | None = None,
) -> int:
    telegram_bot = _build_temp_bot(bot_token_override) or _require_bot()
    kb = build_digest_pager_kb(run_id, page=page, total=total, locale=locale) if total > 1 else None
    try:
        msg = await telegram_bot.send_message(
            telegram_id,
            text,
            parse_mode="HTML",
            disable_web_page_preview=True,
            reply_markup=kb,
        )
        return int(msg.message_id)
    finally:
        if telegram_bot is not bot:
            with contextlib.suppress(Exception):
                await telegram_bot.session.close()


async def send_digest_to_user(
    telegram_id: int,
    run_id: int,
    period_start,
    period_end,
    items: list[dict],
    locale: str | None = None,
    bot_token_override: str | None = None,
) -> int:
    telegram_bot = _build_temp_bot(bot_token_override) or _require_bot()
    current_locale = normalize_locale(locale)
    chunks = render_digest_pages(period_start, period_end, items, locale=current_locale)
    if not chunks:
        chunks = [t(current_locale, "digest_empty_short")]

    sent = 0
    try:
        for idx, part in enumerate(chunks, 1):
            part = (part or "").strip()
            if not part:
                continue

            try:
                log.info(
                    "tg.send_message.call",
                    telegram_id=telegram_id,
                    digest_run_id=run_id,
                    chunk=idx,
                    chunks=len(chunks),
                )
                msg = await telegram_bot.send_message(
                    telegram_id,
                    part,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
                log.info(
                    "tg.send_message.ok",
                    telegram_id=telegram_id,
                    digest_run_id=run_id,
                    message_id=msg.message_id,
                    chunk=idx,
                    chunks=len(chunks),
                )
                sent += 1

            except TelegramBadRequest as e:
                log.error(
                    "tg.send_message.bad_request",
                    telegram_id=telegram_id,
                    digest_run_id=run_id,
                    chunk=idx,
                    chunks=len(chunks),
                    error=getattr(e, "message", str(e)),
                    exc_info=True,
                )
                raise

            except TelegramForbiddenError as e:
                log.error(
                    "tg.send_message.forbidden",
                    telegram_id=telegram_id,
                    digest_run_id=run_id,
                    chunk=idx,
                    chunks=len(chunks),
                    error=getattr(e, "message", str(e)),
                    exc_info=True,
                )
                raise

            except Exception as e:
                log.error(
                    "tg.send_message.failed",
                    telegram_id=telegram_id,
                    digest_run_id=run_id,
                    chunk=idx,
                    chunks=len(chunks),
                    error=str(e),
                    exc_info=True,
                )
                raise
    finally:
        if telegram_bot is not bot:
            with contextlib.suppress(Exception):
                await telegram_bot.session.close()

    return sent
