from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
import re
from typing import Any


ALLOWED_INTENTS = {
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
}

READ_ONLY_INTENTS = {
    "show_settings_overview",
    "show_subscriptions",
    "show_billing",
    "show_help_topic",
}

MUTATING_INTENTS = ALLOWED_INTENTS - READ_ONLY_INTENTS

FILTER_MODES = {"all", "not_interesting", "only_fire", "digest_only"}
LANGUAGE_CODES = {"ru", "en"}
FAQ_TOPICS = {
    "general",
    "settings",
    "subscriptions",
    "billing",
    "forwarding",
    "filters",
    "digest",
    "language",
    "summary",
    "ai_filter",
    "storyline_tracking",
}
INSTRUCTION_PROMPT_MAX_CHARS = int(os.getenv("INSTRUCTION_FILTER_PROMPT_MAX_CHARS", "500"))

TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
OFFSET_RE = re.compile(r"^(?:UTC)?\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?$", re.IGNORECASE)


class NLUValidationError(ValueError):
    pass


@dataclass(slots=True)
class NLUResult:
    intent: str
    slots: dict[str, Any]
    needs_clarification: bool
    clarify_question: str | None
    faq_topic: str | None
    proposed_user_message: str | None


@dataclass(slots=True)
class PendingActionPayload:
    user_id: int
    telegram_id: int
    intent: str
    slots: dict[str, Any]
    locale: str
    summary_text: str
    created_at: str
    expires_at: str
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PendingActionPayload":
        if not isinstance(raw, dict):
            raise NLUValidationError("Pending action payload must be an object")
        return cls(
            user_id=int(raw["user_id"]),
            telegram_id=int(raw["telegram_id"]),
            intent=str(raw["intent"]),
            slots=dict(raw.get("slots") or {}),
            locale=str(raw.get("locale") or "ru"),
            summary_text=str(raw.get("summary_text") or ""),
            created_at=str(raw["created_at"]),
            expires_at=str(raw["expires_at"]),
            request_id=str(raw.get("request_id") or "").strip() or None,
        )


@dataclass(slots=True)
class ExecutorResult:
    text: str
    reply_markup: Any = None
    parse_mode: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_pending_action_payload(
    *,
    user_id: int,
    telegram_id: int,
    intent: str,
    slots: dict[str, Any],
    locale: str,
    summary_text: str,
    ttl_seconds: int,
    request_id: str | None = None,
) -> PendingActionPayload:
    now = datetime.now(timezone.utc)
    expires_at = now.timestamp() + max(1, int(ttl_seconds))
    return PendingActionPayload(
        user_id=int(user_id),
        telegram_id=int(telegram_id),
        intent=str(intent),
        slots=dict(slots or {}),
        locale=str(locale or "ru"),
        summary_text=str(summary_text or "").strip(),
        created_at=now.isoformat(),
        expires_at=datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        request_id=str(request_id or "").strip() or None,
    )


def parse_hhmm(value: str) -> str | None:
    text = str(value or "").strip()
    return text if TIME_RE.match(text) else None


def normalize_utc_offset(value: str | int | float | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        sign = "+" if float(value) >= 0 else "-"
        total_minutes = abs(int(round(float(value) * 60)))
        hours, minutes = divmod(total_minutes, 60)
        if hours > 14 or minutes > 59:
            return None
        return f"{sign}{hours:02d}:{minutes:02d}"

    raw = str(value or "").strip()
    match = OFFSET_RE.match(raw.replace(" ", ""))
    if not match:
        return None
    sign, hours_raw, minutes_raw = match.groups()
    hours = int(hours_raw)
    minutes = int(minutes_raw or "0")
    if hours > 14 or minutes > 59:
        return None
    return f"{sign}{hours:02d}:{minutes:02d}"


def normalize_language_code(value: str | None) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in LANGUAGE_CODES else None


def parse_group_input(value: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    if not raw:
        return "@", "https://t.me/"

    if raw.startswith("@"):
        username = raw[1:]
    elif "t.me/" in raw:
        username = raw.split("t.me/", 1)[1].split("?", 1)[0].strip("/")
    else:
        username = raw.lstrip("@")

    username = re.sub(r"[^A-Za-z0-9_]", "", username)
    handle = f"@{username}" if username else "@"
    link = f"https://t.me/{username}" if username else "https://t.me/"
    return handle, link


def normalize_group_targets(
    *,
    links: list[Any] | None = None,
    handles: list[Any] | None = None,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for source in (handles or []) + (links or []):
        if not isinstance(source, str):
            continue
        handle, link = parse_group_input(source)
        if handle == "@":
            continue
        if link in seen:
            continue
        seen.add(link)
        normalized.append(link)
    return normalized


def validate_nlu_result(raw: dict[str, Any]) -> NLUResult:
    if not isinstance(raw, dict):
        raise NLUValidationError("NLU result must be an object")

    intent = str(raw.get("intent") or "").strip()
    if intent not in ALLOWED_INTENTS:
        raise NLUValidationError(f"Unsupported intent: {intent or 'empty'}")

    slots = raw.get("slots") or {}
    if not isinstance(slots, dict):
        raise NLUValidationError("slots must be an object")
    normalized_slots = dict(slots)

    needs_clarification = bool(raw.get("needs_clarification"))
    clarify_question = str(raw.get("clarify_question") or "").strip() or None
    faq_topic = str(raw.get("faq_topic") or "").strip().lower() or None
    proposed_user_message = str(raw.get("proposed_user_message") or "").strip() or None

    if faq_topic and faq_topic not in FAQ_TOPICS:
        faq_topic = "general"

    if intent == "show_help_topic" and not faq_topic:
        faq_topic = "general"

    if intent in {"add_subscriptions", "remove_subscriptions"}:
        normalized_links = normalize_group_targets(
            links=normalized_slots.get("links"),
            handles=normalized_slots.get("handles"),
        )
        if normalized_links:
            normalized_slots["links"] = normalized_links
            normalized_slots.pop("handles", None)
        elif not needs_clarification:
            raise NLUValidationError(f"{intent} requires links or handles")

    if intent == "set_feed_filter":
        filter_mode = str(normalized_slots.get("filter_mode") or "").strip().lower()
        if filter_mode in FILTER_MODES:
            normalized_slots["filter_mode"] = filter_mode
        elif not needs_clarification:
            raise NLUValidationError("set_feed_filter requires valid filter_mode")

    if intent in {"toggle_forwarding", "toggle_summary"}:
        if "value_bool" not in normalized_slots and not needs_clarification:
            raise NLUValidationError(f"{intent} requires value_bool")
        if "value_bool" in normalized_slots:
            normalized_slots["value_bool"] = bool(normalized_slots.get("value_bool"))

    if intent == "digest_set_time":
        parsed_time = parse_hhmm(str(normalized_slots.get("time_hhmm") or ""))
        if parsed_time:
            normalized_slots["time_hhmm"] = parsed_time
        elif not needs_clarification:
            raise NLUValidationError("digest_set_time requires valid time_hhmm")

    if intent == "digest_set_offset":
        normalized_offset = normalize_utc_offset(normalized_slots.get("utc_offset"))
        if normalized_offset:
            normalized_slots["utc_offset"] = normalized_offset
        elif not needs_clarification:
            raise NLUValidationError("digest_set_offset requires valid utc_offset")

    if intent == "set_language":
        language_code = normalize_language_code(normalized_slots.get("language_code"))
        if language_code:
            normalized_slots["language_code"] = language_code
        elif not needs_clarification:
            raise NLUValidationError("set_language requires language_code")

    if intent == "set_global_instruction_filter":
        prompt_text = str(normalized_slots.get("prompt_text") or "").strip()
        if prompt_text and len(prompt_text) <= INSTRUCTION_PROMPT_MAX_CHARS:
            normalized_slots["prompt_text"] = prompt_text
        elif not needs_clarification:
            raise NLUValidationError("set_global_instruction_filter requires valid prompt_text")

    if needs_clarification and not clarify_question:
        raise NLUValidationError("needs_clarification requires clarify_question")

    return NLUResult(
        intent=intent,
        slots=normalized_slots,
        needs_clarification=needs_clarification,
        clarify_question=clarify_question,
        faq_topic=faq_topic,
        proposed_user_message=proposed_user_message,
    )
