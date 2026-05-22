from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.i18n import normalize_button_label

NEWS_TIME_FOLLOWUP_DELAY = timedelta(days=7)
NEWS_TIME_FOLLOWUP_RETRY_DELAY = timedelta(days=1)

NEWS_TIME_OPTIONS: tuple[tuple[str, int, str], ...] = (
    ("до часа", 30, "under_hour"),
    ("1 час", 60, "1h"),
    ("2 часа", 120, "2h"),
    ("3 часа", 180, "3h"),
    ("4 часа", 240, "4h"),
    ("5 часов", 300, "5h"),
)

NEWS_TIME_OPTION_KEYS: tuple[str, ...] = (
    "news_time_option_under_hour",
    "news_time_option_1h",
    "news_time_option_2h",
    "news_time_option_3h",
    "news_time_option_4h",
    "news_time_option_5h",
)

NEWS_TIME_BUCKET_SLUGS: tuple[str, ...] = tuple(slug for _, _, slug in NEWS_TIME_OPTIONS)

_OPTION_BY_NORMALIZED = {
    normalize_button_label(label): (label, minutes, slug)
    for label, minutes, slug in NEWS_TIME_OPTIONS
}


@dataclass(frozen=True, slots=True)
class NewsTimeAnswer:
    label: str
    minutes: int
    slug: str


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_news_time_answer(value: str | None) -> NewsTimeAnswer | None:
    normalized = normalize_button_label(value)
    match = _OPTION_BY_NORMALIZED.get(normalized)
    if not match:
        return None
    label, minutes, slug = match
    return NewsTimeAnswer(label=label, minutes=minutes, slug=slug)


def parse_time_saved_answer(value: str | None) -> str | None:
    normalized = normalize_button_label(value)
    if normalized == "да":
        return "yes"
    if normalized == "нет":
        return "no"
    return None


def followup_due_at(baseline_answered_at: datetime) -> datetime:
    return baseline_answered_at + NEWS_TIME_FOLLOWUP_DELAY


def should_ask_followup(survey, *, now: datetime) -> bool:
    if not survey or getattr(survey, "time_saved_answered_at", None) or getattr(survey, "followup_abandoned_at", None):
        return False
    due_at = getattr(survey, "followup_due_at", None)
    if due_at is None:
        baseline_answered_at = getattr(survey, "baseline_answered_at", None)
        if baseline_answered_at is None:
            return False
        due_at = followup_due_at(baseline_answered_at)
    if now < due_at:
        return False
    asked_at = getattr(survey, "followup_asked_at", None)
    retry_asked_at = getattr(survey, "followup_retry_asked_at", None)
    if asked_at is None:
        return True
    if retry_asked_at is None and now >= asked_at + NEWS_TIME_FOLLOWUP_RETRY_DELAY:
        return True
    return False


def should_abandon_followup(survey, *, now: datetime) -> bool:
    if not survey or getattr(survey, "time_saved_answered_at", None) or getattr(survey, "followup_abandoned_at", None):
        return False
    retry_asked_at = getattr(survey, "followup_retry_asked_at", None)
    return bool(retry_asked_at and now >= retry_asked_at + NEWS_TIME_FOLLOWUP_RETRY_DELAY)
