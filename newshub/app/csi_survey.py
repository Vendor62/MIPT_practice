from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.i18n import normalize_button_label

CSI_FOLLOWUP_DELAY = timedelta(days=7)
CSI_RETRY_DELAY = timedelta(days=1)
CSI_MIN_DELIVERIES = 5
CSI_MAX_SCORE = 10
CSI_NEWS_TIME_COOLDOWN = timedelta(days=1)
CSI_SCORE_KEYS: tuple[str, ...] = tuple(f"csi_score_{value}" for value in range(1, CSI_MAX_SCORE + 1))


@dataclass(frozen=True, slots=True)
class CsiEligibility:
    first_delivery_at: datetime
    delivery_count: int
    due_at: datetime


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_csi_score(value: str | None) -> int | None:
    normalized = normalize_button_label(value)
    try:
        score = int(normalized)
    except (TypeError, ValueError):
        return None
    if 1 <= score <= CSI_MAX_SCORE:
        return score
    return None


def build_csi_eligibility(
    *,
    first_delivery_at: datetime | None,
    delivery_count: int | None,
) -> CsiEligibility | None:
    if first_delivery_at is None:
        return None
    count = int(delivery_count or 0)
    if count < CSI_MIN_DELIVERIES:
        return None
    return CsiEligibility(
        first_delivery_at=first_delivery_at,
        delivery_count=count,
        due_at=first_delivery_at + CSI_FOLLOWUP_DELAY,
    )


def should_ask_csi(survey, *, now: datetime, eligibility: CsiEligibility | None) -> bool:
    if eligibility is None:
        return False
    if now < eligibility.due_at:
        return False
    if survey and (getattr(survey, "answered_at", None) or getattr(survey, "abandoned_at", None)):
        return False
    asked_at = getattr(survey, "asked_at", None) if survey else None
    retry_asked_at = getattr(survey, "retry_asked_at", None) if survey else None
    if asked_at is None:
        return True
    if retry_asked_at is None and now >= asked_at + CSI_RETRY_DELAY:
        return True
    return False


def should_abandon_csi(survey, *, now: datetime) -> bool:
    if not survey or getattr(survey, "answered_at", None) or getattr(survey, "abandoned_at", None):
        return False
    retry_asked_at = getattr(survey, "retry_asked_at", None)
    return bool(retry_asked_at and now >= retry_asked_at + CSI_RETRY_DELAY)


def is_csi_pending(survey) -> bool:
    return bool(
        survey
        and getattr(survey, "asked_at", None)
        and not getattr(survey, "answered_at", None)
        and not getattr(survey, "abandoned_at", None)
    )
