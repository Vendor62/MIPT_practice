from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.models import BotAssistantRequest
from app.nlu_router import NLUResult

RAW_TEXT_MAX_CHARS = 4000

STATUS_RECEIVED = "received"
STATUS_UNDERSTOOD = "understood"
STATUS_CLARIFICATION_REQUESTED = "clarification_requested"
STATUS_ANSWERED = "answered"
STATUS_ACTION_PROPOSED = "action_proposed"
STATUS_CANCELLED = "cancelled"
STATUS_APPLIED = "applied"
STATUS_FAILED = "failed"
STATUS_EXPIRED = "expired"

OUTCOME_SUCCESS = "success"
OUTCOME_NEEDS_CLARIFICATION = "needs_clarification"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_ERROR = "error"
OUTCOME_EXPIRED = "expired"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _truncate_text(value: str | None) -> str:
    return str(value or "").strip()[:RAW_TEXT_MAX_CHARS]


async def _get_request(session: Any, request_id: str | None) -> BotAssistantRequest | None:
    normalized = str(request_id or "").strip()
    if not normalized:
        return None
    result = await session.execute(select(BotAssistantRequest).where(BotAssistantRequest.request_id == normalized))
    return result.scalar_one_or_none()


async def create_request(
    session: Any,
    *,
    request_id: str,
    telegram_id: int | None,
    user_id: int | None,
    locale: str,
    raw_text: str,
) -> BotAssistantRequest:
    now = _now()
    record = BotAssistantRequest(
        request_id=str(request_id),
        telegram_id=int(telegram_id) if telegram_id is not None else None,
        user_id=int(user_id) if user_id is not None else None,
        locale=str(locale or "ru"),
        raw_text=_truncate_text(raw_text),
        status=STATUS_RECEIVED,
        received_at=now,
        updated_at=now,
    )
    session.add(record)
    await session.flush()
    return record


async def mark_understood(session: Any, *, request_id: str | None, nlu_result: NLUResult) -> None:
    record = await _get_request(session, request_id)
    if not record:
        return
    now = _now()
    record.intent = nlu_result.intent
    record.slots = dict(nlu_result.slots or {})
    record.faq_topic = nlu_result.faq_topic
    record.needs_clarification = bool(nlu_result.needs_clarification)
    record.status = STATUS_UNDERSTOOD
    record.understood_at = now
    record.updated_at = now
    await session.flush()


async def mark_clarification(session: Any, *, request_id: str | None) -> None:
    record = await _get_request(session, request_id)
    if not record:
        return
    now = _now()
    record.status = STATUS_CLARIFICATION_REQUESTED
    record.outcome = OUTCOME_NEEDS_CLARIFICATION
    record.clarified_at = now
    record.updated_at = now
    await session.flush()


async def mark_answered(session: Any, *, request_id: str | None) -> None:
    record = await _get_request(session, request_id)
    if not record:
        return
    now = _now()
    record.status = STATUS_ANSWERED
    record.outcome = OUTCOME_SUCCESS
    record.updated_at = now
    await session.flush()


async def mark_proposed(session: Any, *, request_id: str | None) -> None:
    record = await _get_request(session, request_id)
    if not record:
        return
    now = _now()
    record.status = STATUS_ACTION_PROPOSED
    record.proposed_at = now
    record.updated_at = now
    await session.flush()


async def mark_confirmed(session: Any, *, request_id: str | None) -> None:
    record = await _get_request(session, request_id)
    if not record:
        return
    now = _now()
    record.confirmed_at = now
    record.updated_at = now
    await session.flush()


async def mark_cancelled(session: Any, *, request_id: str | None) -> None:
    record = await _get_request(session, request_id)
    if not record:
        return
    now = _now()
    record.status = STATUS_CANCELLED
    record.outcome = OUTCOME_CANCELLED
    record.cancelled_at = now
    record.updated_at = now
    await session.flush()


async def mark_applied(session: Any, *, request_id: str | None) -> None:
    record = await _get_request(session, request_id)
    if not record:
        return
    now = _now()
    record.status = STATUS_APPLIED
    record.outcome = OUTCOME_SUCCESS
    record.applied_at = now
    record.updated_at = now
    await session.flush()


async def mark_failed(
    session: Any,
    *,
    request_id: str | None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    record = await _get_request(session, request_id)
    if not record:
        return
    now = _now()
    record.status = STATUS_FAILED
    record.outcome = OUTCOME_ERROR
    record.error_code = str(error_code or "").strip() or None
    record.error_message = str(error_message or "").strip()[:1000] or None
    record.failed_at = now
    record.updated_at = now
    await session.flush()


async def mark_expired(session: Any, *, request_id: str | None) -> None:
    record = await _get_request(session, request_id)
    if not record:
        return
    now = _now()
    record.status = STATUS_EXPIRED
    record.outcome = OUTCOME_EXPIRED
    record.failed_at = now
    record.updated_at = now
    await session.flush()


async def safe_mark_failed(
    session: Any,
    *,
    request_id: str | None,
    error_code: str,
    error_message: Any,
) -> None:
    try:
        await mark_failed(
            session,
            request_id=request_id,
            error_code=error_code,
            error_message=repr(error_message),
        )
        await session.commit()
    except Exception:
        await session.rollback()
