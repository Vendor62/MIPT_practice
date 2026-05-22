import asyncio
import os
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.celery_app import celery_app
from app.models import get_session, UserDigestSlot, UserDigestSettings, DigestRun

log = structlog.get_logger()

POLL_SECONDS = int(os.getenv("DIGEST_SCHEDULER_POLL_SECONDS", "30"))
WINDOW_HOURS = 24
BATCH = int(os.getenv("DIGEST_SCHEDULER_BATCH", "50"))


def compute_next_run_after(base_utc: datetime, offset_minutes: int, hour: int, minute: int) -> datetime:
    """Return next day run datetime in UTC for fixed UTC offset and desired local HH:MM."""
    offset = timezone(timedelta(minutes=offset_minutes))
    base_local = base_utc.astimezone(offset)
    next_local = base_local.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=1)
    return next_local.astimezone(timezone.utc)


async def _try_lock_slot(session, slot_id: int) -> bool:
    # Prevent double-processing if multiple schedulers are running.
    res = await session.execute(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": int(slot_id)})
    return bool(res.scalar_one())


async def scheduler_tick() -> None:
    now = datetime.now(timezone.utc)

    async for session in get_session():
        res = await session.execute(
            select(UserDigestSlot)
            .where(
                UserDigestSlot.is_active.is_(True),
                UserDigestSlot.next_run_at.is_not(None),
                UserDigestSlot.next_run_at <= now,
            )
            .order_by(UserDigestSlot.next_run_at.asc())
            .limit(BATCH)
            .with_for_update(skip_locked=True)
        )
        slots = res.scalars().all()

        for slot in slots:
            if not await _try_lock_slot(session, slot.id):
                continue

            settings = await session.get(UserDigestSettings, slot.user_id)

            base_utc = slot.next_run_at or now

            # If disabled or missing offset -> do not send, but move forward.
            if (not settings) or (not settings.enabled) or (settings.utc_offset_minutes is None):
                # If offset is unknown, move by +24h in UTC to avoid hammering.
                slot.next_run_at = base_utc + timedelta(days=1)
                await session.commit()
                continue

            # ---- create digest run ----
            period_end = slot.next_run_at
            if period_end is None:
                # Safety: if next_run_at is missing, just schedule the next one.
                slot.next_run_at = compute_next_run_after(now, settings.utc_offset_minutes, slot.hour, slot.minute)
                await session.commit()
                continue

            period_start = period_end - timedelta(hours=WINDOW_HOURS)

            run = DigestRun(
                slot_id=slot.id,
                user_id=slot.user_id,
                period_start=period_start,
                period_end=period_end,
                status="queued",
            )
            session.add(run)

            slot.last_run_at = period_end
            slot.next_run_at = compute_next_run_after(period_end, settings.utc_offset_minutes, slot.hour, slot.minute)

            try:
                await session.commit()
            except IntegrityError:
                # likely unique constraint collision; rollback and skip
                await session.rollback()
                continue

            celery_app.send_task(
                "app.tasks.build_and_send_digest",
                args=[run.id],
                queue="digest_queue",
            )
            log.info("digest.queued", run_id=run.id, slot_id=slot.id, user_id=slot.user_id)

        break


async def main() -> None:
    log.info("digest_scheduler.start", poll=POLL_SECONDS)
    while True:
        try:
            await scheduler_tick()
        except Exception as e:
            log.error("digest_scheduler.tick_failed", error=str(e), exc_info=True)
        await asyncio.sleep(POLL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
