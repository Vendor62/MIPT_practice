from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from sqlalchemy import case, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    BotAssistantRequest,
    Click,
    Community,
    ExperimentAssignment,
    ExperimentExposure,
    PaymentOrder,
    Post,
    PostInteraction,
    PremiumEntitlementRecord,
    ProductMetricsDaily,
    Reaction,
    User,
    UserInstructionRule,
    StorylineUpdateEvent,
    UserStorylineFollow,
    TelegramOperation,
    UserCsiSurvey,
    UserNewsTimeSurvey,
    UserBillingState,
    UserbotAccount,
    UserCommunity,
    UserDigestSettings,
)
from app.experiments import AA_CURRENT_LOGIC_EXPERIMENT_KEY, AA_VARIANTS
from app.csi_survey import CSI_MAX_SCORE, CSI_MIN_DELIVERIES
from app.news_time_survey import NEWS_TIME_BUCKET_SLUGS, parse_news_time_answer

AUDIT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", "/app/logs/audit.log"))
PRODUCT_METRICS_WINDOW_DAYS = int(os.getenv("PRODUCT_METRICS_WINDOW_DAYS", "7"))
AUDIT_RETENTION_DAYS = int(os.getenv("PRODUCT_METRICS_AUDIT_RETENTION_DAYS", "35"))


@dataclass(frozen=True, slots=True)
class SendRecord:
    sent_at: datetime
    user_id: int
    post_id: int
    event: str | None = None
    telegram_id: int | None = None
    message_id: int | None = None
    delivery_result: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentExposureRecord:
    sent_at: datetime
    user_id: int
    post_id: int
    variant: str


class AuditSendCache:
    def __init__(self, path: Path, retention_days: int = AUDIT_RETENTION_DAYS):
        self.path = path
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._inode: int | None = None
        self._offset = 0
        self._records: dict[tuple[int, int], SendRecord] = {}

    def get_records(self, *, since: datetime) -> list[SendRecord]:
        with self._lock:
            self._refresh_locked()
            return [record for record in self._records.values() if record.sent_at >= since]

    def _refresh_locked(self) -> None:
        if not self.path.exists():
            self._records.clear()
            self._inode = None
            self._offset = 0
            return

        stat = self.path.stat()
        inode = int(getattr(stat, "st_ino", 0) or 0)
        file_size = int(stat.st_size or 0)

        if self._inode != inode or file_size < self._offset:
            self._records.clear()
            self._offset = 0
            self._inode = inode

        with self.path.open("r", encoding="utf-8") as fh:
            fh.seek(self._offset)
            for line in fh:
                self._consume_line_locked(line)
            self._offset = fh.tell()

        self._prune_locked()

    def _consume_line_locked(self, line: str) -> None:
        raw = (line or "").strip()
        if not raw:
            return

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return

        event = str(payload.get("event") or "").strip()
        if event not in {
            "send.ok",
            "send.fallback_text_ok",
            "storyline_send.success",
            "storyline_followup.success",
        }:
            return

        try:
            user_id = int(payload["user_id"])
            post_id = int(payload["post_id"])
        except (KeyError, TypeError, ValueError):
            return

        sent_at = _parse_dt(payload.get("ts"))
        if sent_at is None:
            return

        key = (user_id, post_id)
        current = self._records.get(key)
        if current is not None and current.sent_at <= sent_at:
            return

        self._records[key] = SendRecord(
            sent_at=sent_at,
            user_id=user_id,
            post_id=post_id,
            event=event,
            telegram_id=_to_optional_int(payload.get("telegram_id")),
            message_id=_to_optional_int(payload.get("message_id")),
            delivery_result=str(payload.get("delivery_result") or "").strip() or None,
        )

    def _prune_locked(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, self.retention_days))
        stale_keys = [key for key, record in self._records.items() if record.sent_at < cutoff]
        for key in stale_keys:
            self._records.pop(key, None)


AUDIT_SEND_CACHE = AuditSendCache(AUDIT_LOG_PATH)
STORYLINE_DELIVERY_EVENTS = {"storyline_send.success", "storyline_followup.success"}


def active_premium_subscribers_query(now: datetime):
    return select(func.count(func.distinct(PremiumEntitlementRecord.user_id))).where(
        PremiumEntitlementRecord.is_active.is_(True),
        PremiumEntitlementRecord.source_type == "payment",
        PremiumEntitlementRecord.starts_at <= now,
        PremiumEntitlementRecord.expires_at > now,
    )


def active_promo_premium_subscribers_query(now: datetime):
    return select(func.count(func.distinct(PremiumEntitlementRecord.user_id))).where(
        PremiumEntitlementRecord.is_active.is_(True),
        PremiumEntitlementRecord.source_type == "promo",
        PremiumEntitlementRecord.starts_at <= now,
        PremiumEntitlementRecord.expires_at > now,
    )


def _to_optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _as_float(value) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _finite_float(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return value


def _p95(values: Iterable[float]) -> float:
    ordered = sorted(float(v) for v in values if v is not None)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(v) for v in values if v is not None)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _window_start(now: datetime, days: int) -> datetime:
    return now - timedelta(days=max(1, days))


def compute_storyline_recipient_engagement_metrics(
    *,
    storyline_recipient_user_ids: Iterable[int],
    engaged_user_ids: Iterable[int],
) -> dict[str, float]:
    recipients = {int(user_id) for user_id in storyline_recipient_user_ids if user_id is not None}
    engaged = {int(user_id) for user_id in engaged_user_ids if user_id is not None}
    engaged_recipients = recipients & engaged
    recipient_count = len(recipients)
    return {
        "newshub_product_storyline_recipients_total_7d": float(recipient_count),
        "newshub_product_storyline_recipient_engagement_ratio_7d": (
            float(len(engaged_recipients)) / float(recipient_count) if recipient_count else 0.0
        ),
    }


async def record_product_metrics_daily_snapshot(
    session: AsyncSession,
    metrics: dict[str, float],
    *,
    now: datetime | None = None,
) -> None:
    """Один ряд на календарный день UTC; при каждом скрейпе обновляются значения за «сегодня»."""
    now = now or datetime.now(timezone.utc)
    day = now.astimezone(timezone.utc).date()
    sub = _finite_float(metrics.get("newshub_subscribers_total", 0.0))
    eng = _finite_float(metrics.get("newshub_product_engagement_subscribers_ratio_30d", 0.0))
    avg = _finite_float(metrics.get("newshub_product_avg_subscriptions_per_subscriber", 0.0))

    ins = insert(ProductMetricsDaily).values(
        snapshot_date=day,
        subscribers_total=sub,
        engagement_ratio_30d=eng,
        avg_subscriptions_per_subscriber=avg,
    )
    ins = ins.on_conflict_do_update(
        index_elements=["snapshot_date"],
        set_={
            "subscribers_total": ins.excluded.subscribers_total,
            "engagement_ratio_30d": ins.excluded.engagement_ratio_30d,
            "avg_subscriptions_per_subscriber": ins.excluded.avg_subscriptions_per_subscriber,
            "updated_at": func.now(),
        },
    )
    await session.execute(ins)


async def collect_product_metrics(session: AsyncSession) -> dict[str, float]:
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = _window_start(now, PRODUCT_METRICS_WINDOW_DAYS)
    previous_week_start = today_start - timedelta(days=14)
    previous_week_end = today_start - timedelta(days=7)
    thirty_start = now - timedelta(days=30)
    retention_30_prev_start = now - timedelta(days=60)
    retention_30_prev_end = now - timedelta(days=30)

    metrics: dict[str, float] = {}

    subscriptions_total = await session.scalar(select(func.count(UserCommunity.id)))
    subscribers_total = await session.scalar(select(func.count(func.distinct(UserCommunity.user_id))))
    premium_total = await session.scalar(active_premium_subscribers_query(now))
    promo_premium_total = await session.scalar(active_promo_premium_subscribers_query(now))
    digest_enabled_total = await session.scalar(
        select(func.count(func.distinct(UserDigestSettings.user_id))).where(
            UserDigestSettings.enabled.is_(True),
        )
    )
    channels_total = await session.scalar(select(func.count()).select_from(Community))
    posts_total = await session.scalar(select(func.count()).select_from(Post))
    filters_users_total = await session.scalar(
        select(func.count(User.id)).where(User.feed_filter.in_(("not_interesting", "only_fire")))
    )

    subs_f = float(subscribers_total or 0)
    prem_f = float(premium_total or 0)
    non_prem = subs_f - prem_f
    if non_prem <= 0:
        conversion_pct = 100.0 if prem_f > 0 else 0.0
    else:
        conversion_pct = prem_f / non_prem * 100.0

    metrics["newshub_subscriptions_active_total"] = float(subscriptions_total or 0)
    metrics["newshub_subscribers_total"] = subs_f
    metrics["newshub_premium_subscribers_total"] = prem_f
    metrics["newshub_promo_premium_subscribers_total"] = float(promo_premium_total or 0)
    metrics["newshub_digest_enabled_users_total"] = float(digest_enabled_total or 0)
    metrics["newshub_channels_total"] = float(channels_total or 0)
    metrics["newshub_posts_total"] = float(posts_total or 0)
    telegram_backlog_total = await session.scalar(
        select(func.count(TelegramOperation.id)).where(TelegramOperation.status.in_(("queued", "retrying")))
    )
    telegram_limited_accounts = await session.scalar(
        select(func.count(UserbotAccount.id)).where(UserbotAccount.flood_wait_until > now)
    )
    metrics["newshub_telegram_operation_backlog_total"] = float(telegram_backlog_total or 0)
    metrics["newshub_telegram_flood_wait_accounts_total"] = float(telegram_limited_accounts or 0)

    stale_payment_cutoff = now - timedelta(minutes=10)
    stale_pending_payments = await session.scalar(
        select(func.count()).select_from(PaymentOrder).where(
            PaymentOrder.status.in_(("new", "pending")),
            PaymentOrder.created_at < stale_payment_cutoff,
        )
    )
    confirmed_without_entitlement = await session.scalar(
        text(
            """
            SELECT count(*)
            FROM payment_orders po
            LEFT JOIN premium_entitlements pe
              ON pe.source_type = 'payment'
             AND pe.source_id = po.id
            WHERE po.status = 'confirmed'
              AND po.packs_count > 0
              AND pe.id IS NULL
            """
        )
    )
    metrics["newshub_billing_pending_payments_stale_total"] = float(stale_pending_payments or 0)
    metrics["newshub_billing_confirmed_without_entitlement_total"] = float(confirmed_without_entitlement or 0)
    metrics["newshub_product_conversion_premium_pct"] = conversion_pct
    metrics["newshub_product_avg_subscriptions_per_subscriber"] = (
        float(subscriptions_total or 0) / subs_f if subs_f > 0 else 0.0
    )
    metrics["newshub_product_filters_users_total"] = float(filters_users_total or 0)
    story_tracking_users_total = await session.scalar(
        select(func.count(func.distinct(UserStorylineFollow.user_id))).where(
            UserStorylineFollow.is_active.is_(True),
        )
    )
    metrics["newshub_product_story_tracking_users"] = float(story_tracking_users_total or 0)
    deliveries = AUDIT_SEND_CACHE.get_records(since=week_start)

    engagement_cte = """
        WITH engagement_events AS (
            SELECT user_id, created_at AS ts
            FROM reactions
            WHERE created_at IS NOT NULL
            UNION ALL
            SELECT user_id, COALESCE(last_clicked_at, first_clicked_at) AS ts
            FROM clicks
            WHERE COALESCE(last_clicked_at, first_clicked_at) IS NOT NULL
        )
    """

    retention_7d_subscribers_query = text(
        engagement_cte
        + """
        , previous_week_users AS (
            SELECT DISTINCT ee.user_id
            FROM engagement_events ee
            WHERE ee.ts >= :previous_week_start AND ee.ts < :previous_week_end
              AND EXISTS (SELECT 1 FROM users_communities uc WHERE uc.user_id = ee.user_id)
        ),
        current_week_users AS (
            SELECT DISTINCT ee.user_id
            FROM engagement_events ee
            WHERE ee.ts >= :week_start AND ee.ts < :now_ts
              AND EXISTS (SELECT 1 FROM users_communities uc WHERE uc.user_id = ee.user_id)
        )
        SELECT CASE
            WHEN (SELECT COUNT(*) FROM previous_week_users) = 0 THEN 0.0
            ELSE (
                SELECT COUNT(*)::float
                FROM previous_week_users pwu
                JOIN current_week_users cwu USING (user_id)
            ) / (SELECT COUNT(*)::float FROM previous_week_users)
        END AS value
        """
    )
    retention_7d_sub = await session.scalar(
        retention_7d_subscribers_query,
        {
            "previous_week_start": previous_week_start,
            "previous_week_end": previous_week_end,
            "week_start": week_start,
            "now_ts": now,
        },
    )
    r7 = _as_float(retention_7d_sub)
    metrics["newshub_product_retention_7d_subscribers_ratio"] = r7
    # Короткий алиас для PromQL/Grafana (то же значение, что и ..._subscribers_ratio).
    metrics["newshub_product_retention_7d"] = r7

    retention_30d_subscribers_query = text(
        engagement_cte
        + """
        , prev_period_users AS (
            SELECT DISTINCT ee.user_id
            FROM engagement_events ee
            WHERE ee.ts >= :retention_30_prev_start AND ee.ts < :retention_30_prev_end
              AND EXISTS (SELECT 1 FROM users_communities uc WHERE uc.user_id = ee.user_id)
        ),
        curr_period_users AS (
            SELECT DISTINCT ee.user_id
            FROM engagement_events ee
            WHERE ee.ts >= :retention_30_curr_start AND ee.ts < :now_ts
              AND EXISTS (SELECT 1 FROM users_communities uc WHERE uc.user_id = ee.user_id)
        )
        SELECT CASE
            WHEN (SELECT COUNT(*) FROM prev_period_users) = 0 THEN 0.0
            ELSE (
                SELECT COUNT(*)::float
                FROM prev_period_users ppu
                JOIN curr_period_users cpu USING (user_id)
            ) / (SELECT COUNT(*)::float FROM prev_period_users)
        END AS value
        """
    )
    retention_30d_sub = await session.scalar(
        retention_30d_subscribers_query,
        {
            "retention_30_prev_start": retention_30_prev_start,
            "retention_30_prev_end": retention_30_prev_end,
            "retention_30_curr_start": retention_30_prev_end,
            "now_ts": now,
        },
    )
    metrics["newshub_product_retention_30d_subscribers_ratio"] = _as_float(retention_30d_sub)

    engagement_subscribers_7d_query = text(
        engagement_cte
        + """
        , engaged AS (
            SELECT DISTINCT ee.user_id
            FROM engagement_events ee
            WHERE ee.ts >= :week_start AND ee.ts < :now_ts
              AND EXISTS (SELECT 1 FROM users_communities uc WHERE uc.user_id = ee.user_id)
        ),
        subs AS (
            SELECT COUNT(DISTINCT user_id)::float AS c FROM users_communities
        )
        SELECT COALESCE((SELECT COUNT(*)::float FROM engaged) / NULLIF((SELECT c FROM subs), 0.0), 0.0) AS value
        """
    )
    eng_7 = await session.scalar(engagement_subscribers_7d_query, {"week_start": week_start, "now_ts": now})
    metrics["newshub_product_engagement_subscribers_ratio_7d"] = _as_float(eng_7)
    metrics.update(await _collect_storyline_recipient_engagement_metrics(session, deliveries, week_start, now))

    engagement_subscribers_30d_query = text(
        engagement_cte
        + """
        , engaged AS (
            SELECT DISTINCT ee.user_id
            FROM engagement_events ee
            WHERE ee.ts >= :thirty_start AND ee.ts < :now_ts
              AND EXISTS (SELECT 1 FROM users_communities uc WHERE uc.user_id = ee.user_id)
        ),
        subs AS (
            SELECT COUNT(DISTINCT user_id)::float AS c FROM users_communities
        )
        SELECT COALESCE((SELECT COUNT(*)::float FROM engaged) / NULLIF((SELECT c FROM subs), 0.0), 0.0) AS value
        """
    )
    eng_30 = await session.scalar(engagement_subscribers_30d_query, {"thirty_start": thirty_start, "now_ts": now})
    metrics["newshub_product_engagement_subscribers_ratio_30d"] = _as_float(eng_30)

    positive_ratio_query = select(
        func.coalesce(
            func.sum(case((Reaction.reaction.in_(("fire", "up")), 1), else_=0)),
            0,
        ),
        func.coalesce(
            func.sum(case((Reaction.reaction == "down", 1), else_=0)),
            0,
        ),
        func.count(Reaction.id),
    ).where(Reaction.created_at >= week_start)
    positive_count, negative_count, reaction_count = (await session.execute(positive_ratio_query)).one()
    metrics["newshub_product_positive_ratio_7d"] = (
        float(positive_count or 0.0) / float(reaction_count or 1) if reaction_count else 0.0
    )
    metrics["newshub_product_negative_ratio_7d"] = (
        float(negative_count or 0.0) / float(reaction_count or 1) if reaction_count else 0.0
    )

    delivery_metrics = await _collect_delivery_metrics(session, deliveries, week_start)
    metrics.update(delivery_metrics)
    metrics.update(await _collect_aa_experiment_metrics(session, week_start))
    assistant_metrics = await _collect_assistant_metrics(session, week_start)
    metrics.update(assistant_metrics)
    metrics.update(await _collect_news_time_survey_metrics(session, thirty_start, now))
    metrics.update(await _collect_csi_metrics(session, week_start, thirty_start, now))

    return metrics


async def _collect_csi_metrics(
    session: AsyncSession,
    week_start: datetime,
    thirty_start: datetime,
    now: datetime,
) -> dict[str, float]:
    week_rows = (
        await session.execute(
            select(UserCsiSurvey.score).where(
                UserCsiSurvey.answered_at.is_not(None),
                UserCsiSurvey.delivery_count_at_prompt >= CSI_MIN_DELIVERIES,
                UserCsiSurvey.answered_at >= week_start,
                UserCsiSurvey.answered_at < now,
            )
        )
    ).all()
    thirty_rows = (
        await session.execute(
            select(UserCsiSurvey.score).where(
                UserCsiSurvey.answered_at.is_not(None),
                UserCsiSurvey.delivery_count_at_prompt >= CSI_MIN_DELIVERIES,
                UserCsiSurvey.answered_at >= thirty_start,
                UserCsiSurvey.answered_at < now,
            )
        )
    ).all()
    return compute_csi_metrics(
        scores_7d=[row[0] for row in week_rows],
        scores_30d=[row[0] for row in thirty_rows],
    )


def compute_csi_metrics(
    *,
    scores_7d: Iterable[int | float | None],
    scores_30d: Iterable[int | float | None],
) -> dict[str, float]:
    def _score_pct(scores: Iterable[int | float | None]) -> tuple[float, float]:
        clean = [float(score) for score in scores if score is not None]
        if not clean:
            return 0.0, 0.0
        return (sum(clean) / len(clean) / float(CSI_MAX_SCORE) * 100.0, float(len(clean)))

    csi_7d, responses_7d = _score_pct(scores_7d)
    csi_30d, responses_30d = _score_pct(scores_30d)
    return {
        "newshub_product_csi_7d": csi_7d,
        "newshub_product_csi_responses_total_7d": responses_7d,
        "newshub_product_csi_30d": csi_30d,
        "newshub_product_csi_responses_total_30d": responses_30d,
    }


async def _collect_storyline_recipient_engagement_metrics(
    session: AsyncSession,
    deliveries: Iterable[SendRecord],
    week_start: datetime,
    now: datetime,
) -> dict[str, float]:
    _ = deliveries  # Storyline deliveries are persisted; audit log is only a fallback for generic send metrics.
    storyline_event_ts = func.coalesce(StorylineUpdateEvent.updated_at, StorylineUpdateEvent.created_at)
    storyline_rows = (
        await session.execute(
            select(StorylineUpdateEvent.user_id)
            .distinct()
            .where(
                StorylineUpdateEvent.telegram_message_id.is_not(None),
                storyline_event_ts >= week_start,
                storyline_event_ts < now,
            )
        )
    ).all()
    storyline_user_ids = sorted(
        {int(row[0]) for row in storyline_rows if row[0] is not None}
    )
    if not storyline_user_ids:
        return compute_storyline_recipient_engagement_metrics(
            storyline_recipient_user_ids=[],
            engaged_user_ids=[],
        )

    reaction_rows = (
        await session.execute(
            select(Reaction.user_id)
            .distinct()
            .where(
                Reaction.user_id.in_(storyline_user_ids),
                Reaction.created_at >= week_start,
                Reaction.created_at < now,
            )
        )
    ).all()
    click_rows = (
        await session.execute(
            select(Click.user_id)
            .distinct()
            .where(
                Click.user_id.in_(storyline_user_ids),
                func.coalesce(Click.last_clicked_at, Click.first_clicked_at) >= week_start,
                func.coalesce(Click.last_clicked_at, Click.first_clicked_at) < now,
            )
        )
    ).all()
    engaged_user_ids = {row[0] for row in reaction_rows} | {row[0] for row in click_rows}
    return compute_storyline_recipient_engagement_metrics(
        storyline_recipient_user_ids=storyline_user_ids,
        engaged_user_ids=engaged_user_ids,
    )


async def _collect_news_time_survey_metrics(
    session: AsyncSession,
    since: datetime,
    now: datetime,
) -> dict[str, float]:
    completed_rows = (
        await session.execute(
            select(
                UserNewsTimeSurvey.baseline_answer,
                UserNewsTimeSurvey.baseline_minutes,
                UserNewsTimeSurvey.current_answer,
                UserNewsTimeSurvey.current_minutes,
            )
            .join(
                UserCsiSurvey,
                UserCsiSurvey.user_id == UserNewsTimeSurvey.user_id,
            )
            .where(
                UserCsiSurvey.answered_at.is_not(None),
                UserNewsTimeSurvey.current_answered_at.is_not(None),
                UserNewsTimeSurvey.current_answered_at >= since,
                UserNewsTimeSurvey.current_answered_at < now,
            )
        )
    ).all()
    time_saved_rows = (
        await session.execute(
            select(UserNewsTimeSurvey.time_saved_answer).where(
                UserNewsTimeSurvey.time_saved_answered_at.is_not(None),
                UserNewsTimeSurvey.time_saved_answered_at >= since,
                UserNewsTimeSurvey.time_saved_answered_at < now,
            )
        )
    ).all()
    return compute_news_time_survey_metrics(
        completed_rows=completed_rows,
        time_saved_answers=[row[0] for row in time_saved_rows],
    )


def compute_news_time_survey_metrics(
    *,
    completed_rows: Iterable[tuple[str | None, int | None, str | None, int | None]],
    time_saved_answers: Iterable[str | None],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for slug in NEWS_TIME_BUCKET_SLUGS:
        metrics[f"newshub_product_news_time_baseline_{slug}_total_30d"] = 0.0
        metrics[f"newshub_product_news_time_current_{slug}_total_30d"] = 0.0

    baseline_minutes: list[float] = []
    current_minutes: list[float] = []
    for baseline_answer, baseline_min, current_answer, current_min in completed_rows:
        baseline = parse_news_time_answer(baseline_answer)
        current = parse_news_time_answer(current_answer)
        if baseline:
            metrics[f"newshub_product_news_time_baseline_{baseline.slug}_total_30d"] += 1.0
        if current:
            metrics[f"newshub_product_news_time_current_{current.slug}_total_30d"] += 1.0
        if baseline_min is not None:
            baseline_minutes.append(float(baseline_min))
        if current_min is not None:
            current_minutes.append(float(current_min))

    baseline_avg_hours = (sum(baseline_minutes) / len(baseline_minutes) / 60.0) if baseline_minutes else 0.0
    current_avg_hours = (sum(current_minutes) / len(current_minutes) / 60.0) if current_minutes else 0.0
    metrics["newshub_product_news_time_baseline_avg_hours_30d"] = baseline_avg_hours
    metrics["newshub_product_news_time_current_avg_hours_30d"] = current_avg_hours
    metrics["newshub_product_news_time_avg_change_pct_30d"] = (
        ((current_avg_hours - baseline_avg_hours) / baseline_avg_hours) * 100.0
        if baseline_avg_hours > 0
        else 0.0
    )

    saved_answers = [str(answer or "").strip().lower() for answer in time_saved_answers if answer]
    yes_total = sum(1 for answer in saved_answers if answer == "yes")
    metrics["newshub_product_news_time_yes_conversion_pct_30d"] = (
        float(yes_total) / float(len(saved_answers)) * 100.0 if saved_answers else 0.0
    )
    return metrics


async def _collect_assistant_metrics(session: AsyncSession, since: datetime) -> dict[str, float]:
    query = select(
        func.count(BotAssistantRequest.id),
        func.coalesce(func.sum(case((BotAssistantRequest.understood_at.is_not(None), 1), else_=0)), 0),
        func.coalesce(func.sum(case((BotAssistantRequest.clarified_at.is_not(None), 1), else_=0)), 0),
        func.coalesce(func.sum(case((BotAssistantRequest.proposed_at.is_not(None), 1), else_=0)), 0),
        func.coalesce(func.sum(case((BotAssistantRequest.status == "applied", 1), else_=0)), 0),
        func.coalesce(func.sum(case((BotAssistantRequest.status == "cancelled", 1), else_=0)), 0),
        func.coalesce(func.sum(case((BotAssistantRequest.status == "failed", 1), else_=0)), 0),
    ).where(BotAssistantRequest.received_at >= since)
    row = (await session.execute(query)).one()
    return compute_assistant_metrics(
        requests_total=row[0],
        understood_total=row[1],
        clarification_total=row[2],
        action_proposed_total=row[3],
        action_applied_total=row[4],
        action_cancelled_total=row[5],
        failed_total=row[6],
    )


def compute_assistant_metrics(
    *,
    requests_total: int | float,
    understood_total: int | float,
    clarification_total: int | float,
    action_proposed_total: int | float,
    action_applied_total: int | float,
    action_cancelled_total: int | float,
    failed_total: int | float,
) -> dict[str, float]:
    requests = _as_float(requests_total)
    understood = _as_float(understood_total)
    proposed = _as_float(action_proposed_total)
    applied = _as_float(action_applied_total)
    return {
        "newshub_assistant_requests_total_7d": requests,
        "newshub_assistant_understood_total_7d": understood,
        "newshub_assistant_clarification_total_7d": _as_float(clarification_total),
        "newshub_assistant_action_proposed_total_7d": proposed,
        "newshub_assistant_action_applied_total_7d": applied,
        "newshub_assistant_action_cancelled_total_7d": _as_float(action_cancelled_total),
        "newshub_assistant_failed_total_7d": _as_float(failed_total),
        "newshub_assistant_understood_ratio_7d": understood / requests if requests > 0 else 0.0,
        "newshub_assistant_action_success_ratio_7d": applied / proposed if proposed > 0 else 0.0,
    }


def compute_aa_experiment_metrics(
    exposures: list[ExperimentExposureRecord],
    reactions_by_key: dict[tuple[int, int], tuple[str | None, datetime | None]],
    clicks_by_key: dict[tuple[int, int], datetime | None],
    *,
    assignment_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    prefix = "newshub_experiment_aa_current_logic"
    metrics: dict[str, float] = {}
    assignment_counts = assignment_counts or {}

    for variant in AA_VARIANTS:
        metrics[f"{prefix}_assignments_{variant}_total"] = float(assignment_counts.get(variant, 0))
        metrics[f"{prefix}_exposures_{variant}_total_7d"] = 0.0
        metrics[f"{prefix}_ctr_source_{variant}_7d"] = 0.0
        metrics[f"{prefix}_reaction_delivery_{variant}_7d"] = 0.0
        metrics[f"{prefix}_positive_ratio_{variant}_7d"] = 0.0
        metrics[f"{prefix}_time_to_engage_minutes_avg_{variant}_7d"] = 0.0

    assignment_total = sum(max(0, int(v)) for v in assignment_counts.values())
    if assignment_total:
        a_assignments = float(assignment_counts.get("a", 0))
        b_assignments = float(assignment_counts.get("b", 0))
        metrics[f"{prefix}_assignment_srm_abs_pct"] = abs(a_assignments - b_assignments) / assignment_total * 100.0
        metrics[f"{prefix}_assignment_srm_p_value"] = _two_bucket_srm_p_value(a_assignments, b_assignments)
    else:
        metrics[f"{prefix}_assignment_srm_abs_pct"] = 0.0
        metrics[f"{prefix}_assignment_srm_p_value"] = 1.0

    by_variant: dict[str, list[ExperimentExposureRecord]] = {variant: [] for variant in AA_VARIANTS}
    for exposure in exposures:
        variant = str(exposure.variant or "").strip().lower()
        if variant in by_variant:
            by_variant[variant].append(exposure)

    exposure_total = len(exposures)
    if exposure_total:
        a_exposures = float(len(by_variant.get("a", [])))
        b_exposures = float(len(by_variant.get("b", [])))
        metrics[f"{prefix}_exposure_srm_abs_pct_7d"] = abs(a_exposures - b_exposures) / exposure_total * 100.0
        metrics[f"{prefix}_exposure_srm_p_value_7d"] = _two_bucket_srm_p_value(a_exposures, b_exposures)
    else:
        metrics[f"{prefix}_exposure_srm_abs_pct_7d"] = 0.0
        metrics[f"{prefix}_exposure_srm_p_value_7d"] = 1.0

    for variant, variant_exposures in by_variant.items():
        total = len(variant_exposures)
        metrics[f"{prefix}_exposures_{variant}_total_7d"] = float(total)
        if not total:
            continue

        clicked = 0
        reaction_on_delivery = 0
        positive_reactions = 0
        reactions_total = 0
        engagement_minutes: list[float] = []
        for exposure in variant_exposures:
            key = (exposure.user_id, exposure.post_id)
            reaction_value, reaction_ts = reactions_by_key.get(key, (None, None))
            click_ts = clicks_by_key.get(key)
            if click_ts and click_ts >= exposure.sent_at:
                clicked += 1
            if reaction_ts and reaction_ts >= exposure.sent_at:
                reactions_total += 1
                reaction_on_delivery += 1
                if reaction_value in {"fire", "up"}:
                    positive_reactions += 1

            event_times = [ts for ts in (reaction_ts, click_ts) if ts and ts >= exposure.sent_at]
            if event_times:
                first_engagement = min(event_times)
                engagement_minutes.append(max(0.0, (first_engagement - exposure.sent_at).total_seconds() / 60.0))

        metrics[f"{prefix}_ctr_source_{variant}_7d"] = clicked / total
        metrics[f"{prefix}_reaction_delivery_{variant}_7d"] = reaction_on_delivery / total
        metrics[f"{prefix}_positive_ratio_{variant}_7d"] = (
            positive_reactions / reactions_total if reactions_total else 0.0
        )
        metrics[f"{prefix}_time_to_engage_minutes_avg_{variant}_7d"] = (
            sum(engagement_minutes) / len(engagement_minutes) if engagement_minutes else 0.0
        )

    return metrics


def _two_bucket_srm_p_value(left: float, right: float) -> float:
    total = float(left + right)
    if total <= 0:
        return 1.0
    expected = total / 2.0
    chi_square = ((left - expected) ** 2 + (right - expected) ** 2) / expected
    return max(0.0, min(1.0, math.erfc(math.sqrt(chi_square / 2.0))))


async def _collect_aa_experiment_metrics(session: AsyncSession, since: datetime) -> dict[str, float]:
    assignment_rows = (
        await session.execute(
            select(ExperimentAssignment.variant, func.count(ExperimentAssignment.id))
            .where(ExperimentAssignment.experiment_key == AA_CURRENT_LOGIC_EXPERIMENT_KEY)
            .group_by(ExperimentAssignment.variant)
        )
    ).all()
    assignment_counts = {str(variant or "").strip().lower(): int(count or 0) for variant, count in assignment_rows}

    exposure_rows = (
        await session.execute(
            select(
                ExperimentExposure.user_id,
                ExperimentExposure.post_id,
                ExperimentExposure.variant,
                ExperimentExposure.sent_at,
            ).where(
                ExperimentExposure.experiment_key == AA_CURRENT_LOGIC_EXPERIMENT_KEY,
                ExperimentExposure.sent_at >= since,
            )
        )
    ).all()
    exposures = [
        ExperimentExposureRecord(
            user_id=int(user_id),
            post_id=int(post_id),
            variant=str(variant or "").strip().lower(),
            sent_at=_parse_dt(sent_at) or since,
        )
        for user_id, post_id, variant, sent_at in exposure_rows
    ]
    if not exposures:
        return compute_aa_experiment_metrics([], {}, {}, assignment_counts=assignment_counts)

    user_post_keys = {(record.user_id, record.post_id) for record in exposures}
    post_ids = {record.post_id for record in exposures}
    user_ids = {record.user_id for record in exposures}

    reaction_rows = (
        await session.execute(
            select(Reaction.user_id, Reaction.post_id, Reaction.reaction, Reaction.created_at).where(
                Reaction.created_at >= since,
                Reaction.post_id.in_(post_ids),
                Reaction.user_id.in_(user_ids),
            )
        )
    ).all()
    click_rows = (
        await session.execute(
            select(
                Click.user_id,
                Click.post_id,
                Click.first_clicked_at,
                Click.last_clicked_at,
            ).where(
                func.coalesce(Click.last_clicked_at, Click.first_clicked_at) >= since,
                Click.user_id.in_(user_ids),
                Click.post_id.in_(post_ids),
            )
        )
    ).all()

    reactions_by_key: dict[tuple[int, int], tuple[str | None, datetime | None]] = {}
    for user_id, post_id, reaction, created_at in reaction_rows:
        key = (int(user_id), int(post_id))
        if key in user_post_keys:
            reactions_by_key[key] = (str(reaction or "").strip().lower() or None, _parse_dt(created_at))

    clicks_by_key: dict[tuple[int, int], datetime | None] = {}
    for user_id, post_id, first_clicked_at, last_clicked_at in click_rows:
        key = (int(user_id), int(post_id))
        if key not in user_post_keys:
            continue
        first_ts = _parse_dt(first_clicked_at)
        last_ts = _parse_dt(last_clicked_at)
        click_ts = first_ts or last_ts
        if click_ts is None:
            continue
        current = clicks_by_key.get(key)
        if current is None or click_ts < current:
            clicks_by_key[key] = click_ts

    return compute_aa_experiment_metrics(
        exposures,
        reactions_by_key,
        clicks_by_key,
        assignment_counts=assignment_counts,
    )


async def _collect_delivery_metrics(
    session: AsyncSession,
    deliveries: list[SendRecord],
    since: datetime,
) -> dict[str, float]:
    if not deliveries:
        return compute_delivery_metrics([], {}, {}, {}, {})

    user_post_keys = {(record.user_id, record.post_id) for record in deliveries}
    post_ids = {record.post_id for record in deliveries}
    delivery_user_ids = {record.user_id for record in deliveries}

    # "Personalization" cohort:
    # - users who enabled explicit forwarding filters (only_fire / not_interesting)
    # - OR users who enabled at least one instruction rule (Premium AI filter)
    user_filters_rows = (
        await session.execute(
            select(User.id, User.feed_filter).where(User.id.in_(delivery_user_ids))
        )
    ).all()
    users_with_forward_filters = {
        int(user_id)
        for user_id, feed_filter in user_filters_rows
        if str(feed_filter or "").strip().lower() in ("only_fire", "not_interesting")
    }
    users_with_ai_filter = set(
        (
            await session.execute(
                select(func.distinct(UserInstructionRule.user_id)).where(
                    UserInstructionRule.user_id.in_(delivery_user_ids),
                    UserInstructionRule.is_enabled.is_(True),
                )
            )
        )
        .scalars()
        .all()
    )
    personalized_user_ids = users_with_forward_filters | {int(uid) for uid in users_with_ai_filter}

    reaction_rows = (
        await session.execute(
            select(Reaction.user_id, Reaction.post_id, Reaction.reaction, Reaction.created_at).where(
                Reaction.created_at >= since,
                Reaction.post_id.in_(post_ids),
            )
        )
    ).all()
    click_rows = (
        await session.execute(
            select(
                Click.user_id,
                Click.post_id,
                Click.first_clicked_at,
                Click.last_clicked_at,
            ).where(
                func.coalesce(Click.last_clicked_at, Click.first_clicked_at) >= since,
                Click.user_id.in_(delivery_user_ids),
            )
        )
    ).all()
    interaction_rows = (
        await session.execute(
            select(
                PostInteraction.user_id,
                PostInteraction.post_id,
                PostInteraction.first_interacted_at,
                PostInteraction.last_interacted_at,
            ).where(
                func.coalesce(PostInteraction.last_interacted_at, PostInteraction.first_interacted_at) >= since,
                PostInteraction.user_id.in_(delivery_user_ids),
            )
        )
    ).all()
    post_rows = (
        await session.execute(select(Post.id, Post.timestamp).where(Post.id.in_(post_ids)))
    ).all()

    reactions_by_key: dict[tuple[int, int], tuple[str | None, datetime | None]] = {}
    for user_id, post_id, reaction, created_at in reaction_rows:
        key = (int(user_id), int(post_id))
        if key not in user_post_keys:
            continue
        reactions_by_key[key] = (str(reaction or "").strip().lower() or None, _parse_dt(created_at))

    clicks_by_key: dict[tuple[int, int], datetime | None] = {}
    for user_id, post_id, first_clicked_at, last_clicked_at in click_rows:
        key = (int(user_id), int(post_id))
        if key not in user_post_keys:
            continue
        first_ts = _parse_dt(first_clicked_at)
        last_ts = _parse_dt(last_clicked_at)
        click_ts = first_ts or last_ts
        if click_ts is None:
            continue
        current = clicks_by_key.get(key)
        if current is None or click_ts < current:
            clicks_by_key[key] = click_ts

    interactions_by_key: dict[tuple[int, int], datetime | None] = {}
    for user_id, post_id, first_interacted_at, last_interacted_at in interaction_rows:
        key = (int(user_id), int(post_id))
        if key not in user_post_keys:
            continue
        first_ts = _parse_dt(first_interacted_at)
        last_ts = _parse_dt(last_interacted_at)
        interaction_ts = first_ts or last_ts
        if interaction_ts is None:
            continue
        current = interactions_by_key.get(key)
        if current is None or interaction_ts < current:
            interactions_by_key[key] = interaction_ts

    post_timestamps = {int(post_id): _parse_dt(timestamp) for post_id, timestamp in post_rows}

    return compute_delivery_metrics(
        deliveries,
        reactions_by_key,
        clicks_by_key,
        interactions_by_key,
        post_timestamps,
        personalized_user_ids=personalized_user_ids,
    )


def compute_delivery_metrics(
    deliveries: list[SendRecord],
    reactions_by_key: dict[tuple[int, int], tuple[str | None, datetime | None]],
    clicks_by_key: dict[tuple[int, int], datetime | None],
    interactions_by_key: dict[tuple[int, int], datetime | None],
    post_timestamps: dict[int, datetime | None],
    *,
    personalized_user_ids: set[int] | None = None,
) -> dict[str, float]:
    metrics = {
        "newshub_product_content_engagement_rate_7d": 0.0,
        "newshub_product_precision_at_send_7d": 0.0,
        "newshub_product_regret_rate_7d": 0.0,
        "newshub_product_time_to_engage_minutes_avg_7d": 0.0,
        "newshub_product_ctr_7d": 0.0,
        "newshub_product_ctr_source_7d": 0.0,
        "newshub_product_ctr_reaction_delivery_7d": 0.0,
        "newshub_product_button_interaction_delivery_7d": 0.0,
        "newshub_product_send_accuracy_pct_7d": 0.0,
        "newshub_product_send_accuracy_personalized_pct_7d": 0.0,
        "newshub_product_message_cap_daily_avg_7d": 0.0,
        "newshub_product_latency_seconds_p95_7d": 0.0,
        "newshub_product_latency_seconds_median_7d": 0.0,
        "newshub_product_deliveries_total_7d": 0.0,
    }
    if not deliveries:
        return metrics

    metrics["newshub_product_deliveries_total_7d"] = float(len(deliveries))

    engaged = 0
    positive = 0
    regret = 0
    personalized_total = 0
    personalized_regret = 0
    clicked = 0
    reaction_on_delivery = 0
    button_interaction_on_delivery = 0
    engagement_minutes: list[float] = []
    latency_seconds: list[float] = []
    sends_per_user_day: dict[tuple[int, datetime], int] = {}

    personalized_users = personalized_user_ids or set()
    for delivery in deliveries:
        key = (delivery.user_id, delivery.post_id)
        reaction_value, reaction_ts = reactions_by_key.get(key, (None, None))
        click_ts = clicks_by_key.get(key)
        interaction_ts = interactions_by_key.get(key)

        event_times = [ts for ts in (reaction_ts, click_ts, interaction_ts) if ts and ts >= delivery.sent_at]
        first_engagement = min(event_times) if event_times else None

        if reaction_value == "down" and reaction_ts and reaction_ts >= delivery.sent_at:
            regret += 1
            if delivery.user_id in personalized_users:
                personalized_regret += 1
        elif reaction_value in {"up", "fire"} or (click_ts and click_ts >= delivery.sent_at):
            positive += 1

        if delivery.user_id in personalized_users:
            personalized_total += 1

        if click_ts and click_ts >= delivery.sent_at:
            clicked += 1
        if reaction_ts and reaction_ts >= delivery.sent_at:
            reaction_on_delivery += 1
        if interaction_ts and interaction_ts >= delivery.sent_at:
            button_interaction_on_delivery += 1
        if first_engagement is not None:
            engaged += 1
            engagement_minutes.append(max(0.0, (first_engagement - delivery.sent_at).total_seconds() / 60.0))

        post_ts = post_timestamps.get(delivery.post_id)
        if post_ts and post_ts <= delivery.sent_at:
            latency_seconds.append(max(0.0, (delivery.sent_at - post_ts).total_seconds()))

        day_bucket = delivery.sent_at.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        sends_per_user_day[(delivery.user_id, day_bucket)] = sends_per_user_day.get((delivery.user_id, day_bucket), 0) + 1

    total_deliveries = float(len(deliveries))
    metrics["newshub_product_content_engagement_rate_7d"] = engaged / total_deliveries
    metrics["newshub_product_precision_at_send_7d"] = positive / total_deliveries
    metrics["newshub_product_regret_rate_7d"] = regret / total_deliveries
    metrics["newshub_product_ctr_source_7d"] = clicked / total_deliveries
    metrics["newshub_product_ctr_7d"] = metrics["newshub_product_ctr_source_7d"]
    metrics["newshub_product_ctr_reaction_delivery_7d"] = reaction_on_delivery / total_deliveries
    metrics["newshub_product_button_interaction_delivery_7d"] = button_interaction_on_delivery / total_deliveries
    metrics["newshub_product_send_accuracy_pct_7d"] = (1.0 - regret / total_deliveries) * 100.0
    if personalized_total > 0:
        metrics["newshub_product_send_accuracy_personalized_pct_7d"] = (
            (1.0 - float(personalized_regret) / float(personalized_total)) * 100.0
        )
    metrics["newshub_product_time_to_engage_minutes_avg_7d"] = (
        sum(engagement_minutes) / len(engagement_minutes) if engagement_minutes else 0.0
    )
    metrics["newshub_product_latency_seconds_p95_7d"] = _p95(latency_seconds)
    metrics["newshub_product_latency_seconds_median_7d"] = _median(latency_seconds)
    metrics["newshub_product_message_cap_daily_avg_7d"] = (
        sum(sends_per_user_day.values()) / len(sends_per_user_day) if sends_per_user_day else 0.0
    )
    return metrics


def render_prometheus_metrics(metrics: dict[str, float]) -> str:
    metadata = {
        "newshub_assistant_action_applied_total_7d": "Запросы помощника за 7 дней, завершившиеся успешным применением действия.",
        "newshub_assistant_action_cancelled_total_7d": "Запросы помощника за 7 дней, отменённые пользователем на подтверждении.",
        "newshub_assistant_action_proposed_total_7d": "Запросы помощника за 7 дней, где бот предложил мутирующее действие на подтверждение.",
        "newshub_assistant_action_success_ratio_7d": "Доля предложенных действий помощника за 7 дней, которые были успешно применены.",
        "newshub_assistant_clarification_total_7d": "Запросы помощника за 7 дней, где потребовалось уточнение.",
        "newshub_assistant_failed_total_7d": "Запросы помощника за 7 дней, завершившиеся ошибкой.",
        "newshub_assistant_requests_total_7d": "Свободные private text-запросы к помощнику за 7 дней.",
        "newshub_assistant_understood_ratio_7d": "Доля запросов помощника за 7 дней, которые NLU успешно понял.",
        "newshub_assistant_understood_total_7d": "Запросы помощника за 7 дней, успешно распознанные NLU.",
        "newshub_subscriptions_active_total": "Суммарное число активных подписок на каналы (строки users_communities).",
        "newshub_subscribers_total": "Пользователи с хотя бы одной подпиской на канал; не billing-метрика.",
        "newshub_premium_subscribers_total": "Платные пользователи с активным Premium на текущий момент (premium_entitlements.source_type = payment).",
        "newshub_promo_premium_subscribers_total": "Пользователи с активным Premium, полученным по промокоду (premium_entitlements.source_type = promo).",
        "newshub_billing_pending_payments_stale_total": "Pending/new payment_orders старше 10 минут.",
        "newshub_billing_confirmed_without_entitlement_total": "Confirmed payment_orders без соответствующей premium_entitlements записи.",
        "newshub_digest_enabled_users_total": "Пользователи с включённой доставкой дайджеста.",
        "newshub_channels_total": "Уникальные каналы (communities) в БД.",
        "newshub_posts_total": "Посты в БД.",
        "newshub_product_conversion_premium_pct": "Premium / (подписчики без Premium) * 100.",
        "newshub_product_avg_subscriptions_per_subscriber": "Среднее число подписок на пользователя среди имеющих подписку.",
        "newshub_product_retention_7d": "Retention 7d (доля), только пользователи с подпиской; дубликат для дашборда.",
        "newshub_product_retention_7d_subscribers_ratio": "Retention 7d только среди пользователей с подпиской.",
        "newshub_product_retention_30d_subscribers_ratio": "Retention 30d (скользящие окна 60–30 и 30–0 дней), только с подпиской.",
        "newshub_product_engagement_subscribers_ratio_7d": "Доля пользователей с подпиской с реакцией или кликом за 7 дней.",
        "newshub_product_engagement_subscribers_ratio_30d": "То же за 30 дней (для графика в Prometheus — шаг скрейпа).",
        "newshub_product_storyline_recipients_total_7d": "Пользователи, получившие хотя бы одну storyline-карточку за 7 дней.",
        "newshub_product_storyline_recipient_engagement_ratio_7d": "Доля пользователей, получивших хотя бы одну storyline-карточку за 7 дней, у которых была реакция или клик за 7 дней.",
        "newshub_product_filters_users_total": "Пользователи с фильтром «не интересные» или «только огненные» (не «Всё»).",
        "newshub_product_csi_7d": "CSI за 7 дней: средняя оценка / 10 * 100.",
        "newshub_product_csi_responses_total_7d": "Ответы CSI за 7 дней.",
        "newshub_product_csi_30d": "CSI за 30 дней: средняя оценка / 10 * 100.",
        "newshub_product_csi_responses_total_30d": "Ответы CSI за 30 дней.",
        "newshub_product_story_tracking_users": "Пользователи с активной подпиской на хотя бы один сюжет (users_storyline_follows.is_active=true).",
        "newshub_product_content_engagement_rate_7d": "Доля доставленных карточек за 7 дней с кликом или реакцией.",
        "newshub_product_precision_at_send_7d": "Доля доставок с позитивным исходом (fire/up или клик, без down).",
        "newshub_product_regret_rate_7d": "Доля доставок с реакцией down.",
        "newshub_product_time_to_engage_minutes_avg_7d": "Среднее время до первого взаимодействия по доставкам с откликом, минуты.",
        "newshub_product_ctr_7d": "CTR в источник: доля доставок с кликом по /r/ после отправки (все фичи, где есть трекинг).",
        "newshub_product_ctr_source_7d": "То же, что newshub_product_ctr_7d.",
        "newshub_product_ctr_reaction_delivery_7d": "Доля доставок с реакцией (кнопки) после отправки.",
        "newshub_product_button_interaction_delivery_7d": "Доля доставок с любым button interaction после отправки, включая storyline-кнопки.",
        "newshub_product_send_accuracy_pct_7d": "(1 - доля доставок с down) * 100%.",
        "newshub_product_send_accuracy_personalized_pct_7d": "Точность отправок среди пользователей с фильтрами пересылки или включённым AI-фильтром: (1 - доля доставок с down) * 100%.",
        "newshub_product_message_cap_daily_avg_7d": "Среднее число доставок на user-day за 7 дней.",
        "newshub_product_latency_seconds_p95_7d": "P95 latency post→send, секунды.",
        "newshub_product_latency_seconds_median_7d": "Медиана latency post→send, секунды.",
        "newshub_product_positive_ratio_7d": "Доля fire/up среди реакций за 7 дней.",
        "newshub_product_negative_ratio_7d": "Доля down среди реакций за 7 дней.",
        "newshub_product_deliveries_total_7d": "Доставки за 7 дней (audit send.ok).",
        "newshub_product_news_time_baseline_under_hour_total_30d": "Опрос экономии времени: baseline «до часа», completed cohort за 30 дней.",
        "newshub_product_news_time_baseline_1h_total_30d": "Опрос экономии времени: baseline «1 час», completed cohort за 30 дней.",
        "newshub_product_news_time_baseline_2h_total_30d": "Опрос экономии времени: baseline «2 часа», completed cohort за 30 дней.",
        "newshub_product_news_time_baseline_3h_total_30d": "Опрос экономии времени: baseline «3 часа», completed cohort за 30 дней.",
        "newshub_product_news_time_baseline_4h_total_30d": "Опрос экономии времени: baseline «4 часа», completed cohort за 30 дней.",
        "newshub_product_news_time_baseline_5h_total_30d": "Опрос экономии времени: baseline «5 часов», completed cohort за 30 дней.",
        "newshub_product_news_time_current_under_hour_total_30d": "Опрос экономии времени: current «до часа», completed cohort за 30 дней.",
        "newshub_product_news_time_current_1h_total_30d": "Опрос экономии времени: current «1 час», completed cohort за 30 дней.",
        "newshub_product_news_time_current_2h_total_30d": "Опрос экономии времени: current «2 часа», completed cohort за 30 дней.",
        "newshub_product_news_time_current_3h_total_30d": "Опрос экономии времени: current «3 часа», completed cohort за 30 дней.",
        "newshub_product_news_time_current_4h_total_30d": "Опрос экономии времени: current «4 часа», completed cohort за 30 дней.",
        "newshub_product_news_time_current_5h_total_30d": "Опрос экономии времени: current «5 часов», completed cohort за 30 дней.",
        "newshub_product_news_time_baseline_avg_hours_30d": "Среднее baseline-время в часах; «до часа» считается как 0.5.",
        "newshub_product_news_time_current_avg_hours_30d": "Среднее current-время в часах; «до часа» считается как 0.5.",
        "newshub_product_news_time_avg_change_pct_30d": "Процент изменения среднего времени current к baseline.",
        "newshub_product_news_time_yes_conversion_pct_30d": "Доля ответов «Да» среди ответивших на второй вопрос за 30 дней.",
        "newshub_experiment_aa_current_logic_assignments_a_total": "A/A current logic: пользователи, назначенные в группу A.",
        "newshub_experiment_aa_current_logic_assignments_b_total": "A/A current logic: пользователи, назначенные в группу B.",
        "newshub_experiment_aa_current_logic_assignment_srm_abs_pct": "A/A current logic: абсолютный дисбаланс assignment A/B, проценты.",
        "newshub_experiment_aa_current_logic_assignment_srm_p_value": "A/A current logic: p-value SRM для assignment A/B.",
        "newshub_experiment_aa_current_logic_exposures_a_total_7d": "A/A current logic: успешные exposure группы A за 7 дней.",
        "newshub_experiment_aa_current_logic_exposures_b_total_7d": "A/A current logic: успешные exposure группы B за 7 дней.",
        "newshub_experiment_aa_current_logic_exposure_srm_abs_pct_7d": "A/A current logic: абсолютный дисбаланс exposure A/B за 7 дней, проценты.",
        "newshub_experiment_aa_current_logic_exposure_srm_p_value_7d": "A/A current logic: p-value SRM для exposure A/B за 7 дней.",
        "newshub_experiment_aa_current_logic_ctr_source_a_7d": "A/A current logic: CTR в источник группы A за 7 дней.",
        "newshub_experiment_aa_current_logic_ctr_source_b_7d": "A/A current logic: CTR в источник группы B за 7 дней.",
        "newshub_experiment_aa_current_logic_reaction_delivery_a_7d": "A/A current logic: доля exposure группы A с реакцией за 7 дней.",
        "newshub_experiment_aa_current_logic_reaction_delivery_b_7d": "A/A current logic: доля exposure группы B с реакцией за 7 дней.",
        "newshub_experiment_aa_current_logic_positive_ratio_a_7d": "A/A current logic: доля позитивных реакций группы A за 7 дней.",
        "newshub_experiment_aa_current_logic_positive_ratio_b_7d": "A/A current logic: доля позитивных реакций группы B за 7 дней.",
        "newshub_experiment_aa_current_logic_time_to_engage_minutes_avg_a_7d": "A/A current logic: среднее время до отклика группы A, минуты.",
        "newshub_experiment_aa_current_logic_time_to_engage_minutes_avg_b_7d": "A/A current logic: среднее время до отклика группы B, минуты.",
    }
    lines: list[str] = []
    for name in sorted(metrics):
        lines.append(f"# HELP {name} {metadata.get(name, name)}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {_finite_float(float(metrics[name])):.10g}")
    return "\n".join(lines) + "\n"
