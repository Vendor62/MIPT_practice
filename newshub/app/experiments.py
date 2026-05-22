from __future__ import annotations

import hashlib
import os
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ExperimentAssignment, ExperimentExposure

AA_CURRENT_LOGIC_EXPERIMENT_KEY = os.getenv("AA_CURRENT_LOGIC_EXPERIMENT_KEY", "aa_current_logic_v1")
AA_CURRENT_LOGIC_SALT = os.getenv("AA_CURRENT_LOGIC_SALT", "newshub-aa-current-logic-v1")
AA_VARIANTS = ("a", "b")


def assign_aa_variant(
    user_id: int,
    *,
    experiment_key: str = AA_CURRENT_LOGIC_EXPERIMENT_KEY,
    salt: str = AA_CURRENT_LOGIC_SALT,
) -> str:
    """Stable 50/50 assignment without relying on database row order."""
    payload = f"{experiment_key}:{salt}:{int(user_id)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return AA_VARIANTS[digest[0] % len(AA_VARIANTS)]


async def ensure_aa_assignment(
    session: AsyncSession,
    *,
    user_id: int,
    experiment_key: str = AA_CURRENT_LOGIC_EXPERIMENT_KEY,
) -> str:
    variant = assign_aa_variant(user_id, experiment_key=experiment_key)
    stmt = (
        insert(ExperimentAssignment)
        .values(
            experiment_key=experiment_key,
            user_id=int(user_id),
            variant=variant,
        )
        .on_conflict_do_nothing(
            index_elements=[
                ExperimentAssignment.experiment_key,
                ExperimentAssignment.user_id,
            ]
        )
    )
    await session.execute(stmt)

    stored_variant = await session.scalar(
        select(ExperimentAssignment.variant).where(
            ExperimentAssignment.experiment_key == experiment_key,
            ExperimentAssignment.user_id == int(user_id),
        )
    )
    return str(stored_variant or variant)


async def record_aa_exposure(
    session: AsyncSession,
    *,
    user_id: int,
    post_id: int,
    variant: str,
    sent_at: datetime,
    telegram_message_id: int | None = None,
    delivery_result: str | None = None,
    render_mode: str | None = None,
    content_chars: int | None = None,
    summary_chars: int | None = None,
    has_summary: bool = False,
    experiment_key: str = AA_CURRENT_LOGIC_EXPERIMENT_KEY,
) -> None:
    stmt = (
        insert(ExperimentExposure)
        .values(
            experiment_key=experiment_key,
            user_id=int(user_id),
            post_id=int(post_id),
            variant=str(variant),
            eligible=True,
            render_mode=render_mode,
            content_chars=content_chars,
            summary_chars=summary_chars,
            has_summary=bool(has_summary),
            delivery_result=delivery_result,
            telegram_message_id=telegram_message_id,
            sent_at=sent_at,
        )
        .on_conflict_do_nothing(
            index_elements=[
                ExperimentExposure.experiment_key,
                ExperimentExposure.user_id,
                ExperimentExposure.post_id,
            ]
        )
    )
    await session.execute(stmt)
