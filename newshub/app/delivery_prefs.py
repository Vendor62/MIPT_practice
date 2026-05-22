import os
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import UserDeliveryPreference


PLUS_BOT_TOKEN = (os.getenv("BOT_PLUS_TOKEN", "") or "").strip()
MAIN_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN", "") or "").strip()
ENABLE_BOT_PLUS_ROUTING = os.getenv("ENABLE_BOT_PLUS_ROUTING", "false").lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DeliveryRoute:
    token: str
    target: str
    reason: str


async def get_or_create_delivery_prefs(session: AsyncSession, user_id: int) -> UserDeliveryPreference:
    prefs = await session.get(UserDeliveryPreference, int(user_id))
    if prefs:
        return prefs
    prefs = UserDeliveryPreference(user_id=int(user_id))
    session.add(prefs)
    await session.flush()
    return prefs


async def resolve_delivery_route(
    session: AsyncSession,
    *,
    user_id: int,
    delivery_kind: str,
) -> DeliveryRoute:
    """Resolve delivery bot route for a user and content kind.

    delivery_kind: "digest" | "storyline"
    """
    if not MAIN_BOT_TOKEN:
        return DeliveryRoute(token="", target="none", reason="main_token_missing")

    if not ENABLE_BOT_PLUS_ROUTING:
        return DeliveryRoute(token=MAIN_BOT_TOKEN, target="main", reason="routing_disabled")

    if not PLUS_BOT_TOKEN:
        return DeliveryRoute(token=MAIN_BOT_TOKEN, target="main", reason="plus_token_missing")

    prefs = await session.get(UserDeliveryPreference, int(user_id))
    if not prefs:
        return DeliveryRoute(token=MAIN_BOT_TOKEN, target="main", reason="prefs_missing")

    if not isinstance(getattr(prefs, "plus_bot_connected_at", None), datetime):
        return DeliveryRoute(token=MAIN_BOT_TOKEN, target="main", reason="plus_not_connected")

    wants_plus = False
    if delivery_kind == "digest":
        wants_plus = bool(getattr(prefs, "digest_to_plus_bot", False))
    elif delivery_kind == "storyline":
        wants_plus = bool(getattr(prefs, "storyline_to_plus_bot", False))

    if wants_plus:
        return DeliveryRoute(token=PLUS_BOT_TOKEN, target="plus", reason="user_pref")
    return DeliveryRoute(token=MAIN_BOT_TOKEN, target="main", reason="user_pref_main")


async def bulk_load_delivery_prefs(session: AsyncSession, user_ids: list[int]) -> dict[int, UserDeliveryPreference]:
    normalized_ids = sorted({int(uid) for uid in user_ids if uid is not None})
    if not normalized_ids:
        return {}
    res = await session.execute(
        select(UserDeliveryPreference).where(UserDeliveryPreference.user_id.in_(normalized_ids))
    )
    rows = res.scalars().all()
    return {int(row.user_id): row for row in rows}


def resolve_delivery_route_from_pref(
    *,
    pref: UserDeliveryPreference | None,
    delivery_kind: str,
) -> DeliveryRoute:
    if not MAIN_BOT_TOKEN:
        return DeliveryRoute(token="", target="none", reason="main_token_missing")
    if not ENABLE_BOT_PLUS_ROUTING:
        return DeliveryRoute(token=MAIN_BOT_TOKEN, target="main", reason="routing_disabled")
    if not PLUS_BOT_TOKEN:
        return DeliveryRoute(token=MAIN_BOT_TOKEN, target="main", reason="plus_token_missing")
    if pref is None:
        return DeliveryRoute(token=MAIN_BOT_TOKEN, target="main", reason="prefs_missing")
    if not isinstance(getattr(pref, "plus_bot_connected_at", None), datetime):
        return DeliveryRoute(token=MAIN_BOT_TOKEN, target="main", reason="plus_not_connected")

    wants_plus = False
    if delivery_kind == "digest":
        wants_plus = bool(getattr(pref, "digest_to_plus_bot", False))
    elif delivery_kind == "storyline":
        wants_plus = bool(getattr(pref, "storyline_to_plus_bot", False))

    if wants_plus:
        return DeliveryRoute(token=PLUS_BOT_TOKEN, target="plus", reason="user_pref")
    return DeliveryRoute(token=MAIN_BOT_TOKEN, target="main", reason="user_pref_main")
