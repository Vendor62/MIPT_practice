import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Literal

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

#from app.bot import start_bot
from app.ai.deepseek import close_session
from app.logging import configure_logging
from app.metrics import collect_product_metrics, record_product_metrics_daily_snapshot, render_prometheus_metrics
from app.models import Click, Post, User, UserKeywordStat, get_session
# TODO[showcase]: stub after removing payments/promo/userbot/subscribe modules from public version
# from app.payments.cryptopay_client import verify_webhook_signature
# from app.payments.service import (
#     create_payment_order_for_user,
#     get_current_payment_snapshot,
#     process_cryptopay_webhook,
#     process_tbank_webhook,
# )
# from app.promo import create_promo_code, redeem_promo_code
from app.reco_runtime import update_user_model_from_feedback
# TODO[showcase]: stub after removing payments/promo/userbot/subscribe modules from public version
#from app.userbot import start_userbot, sync_all_channels

app = FastAPI()
configure_logging("app")

log = structlog.get_logger()
PROMO_ADMIN_TOKEN = os.getenv("PROMO_ADMIN_TOKEN", "").strip()
ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN", "").strip()
CRYPTOPAY_WEBHOOK_SECRET = os.getenv("CRYPTOPAY_WEBHOOK_SECRET", "").strip()


def _require_admin(authorization: str | None) -> None:
    if not ADMIN_API_TOKEN:
        raise HTTPException(status_code=404, detail="Not found")
    expected = f"Bearer {ADMIN_API_TOKEN}"
    if not authorization or authorization.strip() != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


class CreatePaymentRequest(BaseModel):
    telegram_id: int
    extra_packs: int = Field(default=0, ge=0, le=100)
    payment_method: Literal["tbank", "cryptopay_usdt", "telegram_stars"] = "tbank"
    payment_term_days: Literal[30, 90, 180, 365] = 30


class PromoCreateRequest(BaseModel):
    admin_token: str
    code: str
    grant_type: Literal["premium", "extra_groups"] = "premium"
    premium_days: int | None = Field(default=30, ge=1, le=3650)
    extra_groups: int = Field(default=0, ge=0, le=1000)
    valid_from: str | None = None
    valid_until: str | None = None
    max_activations: int | None = Field(default=None, ge=1, le=1_000_000)
    per_user_limit: int = Field(default=1, ge=1, le=100)
    title: str | None = None


class PromoRedeemRequest(BaseModel):
    telegram_id: int
    code: str


class TBankWebhookPayload(BaseModel):
    TerminalKey: str
    OrderId: str
    Success: bool
    Status: str
    PaymentId: int
    ErrorCode: str
    Amount: int
    Token: str
    RebillId: int | None = None
    CardId: int | None = None
    Pan: str | None = None
    ExpDate: str | None = None
    Message: str | None = None
    Details: str | None = None


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _cryptopay_confirmation_mode() -> str:
    return "webhook_and_polling" if CRYPTOPAY_WEBHOOK_SECRET else "polling"


def _payment_confirmation_mode(provider: str | None) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "cryptopay":
        return _cryptopay_confirmation_mode()
    if normalized == "telegram_stars":
        return "telegram_update"
    return "webhook"


@app.on_event("shutdown")
async def shutdown_close_clients():
    await close_session()


@app.get("/")
async def root():
    return {"msg": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics(session: AsyncSession = Depends(get_session)):
    payload = await collect_product_metrics(session)
    try:
        await record_product_metrics_daily_snapshot(session, payload)
        await session.commit()
    except Exception:
        await session.rollback()
        log.exception("product_metrics_daily_snapshot_failed")
    return PlainTextResponse(
        render_prometheus_metrics(payload),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/users/")
async def get_users(
    authorization: str | None = Header(default=None),
    session: AsyncSession = Depends(get_session),
):
    _require_admin(authorization)
    result = await session.execute(select(User))
    users = result.scalars().all()
    return [
        {
            "id": u.id,
            "telegram_id": u.telegram_id,
            "first_name": u.first_name,
            "last_name": u.last_name,
            "username": u.username,
            "email": u.email,
            "summary_enabled": bool(getattr(u, "summary_enabled", False)),
            "engagement_score": float(getattr(u, "engagement_score", 0.0) or 0.0),
        }
        for u in users
    ]


@app.get("/r/{post_id}")
async def redirect_post(
    post_id: int,
    tg: int | None = None,
    u: int | None = None,
    source: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    content_link = post.content_link or {}
    target_url = content_link.get("url")
    if not target_url:
        raise HTTPException(status_code=404, detail="Source URL not available")

    # Новый формат трекинга: tg=<telegram_id>; старый u=<internal user_id> поддерживаем.
    user = None
    if tg:
        user_result = await session.execute(select(User).where(User.telegram_id == tg))
        user = user_result.scalar_one_or_none()
    elif u:
        user_result = await session.execute(select(User).where(User.id == u))
        user = user_result.scalar_one_or_none()

    # Не валим редирект, если пользователь не найден/метрика не записалась.
    if user:
        try:
            # Минимальный и максимально надёжный путь: один UPSERT сырым SQL.
            # Это специально изолировано от остальных апдейтов (модель/keywords),
            # чтобы трекинг кликов работал даже если что-то ещё падает.
            await session.execute(
                text(
                    """
                    INSERT INTO clicks (user_id, post_id, community_id, source, click_count, first_clicked_at, last_clicked_at)
                    VALUES (:user_id, :post_id, :community_id, :source, 1, now(), now())
                    ON CONFLICT (user_id, post_id)
                    DO UPDATE SET
                      click_count = clicks.click_count + 1,
                      last_clicked_at = now(),
                      source = EXCLUDED.source;
                    """
                ),
                {
                    "user_id": int(user.id),
                    "post_id": int(post_id),
                    "community_id": int(post.community_id) if post.community_id is not None else None,
                    "source": str(source) if source else None,
                },
            )

            await session.execute(
                update(User)
                .where(User.id == user.id)
                .values(engagement_score=func.coalesce(User.engagement_score, 0.0) + 0.2)
            )

            await session.commit()
            log.info(
                "click_redirect_recorded",
                user_id=int(user.id),
                telegram_id=int(user.telegram_id),
                post_id=int(post_id),
                community_id=int(post.community_id) if post.community_id is not None else None,
                source=str(source) if source else None,
            )
        except Exception as e:
            await session.rollback()
            log.exception("click_tracking_failed_but_redirect_continues", post_id=post_id, error=str(e))

    return RedirectResponse(url=target_url)


@app.post("/billing/create-payment")
async def create_billing_payment(
    body: CreatePaymentRequest,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(User).where(User.telegram_id == body.telegram_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        order = await create_payment_order_for_user(
            session,
            user=user,
            extra_packs=body.extra_packs,
            payment_method=body.payment_method,
            payment_term_days=body.payment_term_days,
        )
        snapshot = await get_current_payment_snapshot(session, user)
        await session.commit()
        return {
            "ok": True,
            "order_id": order.id,
            "status": order.status,
            "provider": order.provider,
            "quote_currency": order.quote_currency,
            "quote_amount": str(order.quote_amount) if order.quote_amount is not None else None,
            "amount_rub": int(order.amount_rub or 0),
            "packs_count": int(order.packs_count or 0),
            "payment_term_days": body.payment_term_days,
            "period_start": order.period_start.isoformat() if order.period_start else None,
            "period_end": order.period_end.isoformat() if order.period_end else None,
            "payment_url": order.payment_url,
            "payment_confirmation_mode": _payment_confirmation_mode(order.provider),
            "bot_invoice": order.raw_init_response if order.provider == "telegram_stars" else None,
            "snapshot": snapshot,
        }
    except Exception as e:
        await session.rollback()
        log.error("billing.create_payment_failed", telegram_id=body.telegram_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/billing/status/{telegram_id}")
async def billing_status(
    telegram_id: int,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(User).where(User.telegram_id == telegram_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    snapshot = await get_current_payment_snapshot(session, user)
    return {
        "ok": True,
        "snapshot": snapshot,
        "payment_confirmation_mode": _payment_confirmation_mode((snapshot.get("pending_payment") or {}).get("provider")),
    }


@app.post("/billing/tbank/webhook")
async def billing_tbank_webhook(
    payload: TBankWebhookPayload,
    session: AsyncSession = Depends(get_session),
):
    result = await process_tbank_webhook(session, payload.model_dump())
    if not result.get("ok"):
        await session.rollback()
        raise HTTPException(status_code=400, detail=result.get("error", "webhook_error"))
    await session.commit()
    return {"ok": True}


@app.post("/billing/cryptopay/webhook/{path_secret}")
async def billing_cryptopay_webhook(
    path_secret: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    if not CRYPTOPAY_WEBHOOK_SECRET:
        raise HTTPException(status_code=404, detail="webhook_not_configured")
    if path_secret != CRYPTOPAY_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="invalid_secret")

    raw_body = await request.body()
    signature = request.headers.get("crypto-pay-api-signature")
    if not verify_webhook_signature(raw_body, signature):
        await session.rollback()
        raise HTTPException(status_code=400, detail="invalid_signature")

    try:
        payload = json.loads(raw_body.decode("utf-8") or "{}")
    except Exception:
        await session.rollback()
        raise HTTPException(status_code=400, detail="invalid_json")

    result = await process_cryptopay_webhook(session, payload)
    if not result.get("ok"):
        await session.rollback()
        raise HTTPException(status_code=400, detail=result.get("error", "webhook_error"))
    await session.commit()
    return {"ok": True}


@app.post("/billing/promo/create")
async def billing_promo_create(
    body: PromoCreateRequest,
    session: AsyncSession = Depends(get_session),
):
    if not PROMO_ADMIN_TOKEN or body.admin_token != PROMO_ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        promo = await create_promo_code(
            session,
            code=body.code,
            grant_type=body.grant_type,
            premium_days=body.premium_days,
            extra_groups=int(body.extra_groups),
            valid_from=_parse_iso_dt(body.valid_from),
            valid_until=_parse_iso_dt(body.valid_until),
            max_activations=body.max_activations,
            per_user_limit=int(body.per_user_limit),
            title=body.title,
        )
        await session.commit()
        return {
            "ok": True,
            "promo": {
                "id": promo.id,
                "code": promo.code,
                "grant_type": promo.grant_type,
                "premium_days": promo.premium_days,
                "extra_groups": promo.extra_groups,
                "valid_from": promo.valid_from,
                "valid_until": promo.valid_until,
                "max_activations": promo.max_activations,
                "per_user_limit": promo.per_user_limit,
                "is_active": promo.is_active,
            },
        }
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/billing/promo/redeem")
async def billing_promo_redeem(
    body: PromoRedeemRequest,
    session: AsyncSession = Depends(get_session),
):
    user = (
        await session.execute(select(User).where(User.telegram_id == body.telegram_id))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        grant = await redeem_promo_code(session, user=user, code=body.code)
        snapshot = await get_current_payment_snapshot(session, user)
        await session.commit()
        return {
            "ok": True,
            "grant": {
                "id": grant.id,
                "code": grant.code,
                "grant_type": grant.grant_type,
                "premium_days": grant.premium_days,
                "extra_groups": grant.extra_groups,
                "starts_at": grant.starts_at,
                "expires_at": grant.expires_at,
            },
            "snapshot": snapshot,
        }
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/billing/success")
async def billing_success():
    return HTMLResponse(
        content="""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Оплата успешна</title>
  <style>
    body { margin:0; font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0f172a; color:#e2e8f0; }
    .wrap { min-height:100vh; display:flex; align-items:center; justify-content:center; padding:24px; }
    .card { width:100%; max-width:520px; background:#111827; border:1px solid #1f2937; border-radius:16px; padding:24px; box-shadow:0 10px 30px rgba(0,0,0,.35); }
    .ok { font-size:40px; line-height:1; margin-bottom:10px; }
    h1 { margin:0 0 10px; font-size:24px; color:#22c55e; }
    p { margin:0 0 16px; color:#cbd5e1; }
    .btn { display:inline-block; padding:10px 14px; border-radius:10px; text-decoration:none; background:#22c55e; color:#06210f; font-weight:700; }
    .hint { margin-top:12px; font-size:13px; color:#94a3b8; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="ok">✅</div>
      <h1>Оплата прошла успешно</h1>
      <p>Пакеты подписок начислены. Вернись в Telegram и нажми «Обновить статус оплаты».</p>
      <a class="btn" href="https://t.me/TeleNewsHubBot">Открыть бота</a>
      <div class="hint">Если статус в боте не обновился сразу — подожди 5–10 секунд и нажми обновление еще раз.</div>
    </div>
  </div>
</body>
</html>
        """.strip()
    )


@app.get("/billing/fail")
async def billing_fail():
    return HTMLResponse(
        content="""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Оплата не завершена</title>
  <style>
    body { margin:0; font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#0f172a; color:#e2e8f0; }
    .wrap { min-height:100vh; display:flex; align-items:center; justify-content:center; padding:24px; }
    .card { width:100%; max-width:520px; background:#111827; border:1px solid #1f2937; border-radius:16px; padding:24px; box-shadow:0 10px 30px rgba(0,0,0,.35); }
    .bad { font-size:40px; line-height:1; margin-bottom:10px; }
    h1 { margin:0 0 10px; font-size:24px; color:#f59e0b; }
    p { margin:0 0 16px; color:#cbd5e1; }
    .btn { display:inline-block; padding:10px 14px; border-radius:10px; text-decoration:none; background:#f59e0b; color:#2c1a00; font-weight:700; }
    .hint { margin-top:12px; font-size:13px; color:#94a3b8; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="bad">⚠️</div>
      <h1>Оплата не завершена</h1>
      <p>Платеж был отменен или отклонен. Вернись в Telegram и создай новую ссылку на оплату.</p>
      <a class="btn" href="https://t.me/TeleNewsHubBot">Вернуться в бота</a>
      <div class="hint">Если ты уверен, что оплатил — нажми в боте «Обновить статус оплаты».</div>
    </div>
  </div>
</body>
</html>
        """.strip()
    )

# Schema is managed by Alembic (`alembic upgrade head`); see
# app.models.ensure_db_schema for history.

# @app.on_event("startup")
# async def startup_start_bot():
#     asyncio.create_task(start_bot())
#     log.info("Startup: bot polling task created")


# @app.on_event("startup")
# async def startup_start_userbot():
#     log.info("Startup: userbot runner starting")

#     async def userbot_runner():
#         client = None
#         while True:
#             try:
#                 client = await start_userbot()
#                 await sync_all_channels()
#                 await client.run_until_disconnected()

#             except asyncio.CancelledError:
#                 log.info("Userbot runner cancelled, exiting")
#                 if client is not None:
#                     try:
#                         await client.disconnect()
#                     except Exception as e:
#                         log.error("Userbot disconnect error", error=str(e))
#                 raise

#             except Exception as e:
#                 log.error("Userbot runner error", error=str(e), exc_info=True)
#                 await asyncio.sleep(10)

#     asyncio.create_task(userbot_runner())
