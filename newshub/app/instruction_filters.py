from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.deepseek import (
    DeepseekAuthError,
    DeepseekRetryableError,
    evaluate_instruction_filter,
)
# TODO[showcase]: stub after removing payments/promo/userbot/subscribe modules from public version
# from app.billing import recalculate_user_billing_state
from app.models import Community, InstructionFilterDecision, User, UserCommunity, UserInstructionRule
# from app.payments.service import (
#     repair_confirmed_payment_entitlement_for_user,
#     sync_pending_payment_for_user,
#     user_has_syncable_pending_payment,
# )

log = structlog.get_logger()

INSTRUCTION_FILTER_ENABLED = os.getenv("INSTRUCTION_FILTER_ENABLED", "true").lower() == "true"
INSTRUCTION_FILTER_PROMPT_MAX_CHARS = int(os.getenv("INSTRUCTION_FILTER_PROMPT_MAX_CHARS", "500"))
INSTRUCTION_FILTER_REASON_MAX_CHARS = int(os.getenv("INSTRUCTION_FILTER_REASON_MAX_CHARS", "240"))
INSTRUCTION_FILTER_MAX_RETRIES = int(os.getenv("INSTRUCTION_FILTER_MAX_RETRIES", "1"))
INSTRUCTION_FILTER_TRIAL_DAYS = int(os.getenv("INSTRUCTION_FILTER_TRIAL_DAYS", "7"))

INSTRUCTION_SCOPE_GLOBAL = "global"
INSTRUCTION_SCOPE_COMMUNITY = "community"
INSTRUCTION_DECISION_ALLOW = "allow"
INSTRUCTION_DECISION_BLOCK = "block"
INSTRUCTION_DECISION_SOFT_BYPASS = "soft_bypass"


class InstructionPromptError(ValueError):
    pass


@dataclass(slots=True)
class ResolvedInstructionRule:
    user_id: int
    community_id: int | None
    scope: str
    prompt_text: str
    prompt_hash: str
    rule_id: int | None = None


@dataclass(slots=True)
class InstructionEvaluation:
    decision: str
    reason_short: str
    raw_response: dict[str, Any] | None
    latency_ms: int | None
    error_code: str | None = None

    @property
    def llm_allow(self) -> bool:
        return self.decision == INSTRUCTION_DECISION_ALLOW

    @property
    def soft_bypass(self) -> bool:
        return self.decision == INSTRUCTION_DECISION_SOFT_BYPASS


def merge_instruction_and_reco(*, reco_send: bool, instruction_decision: str | None) -> tuple[bool, bool]:
    if not instruction_decision:
        return reco_send, False
    if instruction_decision == INSTRUCTION_DECISION_ALLOW:
        return reco_send, False
    if instruction_decision == INSTRUCTION_DECISION_SOFT_BYPASS:
        return reco_send, True
    return False, False


def normalize_instruction_prompt(prompt_text: str) -> str:
    normalized = str(prompt_text or "").strip()
    if not normalized:
        raise InstructionPromptError("empty_prompt")
    if len(normalized) > INSTRUCTION_FILTER_PROMPT_MAX_CHARS:
        raise InstructionPromptError("prompt_too_long")
    return normalized


def prompt_hash_for_text(prompt_text: str) -> str:
    normalized = normalize_instruction_prompt(prompt_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_reason(value: Any) -> str:
    reason = str(value or "").strip()
    if not reason:
        return "No reason provided"
    return reason[:INSTRUCTION_FILTER_REASON_MAX_CHARS]


def user_has_instruction_filter_trial_access(user: Any, *, now: datetime | None = None) -> bool:
    if INSTRUCTION_FILTER_TRIAL_DAYS <= 0:
        return False
    created_at = getattr(user, "created_at", None)
    if not isinstance(created_at, datetime):
        return False

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    else:
        created_at = created_at.astimezone(timezone.utc)

    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    trial_start = now_utc - timedelta(days=INSTRUCTION_FILTER_TRIAL_DAYS)
    return trial_start <= created_at <= now_utc


async def user_has_instruction_filter_access(
    session: AsyncSession,
    user_id: int,
    *,
    attempt_payment_sync: bool = True,
    commit_on_sync: bool = False,
) -> bool:
    _state, req, _plan = await recalculate_user_billing_state(session, user_id)
    if bool(getattr(req, "premium_active", False)):
        return True

    user = await session.get(User, int(user_id))
    if not user:
        return False
    if user_has_instruction_filter_trial_access(user):
        return True

    if not attempt_payment_sync:
        return False

    synced = False
    try:
        repaired = await repair_confirmed_payment_entitlement_for_user(session, user)
        synced = synced or repaired > 0
        if await user_has_syncable_pending_payment(session, int(user_id)):
            result = await sync_pending_payment_for_user(session, user)
            synced = synced or str(result.get("status") or "").strip().lower() == "confirmed"
    except Exception as exc:
        log.warning(
            "instruction_filter.payment_access_sync_failed",
            user_id=user_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return False

    _state, req, _plan = await recalculate_user_billing_state(session, user_id)
    has_access = bool(getattr(req, "premium_active", False))
    if has_access and synced and commit_on_sync:
        await session.commit()
    return has_access


async def upsert_global_instruction_rule(
    session: AsyncSession,
    *,
    user_id: int,
    prompt_text: str,
) -> UserInstructionRule:
    normalized = normalize_instruction_prompt(prompt_text)
    result = await session.execute(
        select(UserInstructionRule).where(
            UserInstructionRule.user_id == user_id,
            UserInstructionRule.scope == INSTRUCTION_SCOPE_GLOBAL,
        )
    )
    rule = result.scalar_one_or_none()
    if not rule:
        rule = UserInstructionRule(
            user_id=user_id,
            scope=INSTRUCTION_SCOPE_GLOBAL,
            community_id=None,
        )
        session.add(rule)
    rule.prompt_text = normalized
    rule.is_enabled = True
    await session.flush()
    return rule


async def upsert_community_instruction_rule(
    session: AsyncSession,
    *,
    user_id: int,
    community_id: int,
    prompt_text: str,
) -> UserInstructionRule:
    normalized = normalize_instruction_prompt(prompt_text)
    result = await session.execute(
        select(UserInstructionRule).where(
            UserInstructionRule.user_id == user_id,
            UserInstructionRule.scope == INSTRUCTION_SCOPE_COMMUNITY,
            UserInstructionRule.community_id == community_id,
        )
    )
    rule = result.scalar_one_or_none()
    if not rule:
        rule = UserInstructionRule(
            user_id=user_id,
            scope=INSTRUCTION_SCOPE_COMMUNITY,
            community_id=community_id,
        )
        session.add(rule)
    rule.prompt_text = normalized
    rule.is_enabled = True
    await session.flush()
    return rule


async def disable_global_instruction_rule(session: AsyncSession, *, user_id: int) -> bool:
    result = await session.execute(
        select(UserInstructionRule).where(
            UserInstructionRule.user_id == user_id,
            UserInstructionRule.scope == INSTRUCTION_SCOPE_GLOBAL,
            UserInstructionRule.is_enabled.is_(True),
        )
    )
    rule = result.scalar_one_or_none()
    if not rule:
        return False
    rule.is_enabled = False
    await session.flush()
    return True


async def disable_community_instruction_rule(
    session: AsyncSession,
    *,
    user_id: int,
    community_id: int,
) -> bool:
    result = await session.execute(
        select(UserInstructionRule).where(
            UserInstructionRule.user_id == user_id,
            UserInstructionRule.scope == INSTRUCTION_SCOPE_COMMUNITY,
            UserInstructionRule.community_id == community_id,
            UserInstructionRule.is_enabled.is_(True),
        )
    )
    rule = result.scalar_one_or_none()
    if not rule:
        return False
    rule.is_enabled = False
    await session.flush()
    return True


async def get_instruction_rules_for_user(
    session: AsyncSession,
    *,
    user_id: int,
) -> list[UserInstructionRule]:
    result = await session.execute(
        select(UserInstructionRule)
        .where(
            UserInstructionRule.user_id == user_id,
            UserInstructionRule.is_enabled.is_(True),
        )
        .order_by(UserInstructionRule.scope.asc(), UserInstructionRule.id.asc())
    )
    return list(result.scalars().all())


async def get_subscribed_instruction_targets(
    session: AsyncSession,
    *,
    user_id: int,
) -> list[Community]:
    result = await session.execute(
        select(Community)
        .join(UserCommunity, UserCommunity.community_id == Community.id)
        .where(UserCommunity.user_id == user_id, Community.is_active.is_(True))
        .order_by(Community.link.asc(), Community.id.asc())
    )
    return list(result.scalars().all())


async def get_rule_bound_communities(
    session: AsyncSession,
    *,
    user_id: int,
) -> list[Community]:
    result = await session.execute(
        select(Community)
        .join(UserInstructionRule, UserInstructionRule.community_id == Community.id)
        .where(
            UserInstructionRule.user_id == user_id,
            UserInstructionRule.scope == INSTRUCTION_SCOPE_COMMUNITY,
            UserInstructionRule.is_enabled.is_(True),
        )
        .order_by(Community.link.asc(), Community.id.asc())
    )
    return list(result.scalars().all())


async def resolve_effective_instruction_rule(
    session: AsyncSession,
    *,
    user_id: int,
    community_id: int,
) -> ResolvedInstructionRule | None:
    rules = await resolve_effective_instruction_rules_for_users(
        session,
        user_ids=[user_id],
        community_id=community_id,
    )
    return rules.get(user_id)


async def resolve_effective_instruction_rules_for_users(
    session: AsyncSession,
    *,
    user_ids: list[int],
    community_id: int,
) -> dict[int, ResolvedInstructionRule]:
    if not user_ids:
        return {}
    result = await session.execute(
        select(UserInstructionRule).where(
            UserInstructionRule.user_id.in_(user_ids),
            UserInstructionRule.is_enabled.is_(True),
            (
                (UserInstructionRule.scope == INSTRUCTION_SCOPE_GLOBAL)
                | (
                    (UserInstructionRule.scope == INSTRUCTION_SCOPE_COMMUNITY)
                    & (UserInstructionRule.community_id == community_id)
                )
            ),
        )
    )
    rules = list(result.scalars().all())
    globals_by_user: dict[int, UserInstructionRule] = {}
    community_by_user: dict[int, UserInstructionRule] = {}
    for rule in rules:
        if rule.scope == INSTRUCTION_SCOPE_COMMUNITY and int(rule.community_id or 0) == int(community_id):
            community_by_user[int(rule.user_id)] = rule
        elif rule.scope == INSTRUCTION_SCOPE_GLOBAL:
            globals_by_user[int(rule.user_id)] = rule

    resolved: dict[int, ResolvedInstructionRule] = {}
    for user_id in user_ids:
        rule = community_by_user.get(int(user_id)) or globals_by_user.get(int(user_id))
        if not rule:
            continue
        prompt_text = normalize_instruction_prompt(rule.prompt_text)
        resolved[int(user_id)] = ResolvedInstructionRule(
            user_id=int(user_id),
            community_id=int(rule.community_id) if rule.community_id is not None else None,
            scope=str(rule.scope),
            prompt_text=prompt_text,
            prompt_hash=prompt_hash_for_text(prompt_text),
            rule_id=int(rule.id) if rule.id is not None else None,
        )
    return resolved


async def get_or_create_instruction_decision(
    session: AsyncSession,
    *,
    user: User,
    post_id: int,
    community: Community,
    source_title: str | None,
    post_title: str | None,
    post_text: str | None,
    resolved_rule: ResolvedInstructionRule,
) -> InstructionEvaluation:
    existing_res = await session.execute(
        select(InstructionFilterDecision).where(
            InstructionFilterDecision.user_id == user.id,
            InstructionFilterDecision.post_id == post_id,
            InstructionFilterDecision.effective_prompt_hash == resolved_rule.prompt_hash,
        )
    )
    existing = existing_res.scalar_one_or_none()
    if existing:
        return InstructionEvaluation(
            decision=str(existing.decision),
            reason_short=_normalize_reason(existing.reason_short),
            raw_response=existing.raw_response if isinstance(existing.raw_response, dict) else None,
            latency_ms=int(existing.latency_ms) if existing.latency_ms is not None else None,
            error_code=str(existing.error_code) if existing.error_code else None,
        )

    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(INSTRUCTION_FILTER_MAX_RETRIES + 1):
        try:
            raw_response = await evaluate_instruction_filter(
                source_title=source_title,
                source_link=getattr(community, "link", None),
                post_title=post_title,
                post_text=post_text,
                prompt_text=resolved_rule.prompt_text,
            )
            decision = str((raw_response or {}).get("decision") or "").strip().lower()
            normalized_decision = (
                INSTRUCTION_DECISION_ALLOW if decision == INSTRUCTION_DECISION_ALLOW else INSTRUCTION_DECISION_BLOCK
            )
            evaluation = InstructionEvaluation(
                decision=normalized_decision,
                reason_short=_normalize_reason((raw_response or {}).get("reason_short")),
                raw_response=raw_response if isinstance(raw_response, dict) else None,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error_code=None,
            )
            break
        except (DeepseekRetryableError, DeepseekAuthError, RuntimeError) as exc:
            last_error = exc
            retryable = isinstance(exc, DeepseekRetryableError) and attempt < INSTRUCTION_FILTER_MAX_RETRIES
            if retryable:
                continue
            evaluation = InstructionEvaluation(
                decision=INSTRUCTION_DECISION_SOFT_BYPASS,
                reason_short="Instruction filter temporarily bypassed",
                raw_response={"error": str(exc), "error_type": type(exc).__name__},
                latency_ms=int((time.perf_counter() - started) * 1000),
                error_code=type(exc).__name__,
            )
            break

    if last_error:
        log.warning(
            "instruction_filter.llm_failed",
            user_id=user.id,
            post_id=post_id,
            community_id=getattr(community, "id", None),
            error=str(last_error),
            error_type=type(last_error).__name__,
        )

    record = InstructionFilterDecision(
        user_id=user.id,
        post_id=post_id,
        community_id=getattr(community, "id", None),
        effective_scope=resolved_rule.scope,
        effective_prompt_hash=resolved_rule.prompt_hash,
        decision=evaluation.decision,
        reason_short=evaluation.reason_short,
        raw_response=evaluation.raw_response,
        latency_ms=evaluation.latency_ms,
        error_code=evaluation.error_code,
    )
    session.add(record)
    await session.flush()
    return evaluation
