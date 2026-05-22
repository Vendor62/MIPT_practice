"""add billing tables

Revision ID: 8b1d2c7f0e44
Revises: 4f6c2a9b1d3e
Create Date: 2026-03-03 12:00:00.000000
"""

from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "8b1d2c7f0e44"
down_revision = "4f6c2a9b1d3e"
branch_labels = None
depends_on = None


def _month_period(now_utc: datetime) -> tuple[datetime, datetime]:
    period_start = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if period_start.month == 12:
        period_end = period_start.replace(year=period_start.year + 1, month=1)
    else:
        period_end = period_start.replace(month=period_start.month + 1)
    return period_start, period_end


def upgrade():
    op.create_table(
        "billing_plans",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("free_limit", sa.Integer(), nullable=False, server_default=sa.text("7")),
        sa.Column("pack_size", sa.Integer(), nullable=False, server_default=sa.text("10")),
        sa.Column("pack_price_rub", sa.Integer(), nullable=False, server_default=sa.text("49")),
        sa.Column("period_type", sa.String(), nullable=False, server_default="monthly"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_billing_plans_id", "billing_plans", ["id"], unique=False)
    op.create_index("ix_billing_plans_code", "billing_plans", ["code"], unique=True)

    op.create_table(
        "user_billing_states",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("active_subscriptions", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("required_packs", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("paid_packs", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("required_amount_rub", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("paid_amount_rub", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("status", sa.String(), nullable=False, server_default="free"),
        sa.Column("last_degraded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "period_start", name="uq_user_billing_state_period"),
    )
    op.create_index("ix_user_billing_states_id", "user_billing_states", ["id"], unique=False)
    op.create_index("ix_user_billing_states_user_id", "user_billing_states", ["user_id"], unique=False)
    op.create_index("ix_user_billing_states_period_start", "user_billing_states", ["period_start"], unique=False)
    op.create_index("ix_user_billing_states_period_end", "user_billing_states", ["period_end"], unique=False)
    op.create_index("ix_user_billing_states_status", "user_billing_states", ["status"], unique=False)

    op.create_table(
        "payment_orders",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("merchant_order_id", sa.String(), nullable=False),
        sa.Column("amount_rub", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("packs_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("status", sa.String(), nullable=False, server_default="new"),
        sa.Column("external_payment_id", sa.String(), nullable=True),
        sa.Column("payment_url", sa.Text(), nullable=True),
        sa.Column("fail_reason", sa.Text(), nullable=True),
        sa.Column("raw_init_response", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("raw_last_state", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key", name="uq_payment_orders_idempotency"),
        sa.UniqueConstraint("merchant_order_id", name="uq_payment_orders_merchant_order_id"),
    )
    op.create_index("ix_payment_orders_id", "payment_orders", ["id"], unique=False)
    op.create_index("ix_payment_orders_user_id", "payment_orders", ["user_id"], unique=False)
    op.create_index("ix_payment_orders_period_start", "payment_orders", ["period_start"], unique=False)
    op.create_index("ix_payment_orders_period_end", "payment_orders", ["period_end"], unique=False)
    op.create_index("ix_payment_orders_status", "payment_orders", ["status"], unique=False)
    op.create_index("ix_payment_orders_external_payment_id", "payment_orders", ["external_payment_id"], unique=False)

    op.create_table(
        "payment_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("payment_order_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["payment_order_id"], ["payment_orders.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_payment_events_id", "payment_events", ["id"], unique=False)
    op.create_index("ix_payment_events_payment_order_id", "payment_events", ["payment_order_id"], unique=False)
    op.create_index("ix_payment_events_event_type", "payment_events", ["event_type"], unique=False)

    op.execute(
        """
        INSERT INTO billing_plans (code, title, free_limit, pack_size, pack_price_rub, period_type, is_active)
        VALUES ('main', 'NewsHub Main Plan', 7, 10, 49, 'monthly', true)
        ON CONFLICT (code) DO NOTHING
        """
    )

    now_utc = datetime.now(timezone.utc)
    period_start, period_end = _month_period(now_utc)
    op.execute(
        sa.text(
            """
            INSERT INTO user_billing_states (
                user_id,
                period_start,
                period_end,
                active_subscriptions,
                required_packs,
                paid_packs,
                required_amount_rub,
                paid_amount_rub,
                status
            )
            SELECT
                u.id,
                :period_start,
                :period_end,
                COALESCE(subq.active_subscriptions, 0),
                CASE
                    WHEN COALESCE(subq.active_subscriptions, 0) <= 7 THEN 0
                    ELSE CEIL((COALESCE(subq.active_subscriptions, 0) - 7) / 10.0)::int
                END,
                0,
                CASE
                    WHEN COALESCE(subq.active_subscriptions, 0) <= 7 THEN 0
                    ELSE CEIL((COALESCE(subq.active_subscriptions, 0) - 7) / 10.0)::int * 49
                END,
                0,
                CASE
                    WHEN COALESCE(subq.active_subscriptions, 0) <= 7 THEN 'free'
                    ELSE 'payment_required'
                END
            FROM users u
            LEFT JOIN (
                SELECT user_id, COUNT(*)::int AS active_subscriptions
                FROM users_communities
                GROUP BY user_id
            ) subq ON subq.user_id = u.id
            ON CONFLICT (user_id, period_start) DO NOTHING
            """
        ).bindparams(period_start=period_start, period_end=period_end)
    )


def downgrade():
    op.drop_index("ix_payment_events_event_type", table_name="payment_events")
    op.drop_index("ix_payment_events_payment_order_id", table_name="payment_events")
    op.drop_index("ix_payment_events_id", table_name="payment_events")
    op.drop_table("payment_events")

    op.drop_index("ix_payment_orders_external_payment_id", table_name="payment_orders")
    op.drop_index("ix_payment_orders_status", table_name="payment_orders")
    op.drop_index("ix_payment_orders_period_end", table_name="payment_orders")
    op.drop_index("ix_payment_orders_period_start", table_name="payment_orders")
    op.drop_index("ix_payment_orders_user_id", table_name="payment_orders")
    op.drop_index("ix_payment_orders_id", table_name="payment_orders")
    op.drop_table("payment_orders")

    op.drop_index("ix_user_billing_states_status", table_name="user_billing_states")
    op.drop_index("ix_user_billing_states_period_end", table_name="user_billing_states")
    op.drop_index("ix_user_billing_states_period_start", table_name="user_billing_states")
    op.drop_index("ix_user_billing_states_user_id", table_name="user_billing_states")
    op.drop_index("ix_user_billing_states_id", table_name="user_billing_states")
    op.drop_table("user_billing_states")

    op.drop_index("ix_billing_plans_code", table_name="billing_plans")
    op.drop_index("ix_billing_plans_id", table_name="billing_plans")
    op.drop_table("billing_plans")
