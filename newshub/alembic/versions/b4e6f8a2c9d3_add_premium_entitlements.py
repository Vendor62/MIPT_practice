"""add premium entitlements

Revision ID: b4e6f8a2c9d3
Revises: e9b7c1d3a5f2
Create Date: 2026-04-28 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "b4e6f8a2c9d3"
down_revision = "e9b7c1d3a5f2"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "premium_entitlements",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_type", "source_id", name="uq_premium_entitlements_source"),
    )
    op.create_index("ix_premium_entitlements_id", "premium_entitlements", ["id"], unique=False)
    op.create_index("ix_premium_entitlements_user_id", "premium_entitlements", ["user_id"], unique=False)
    op.create_index("ix_premium_entitlements_source_type", "premium_entitlements", ["source_type"], unique=False)
    op.create_index("ix_premium_entitlements_source_id", "premium_entitlements", ["source_id"], unique=False)
    op.create_index("ix_premium_entitlements_starts_at", "premium_entitlements", ["starts_at"], unique=False)
    op.create_index("ix_premium_entitlements_expires_at", "premium_entitlements", ["expires_at"], unique=False)
    op.create_index("ix_premium_entitlements_is_active", "premium_entitlements", ["is_active"], unique=False)

    op.add_column(
        "promo_codes",
        sa.Column("grant_type", sa.String(), nullable=False, server_default="extra_groups"),
    )
    op.add_column("promo_codes", sa.Column("premium_days", sa.Integer(), nullable=True))
    op.create_index("ix_promo_codes_grant_type", "promo_codes", ["grant_type"], unique=False)

    op.add_column(
        "user_promo_grants",
        sa.Column("grant_type", sa.String(), nullable=False, server_default="extra_groups"),
    )
    op.add_column("user_promo_grants", sa.Column("premium_days", sa.Integer(), nullable=True))
    op.create_index("ix_user_promo_grants_grant_type", "user_promo_grants", ["grant_type"], unique=False)

    op.execute(
        sa.text(
            """
            INSERT INTO premium_entitlements (
                user_id, source_type, source_id, starts_at, expires_at, is_active, created_at, updated_at
            )
            SELECT
                user_id,
                'payment',
                id,
                period_start,
                period_end,
                true,
                COALESCE(confirmed_at, created_at, now()),
                now()
            FROM payment_orders
            WHERE status = 'confirmed'
              AND period_start IS NOT NULL
              AND period_end IS NOT NULL
            ON CONFLICT (source_type, source_id) DO NOTHING
            """
        )
    )

    op.alter_column("promo_codes", "grant_type", server_default="premium")
    op.alter_column("promo_codes", "premium_days", server_default=sa.text("30"))
    op.alter_column("user_promo_grants", "grant_type", server_default="premium")


def downgrade():
    op.drop_index("ix_user_promo_grants_grant_type", table_name="user_promo_grants")
    op.drop_column("user_promo_grants", "premium_days")
    op.drop_column("user_promo_grants", "grant_type")

    op.drop_index("ix_promo_codes_grant_type", table_name="promo_codes")
    op.drop_column("promo_codes", "premium_days")
    op.drop_column("promo_codes", "grant_type")

    op.drop_index("ix_premium_entitlements_is_active", table_name="premium_entitlements")
    op.drop_index("ix_premium_entitlements_expires_at", table_name="premium_entitlements")
    op.drop_index("ix_premium_entitlements_starts_at", table_name="premium_entitlements")
    op.drop_index("ix_premium_entitlements_source_id", table_name="premium_entitlements")
    op.drop_index("ix_premium_entitlements_source_type", table_name="premium_entitlements")
    op.drop_index("ix_premium_entitlements_user_id", table_name="premium_entitlements")
    op.drop_index("ix_premium_entitlements_id", table_name="premium_entitlements")
    op.drop_table("premium_entitlements")
