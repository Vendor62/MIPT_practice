"""add promo codes and user grants

Revision ID: 9c6a2f8b7d11
Revises: dbbdd5b4ff58
Create Date: 2026-03-11 11:20:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "9c6a2f8b7d11"
down_revision = "dbbdd5b4ff58"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "promo_codes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("extra_groups", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("max_activations", sa.Integer(), nullable=True),
        sa.Column("per_user_limit", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )
    op.create_index("ix_promo_codes_id", "promo_codes", ["id"], unique=False)
    op.create_index("ix_promo_codes_code", "promo_codes", ["code"], unique=True)

    op.create_table(
        "user_promo_grants",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("promo_code_id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(), nullable=False),
        sa.Column("extra_groups", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["promo_code_id"], ["promo_codes.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_promo_grants_id", "user_promo_grants", ["id"], unique=False)
    op.create_index("ix_user_promo_grants_user_id", "user_promo_grants", ["user_id"], unique=False)
    op.create_index("ix_user_promo_grants_promo_code_id", "user_promo_grants", ["promo_code_id"], unique=False)
    op.create_index("ix_user_promo_grants_code", "user_promo_grants", ["code"], unique=False)
    op.create_index("ix_user_promo_grants_is_active", "user_promo_grants", ["is_active"], unique=False)


def downgrade():
    op.drop_index("ix_user_promo_grants_is_active", table_name="user_promo_grants")
    op.drop_index("ix_user_promo_grants_code", table_name="user_promo_grants")
    op.drop_index("ix_user_promo_grants_promo_code_id", table_name="user_promo_grants")
    op.drop_index("ix_user_promo_grants_user_id", table_name="user_promo_grants")
    op.drop_index("ix_user_promo_grants_id", table_name="user_promo_grants")
    op.drop_table("user_promo_grants")

    op.drop_index("ix_promo_codes_code", table_name="promo_codes")
    op.drop_index("ix_promo_codes_id", table_name="promo_codes")
    op.drop_table("promo_codes")
