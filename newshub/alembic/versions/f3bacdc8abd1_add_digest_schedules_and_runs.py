"""add digest schedules and runs

Revision ID: f3bacdc8abd1
Revises: 82d4d2ce9cce
Create Date: 2026-02-07 13:06:50.080021

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3bacdc8abd1'
down_revision: Union[str, None] = '82d4d2ce9cce'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_digest_settings",
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("timezone", sa.String(), nullable=False, server_default="Europe/Moscow"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "user_digest_slots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("hour", sa.SmallInteger(), nullable=False),
        sa.Column("minute", sa.SmallInteger(), nullable=False),
        sa.Column("days_mask", sa.Integer(), nullable=False, server_default=sa.text("127")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "hour", "minute", name="uq_user_digest_slot_time"),
    )
    op.create_index("ix_user_digest_slots_next_run_at", "user_digest_slots", ["next_run_at"])
    op.create_index("ix_user_digest_slots_user_id", "user_digest_slots", ["user_id"])

    op.create_table(
        "digest_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("slot_id", sa.Integer(), sa.ForeignKey("user_digest_slots.id"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="created"),
        sa.Column("sent_messages", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("slot_id", "period_end", name="uq_digest_run_slot_period_end"),
    )
    op.create_index("ix_digest_runs_slot_id", "digest_runs", ["slot_id"])
    op.create_index("ix_digest_runs_user_id", "digest_runs", ["user_id"])
    op.create_index("ix_digest_runs_status", "digest_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_digest_runs_status", table_name="digest_runs")
    op.drop_index("ix_digest_runs_user_id", table_name="digest_runs")
    op.drop_index("ix_digest_runs_slot_id", table_name="digest_runs")
    op.drop_table("digest_runs")

    op.drop_index("ix_user_digest_slots_user_id", table_name="user_digest_slots")
    op.drop_index("ix_user_digest_slots_next_run_at", table_name="user_digest_slots")
    op.drop_table("user_digest_slots")

    op.drop_table("user_digest_settings")
