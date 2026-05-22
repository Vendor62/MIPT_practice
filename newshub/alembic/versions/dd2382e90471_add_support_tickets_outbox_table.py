"""add support tickets outbox table

Revision ID: dd2382e90471
Revises: 2d4cba0254be
Create Date: 2026-05-01 23:16:32.832646

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dd2382e90471'
down_revision: Union[str, None] = '2d4cba0254be'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "support_tickets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("telegram_id", sa.BigInteger(), nullable=False),
        sa.Column("username", sa.String(), nullable=True),
        sa.Column("premium_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("user_registered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="accepted"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id", name="uq_support_tickets_request_id"),
    )
    op.create_index(op.f("ix_support_tickets_id"), "support_tickets", ["id"], unique=False)
    op.create_index(op.f("ix_support_tickets_request_id"), "support_tickets", ["request_id"], unique=True)
    op.create_index(op.f("ix_support_tickets_user_id"), "support_tickets", ["user_id"], unique=False)
    op.create_index(op.f("ix_support_tickets_telegram_id"), "support_tickets", ["telegram_id"], unique=False)
    op.create_index(op.f("ix_support_tickets_status"), "support_tickets", ["status"], unique=False)
    op.create_index(op.f("ix_support_tickets_next_retry_at"), "support_tickets", ["next_retry_at"], unique=False)
    op.create_index(op.f("ix_support_tickets_delivered_at"), "support_tickets", ["delivered_at"], unique=False)
    op.create_index(op.f("ix_support_tickets_created_at"), "support_tickets", ["created_at"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_support_tickets_created_at"), table_name="support_tickets")
    op.drop_index(op.f("ix_support_tickets_delivered_at"), table_name="support_tickets")
    op.drop_index(op.f("ix_support_tickets_next_retry_at"), table_name="support_tickets")
    op.drop_index(op.f("ix_support_tickets_status"), table_name="support_tickets")
    op.drop_index(op.f("ix_support_tickets_telegram_id"), table_name="support_tickets")
    op.drop_index(op.f("ix_support_tickets_user_id"), table_name="support_tickets")
    op.drop_index(op.f("ix_support_tickets_request_id"), table_name="support_tickets")
    op.drop_index(op.f("ix_support_tickets_id"), table_name="support_tickets")
    op.drop_table("support_tickets")
