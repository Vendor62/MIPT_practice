"""add dispatch_deliveries outbox table

Revision ID: e3d9b1a2f6c7
Revises: b7c3d9e1f2a4, c9a1f2e3b4d5
Create Date: 2026-04-21 19:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = "e3d9b1a2f6c7"
down_revision = ("b7c3d9e1f2a4", "c9a1f2e3b4d5")
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Outbox of per-(post, user) dispatch attempts. Enforces idempotent
    # delivery: the dispatch task claims a row via INSERT ... ON CONFLICT
    # DO NOTHING RETURNING; on Celery retry the duplicate claim returns no
    # rows and that user is skipped. On send failure the row is deleted so
    # the retry can re-attempt.
    op.create_table(
        "dispatch_deliveries",
        sa.Column("post_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("telegram_message_id", sa.BigInteger(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.PrimaryKeyConstraint("post_id", "user_id", name="pk_dispatch_deliveries"),
    )
    op.create_index(
        "ix_dispatch_deliveries_user_sent",
        "dispatch_deliveries",
        ["user_id", "sent_at"],
    )
    op.create_index(
        "ix_dispatch_deliveries_post",
        "dispatch_deliveries",
        ["post_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_dispatch_deliveries_post", table_name="dispatch_deliveries")
    op.drop_index("ix_dispatch_deliveries_user_sent", table_name="dispatch_deliveries")
    op.drop_table("dispatch_deliveries")
