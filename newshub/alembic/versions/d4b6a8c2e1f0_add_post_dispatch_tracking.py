"""add post dispatch tracking fields

Revision ID: d4b6a8c2e1f0
Revises: c8f4d2a1b6e0
Create Date: 2026-03-15 05:55:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "d4b6a8c2e1f0"
down_revision = "c8f4d2a1b6e0"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("posts", sa.Column("dispatch_enqueued_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("posts", sa.Column("dispatch_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("posts", sa.Column("dispatch_finished_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("posts", sa.Column("dispatch_error", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("posts", "dispatch_error")
    op.drop_column("posts", "dispatch_finished_at")
    op.drop_column("posts", "dispatch_started_at")
    op.drop_column("posts", "dispatch_enqueued_at")
