"""add ie status fields to posts

Revision ID: dbbdd5b4ff58
Revises: 8b1d2c7f0e44
Create Date: 2026-03-11 15:49:04.637595

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dbbdd5b4ff58'
down_revision: Union[str, None] = '8b1d2c7f0e44'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column("posts", sa.Column("ie_status", sa.String(), nullable=True))
    op.add_column("posts", sa.Column("ie_enqueued_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("posts", sa.Column("ie_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("posts", sa.Column("ie_processed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("posts", sa.Column("ie_error", sa.Text(), nullable=True))
    op.add_column(
        "posts",
        sa.Column("ie_retry_count", sa.Integer(), nullable=False, server_default="0")
    )

    op.create_index("ix_posts_ie_status", "posts", ["ie_status"], unique=False)
    op.create_index("idx_posts_ie_status_id", "posts", ["ie_status", "id"], unique=False)
    op.create_index("idx_posts_ie_processed_at", "posts", ["ie_processed_at"], unique=False)


def downgrade():
    op.drop_index("idx_posts_ie_processed_at", table_name="posts")
    op.drop_index("idx_posts_ie_status_id", table_name="posts")
    op.drop_index("ix_posts_ie_status", table_name="posts")

    op.drop_column("posts", "ie_retry_count")
    op.drop_column("posts", "ie_error")
    op.drop_column("posts", "ie_processed_at")
    op.drop_column("posts", "ie_started_at")
    op.drop_column("posts", "ie_enqueued_at")
    op.drop_column("posts", "ie_status")
