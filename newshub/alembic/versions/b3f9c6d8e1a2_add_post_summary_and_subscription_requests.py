"""add_post_summary_and_subscription_requests

Revision ID: b3f9c6d8e1a2
Revises: 9d5f8e2c0f1a
Create Date: 2026-01-31 18:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b3f9c6d8e1a2"
down_revision: Union[str, None] = "9d5f8e2c0f1a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("posts", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column("posts", sa.Column("summary_created_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_posts_summary_created_at", "posts", ["summary_created_at"], unique=False)

    op.create_table(
        "subscription_requests",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("group_link", sa.String(), nullable=False),
        sa.Column("group_handle", sa.String(), nullable=True),
        sa.Column("status", sa.String(), server_default="pending", nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_subscription_requests_user_id", "subscription_requests", ["user_id"], unique=False)
    op.create_index("ix_subscription_requests_group_link", "subscription_requests", ["group_link"], unique=False)
    op.create_index("ix_subscription_requests_status", "subscription_requests", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_subscription_requests_status", table_name="subscription_requests")
    op.drop_index("ix_subscription_requests_group_link", table_name="subscription_requests")
    op.drop_index("ix_subscription_requests_user_id", table_name="subscription_requests")
    op.drop_table("subscription_requests")

    op.drop_index("ix_posts_summary_created_at", table_name="posts")
    op.drop_column("posts", "summary_created_at")
    op.drop_column("posts", "summary")
