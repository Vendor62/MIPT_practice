"""add_media_columns_and_subscription_notify

Revision ID: c4e1b9f3a6d7
Revises: b3f9c6d8e1a2
Create Date: 2026-01-31 18:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c4e1b9f3a6d7"
down_revision: Union[str, None] = "b3f9c6d8e1a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE posts ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ")
    op.execute("ALTER TABLE posts ADD COLUMN IF NOT EXISTS processed_content TEXT")
    op.execute("ALTER TABLE posts ADD COLUMN IF NOT EXISTS media_path TEXT")
    op.execute("ALTER TABLE posts ADD COLUMN IF NOT EXISTS media_mime VARCHAR")
    op.execute("ALTER TABLE posts ADD COLUMN IF NOT EXISTS media_created_at TIMESTAMPTZ")

    op.execute("ALTER TABLE subscription_requests ADD COLUMN IF NOT EXISTS notify_chat_id BIGINT")
    op.execute("ALTER TABLE subscription_requests ADD COLUMN IF NOT EXISTS notify_message_id BIGINT")


def downgrade() -> None:
    op.drop_column("subscription_requests", "notify_message_id")
    op.drop_column("subscription_requests", "notify_chat_id")

    op.drop_column("posts", "media_created_at")
    op.drop_column("posts", "media_mime")
    op.drop_column("posts", "media_path")
    op.drop_column("posts", "processed_content")
    op.drop_column("posts", "timestamp")
