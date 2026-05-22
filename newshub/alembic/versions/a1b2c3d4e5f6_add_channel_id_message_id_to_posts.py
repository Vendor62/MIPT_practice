"""add channel_id message_id to posts

Revision ID: a1b2c3d4e5f6
Revises: 9c6a2f8b7d11
Create Date: 2026-03-12 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "a1b2c3d4e5f6"
down_revision = "9c6a2f8b7d11"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("posts", sa.Column("channel_id", sa.String(), nullable=True))
    op.add_column("posts", sa.Column("message_id", sa.BigInteger(), nullable=True))
    op.create_index("ix_posts_channel_id", "posts", ["channel_id"])

    # Backfill from content_link JSONB
    op.execute("""
        UPDATE posts
        SET channel_id = content_link->>'channel',
            message_id = (content_link->>'id')::bigint
        WHERE content_link IS NOT NULL
          AND content_link->>'channel' IS NOT NULL
          AND content_link->>'id' IS NOT NULL
    """)

    # Deduplicate: keep latest post per (channel_id, message_id)
    op.execute("""
        DELETE FROM posts p1
        USING posts p2
        WHERE p1.channel_id IS NOT NULL
          AND p1.message_id IS NOT NULL
          AND p1.channel_id = p2.channel_id
          AND p1.message_id = p2.message_id
          AND p1.id < p2.id
    """)

    op.create_unique_constraint("uq_posts_channel_message", "posts", ["channel_id", "message_id"])


def downgrade() -> None:
    op.drop_constraint("uq_posts_channel_message", "posts", type_="unique")
    op.drop_index("ix_posts_channel_id", "posts")
    op.drop_column("posts", "message_id")
    op.drop_column("posts", "channel_id")
