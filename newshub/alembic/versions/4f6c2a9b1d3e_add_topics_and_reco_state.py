"""add topics and recommendation state tables

Revision ID: 4f6c2a9b1d3e
Revises: 36ea519df1ef
Create Date: 2026-03-02 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "4f6c2a9b1d3e"
down_revision = "36ea519df1ef"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS feed_filter VARCHAR DEFAULT 'all'")
    op.execute("UPDATE users SET feed_filter='all' WHERE feed_filter IS NULL OR feed_filter=''")
    op.alter_column("users", "feed_filter", existing_type=sa.String(), nullable=False, server_default="all")

    op.create_table(
        "post_topics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("post_id", sa.Integer(), nullable=False),
        sa.Column("topic", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("source", sa.String(), nullable=False, server_default="deepseek"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["post_id"], ["posts.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("post_id", "topic", name="uq_post_topics_post_topic"),
    )
    op.create_index("ix_post_topics_id", "post_topics", ["id"], unique=False)
    op.create_index("ix_post_topics_post_id", "post_topics", ["post_id"], unique=False)
    op.create_index("ix_post_topics_topic", "post_topics", ["topic"], unique=False)

    op.create_table(
        "user_model_states",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("weights", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("bias", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("samples_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("user_id"),
    )

    op.create_table(
        "user_embedding_profiles",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("vector", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("dim", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("samples_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("user_id"),
    )


def downgrade():
    op.drop_table("user_embedding_profiles")
    op.drop_table("user_model_states")
    op.drop_index("ix_post_topics_topic", table_name="post_topics")
    op.drop_index("ix_post_topics_post_id", table_name="post_topics")
    op.drop_index("ix_post_topics_id", table_name="post_topics")
    op.drop_table("post_topics")
