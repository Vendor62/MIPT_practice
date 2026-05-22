"""add post_interactions table for button engagement

Revision ID: fa12c3d4e5b6
Revises: dd2382e90471
Create Date: 2026-05-07 19:40:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = "fa12c3d4e5b6"
down_revision = "dd2382e90471"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "post_interactions",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("post_id", sa.Integer(), nullable=False),
        sa.Column("community_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("interaction_count", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("first_interacted_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_interacted_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["post_id"], ["posts.id"]),
        sa.ForeignKeyConstraint(["community_id"], ["communities.id"]),
        sa.UniqueConstraint(
            "user_id",
            "post_id",
            "action",
            "source",
            name="uq_post_interactions_user_post_action_source",
        ),
    )
    op.create_index("ix_post_interactions_user_id", "post_interactions", ["user_id"])
    op.create_index("ix_post_interactions_post_id", "post_interactions", ["post_id"])
    op.create_index("ix_post_interactions_community_id", "post_interactions", ["community_id"])
    op.create_index("ix_post_interactions_action", "post_interactions", ["action"])
    op.create_index("ix_post_interactions_source", "post_interactions", ["source"])


def downgrade() -> None:
    op.drop_index("ix_post_interactions_source", table_name="post_interactions")
    op.drop_index("ix_post_interactions_action", table_name="post_interactions")
    op.drop_index("ix_post_interactions_community_id", table_name="post_interactions")
    op.drop_index("ix_post_interactions_post_id", table_name="post_interactions")
    op.drop_index("ix_post_interactions_user_id", table_name="post_interactions")
    op.drop_table("post_interactions")
