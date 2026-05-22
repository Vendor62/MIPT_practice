"""add user storyline follows

Revision ID: 5b0f8a7c1d2e
Revises: f2c7d9a1b4e8
Create Date: 2026-04-08 10:27:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "5b0f8a7c1d2e"
down_revision = "f2c7d9a1b4e8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if "user_storyline_follows" not in table_names:
        op.create_table(
            "user_storyline_follows",
            sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("source_post_id", sa.Integer(), sa.ForeignKey("posts.id"), nullable=True),
            sa.Column("storyline_id", sa.String(), nullable=False),
            sa.Column("story_family_id", sa.String(), nullable=False),
            sa.Column("family_root_storyline_id", sa.String(), nullable=False),
            sa.Column("storyline_title", sa.String(), nullable=True),
            sa.Column(
                "branch_mode",
                sa.String(),
                nullable=False,
                server_default="root_only",
            ),
            sa.Column(
                "selected_branch_ids",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default=sa.text("'[]'::jsonb"),
            ),
            sa.Column(
                "is_active",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("true"),
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.UniqueConstraint(
                "user_id",
                "family_root_storyline_id",
                name="uq_user_storyline_follows_user_family_root",
            ),
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("user_storyline_follows")}
    desired_indexes = [
        ("ix_user_storyline_follows_user_id", ["user_id"]),
        ("ix_user_storyline_follows_source_post_id", ["source_post_id"]),
        ("ix_user_storyline_follows_storyline_id", ["storyline_id"]),
        ("ix_user_storyline_follows_story_family_id", ["story_family_id"]),
        ("ix_user_storyline_follows_family_root_storyline_id", ["family_root_storyline_id"]),
        ("ix_user_storyline_follows_is_active", ["is_active"]),
    ]
    for index_name, columns in desired_indexes:
        if index_name not in existing_indexes:
            op.create_index(index_name, "user_storyline_follows", columns, unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if "user_storyline_follows" not in table_names:
        return
    existing_indexes = {idx["name"] for idx in inspector.get_indexes("user_storyline_follows")}
    for index_name in [
        "ix_user_storyline_follows_is_active",
        "ix_user_storyline_follows_family_root_storyline_id",
        "ix_user_storyline_follows_story_family_id",
        "ix_user_storyline_follows_storyline_id",
        "ix_user_storyline_follows_source_post_id",
        "ix_user_storyline_follows_user_id",
    ]:
        if index_name in existing_indexes:
            op.drop_index(index_name, table_name="user_storyline_follows")
    op.drop_table("user_storyline_follows")
