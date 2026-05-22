"""add storyline update events

Revision ID: c1d2e3f4a6b8
Revises: b1d2e3f4a5b7, fa12c3d4e5b6
Create Date: 2026-05-13 15:20:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "c1d2e3f4a6b8"
down_revision = ("b1d2e3f4a5b7", "fa12c3d4e5b6")
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if "storyline_update_events" not in table_names:
        op.create_table(
            "storyline_update_events",
            sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("follow_id", sa.Integer(), sa.ForeignKey("user_storyline_follows.id"), nullable=False),
            sa.Column("family_root_storyline_id", sa.String(), nullable=False),
            sa.Column("event_key", sa.String(), nullable=False),
            sa.Column("canonical_post_id", sa.Integer(), sa.ForeignKey("posts.id"), nullable=True),
            sa.Column("last_post_id", sa.Integer(), sa.ForeignKey("posts.id"), nullable=True),
            sa.Column("telegram_message_id", sa.BigInteger(), nullable=True),
            sa.Column("delivery_target", sa.String(), nullable=False, server_default="main"),
            sa.Column("delivery_result", sa.String(), nullable=True),
            sa.Column("title", sa.String(), nullable=True),
            sa.Column("summary", sa.Text(), nullable=True),
            sa.Column(
                "sources",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default=sa.text("'[]'::jsonb"),
            ),
            sa.Column("source_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column("first_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint(
                "user_id",
                "follow_id",
                "family_root_storyline_id",
                "event_key",
                "delivery_target",
                name="uq_storyline_update_events_user_follow_event_target",
            ),
        )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("storyline_update_events")}
    desired_indexes = [
        ("ix_storyline_update_events_user_id", ["user_id"]),
        ("ix_storyline_update_events_follow_id", ["follow_id"]),
        ("ix_storyline_update_events_family_root_storyline_id", ["family_root_storyline_id"]),
        ("ix_storyline_update_events_event_key", ["event_key"]),
        ("ix_storyline_update_events_canonical_post_id", ["canonical_post_id"]),
        ("ix_storyline_update_events_last_post_id", ["last_post_id"]),
        ("ix_storyline_update_events_delivery_target", ["delivery_target"]),
    ]
    for index_name, columns in desired_indexes:
        if index_name not in existing_indexes:
            op.create_index(index_name, "storyline_update_events", columns, unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())
    if "storyline_update_events" not in table_names:
        return
    existing_indexes = {idx["name"] for idx in inspector.get_indexes("storyline_update_events")}
    for index_name in [
        "ix_storyline_update_events_delivery_target",
        "ix_storyline_update_events_last_post_id",
        "ix_storyline_update_events_canonical_post_id",
        "ix_storyline_update_events_event_key",
        "ix_storyline_update_events_family_root_storyline_id",
        "ix_storyline_update_events_follow_id",
        "ix_storyline_update_events_user_id",
    ]:
        if index_name in existing_indexes:
            op.drop_index(index_name, table_name="storyline_update_events")
    op.drop_table("storyline_update_events")
