"""add bot assistant requests

Revision ID: d2f4a6b8c0e1
Revises: 8b1d2c7f0e44, a7c2d5e9f1b4
Create Date: 2026-04-23 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "d2f4a6b8c0e1"
down_revision = ("8b1d2c7f0e44", "a7c2d5e9f1b4")
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "bot_assistant_requests" not in tables:
        op.create_table(
            "bot_assistant_requests",
            sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
            sa.Column("request_id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("telegram_id", sa.BigInteger(), nullable=True),
            sa.Column("locale", sa.String(), nullable=False, server_default="ru"),
            sa.Column("raw_text", sa.Text(), nullable=False),
            sa.Column("intent", sa.String(), nullable=True),
            sa.Column("slots", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("faq_topic", sa.String(), nullable=True),
            sa.Column("needs_clarification", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("status", sa.String(), nullable=False, server_default="received"),
            sa.Column("outcome", sa.String(), nullable=True),
            sa.Column("error_code", sa.String(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("received_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("understood_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("clarified_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("proposed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
            sa.UniqueConstraint("request_id", name="uq_bot_assistant_requests_request_id"),
        )

    indexes = {idx["name"] for idx in inspector.get_indexes("bot_assistant_requests")}
    for name, columns, unique in [
        ("ix_bot_assistant_requests_id", ["id"], False),
        ("ix_bot_assistant_requests_request_id", ["request_id"], True),
        ("ix_bot_assistant_requests_user_id", ["user_id"], False),
        ("ix_bot_assistant_requests_telegram_id", ["telegram_id"], False),
        ("ix_bot_assistant_requests_intent", ["intent"], False),
        ("ix_bot_assistant_requests_faq_topic", ["faq_topic"], False),
        ("ix_bot_assistant_requests_needs_clarification", ["needs_clarification"], False),
        ("ix_bot_assistant_requests_status", ["status"], False),
        ("ix_bot_assistant_requests_outcome", ["outcome"], False),
        ("ix_bot_assistant_requests_error_code", ["error_code"], False),
        ("ix_bot_assistant_requests_received_at", ["received_at"], False),
    ]:
        if name not in indexes:
            op.create_index(name, "bot_assistant_requests", columns, unique=unique)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "bot_assistant_requests" not in tables:
        return

    indexes = {idx["name"] for idx in inspector.get_indexes("bot_assistant_requests")}
    for name in [
        "ix_bot_assistant_requests_received_at",
        "ix_bot_assistant_requests_error_code",
        "ix_bot_assistant_requests_outcome",
        "ix_bot_assistant_requests_status",
        "ix_bot_assistant_requests_needs_clarification",
        "ix_bot_assistant_requests_faq_topic",
        "ix_bot_assistant_requests_intent",
        "ix_bot_assistant_requests_telegram_id",
        "ix_bot_assistant_requests_user_id",
        "ix_bot_assistant_requests_request_id",
        "ix_bot_assistant_requests_id",
    ]:
        if name in indexes:
            op.drop_index(name, table_name="bot_assistant_requests")
    op.drop_table("bot_assistant_requests")
