"""add user news time surveys

Revision ID: e8f2a4c6b9d1
Revises: d7a9c4e6f2b1
Create Date: 2026-04-30 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "e8f2a4c6b9d1"
down_revision = "d7a9c4e6f2b1"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_news_time_surveys",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("baseline_answer", sa.String(), nullable=True),
        sa.Column("baseline_minutes", sa.Integer(), nullable=True),
        sa.Column("baseline_answered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("followup_due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("followup_asked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("followup_retry_asked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("followup_abandoned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("time_saved_answer", sa.String(), nullable=True),
        sa.Column("time_saved_answered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("current_answer", sa.String(), nullable=True),
        sa.Column("current_minutes", sa.Integer(), nullable=True),
        sa.Column("current_answered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", name="uq_user_news_time_surveys_user_id"),
    )
    op.create_index(op.f("ix_user_news_time_surveys_id"), "user_news_time_surveys", ["id"], unique=False)
    op.create_index(op.f("ix_user_news_time_surveys_user_id"), "user_news_time_surveys", ["user_id"], unique=False)
    op.create_index(
        op.f("ix_user_news_time_surveys_baseline_answered_at"),
        "user_news_time_surveys",
        ["baseline_answered_at"],
        unique=False,
    )
    op.create_index(op.f("ix_user_news_time_surveys_followup_due_at"), "user_news_time_surveys", ["followup_due_at"], unique=False)
    op.create_index(op.f("ix_user_news_time_surveys_time_saved_answer"), "user_news_time_surveys", ["time_saved_answer"], unique=False)
    op.create_index(
        op.f("ix_user_news_time_surveys_time_saved_answered_at"),
        "user_news_time_surveys",
        ["time_saved_answered_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_user_news_time_surveys_current_answered_at"),
        "user_news_time_surveys",
        ["current_answered_at"],
        unique=False,
    )


def downgrade():
    op.drop_index(op.f("ix_user_news_time_surveys_current_answered_at"), table_name="user_news_time_surveys")
    op.drop_index(op.f("ix_user_news_time_surveys_time_saved_answered_at"), table_name="user_news_time_surveys")
    op.drop_index(op.f("ix_user_news_time_surveys_time_saved_answer"), table_name="user_news_time_surveys")
    op.drop_index(op.f("ix_user_news_time_surveys_followup_due_at"), table_name="user_news_time_surveys")
    op.drop_index(op.f("ix_user_news_time_surveys_baseline_answered_at"), table_name="user_news_time_surveys")
    op.drop_index(op.f("ix_user_news_time_surveys_user_id"), table_name="user_news_time_surveys")
    op.drop_index(op.f("ix_user_news_time_surveys_id"), table_name="user_news_time_surveys")
    op.drop_table("user_news_time_surveys")
