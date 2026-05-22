"""add user csi surveys

Revision ID: f1a2b3c4d5e6
Revises: e8f2a4c6b9d1
Create Date: 2026-04-30 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "f1a2b3c4d5e6"
down_revision = "e8f2a4c6b9d1"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_csi_surveys",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("first_delivery_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_count_at_prompt", sa.Integer(), nullable=True),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("asked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retry_asked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("abandoned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", name="uq_user_csi_surveys_user_id"),
    )
    op.create_index(op.f("ix_user_csi_surveys_id"), "user_csi_surveys", ["id"], unique=False)
    op.create_index(op.f("ix_user_csi_surveys_user_id"), "user_csi_surveys", ["user_id"], unique=False)
    op.create_index(op.f("ix_user_csi_surveys_first_delivery_at"), "user_csi_surveys", ["first_delivery_at"], unique=False)
    op.create_index(op.f("ix_user_csi_surveys_due_at"), "user_csi_surveys", ["due_at"], unique=False)
    op.create_index(op.f("ix_user_csi_surveys_answered_at"), "user_csi_surveys", ["answered_at"], unique=False)


def downgrade():
    op.drop_index(op.f("ix_user_csi_surveys_answered_at"), table_name="user_csi_surveys")
    op.drop_index(op.f("ix_user_csi_surveys_due_at"), table_name="user_csi_surveys")
    op.drop_index(op.f("ix_user_csi_surveys_first_delivery_at"), table_name="user_csi_surveys")
    op.drop_index(op.f("ix_user_csi_surveys_user_id"), table_name="user_csi_surveys")
    op.drop_index(op.f("ix_user_csi_surveys_id"), table_name="user_csi_surveys")
    op.drop_table("user_csi_surveys")
