"""add aa experiment infra

Revision ID: a9b8c7d6e5f4
Revises: f1a2b3c4d5e6
Create Date: 2026-04-30 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "a9b8c7d6e5f4"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "experiment_assignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("experiment_key", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("variant", sa.String(), nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("experiment_key", "user_id", name="uq_experiment_assignments_key_user"),
    )
    op.create_index(op.f("ix_experiment_assignments_id"), "experiment_assignments", ["id"], unique=False)
    op.create_index(op.f("ix_experiment_assignments_experiment_key"), "experiment_assignments", ["experiment_key"], unique=False)
    op.create_index(op.f("ix_experiment_assignments_user_id"), "experiment_assignments", ["user_id"], unique=False)
    op.create_index(op.f("ix_experiment_assignments_variant"), "experiment_assignments", ["variant"], unique=False)

    op.create_table(
        "experiment_exposures",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("experiment_key", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("post_id", sa.Integer(), nullable=False),
        sa.Column("variant", sa.String(), nullable=False),
        sa.Column("eligible", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("render_mode", sa.String(), nullable=True),
        sa.Column("content_chars", sa.Integer(), nullable=True),
        sa.Column("summary_chars", sa.Integer(), nullable=True),
        sa.Column("has_summary", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("delivery_result", sa.String(), nullable=True),
        sa.Column("telegram_message_id", sa.BigInteger(), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["post_id"], ["posts.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("experiment_key", "user_id", "post_id", name="uq_experiment_exposures_key_user_post"),
    )
    op.create_index(op.f("ix_experiment_exposures_id"), "experiment_exposures", ["id"], unique=False)
    op.create_index(op.f("ix_experiment_exposures_experiment_key"), "experiment_exposures", ["experiment_key"], unique=False)
    op.create_index(op.f("ix_experiment_exposures_user_id"), "experiment_exposures", ["user_id"], unique=False)
    op.create_index(op.f("ix_experiment_exposures_post_id"), "experiment_exposures", ["post_id"], unique=False)
    op.create_index(op.f("ix_experiment_exposures_variant"), "experiment_exposures", ["variant"], unique=False)
    op.create_index(op.f("ix_experiment_exposures_sent_at"), "experiment_exposures", ["sent_at"], unique=False)


def downgrade():
    op.drop_index(op.f("ix_experiment_exposures_sent_at"), table_name="experiment_exposures")
    op.drop_index(op.f("ix_experiment_exposures_variant"), table_name="experiment_exposures")
    op.drop_index(op.f("ix_experiment_exposures_post_id"), table_name="experiment_exposures")
    op.drop_index(op.f("ix_experiment_exposures_user_id"), table_name="experiment_exposures")
    op.drop_index(op.f("ix_experiment_exposures_experiment_key"), table_name="experiment_exposures")
    op.drop_index(op.f("ix_experiment_exposures_id"), table_name="experiment_exposures")
    op.drop_table("experiment_exposures")
    op.drop_index(op.f("ix_experiment_assignments_variant"), table_name="experiment_assignments")
    op.drop_index(op.f("ix_experiment_assignments_user_id"), table_name="experiment_assignments")
    op.drop_index(op.f("ix_experiment_assignments_experiment_key"), table_name="experiment_assignments")
    op.drop_index(op.f("ix_experiment_assignments_id"), table_name="experiment_assignments")
    op.drop_table("experiment_assignments")
