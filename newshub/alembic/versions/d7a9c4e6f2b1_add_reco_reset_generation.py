"""add reco reset generation

Revision ID: d7a9c4e6f2b1
Revises: b4e6f8a2c9d3
Create Date: 2026-04-30 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "d7a9c4e6f2b1"
down_revision = "b4e6f8a2c9d3"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "user_model_states",
        sa.Column("generation", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column("user_model_states", sa.Column("reset_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "user_embedding_profiles",
        sa.Column("generation", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column("user_embedding_profiles", sa.Column("reset_at", sa.DateTime(timezone=True), nullable=True))


def downgrade():
    op.drop_column("user_embedding_profiles", "reset_at")
    op.drop_column("user_embedding_profiles", "generation")
    op.drop_column("user_model_states", "reset_at")
    op.drop_column("user_model_states", "generation")
