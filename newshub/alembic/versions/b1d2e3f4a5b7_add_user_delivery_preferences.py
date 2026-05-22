"""add user delivery preferences for plus bot routing

Revision ID: b1d2e3f4a5b7
Revises: ab12cd34ef56
Create Date: 2026-05-13 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b1d2e3f4a5b7"
down_revision = "ab12cd34ef56"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_delivery_preferences",
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), primary_key=True, nullable=False),
        sa.Column("digest_to_plus_bot", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("storyline_to_plus_bot", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("plus_bot_connected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("user_delivery_preferences")
