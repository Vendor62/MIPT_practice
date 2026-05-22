"""add storyline debug enabled to users

Revision ID: 8f4c2d1a9b77
Revises: 5b0f8a7c1d2e
Create Date: 2026-04-08 10:40:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "8f4c2d1a9b77"
down_revision = "5b0f8a7c1d2e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("users")}
    if "storyline_debug_enabled" not in columns:
        op.add_column(
            "users",
            sa.Column(
                "storyline_debug_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("users")}
    if "storyline_debug_enabled" in columns:
        op.drop_column("users", "storyline_debug_enabled")
