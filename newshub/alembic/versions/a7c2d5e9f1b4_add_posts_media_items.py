"""add posts media_items

Revision ID: a7c2d5e9f1b4
Revises: f4a8c1d2e9b0
Create Date: 2026-04-22 10:26:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "a7c2d5e9f1b4"
down_revision = "f4a8c1d2e9b0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("posts")}

    if "media_items" not in columns:
        op.add_column("posts", sa.Column("media_items", sa.JSON(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("posts")}

    if "media_items" in columns:
        op.drop_column("posts", "media_items")
