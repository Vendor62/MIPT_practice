"""add posts grouped_id

Revision ID: f4a8c1d2e9b0
Revises: e3d9b1a2f6c7
Create Date: 2026-04-22 10:15:00
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "f4a8c1d2e9b0"
down_revision = "e3d9b1a2f6c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("posts")}
    indexes = {index["name"] for index in inspector.get_indexes("posts")}

    if "grouped_id" not in columns:
        op.add_column("posts", sa.Column("grouped_id", sa.BigInteger(), nullable=True))
    if "ix_posts_grouped_id" not in indexes:
        op.create_index("ix_posts_grouped_id", "posts", ["grouped_id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("posts")}
    indexes = {index["name"] for index in inspector.get_indexes("posts")}

    if "ix_posts_grouped_id" in indexes:
        op.drop_index("ix_posts_grouped_id", table_name="posts")
    if "grouped_id" in columns:
        op.drop_column("posts", "grouped_id")
