"""add utc offset for digest settings

Revision ID: 18602e8c6a00
Revises: f3bacdc8abd1
Create Date: 2026-02-07 13:28:56.228258

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '18602e8c6a00'
down_revision: Union[str, None] = 'f3bacdc8abd1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "user_digest_settings",
        sa.Column("utc_offset_minutes", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_digest_settings", "utc_offset_minutes")
