"""add digest_run pages

Revision ID: 333533edaa36
Revises: 18602e8c6a00
Create Date: 2026-02-13 18:39:54.401384

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '333533edaa36'
down_revision: Union[str, None] = '18602e8c6a00'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "digest_runs",
        sa.Column("pages", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

def downgrade() -> None:
    op.drop_column("digest_runs", "pages")
