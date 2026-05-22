"""add digest schedules and runs

Revision ID: 82d4d2ce9cce
Revises: c4e1b9f3a6d7
Create Date: 2026-02-07 13:04:48.321735

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '82d4d2ce9cce'
down_revision: Union[str, None] = 'c4e1b9f3a6d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
