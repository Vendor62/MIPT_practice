"""add chat_id to communities

Revision ID: 36ea519df1ef
Revises: 333533edaa36
Create Date: 2026-02-14 16:00:59.818475

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision = "36ea519df1ef"
down_revision = "333533edaa36"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("communities", sa.Column("chat_id", sa.BigInteger(), nullable=True))
    op.create_index("ix_communities_chat_id", "communities", ["chat_id"], unique=False)
    # partial unique (Postgres): один chat_id -> одна community
    op.execute(
        "CREATE UNIQUE INDEX uq_communities_chat_id_not_null "
        "ON communities (chat_id) WHERE chat_id IS NOT NULL"
    )

def downgrade():
    op.execute("DROP INDEX IF EXISTS uq_communities_chat_id_not_null")
    op.drop_index("ix_communities_chat_id", table_name="communities")
    op.drop_column("communities", "chat_id")

