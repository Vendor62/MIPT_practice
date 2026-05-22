"""add userbot slots for communities and subscription requests

Revision ID: c8f4d2a1b6e0
Revises: a1b2c3d4e5f6
Create Date: 2026-03-14 14:45:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "c8f4d2a1b6e0"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "communities",
        sa.Column("userbot_slot", sa.String(), nullable=False, server_default="primary"),
    )
    op.add_column(
        "subscription_requests",
        sa.Column("userbot_slot", sa.String(), nullable=False, server_default="primary"),
    )
    op.create_index("ix_communities_userbot_slot", "communities", ["userbot_slot"], unique=False)
    op.create_index("ix_subscription_requests_userbot_slot", "subscription_requests", ["userbot_slot"], unique=False)
    op.execute("UPDATE communities SET userbot_slot='primary' WHERE userbot_slot IS NULL OR userbot_slot='';")
    op.execute("UPDATE subscription_requests SET userbot_slot='primary' WHERE userbot_slot IS NULL OR userbot_slot='';")


def downgrade():
    op.drop_index("ix_subscription_requests_userbot_slot", table_name="subscription_requests")
    op.drop_index("ix_communities_userbot_slot", table_name="communities")
    op.drop_column("subscription_requests", "userbot_slot")
    op.drop_column("communities", "userbot_slot")
