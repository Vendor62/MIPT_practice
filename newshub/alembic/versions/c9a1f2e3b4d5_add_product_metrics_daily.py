"""add product_metrics_daily for Grafana 30d charts

Revision ID: c9a1f2e3b4d5
Revises: f2c7d9a1b4e8
Create Date: 2026-04-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c9a1f2e3b4d5"
down_revision: Union[str, None] = "f2c7d9a1b4e8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "product_metrics_daily",
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("subscribers_total", sa.Float(), nullable=False, server_default="0"),
        sa.Column("engagement_ratio_30d", sa.Float(), nullable=False, server_default="0"),
        sa.Column("avg_subscriptions_per_subscriber", sa.Float(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("snapshot_date"),
    )


def downgrade() -> None:
    op.drop_table("product_metrics_daily")
