"""add asked timestamps for baseline/current news-time survey stages

Revision ID: ab12cd34ef56
Revises: f1a2b3c4d5e6
Create Date: 2026-05-09 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "ab12cd34ef56"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("user_news_time_surveys", sa.Column("baseline_asked_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("user_news_time_surveys", sa.Column("baseline_retry_asked_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("user_news_time_surveys", sa.Column("current_asked_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("user_news_time_surveys", sa.Column("current_retry_asked_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("user_news_time_surveys", "current_retry_asked_at")
    op.drop_column("user_news_time_surveys", "current_asked_at")
    op.drop_column("user_news_time_surveys", "baseline_retry_asked_at")
    op.drop_column("user_news_time_surveys", "baseline_asked_at")
