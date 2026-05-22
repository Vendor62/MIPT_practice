from alembic import op
import sqlalchemy as sa


revision = "f2c7d9a1b4e8"
down_revision = "e1f3a5b7c9d1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "last_live_feed_filter",
            sa.String(),
            nullable=False,
            server_default="all",
        ),
    )
    op.execute(
        """
        UPDATE users
        SET last_live_feed_filter = CASE
            WHEN feed_filter IN ('all', 'not_interesting', 'only_fire') THEN feed_filter
            ELSE 'all'
        END
        """
    )


def downgrade() -> None:
    op.drop_column("users", "last_live_feed_filter")
