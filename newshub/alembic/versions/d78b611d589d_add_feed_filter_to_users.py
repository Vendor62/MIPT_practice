from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "d78b611d589d"
down_revision = "36ea519df1ef"  # поставь сюда текущий head из alembic history
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "feed_filter",
            sa.String(),
            nullable=False,
            server_default="all",
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "feed_filter")
