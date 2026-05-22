from alembic import op
import sqlalchemy as sa


revision = "e1f3a5b7c9d1"
down_revision = ("d4b6a8c2e1f0", "d78b611d589d")
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("language_code", sa.String(), nullable=False, server_default="ru"),
    )


def downgrade() -> None:
    op.drop_column("users", "language_code")
