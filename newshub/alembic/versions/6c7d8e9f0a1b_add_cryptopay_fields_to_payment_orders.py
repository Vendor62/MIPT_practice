"""add cryptopay fields to payment orders

Revision ID: 6c7d8e9f0a1b
Revises: 5b0f8a7c1d2e
Create Date: 2026-04-08 13:15:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "6c7d8e9f0a1b"
down_revision = "5b0f8a7c1d2e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("payment_orders")}

    if "provider" not in columns:
        op.add_column(
            "payment_orders",
            sa.Column("provider", sa.String(), nullable=False, server_default="tbank"),
        )
    if "quote_currency" not in columns:
        op.add_column(
            "payment_orders",
            sa.Column("quote_currency", sa.String(), nullable=False, server_default="RUB"),
        )
    if "quote_amount" not in columns:
        op.add_column(
            "payment_orders",
            sa.Column("quote_amount", sa.Numeric(20, 9), nullable=True),
        )
    if "price_code" not in columns:
        op.add_column(
            "payment_orders",
            sa.Column("price_code", sa.String(), nullable=False, server_default="premium_rub"),
        )

    op.execute(
        """
        UPDATE payment_orders
        SET provider = COALESCE(NULLIF(provider, ''), 'tbank'),
            quote_currency = COALESCE(NULLIF(quote_currency, ''), 'RUB'),
            quote_amount = COALESCE(quote_amount, amount_rub::numeric),
            price_code = COALESCE(NULLIF(price_code, ''), 'premium_rub')
        WHERE provider IS NULL
           OR provider = ''
           OR quote_currency IS NULL
           OR quote_currency = ''
           OR quote_amount IS NULL
           OR price_code IS NULL
           OR price_code = '';
        """
    )

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("payment_orders")}
    if "ix_payment_orders_provider" not in existing_indexes:
        op.create_index("ix_payment_orders_provider", "payment_orders", ["provider"], unique=False)
    if "ix_payment_orders_price_code" not in existing_indexes:
        op.create_index("ix_payment_orders_price_code", "payment_orders", ["price_code"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("payment_orders")}
    existing_indexes = {idx["name"] for idx in inspector.get_indexes("payment_orders")}

    if "ix_payment_orders_price_code" in existing_indexes:
        op.drop_index("ix_payment_orders_price_code", table_name="payment_orders")
    if "ix_payment_orders_provider" in existing_indexes:
        op.drop_index("ix_payment_orders_provider", table_name="payment_orders")

    if "price_code" in columns:
        op.drop_column("payment_orders", "price_code")
    if "quote_amount" in columns:
        op.drop_column("payment_orders", "quote_amount")
    if "quote_currency" in columns:
        op.drop_column("payment_orders", "quote_currency")
    if "provider" in columns:
        op.drop_column("payment_orders", "provider")
