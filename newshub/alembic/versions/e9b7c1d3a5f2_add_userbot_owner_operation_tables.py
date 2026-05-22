"""add userbot owner operation tables

Revision ID: e9b7c1d3a5f2
Revises: d2f4a6b8c0e1
Create Date: 2026-04-24 09:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "e9b7c1d3a5f2"
down_revision = "d2f4a6b8c0e1"
branch_labels = None
depends_on = None


def _columns(inspector, table_name: str) -> set[str]:
    return {column["name"] for column in inspector.get_columns(table_name)}


def _indexes(inspector, table_name: str) -> set[str]:
    return {index["name"] for index in inspector.get_indexes(table_name)}


def _unique_constraints(inspector, table_name: str) -> set[str]:
    return {constraint["name"] for constraint in inspector.get_unique_constraints(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    community_columns = _columns(inspector, "communities")
    for name, column in [
        ("access_hash", sa.Column("access_hash", sa.BigInteger(), nullable=True)),
        ("username", sa.Column("username", sa.String(), nullable=True)),
        ("last_resolved_at", sa.Column("last_resolved_at", sa.DateTime(timezone=True), nullable=True)),
        ("resolve_status", sa.Column("resolve_status", sa.String(), nullable=True)),
        ("last_telegram_error", sa.Column("last_telegram_error", sa.Text(), nullable=True)),
    ]:
        if name not in community_columns:
            op.add_column("communities", column)

    community_indexes = _indexes(inspector, "communities")
    for name, columns in [
        ("ix_communities_username", ["username"]),
        ("ix_communities_resolve_status", ["resolve_status"]),
    ]:
        if name not in community_indexes:
            op.create_index(name, "communities", columns, unique=False)

    if "community_aliases" not in tables:
        op.create_table(
            "community_aliases",
            sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
            sa.Column("community_id", sa.Integer(), sa.ForeignKey("communities.id"), nullable=False),
            sa.Column("alias_type", sa.String(), nullable=False),
            sa.Column("value_normalized", sa.String(), nullable=False),
            sa.Column("last_seen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.UniqueConstraint("value_normalized", name="uq_community_aliases_value_normalized"),
        )
    alias_indexes = _indexes(sa.inspect(bind), "community_aliases")
    for name, columns in [
        ("ix_community_aliases_id", ["id"]),
        ("ix_community_aliases_community_id", ["community_id"]),
        ("ix_community_aliases_alias_type", ["alias_type"]),
        ("ix_community_aliases_value_normalized", ["value_normalized"]),
    ]:
        if name not in alias_indexes:
            op.create_index(name, "community_aliases", columns, unique=False)

    # Backfill canonical link aliases from existing communities. Lower-casing
    # matches app-side normalization and makes the fast path immediately useful.
    op.execute(
        """
        INSERT INTO community_aliases (community_id, alias_type, value_normalized, last_seen_at)
        SELECT id, 'link', lower(trim(trailing '/' from link)), now()
        FROM communities
        WHERE link IS NOT NULL AND trim(link) <> ''
        ON CONFLICT (value_normalized) DO NOTHING
        """
    )

    if "userbot_accounts" not in tables:
        op.create_table(
            "userbot_accounts",
            sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
            sa.Column("slot", sa.String(), nullable=False),
            sa.Column("session_name", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, server_default="active"),
            sa.Column("flood_wait_until", sa.DateTime(timezone=True), nullable=True),
            sa.Column("next_resolve_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("next_join_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.UniqueConstraint("slot", name="uq_userbot_accounts_slot"),
        )
    account_indexes = _indexes(sa.inspect(bind), "userbot_accounts")
    for name, columns, unique in [
        ("ix_userbot_accounts_id", ["id"], False),
        ("ix_userbot_accounts_slot", ["slot"], True),
        ("ix_userbot_accounts_status", ["status"], False),
        ("ix_userbot_accounts_flood_wait_until", ["flood_wait_until"], False),
        ("ix_userbot_accounts_next_resolve_at", ["next_resolve_at"], False),
        ("ix_userbot_accounts_next_join_at", ["next_join_at"], False),
    ]:
        if name not in account_indexes:
            op.create_index(name, "userbot_accounts", columns, unique=unique)

    if "telegram_operations" not in tables:
        op.create_table(
            "telegram_operations",
            sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
            sa.Column("kind", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, server_default="queued"),
            sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
            sa.Column("run_after", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("userbot_slot", sa.String(), nullable=True),
            sa.Column("community_id", sa.Integer(), sa.ForeignKey("communities.id"), nullable=True),
            sa.Column("subscription_request_id", sa.Integer(), sa.ForeignKey("subscription_requests.id"), nullable=True),
            sa.Column("target", sa.String(), nullable=False),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        )
    operation_indexes = _indexes(sa.inspect(bind), "telegram_operations")
    for name, columns in [
        ("ix_telegram_operations_id", ["id"]),
        ("ix_telegram_operations_kind", ["kind"]),
        ("ix_telegram_operations_status", ["status"]),
        ("ix_telegram_operations_priority", ["priority"]),
        ("ix_telegram_operations_run_after", ["run_after"]),
        ("ix_telegram_operations_userbot_slot", ["userbot_slot"]),
        ("ix_telegram_operations_community_id", ["community_id"]),
        ("ix_telegram_operations_subscription_request_id", ["subscription_request_id"]),
        ("ix_telegram_operations_target", ["target"]),
    ]:
        if name not in operation_indexes:
            op.create_index(name, "telegram_operations", columns, unique=False)

    # Remove duplicate user/community rows before enforcing idempotency.
    op.execute(
        """
        DELETE FROM users_communities uc
        USING users_communities dup
        WHERE uc.user_id = dup.user_id
          AND uc.community_id = dup.community_id
          AND uc.id > dup.id
        """
    )
    user_community_uniques = _unique_constraints(sa.inspect(bind), "users_communities")
    if "uq_users_communities_user_community" not in user_community_uniques:
        op.create_unique_constraint(
            "uq_users_communities_user_community",
            "users_communities",
            ["user_id", "community_id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "users_communities" in tables:
        uniques = _unique_constraints(inspector, "users_communities")
        if "uq_users_communities_user_community" in uniques:
            op.drop_constraint("uq_users_communities_user_community", "users_communities", type_="unique")

    for table_name in ["telegram_operations", "userbot_accounts", "community_aliases"]:
        if table_name in tables:
            op.drop_table(table_name)

    inspector = sa.inspect(bind)
    if "communities" in set(inspector.get_table_names()):
        indexes = _indexes(inspector, "communities")
        for name in ["ix_communities_resolve_status", "ix_communities_username"]:
            if name in indexes:
                op.drop_index(name, table_name="communities")
        columns = _columns(inspector, "communities")
        for name in [
            "last_telegram_error",
            "resolve_status",
            "last_resolved_at",
            "username",
            "access_hash",
        ]:
            if name in columns:
                op.drop_column("communities", name)
