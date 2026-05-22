"""add instruction filter tables

Revision ID: b7c3d9e1f2a4
Revises: 6c7d8e9f0a1b, 8f4c2d1a9b77
Create Date: 2026-04-21 15:20:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "b7c3d9e1f2a4"
down_revision = ("6c7d8e9f0a1b", "8f4c2d1a9b77")
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "user_instruction_rules" not in tables:
        op.create_table(
            "user_instruction_rules",
            sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("scope", sa.String(), nullable=False),
            sa.Column("community_id", sa.Integer(), sa.ForeignKey("communities.id"), nullable=True),
            sa.Column("prompt_text", sa.Text(), nullable=False),
            sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
        )

    if "instruction_filter_decisions" not in tables:
        op.create_table(
            "instruction_filter_decisions",
            sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("post_id", sa.Integer(), sa.ForeignKey("posts.id"), nullable=False),
            sa.Column("community_id", sa.Integer(), sa.ForeignKey("communities.id"), nullable=True),
            sa.Column("effective_scope", sa.String(), nullable=False),
            sa.Column("effective_prompt_hash", sa.String(), nullable=False),
            sa.Column("decision", sa.String(), nullable=False),
            sa.Column("reason_short", sa.String(), nullable=True),
            sa.Column("raw_response", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("latency_ms", sa.Integer(), nullable=True),
            sa.Column("error_code", sa.String(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.func.now()),
            sa.UniqueConstraint(
                "user_id",
                "post_id",
                "effective_prompt_hash",
                name="uq_instruction_filter_decision_prompt",
            ),
        )

    user_rule_indexes = {idx["name"] for idx in inspector.get_indexes("user_instruction_rules")}
    if "ix_user_instruction_rules_user_id" not in user_rule_indexes:
        op.create_index("ix_user_instruction_rules_user_id", "user_instruction_rules", ["user_id"], unique=False)
    if "ix_user_instruction_rules_scope" not in user_rule_indexes:
        op.create_index("ix_user_instruction_rules_scope", "user_instruction_rules", ["scope"], unique=False)
    if "ix_user_instruction_rules_community_id" not in user_rule_indexes:
        op.create_index("ix_user_instruction_rules_community_id", "user_instruction_rules", ["community_id"], unique=False)
    if "ix_user_instruction_rules_is_enabled" not in user_rule_indexes:
        op.create_index("ix_user_instruction_rules_is_enabled", "user_instruction_rules", ["is_enabled"], unique=False)

    decision_indexes = {idx["name"] for idx in inspector.get_indexes("instruction_filter_decisions")}
    if "ix_instruction_filter_decisions_user_id" not in decision_indexes:
        op.create_index("ix_instruction_filter_decisions_user_id", "instruction_filter_decisions", ["user_id"], unique=False)
    if "ix_instruction_filter_decisions_post_id" not in decision_indexes:
        op.create_index("ix_instruction_filter_decisions_post_id", "instruction_filter_decisions", ["post_id"], unique=False)
    if "ix_instruction_filter_decisions_community_id" not in decision_indexes:
        op.create_index("ix_instruction_filter_decisions_community_id", "instruction_filter_decisions", ["community_id"], unique=False)
    if "ix_instruction_filter_decisions_effective_scope" not in decision_indexes:
        op.create_index("ix_instruction_filter_decisions_effective_scope", "instruction_filter_decisions", ["effective_scope"], unique=False)
    if "ix_instruction_filter_decisions_effective_prompt_hash" not in decision_indexes:
        op.create_index(
            "ix_instruction_filter_decisions_effective_prompt_hash",
            "instruction_filter_decisions",
            ["effective_prompt_hash"],
            unique=False,
        )
    if "ix_instruction_filter_decisions_decision" not in decision_indexes:
        op.create_index("ix_instruction_filter_decisions_decision", "instruction_filter_decisions", ["decision"], unique=False)
    if "ix_instruction_filter_decisions_lookup" not in decision_indexes:
        op.create_index(
            "ix_instruction_filter_decisions_lookup",
            "instruction_filter_decisions",
            ["user_id", "post_id", "effective_prompt_hash"],
            unique=False,
        )

    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_user_instruction_rules_global
        ON user_instruction_rules (user_id)
        WHERE scope = 'global' AND is_enabled = true;
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_user_instruction_rules_community
        ON user_instruction_rules (user_id, community_id)
        WHERE scope = 'community' AND community_id IS NOT NULL AND is_enabled = true;
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "instruction_filter_decisions" in tables:
        decision_indexes = {idx["name"] for idx in inspector.get_indexes("instruction_filter_decisions")}
        for name in [
            "ix_instruction_filter_decisions_lookup",
            "ix_instruction_filter_decisions_decision",
            "ix_instruction_filter_decisions_effective_prompt_hash",
            "ix_instruction_filter_decisions_effective_scope",
            "ix_instruction_filter_decisions_community_id",
            "ix_instruction_filter_decisions_post_id",
            "ix_instruction_filter_decisions_user_id",
        ]:
            if name in decision_indexes:
                op.drop_index(name, table_name="instruction_filter_decisions")
        op.drop_table("instruction_filter_decisions")

    if "user_instruction_rules" in tables:
        user_rule_indexes = {idx["name"] for idx in inspector.get_indexes("user_instruction_rules")}
        for name in [
            "uq_user_instruction_rules_community",
            "uq_user_instruction_rules_global",
            "ix_user_instruction_rules_is_enabled",
            "ix_user_instruction_rules_community_id",
            "ix_user_instruction_rules_scope",
            "ix_user_instruction_rules_user_id",
        ]:
            if name in user_rule_indexes:
                op.drop_index(name, table_name="user_instruction_rules")
            elif name.startswith("uq_user_instruction_rules_"):
                op.execute(f"DROP INDEX IF EXISTS {name}")
        op.drop_table("user_instruction_rules")
