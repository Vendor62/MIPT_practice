import os

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    BigInteger,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
    text,
    Text,
    SmallInteger
)
from sqlalchemy.dialects.postgresql import JSONB, JSON
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

load_dotenv()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(BigInteger, unique=True, index=True, nullable=False)

    first_name = Column(String)
    last_name = Column(String)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    language_code = Column(String, nullable=False, default="ru", server_default="ru")
    storyline_debug_enabled = Column(Boolean, default=False, nullable=False, server_default=text("false"))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # from _g
    summary_enabled = Column(Boolean, default=False)
    feed_filter = Column(String, default="all", nullable=False)
    last_live_feed_filter = Column(String, default="all", nullable=False, server_default="all")
    engagement_score = Column(Float, default=0.0)

    reactions = relationship("Reaction", back_populates="user")
    communities = relationship("UserCommunity", back_populates="user")
    news_time_survey = relationship("UserNewsTimeSurvey", back_populates="user", uselist=False)
    csi_survey = relationship("UserCsiSurvey", back_populates="user", uselist=False)
    experiment_assignments = relationship("ExperimentAssignment", back_populates="user")
    experiment_exposures = relationship("ExperimentExposure", back_populates="user")


class Community(Base):
    __tablename__ = "communities"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(BigInteger, nullable=True)
    access_hash = Column(BigInteger, nullable=True)
    username = Column(String, nullable=True, index=True)

    name = Column(String, unique=True)
    description = Column(String)
    is_active = Column(Boolean, default=True)
    link = Column(String)
    userbot_slot = Column(String, nullable=False, default="primary", index=True)
    last_resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolve_status = Column(String, nullable=True, index=True)
    last_telegram_error = Column(Text, nullable=True)

    last_parsed_id = Column(BigInteger, nullable=True)

    posts = relationship("Post", back_populates="community")
    users = relationship("UserCommunity", back_populates="community")


class UserCommunity(Base):
    __tablename__ = "users_communities"
    __table_args__ = (
        UniqueConstraint("user_id", "community_id", name="uq_users_communities_user_community"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    community_id = Column(Integer, ForeignKey("communities.id"), index=True)

    user = relationship("User", back_populates="communities")
    community = relationship("Community", back_populates="users")


class CommunityAlias(Base):
    __tablename__ = "community_aliases"
    __table_args__ = (
        UniqueConstraint("value_normalized", name="uq_community_aliases_value_normalized"),
    )

    id = Column(Integer, primary_key=True, index=True)
    community_id = Column(Integer, ForeignKey("communities.id"), nullable=False, index=True)
    alias_type = Column(String, nullable=False, index=True)
    value_normalized = Column(String, nullable=False, index=True)
    last_seen_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserbotAccount(Base):
    __tablename__ = "userbot_accounts"

    id = Column(Integer, primary_key=True, index=True)
    slot = Column(String, unique=True, nullable=False, index=True)
    session_name = Column(String, nullable=False)
    status = Column(String, nullable=False, default="active", server_default="active", index=True)
    flood_wait_until = Column(DateTime(timezone=True), nullable=True, index=True)
    next_resolve_at = Column(DateTime(timezone=True), nullable=True, index=True)
    next_join_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_error = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TelegramOperation(Base):
    __tablename__ = "telegram_operations"

    id = Column(Integer, primary_key=True, index=True)
    kind = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default="queued", server_default="queued", index=True)
    priority = Column(Integer, nullable=False, default=100, server_default="100", index=True)
    run_after = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    attempts = Column(Integer, nullable=False, default=0, server_default="0")
    userbot_slot = Column(String, nullable=True, index=True)
    community_id = Column(Integer, ForeignKey("communities.id"), nullable=True, index=True)
    subscription_request_id = Column(Integer, ForeignKey("subscription_requests.id"), nullable=True, index=True)
    target = Column(String, nullable=False, index=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ProductMetricsDaily(Base):
    """Один снимок продуктовых метрик на календарный день (UTC); обновляется при каждом скрейпе /metrics."""

    __tablename__ = "product_metrics_daily"

    snapshot_date = Column(Date, primary_key=True, nullable=False)
    subscribers_total = Column(Float, nullable=False, default=0.0)
    engagement_ratio_30d = Column(Float, nullable=False, default=0.0)
    avg_subscriptions_per_subscriber = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserNewsTimeSurvey(Base):
    __tablename__ = "user_news_time_surveys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)

    baseline_answer = Column(String, nullable=True)
    baseline_minutes = Column(Integer, nullable=True)
    baseline_asked_at = Column(DateTime(timezone=True), nullable=True)
    baseline_retry_asked_at = Column(DateTime(timezone=True), nullable=True)
    baseline_answered_at = Column(DateTime(timezone=True), nullable=True, index=True)
    followup_due_at = Column(DateTime(timezone=True), nullable=True, index=True)
    followup_asked_at = Column(DateTime(timezone=True), nullable=True)
    followup_retry_asked_at = Column(DateTime(timezone=True), nullable=True)
    followup_abandoned_at = Column(DateTime(timezone=True), nullable=True)
    time_saved_answer = Column(String, nullable=True, index=True)
    time_saved_answered_at = Column(DateTime(timezone=True), nullable=True, index=True)
    current_answer = Column(String, nullable=True)
    current_minutes = Column(Integer, nullable=True)
    current_asked_at = Column(DateTime(timezone=True), nullable=True)
    current_retry_asked_at = Column(DateTime(timezone=True), nullable=True)
    current_answered_at = Column(DateTime(timezone=True), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="news_time_survey")


class UserCsiSurvey(Base):
    __tablename__ = "user_csi_surveys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    first_delivery_at = Column(DateTime(timezone=True), nullable=True, index=True)
    delivery_count_at_prompt = Column(Integer, nullable=True)
    due_at = Column(DateTime(timezone=True), nullable=True, index=True)
    asked_at = Column(DateTime(timezone=True), nullable=True)
    retry_asked_at = Column(DateTime(timezone=True), nullable=True)
    abandoned_at = Column(DateTime(timezone=True), nullable=True)
    score = Column(Integer, nullable=True)
    answered_at = Column(DateTime(timezone=True), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="csi_survey")


class BotAssistantRequest(Base):
    __tablename__ = "bot_assistant_requests"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    telegram_id = Column(BigInteger, nullable=True, index=True)
    locale = Column(String, nullable=False, default="ru", server_default="ru")
    raw_text = Column(Text, nullable=False)
    intent = Column(String, nullable=True, index=True)
    slots = Column(JSONB, nullable=True)
    faq_topic = Column(String, nullable=True, index=True)
    needs_clarification = Column(Boolean, nullable=False, default=False, server_default=text("false"), index=True)
    status = Column(String, nullable=False, default="received", server_default="received", index=True)
    outcome = Column(String, nullable=True, index=True)
    error_code = Column(String, nullable=True, index=True)
    error_message = Column(Text, nullable=True)
    received_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    understood_at = Column(DateTime(timezone=True), nullable=True)
    clarified_at = Column(DateTime(timezone=True), nullable=True)
    proposed_at = Column(DateTime(timezone=True), nullable=True)
    confirmed_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    applied_at = Column(DateTime(timezone=True), nullable=True)
    failed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Post(Base):
    __tablename__ = "posts"
    __table_args__ = (
        UniqueConstraint("channel_id", "message_id", name="uq_posts_channel_message"),
    )

    id = Column(Integer, primary_key=True, index=True)

    title = Column(String)
    content = Column(String)

    # e.g. {"id": 123, "channel": "username", "url": "https://t.me/.../123"}
    content_link = Column(JSON)

    channel_id = Column(String, nullable=True, index=True)
    message_id = Column(BigInteger, nullable=True)
    grouped_id = Column(BigInteger, nullable=True, index=True)

    # from _g: lightweight keywords for quick personalization
    keywords = Column(JSON)

    is_active = Column(Boolean, default=True)
    community_id = Column(Integer, ForeignKey("communities.id"), index=True)
    community = relationship("Community", back_populates="posts")

    # from _s: ingestion timestamp + enrichment outputs
    timestamp = Column(DateTime(timezone=True))
    processed_content = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    summary_created_at = Column(DateTime(timezone=True), nullable=True)
    dispatch_enqueued_at = Column(DateTime(timezone=True), nullable=True)
    dispatch_started_at = Column(DateTime(timezone=True), nullable=True)
    dispatch_finished_at = Column(DateTime(timezone=True), nullable=True)
    dispatch_error = Column(Text, nullable=True)

    reactions = relationship("Reaction", back_populates="post")
    entities = relationship("Entity", back_populates="post", cascade="all, delete-orphan")

    media_path = Column(Text, nullable=True)
    media_mime = Column(String, nullable=True)
    media_items = Column(JSON, nullable=True)
    media_created_at = Column(DateTime(timezone=True), nullable=True)


class PostTopic(Base):
    __tablename__ = "post_topics"
    __table_args__ = (
        UniqueConstraint("post_id", "topic", name="uq_post_topics_post_topic"),
    )

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), index=True, nullable=False)
    topic = Column(String, index=True, nullable=False)
    confidence = Column(Float, nullable=True)
    source = Column(String, nullable=False, server_default=text("'deepseek'"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SubscriptionRequest(Base):
    __tablename__ = "subscription_requests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    group_link = Column(String, index=True, nullable=False)
    group_handle = Column(String, nullable=True)
    status = Column(String, index=True, nullable=False, default="pending")
    userbot_slot = Column(String, nullable=False, default="primary", index=True)
    error = Column(String, nullable=True)
    notify_chat_id = Column(BigInteger, nullable=True)
    notify_message_id = Column(BigInteger, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Reaction(Base):
    __tablename__ = "reactions"
    __table_args__ = (
        UniqueConstraint("user_id", "post_id", name="uq_reactions_user_post"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), index=True)

    reaction = Column(String)

    # from _g: timestamps for audit/analytics
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="reactions")
    post = relationship("Post", back_populates="reactions")


class Click(Base):
    __tablename__ = "clicks"
    __table_args__ = (
        UniqueConstraint("user_id", "post_id", name="uq_clicks_user_post"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), index=True)
    community_id = Column(Integer, ForeignKey("communities.id"), index=True)

    source = Column(String, nullable=True)

    click_count = Column(Integer, default=1)
    first_clicked_at = Column(DateTime(timezone=True), server_default=func.now())
    last_clicked_at = Column(DateTime(timezone=True), server_default=func.now())


class PostInteraction(Base):
    __tablename__ = "post_interactions"
    __table_args__ = (
        UniqueConstraint("user_id", "post_id", "action", "source", name="uq_post_interactions_user_post_action_source"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    post_id = Column(Integer, ForeignKey("posts.id"), index=True, nullable=False)
    community_id = Column(Integer, ForeignKey("communities.id"), index=True, nullable=True)

    action = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False, default="", server_default=text("''"), index=True)

    interaction_count = Column(Integer, default=1, nullable=False, server_default=text("1"))
    first_interacted_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_interacted_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class UserKeywordStat(Base):
    __tablename__ = "user_keyword_stats"
    __table_args__ = (
        UniqueConstraint("user_id", "keyword", name="uq_user_keyword"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)

    keyword = Column(String, index=True)

    clicks_count = Column(Integer, default=0)
    reactions_count = Column(Integer, default=0)
    score = Column(Float, default=0.0)

    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserStorylineFollow(Base):
    __tablename__ = "user_storyline_follows"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "family_root_storyline_id",
            name="uq_user_storyline_follows_user_family_root",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    source_post_id = Column(Integer, ForeignKey("posts.id"), index=True, nullable=True)

    storyline_id = Column(String, index=True, nullable=False)
    story_family_id = Column(String, index=True, nullable=False)
    family_root_storyline_id = Column(String, index=True, nullable=False)
    storyline_title = Column(String, nullable=True)

    branch_mode = Column(String, nullable=False, default="root_only", server_default="root_only")
    selected_branch_ids = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    is_active = Column(Boolean, nullable=False, default=True, server_default=text("true"))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Entity(Base):
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, index=True)

    post_id = Column(Integer, ForeignKey("posts.id"), index=True, nullable=False)

    # PER, LOC, ORG, etc
    type = Column(String, index=True)
    text = Column(String, nullable=False)

    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)

    confidence = Column(Float, nullable=True)

    post = relationship("Post", back_populates="entities")


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), index=True, nullable=False, unique=True)
    embedding = Column(JSON, nullable=False)
    model_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
class UserDigestSettings(Base):
    __tablename__ = "user_digest_settings"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    enabled = Column(Boolean, default=True, nullable=False)

    # IANA timezone, e.g. "Europe/Moscow"
    timezone = Column(String, nullable=False, default="Europe/Moscow")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    utc_offset_minutes = Column(Integer, nullable=True)


class UserDeliveryPreference(Base):
    __tablename__ = "user_delivery_preferences"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    digest_to_plus_bot = Column(Boolean, nullable=False, default=False, server_default=text("false"))
    storyline_to_plus_bot = Column(Boolean, nullable=False, default=False, server_default=text("false"))
    plus_bot_connected_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

class UserDigestSlot(Base):
    __tablename__ = "user_digest_slots"
    __table_args__ = (
        UniqueConstraint("user_id", "hour", "minute", name="uq_user_digest_slot_time"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)

    hour = Column(SmallInteger, nullable=False)    # 0..23
    minute = Column(SmallInteger, nullable=False)  # 0..59

    # bitmask: Mon=1<<0 .. Sun=1<<6 (MVP: default all days)
    days_mask = Column(Integer, nullable=False, server_default=text("127"))

    is_active = Column(Boolean, default=True, nullable=False)

    next_run_at = Column(DateTime(timezone=True), index=True, nullable=True)
    last_run_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DigestRun(Base):
    __tablename__ = "digest_runs"
    __table_args__ = (
        UniqueConstraint("slot_id", "period_end", name="uq_digest_run_slot_period_end"),
    )

    id = Column(Integer, primary_key=True, index=True)

    slot_id = Column(Integer, ForeignKey("user_digest_slots.id"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)

    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)

    status = Column(String, index=True, nullable=False, default="created")
    sent_messages = Column(Integer, nullable=False, default=0)

    error = Column(Text, nullable=True)
    pages = Column(JSONB, nullable=True) 
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DispatchDelivery(Base):
    """Per-(post, user) outbox for idempotent dispatch.

    The dispatch task claims a row with ``INSERT ... ON CONFLICT DO NOTHING
    RETURNING``; if no row is returned the user was already processed in a
    prior attempt and must be skipped to avoid duplicate TG messages.
    On send failure the row is deleted so the Celery retry can reclaim.
    """

    __tablename__ = "dispatch_deliveries"

    post_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    sent_at = Column(DateTime(timezone=True), nullable=True)
    telegram_message_id = Column(BigInteger, nullable=True)
    attempt_count = Column(Integer, nullable=False, server_default=text("0"))


class StorylineUpdateEvent(Base):
    __tablename__ = "storyline_update_events"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "follow_id",
            "family_root_storyline_id",
            "event_key",
            "delivery_target",
            name="uq_storyline_update_events_user_follow_event_target",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    follow_id = Column(Integer, ForeignKey("user_storyline_follows.id"), nullable=False, index=True)
    family_root_storyline_id = Column(String, nullable=False, index=True)
    event_key = Column(String, nullable=False, index=True)
    canonical_post_id = Column(Integer, ForeignKey("posts.id"), nullable=True, index=True)
    last_post_id = Column(Integer, ForeignKey("posts.id"), nullable=True, index=True)
    telegram_message_id = Column(BigInteger, nullable=True)
    delivery_target = Column(String, nullable=False, default="main", server_default="main", index=True)
    delivery_result = Column(String, nullable=True)
    title = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    sources = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    source_count = Column(Integer, nullable=False, default=0, server_default=text("0"))
    first_seen_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_seen_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class ExperimentAssignment(Base):
    __tablename__ = "experiment_assignments"
    __table_args__ = (
        UniqueConstraint("experiment_key", "user_id", name="uq_experiment_assignments_key_user"),
    )

    id = Column(Integer, primary_key=True, index=True)
    experiment_key = Column(String, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    variant = Column(String, nullable=False, index=True)
    assigned_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", back_populates="experiment_assignments")


class ExperimentExposure(Base):
    __tablename__ = "experiment_exposures"
    __table_args__ = (
        UniqueConstraint("experiment_key", "user_id", "post_id", name="uq_experiment_exposures_key_user_post"),
    )

    id = Column(Integer, primary_key=True, index=True)
    experiment_key = Column(String, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False, index=True)
    variant = Column(String, nullable=False, index=True)
    eligible = Column(Boolean, nullable=False, default=True, server_default=text("true"))
    render_mode = Column(String, nullable=True)
    content_chars = Column(Integer, nullable=True)
    summary_chars = Column(Integer, nullable=True)
    has_summary = Column(Boolean, nullable=False, default=False, server_default=text("false"))
    delivery_result = Column(String, nullable=True)
    telegram_message_id = Column(BigInteger, nullable=True)
    sent_at = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", back_populates="experiment_exposures")
    post = relationship("Post")


class UserModelState(Base):
    __tablename__ = "user_model_states"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    weights = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    bias = Column(Float, nullable=False, server_default=text("0"))
    samples_count = Column(Integer, nullable=False, server_default=text("0"))
    generation = Column(Integer, nullable=False, default=0, server_default=text("0"))
    reset_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserEmbeddingProfile(Base):
    __tablename__ = "user_embedding_profiles"

    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    vector = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    dim = Column(Integer, nullable=False, server_default=text("0"))
    samples_count = Column(Integer, nullable=False, server_default=text("0"))
    generation = Column(Integer, nullable=False, default=0, server_default=text("0"))
    reset_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class BillingPlan(Base):
    __tablename__ = "billing_plans"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, nullable=False, index=True)
    title = Column(String, nullable=False)
    free_limit = Column(Integer, nullable=False, default=7)
    pack_size = Column(Integer, nullable=False, default=10)
    pack_price_rub = Column(Integer, nullable=False, default=49)
    period_type = Column(String, nullable=False, default="monthly")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserBillingState(Base):
    __tablename__ = "user_billing_states"
    __table_args__ = (
        UniqueConstraint("user_id", "period_start", name="uq_user_billing_state_period"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False, index=True)
    active_subscriptions = Column(Integer, nullable=False, default=0)
    required_packs = Column(Integer, nullable=False, default=0)
    paid_packs = Column(Integer, nullable=False, default=0)
    required_amount_rub = Column(Integer, nullable=False, default=0)
    paid_amount_rub = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="free", index=True)
    last_degraded_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PaymentOrder(Base):
    __tablename__ = "payment_orders"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_payment_orders_idempotency"),
        UniqueConstraint("merchant_order_id", name="uq_payment_orders_merchant_order_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    period_end = Column(DateTime(timezone=True), nullable=False, index=True)
    idempotency_key = Column(String, nullable=False, index=True)
    merchant_order_id = Column(String, nullable=False, index=True)
    amount_rub = Column(Integer, nullable=False, default=0)
    packs_count = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="new", index=True)
    provider = Column(String, nullable=False, default="tbank", server_default="tbank", index=True)
    quote_currency = Column(String, nullable=False, default="RUB", server_default="RUB")
    quote_amount = Column(Numeric(20, 9), nullable=True)
    price_code = Column(String, nullable=False, default="premium_rub", server_default="premium_rub", index=True)
    external_payment_id = Column(String, nullable=True, index=True)
    payment_url = Column(Text, nullable=True)
    fail_reason = Column(Text, nullable=True)
    raw_init_response = Column(JSONB, nullable=True)
    raw_last_state = Column(JSONB, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    confirmed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PaymentEvent(Base):
    __tablename__ = "payment_events"

    id = Column(Integer, primary_key=True, index=True)
    payment_order_id = Column(Integer, ForeignKey("payment_orders.id"), nullable=False, index=True)
    event_type = Column(String, nullable=False, index=True)
    payload = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PremiumEntitlementRecord(Base):
    __tablename__ = "premium_entitlements"
    __table_args__ = (
        UniqueConstraint("source_type", "source_id", name="uq_premium_entitlements_source"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    source_type = Column(String, nullable=False, index=True)
    source_id = Column(Integer, nullable=False, index=True)
    starts_at = Column(DateTime(timezone=True), nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    is_active = Column(Boolean, nullable=False, default=True, server_default=text("true"), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserInstructionRule(Base):
    __tablename__ = "user_instruction_rules"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    scope = Column(String, nullable=False, index=True)
    community_id = Column(Integer, ForeignKey("communities.id"), nullable=True, index=True)
    prompt_text = Column(Text, nullable=False)
    is_enabled = Column(Boolean, nullable=False, default=True, server_default=text("true"), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class InstructionFilterDecision(Base):
    __tablename__ = "instruction_filter_decisions"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "post_id",
            "effective_prompt_hash",
            name="uq_instruction_filter_decision_prompt",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), nullable=False, index=True)
    community_id = Column(Integer, ForeignKey("communities.id"), nullable=True, index=True)
    effective_scope = Column(String, nullable=False, index=True)
    effective_prompt_hash = Column(String, nullable=False, index=True)
    decision = Column(String, nullable=False, index=True)
    reason_short = Column(String, nullable=True)
    raw_response = Column(JSONB, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    error_code = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PromoCode(Base):
    __tablename__ = "promo_codes"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, nullable=False, index=True)
    title = Column(String, nullable=True)
    grant_type = Column(String, nullable=False, default="premium", server_default="premium", index=True)
    premium_days = Column(Integer, nullable=True, default=30, server_default=text("30"))
    extra_groups = Column(Integer, nullable=False, default=0)
    valid_from = Column(DateTime(timezone=True), nullable=True)
    valid_until = Column(DateTime(timezone=True), nullable=True)
    max_activations = Column(Integer, nullable=True)
    per_user_limit = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class UserPromoGrant(Base):
    __tablename__ = "user_promo_grants"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    promo_code_id = Column(Integer, ForeignKey("promo_codes.id"), nullable=False, index=True)
    code = Column(String, nullable=False, index=True)
    grant_type = Column(String, nullable=False, default="premium", server_default="premium", index=True)
    premium_days = Column(Integer, nullable=True)
    extra_groups = Column(Integer, nullable=False, default=0)
    starts_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    activated_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SupportTicket(Base):
    __tablename__ = "support_tickets"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    telegram_id = Column(BigInteger, nullable=False, index=True)
    username = Column(String, nullable=True)
    premium_active = Column(Boolean, nullable=False, default=False, server_default=text("false"))
    user_registered_at = Column(DateTime(timezone=True), nullable=True)
    question_text = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="accepted", server_default="accepted", index=True)
    attempts = Column(Integer, nullable=False, default=0, server_default=text("0"))
    next_retry_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True, index=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())



# --- DB session ---

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME") or os.getenv("POSTGRES_DB")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_echo = os.getenv("DB_ECHO", "false").lower() == "true"
db_pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
db_max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))

engine_dsn = f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

engine = create_async_engine(
    engine_dsn,
    echo=db_echo,
    pool_size=db_pool_size,
    max_overflow=db_max_overflow,
    pool_pre_ping=True,
)

async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session


async def ensure_db_schema() -> None:
    """No-op: schema is managed exclusively by Alembic.

    Historically this function ran a pile of ``ALTER TABLE ... ADD COLUMN IF
    NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS`` / ``UPDATE`` statements at
    startup. Every one of those changes is now represented by an Alembic
    migration under ``alembic/versions/`` (media columns, dispatch tracking,
    feed_filter, last_live_feed_filter, userbot_slots, cryptopay fields on
    payment_orders, instruction filter indexes, etc.), so running the DDL
    again at boot is redundant and actively harmful:

    * ``ALTER TABLE`` takes an ACCESS EXCLUSIVE lock, stalling all readers
      and writers for the duration of the command on busy tables.
    * ``CREATE INDEX`` without ``CONCURRENTLY`` fully locks the table.
    * ``DROP COLUMN embeddings`` could silently destroy data on a replica
      that was rolled back and is running an older image.
    * Multiple workers starting in parallel race to run the same DDL.

    The function is intentionally kept (and callers / imports preserved)
    so tests and any stray callers don't break; it simply does nothing.
    Use ``alembic upgrade head`` to apply schema changes.
    """
    return None
