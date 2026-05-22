from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Post,
    PostTopic,
    User,
    UserEmbeddingProfile,
    UserKeywordStat,
    UserModelState,
)
from app.reco import (
    FEATURE_NAMES,
    clamp01,
    cosine_similarity,
    ema_update_vector,
    online_logistic_update,
)
from app.reco_embeddings import get_post_embedding


RECO_LR = 0.15
RECO_L2 = 0.001
EMBEDDING_EMA_BETA = 0.9


def _to_vector(raw) -> list[float]:
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for item in raw:
        try:
            out.append(float(item))
        except Exception:
            return []
    return out


async def build_feedback_features(session: AsyncSession, user: User, post: Post) -> dict[str, float]:
    features = {name: 0.0 for name in FEATURE_NAMES}

    kws = [str(k).strip().lower() for k in (post.keywords or []) if isinstance(k, str)]
    kws = [k for k in kws if k]
    if kws:
        kw_res = await session.execute(
            select(UserKeywordStat.score).where(
                UserKeywordStat.user_id == user.id,
                UserKeywordStat.keyword.in_(kws),
            )
        )
        vals = [max(0.0, float(row[0] or 0.0)) for row in kw_res.all()]
        if vals:
            features["keyword_affinity"] = clamp01((sum(vals) / len(vals)) / 3.0)

    topics_res = await session.execute(select(PostTopic.topic).where(PostTopic.post_id == post.id))
    topics = [str(row[0]).strip().lower() for row in topics_res.all() if row[0]]
    if topics:
        t_res = await session.execute(
            select(UserKeywordStat.score).where(
                UserKeywordStat.user_id == user.id,
                UserKeywordStat.keyword.in_(topics),
            )
        )
        t_vals = [max(0.0, float(row[0] or 0.0)) for row in t_res.all()]
        if t_vals:
            features["topic_affinity"] = clamp01((sum(t_vals) / len(t_vals)) / 3.0)

    features["engagement"] = clamp01(float(getattr(user, "engagement_score", 0.0) or 0.0) / 20.0)
    features["has_summary"] = 1.0 if bool(post.summary) else 0.0
    features["has_media"] = 1.0 if bool(post.media_path) else 0.0
    features["freshness"] = 0.4

    if post.timestamp:
        ts = post.timestamp if post.timestamp.tzinfo else post.timestamp.replace(tzinfo=timezone.utc)
        age_hours = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0)
        features["freshness"] = clamp01(1.0 - (age_hours / 72.0))

    post_embedding = await get_post_embedding(session, int(post.id))
    if post_embedding:
        profile = await session.get(UserEmbeddingProfile, user.id)
        if profile and isinstance(profile.vector, list) and profile.dim == len(post_embedding):
            features["embedding_similarity"] = cosine_similarity(_to_vector(profile.vector), post_embedding)
    return features


async def update_user_model_from_feedback(
    session: AsyncSession,
    *,
    user_id: int,
    post_id: int,
    label: int,
) -> None:
    user = await session.get(User, user_id)
    post = await session.get(Post, post_id)
    if not user or not post:
        return

    features = await build_feedback_features(session, user, post)
    state = await session.get(UserModelState, user.id)
    if not state:
        state = UserModelState(user_id=user.id, weights={}, bias=0.0, samples_count=0)
        session.add(state)
        await session.flush()

    weights = state.weights if isinstance(state.weights, dict) else {}
    next_weights, next_bias = online_logistic_update(
        weights=weights,
        bias=float(state.bias or 0.0),
        features=features,
        label=int(label),
        learning_rate=RECO_LR,
        l2=RECO_L2,
    )
    state.weights = next_weights
    state.bias = float(next_bias)
    state.samples_count = int(state.samples_count or 0) + 1

    post_embedding = await get_post_embedding(session, int(post.id))
    if post_embedding:
        profile = await session.get(UserEmbeddingProfile, user.id)
        if not profile:
            profile = UserEmbeddingProfile(
                user_id=user.id,
                vector=post_embedding,
                dim=len(post_embedding),
                samples_count=1,
            )
            session.add(profile)
        else:
            current = _to_vector(profile.vector)
            sample = post_embedding if int(label) > 0 else [-1.0 * x for x in post_embedding]
            profile.vector = ema_update_vector(current, sample, EMBEDDING_EMA_BETA)
            profile.dim = len(post_embedding)
            profile.samples_count = int(profile.samples_count or 0) + 1


async def reset_user_recommendations(
    session: AsyncSession,
    *,
    user_id: int,
    reset_at: datetime | None = None,
) -> tuple[int, int]:
    reset_at = reset_at or datetime.now(timezone.utc)
    user = await session.get(User, user_id)
    if user:
        user.engagement_score = 0.0
        session.add(user)

    state = await session.get(UserModelState, user_id)
    if not state:
        state = UserModelState(user_id=user_id, weights={}, bias=0.0, samples_count=0)
    state.weights = {}
    state.bias = 0.0
    state.samples_count = 0
    state.generation = int(getattr(state, "generation", 0) or 0) + 1
    state.reset_at = reset_at
    session.add(state)

    profile = await session.get(UserEmbeddingProfile, user_id)
    if not profile:
        profile = UserEmbeddingProfile(user_id=user_id, vector=[], dim=0, samples_count=0)
    profile.vector = []
    profile.dim = 0
    profile.samples_count = 0
    profile.generation = int(getattr(profile, "generation", 0) or 0) + 1
    profile.reset_at = reset_at
    session.add(profile)

    deleted = await session.execute(delete(UserKeywordStat).where(UserKeywordStat.user_id == user_id))
    return int(state.generation or 0), int(deleted.rowcount or 0)
