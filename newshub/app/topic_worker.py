from datetime import datetime, timezone

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.deepseek import extract_post_topics
from app.models import Post, PostTopic

log = structlog.get_logger()


async def process_post_topics(session: AsyncSession, post_id: int) -> list[str]:
    post_res = await session.execute(select(Post).where(Post.id == post_id))
    post = post_res.scalar_one_or_none()
    if not post:
        return []

    raw_text = (post.content or "").strip()
    topics: list[str] = []
    if raw_text:
        topics = await extract_post_topics(raw_text, max_topics=10)
    if not topics:
        topics = ["unknown"]

    await session.execute(delete(PostTopic).where(PostTopic.post_id == post_id))
    for topic in topics[:10]:
        session.add(
            PostTopic(
                post_id=post_id,
                topic=topic,
                confidence=None,
                source="deepseek",
                created_at=datetime.now(timezone.utc),
            )
        )
    await session.commit()
    log.info("topic_worker.processed", post_id=post_id, topics_count=len(topics), topics=topics)
    return topics
