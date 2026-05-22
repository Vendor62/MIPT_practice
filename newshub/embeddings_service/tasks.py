import os
import asyncio
import threading
import structlog

from datetime import datetime
from celery import shared_task
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint, func, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sentence_transformers import SentenceTransformer

from celery_app import celery_app

log = structlog.get_logger()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")

EMBEDDINGS_MODEL_NAME = os.getenv(
    "EMBEDDINGS_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDINGS_SCAN_LIMIT = int(os.getenv("EMBEDDINGS_SCAN_LIMIT", "50"))
EMBEDDINGS_JOB_DELAY = float(os.getenv("EMBEDDINGS_JOB_DELAY", "1"))
EMBEDDINGS_MIN_POST_ID = int(os.getenv("EMBEDDINGS_MIN_POST_ID", "0"))

engine_dsn = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"
engine = create_async_engine(engine_dsn, pool_pre_ping=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String)
    processed_content = Column(String)
    timestamp = Column(DateTime(timezone=True))


class Embedding(Base):
    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint("post_id", name="uq_embeddings_post_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("posts.id"), index=True, nullable=False, unique=True)
    embedding = Column(JSONB, nullable=False)
    model_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


_model = None
_schema_ready = False
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_ready = threading.Event()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)
    return _model


async def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                """
                ALTER TABLE posts
                    ADD COLUMN IF NOT EXISTS processed_content TEXT,
                    ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ;
                """
            )
        )
        await conn.execute(text("ALTER TABLE posts DROP COLUMN IF EXISTS embeddings;"))
    _schema_ready = True


def _start_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop_ready.set()
    _loop.run_forever()


def _ensure_loop():
    global _loop_thread
    if _loop_thread and _loop_ready.is_set():
        return
    _loop_thread = threading.Thread(target=_start_loop, daemon=True)
    _loop_thread.start()
    _loop_ready.wait()


def _run_async(coro):
    _ensure_loop()
    assert _loop is not None
    fut = asyncio.run_coroutine_threadsafe(coro, _loop)
    return fut.result()


@celery_app.task(name="embeddings.scan_new_posts")
def scan_new_posts():
    async def _scan():
        await _ensure_schema()
        async with async_session() as session:
            stmt = (
                select(Post.id, Post.processed_content, Post.content, Post.timestamp)
                .outerjoin(Embedding, Embedding.post_id == Post.id)
                .where(Embedding.post_id.is_(None))
                .where(Post.content.isnot(None))
                .where(Post.id > EMBEDDINGS_MIN_POST_ID)
                .order_by(Post.id.asc())
                .limit(EMBEDDINGS_SCAN_LIMIT)
            )
            rows = (await session.execute(stmt)).all()

        if not rows:
            return 0

        for idx, row in enumerate(rows):
            post_id, processed_content, content, timestamp = row
            payload = {
                "post_id": post_id,
                "text": processed_content or content,
                "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else None,
            }
            delay = int(idx * EMBEDDINGS_JOB_DELAY)
            embed_post.apply_async(args=[payload], countdown=delay, queue="embeddings_queue")

        return len(rows)

    return _run_async(_scan())


@celery_app.task(name="embeddings.embed_post")
def embed_post(payload: dict):
    async def _work():
        await _ensure_schema()
        text = payload.get("text")
        post_id = payload.get("post_id")
        if not post_id or not text:
            return None

        model = _get_model()
        embedding = model.encode(text).tolist()

        async with async_session() as session:
            stmt = (
                pg_insert(Embedding)
                .values(
                    post_id=post_id,
                    embedding=embedding,
                    model_name=EMBEDDINGS_MODEL_NAME,
                )
                .on_conflict_do_update(
                    index_elements=["post_id"],
                    set_={"embedding": embedding, "model_name": EMBEDDINGS_MODEL_NAME},
                )
            )
            await session.execute(stmt)
            await session.commit()
        return post_id

    return _run_async(_work())
