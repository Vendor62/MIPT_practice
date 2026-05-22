import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "rpc://")

celery_app = Celery(
    "embeddings_service",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["tasks", "ie_pipeline"],   # ← оба модуля
)

celery_app.conf.update(
    task_soft_time_limit=900,
    task_time_limit=1200,
    task_ignore_result=True,
    result_backend=None,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_heartbeat=int(os.getenv("CELERY_BROKER_HEARTBEAT", "120")),
    broker_pool_limit=5,
    broker_connection_retry_on_startup=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,
)

celery_app.conf.beat_schedule = {
    "embeddings-scan-new-posts": {
        "task": "embeddings.scan_new_posts",
        "schedule": 30.0,
    },
    "ie-scan-unprocessed": {
        "task": "ie.scan_unprocessed_posts",
        "schedule": 60.0,
    },
}
