import os

from celery import Celery
from celery.schedules import crontab

from app.logging import configure_logging
from dotenv import load_dotenv 

load_dotenv()


CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "rpc://")

celery_app = Celery(
    "newshub",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.tasks"],
)

celery_app.conf.update(
    # Глобальные таймауты для всех задач
    task_soft_time_limit=900,       # 15 минут мягкий лимит (логирует warning)
    task_time_limit=1200,          # 20 минут жёсткий лимит (убивает задачу)
    
    task_ignore_result=True,
    result_backend=None,
    
    # Prefetch: забирать по 1 задаче (чтобы при restart не терять пул)
    worker_prefetch_multiplier=1,
    
    # Acknowledge после выполнения (не до), чтобы не терять задачи при падении воркера
    task_acks_late=True,
    
    # Requeue если воркер упал
    task_reject_on_worker_lost=True,
    
    # Логировать таймауты
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",

    # Настройки брокера — ДОБАВЛЕНО
    broker_heartbeat=30,
    broker_pool_limit=5,
    task_default_queue="dispatch_queue",
)

configure_logging("celery")

celery_app.conf.task_routes = {
    "app.tasks.handle_subscription_request_task": {"queue": "subscribe_queue"},
    "app.tasks.dispatch_post_to_subscribers": {"queue": "dispatch_queue"},
    "app.tasks.dispatch_storyline_updates_for_post": {"queue": "dispatch_queue"},
    "app.tasks.summarize_post_task": {"queue": "summarize_queue"},
    "app.tasks.extract_post_topics_task": {"queue": "topic_queue"},
    "app.tasks.preprocess_and_embed_task": {"queue": "dispatch_queue"},
    "app.tasks.build_and_send_digest": {"queue": "digest_queue"},
    "app.tasks.digest_scheduler_tick": {"queue": "digest_queue"},
    "app.tasks.billing_monthly_rollover": {"queue": "dispatch_queue"},
    "app.tasks.billing_sync_pending_payments": {"queue": "dispatch_queue"},
    "app.tasks.billing_renewal_reminders": {"queue": "dispatch_queue"},
    "app.tasks.premium_week_after_first_subscription_reminders": {"queue": "dispatch_queue"},
    "app.tasks.cleanup_summaries": {"queue": "dispatch_queue"},
    "app.tasks.detect_missing_dispatch": {"queue": "dispatch_queue"},
}

DIGEST_SCHEDULER_POLL = int(os.getenv("DIGEST_SCHEDULER_POLL_SECONDS", "30"))
PAYMENT_SYNC_POLL = int(os.getenv("PAYMENT_SYNC_POLL_SECONDS", "60"))

celery_app.conf.beat_schedule = {
    "cleanup_summaries_daily": {
        "task": "app.tasks.cleanup_summaries",
        "schedule": crontab(minute=0, hour=3),
        "options": {"queue": "dispatch_queue"},
    },
    "billing_monthly_rollover_daily": {
        "task": "app.tasks.billing_monthly_rollover",
        "schedule": crontab(minute=30, hour=3),
    },
    "billing_sync_pending_payments_every_minute": {
        "task": "app.tasks.billing_sync_pending_payments",
        "schedule": float(PAYMENT_SYNC_POLL),
        "options": {
            "queue": "dispatch_queue",
            "expires": PAYMENT_SYNC_POLL - 5 if PAYMENT_SYNC_POLL > 10 else PAYMENT_SYNC_POLL,
        },
    },
    "billing_renewal_reminders_daily": {
        "task": "app.tasks.billing_renewal_reminders",
        "schedule": crontab(minute=0, hour=10),
        "options": {"queue": "dispatch_queue"},
    },
    "premium_week_after_first_subscription_reminders_daily": {
        "task": "app.tasks.premium_week_after_first_subscription_reminders",
        "schedule": crontab(minute=15, hour=10),
        "options": {"queue": "dispatch_queue"},
    },
    "digest_scheduler_tick": {
        "task": "app.tasks.digest_scheduler_tick",
        "schedule": float(DIGEST_SCHEDULER_POLL),
        "options": {"expires": DIGEST_SCHEDULER_POLL - 5 if DIGEST_SCHEDULER_POLL > 10 else DIGEST_SCHEDULER_POLL},
    },
    "detect_missing_dispatch_every_minute": {
        "task": "app.tasks.detect_missing_dispatch",
        "schedule": 60.0,
        "options": {"queue": "dispatch_queue", "expires": 55},
    },
}
