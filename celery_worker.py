"""
Celery application factory and queue routing configuration.
Three task queues: analysis_queue, cleaning_queue, query_queue.
All tasks: max_retries=3, exponential backoff, acks_late=True, idempotent.
"""
from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "dataset_intelligence",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# ── Celery configuration ─────────────────────────────────────
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,

    # Retry defaults
    task_default_retry_delay=5,
    task_max_retries=3,

    # Queue routing
    task_routes={
        "app.workers.analyze_task.analyze_dataset": {"queue": "analysis_queue"},
        "app.workers.clean_task.clean_dataset": {"queue": "cleaning_queue"},
        "app.workers.query_task.execute_query": {"queue": "query_queue"},
    },

    # Queue definitions (for worker startup)
    task_queues={
        "analysis_queue": {"exchange": "analysis_queue", "routing_key": "analysis_queue"},
        "cleaning_queue": {"exchange": "cleaning_queue", "routing_key": "cleaning_queue"},
        "query_queue": {"exchange": "query_queue", "routing_key": "query_queue"},
    },

    # Result expiry
    result_expires=3600,  # 1 hour
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.workers"])
