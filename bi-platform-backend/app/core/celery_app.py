"""Celery configuration."""
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "bi_platform",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.training",
        "app.tasks.analysis",
        "app.tasks.predictions",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    worker_concurrency=settings.CELERY_WORKER_CONCURRENCY,
    result_expires=3600 * 24 * 7,  # 7 days
)
