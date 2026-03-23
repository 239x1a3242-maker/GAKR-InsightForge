"""Celery tasks."""
from app.tasks.training import train_model_task
from app.tasks.analysis import run_analysis_task

__all__ = ['train_model_task', 'run_analysis_task']
