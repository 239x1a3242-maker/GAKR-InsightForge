"""Training tasks."""
import asyncio
from celery import shared_task
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.core.config import settings
from app.services.model_service import ModelService
from app.core.redis import redis_client
import json
import uuid


@shared_task(bind=True, max_retries=3)
def train_model_task(
    self,
    dataset_id: str,
    target_column: str,
    task_type: str,
    config: dict,
    job_id: str
):
    """Train model async task."""
    import asyncio
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _train_model_async(
                dataset_id, target_column, task_type, config, job_id, self
            )
        )
        return result
    finally:
        loop.close()


async def _train_model_async(
    dataset_id: str,
    target_column: str,
    task_type: str,
    config: dict,
    job_id: str,
    task
):
    """Async training function."""
    # Create database session
    engine = create_async_engine(settings.DATABASE_URL)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    
    async with SessionLocal() as db:
        try:
            # Update status to running
            await _update_job_status(job_id, {
                'status': 'running',
                'progress': 10,
                'current_step': 'Loading dataset...',
            })
            
            # Train model
            await _update_job_status(job_id, {
                'progress': 30,
                'current_step': 'Training AutoML model...',
            })
            
            version = await ModelService.train_model(
                db,
                uuid.UUID(dataset_id),
                target_column,
                config,
                task_type,
            )
            
            await _update_job_status(job_id, {
                'status': 'completed',
                'progress': 100,
                'current_step': 'Training complete',
                'model_id': str(version.id),
                'version_number': version.version_number,
            })
            
            return {
                'status': 'completed',
                'model_id': str(version.id),
                'version_number': version.version_number,
            }
            
        except Exception as e:
            await _update_job_status(job_id, {
                'status': 'failed',
                'progress': 0,
                'current_step': f'Error: {str(e)}',
                'error': str(e),
            })
            
            # Retry on failure
            task.retry(countdown=60, exc=e)
            
        finally:
            await engine.dispose()


async def _update_job_status(job_id: str, status_data: dict):
    """Update job status in Redis."""
    try:
        from app.core.redis import redis_client
        if redis_client:
            await redis_client.setex(
                f"training_job:{job_id}",
                3600 * 24,  # 24 hours
                json.dumps(status_data)
            )
    except Exception as e:
        print(f"Failed to update job status: {e}")
