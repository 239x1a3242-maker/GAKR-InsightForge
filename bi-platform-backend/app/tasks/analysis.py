"""Analysis tasks."""
from celery import shared_task
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.core.config import settings
from app.services.analysis_service import AnalysisService
import uuid


@shared_task(bind=True, max_retries=2)
def run_analysis_task(
    self,
    dataset_id: str,
    analysis_type: str,
    params: dict
):
    """Run analysis async task."""
    import asyncio
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _run_analysis_async(dataset_id, analysis_type, params)
        )
        return result
    finally:
        loop.close()


async def _run_analysis_async(
    dataset_id: str,
    analysis_type: str,
    params: dict
):
    """Async analysis function."""
    engine = create_async_engine(settings.DATABASE_URL)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    
    async with SessionLocal() as db:
        try:
            dataset_uuid = uuid.UUID(dataset_id)
            
            if analysis_type == 'descriptive':
                result = await AnalysisService.descriptive_analytics(
                    db, dataset_uuid, params.get('columns')
                )
            elif analysis_type == 'diagnostic':
                result = await AnalysisService.diagnostic_analytics(
                    db, dataset_uuid, params.get('target_column')
                )
            elif analysis_type == 'predictive':
                result = await AnalysisService.predictive_analytics(
                    db, dataset_uuid, params.get('target_column')
                )
            elif analysis_type == 'prescriptive':
                result = await AnalysisService.prescriptive_analytics(
                    db, dataset_uuid, params.get('target_column')
                )
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            return {
                'status': 'completed',
                'analysis_type': analysis_type,
                'result': result,
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'analysis_type': analysis_type,
                'error': str(e),
            }
        finally:
            await engine.dispose()
