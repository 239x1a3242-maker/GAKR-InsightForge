"""Models API."""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from typing import List, Optional
import uuid
import json

from app.core.database import get_db
from app.core.redis import get_redis
from app.api.auth import get_current_user
from app.models.user import User
from app.models.model_registry import ModelRegistry, ModelVersion, ModelMetrics, ModelStatus
from app.models.dataset import Dataset
from app.services.model_service import ModelService
from app.services.dataset_service import DatasetService
from app.tasks.training import train_model_task
from app.schemas.model import (
    TrainingRequest, TrainingResponse, TrainingStatusResponse,
    ModelRegistryResponse, ModelVersionResponse, ModelMetricsResponse,
    PromoteRequest, PromoteResponse, LeaderboardEntry,
    ShapGlobalResponse, ShapLocalRequest, ShapLocalResponse,
    DriftReport
)

router = APIRouter()


@router.post("/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Start model training."""
    # Verify dataset exists
    dataset = await DatasetService.get_dataset(db, request.dataset_id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Load data to verify targets
    df = DatasetService.load_dataset(dataset.file_path)
    
    invalid_targets = [t for t in request.targets if t not in df.columns]
    if invalid_targets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid target columns: {', '.join(invalid_targets)}"
        )
    
    # Create job IDs for each target
    job_ids = []
    model_ids = []
    
    for target in request.targets:
        # Detect task type
        task_detection = await DatasetService.detect_task_type(df, target)
        task_type = task_detection['task_type']
        
        # Get or create registry
        registry = await ModelService.get_or_create_registry(
            db, request.dataset_id, target, task_type
        )
        
        # Get next version number
        version_number = await ModelService.get_next_version_number(db, registry.id)
        
        # Create version placeholder
        version = ModelVersion(
            registry_id=registry.id,
            version_number=version_number,
            status=ModelStatus.STAGING,
            model_path="",  # Will be updated after training
            model_name="pending",
            schema_hash="pending",
            feature_columns=[],
            training_config=request.config.dict(),
        )
        db.add(version)
        await db.commit()
        await db.refresh(version)
        
        model_ids.append(version.id)
        
        # Create job ID
        job_id = str(uuid.uuid4())
        job_ids.append(job_id)
        
        # Start training task
        train_model_task.delay(
            dataset_id=str(request.dataset_id),
            target_column=target,
            task_type=task_type,
            config=request.config.dict(),
            job_id=job_id,
        )
    
    return {
        "job_id": job_ids[0] if job_ids else "",
        "status": "queued",
        "message": f"Training started for {len(request.targets)} target(s)",
        "model_ids": model_ids,
    }


@router.get("/training-jobs/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(
    job_id: str,
    redis=Depends(get_redis)
):
    """Get training job status."""
    try:
        status_data = await redis.get(f"training_job:{job_id}")
        if status_data:
            data = json.loads(status_data)
            return {
                "job_id": job_id,
                **data,
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "completed_at": data.get("completed_at"),
            }
        else:
            return {
                "job_id": job_id,
                "status": "unknown",
                "progress": 0,
                "current_step": "Job not found",
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("", response_model=List[ModelRegistryResponse])
async def list_models(
    dataset_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all model registries."""
    query = select(ModelRegistry)
    
    if dataset_id:
        query = query.where(ModelRegistry.dataset_id == dataset_id)
    
    query = query.order_by(desc(ModelRegistry.updated_at))
    
    result = await db.execute(query)
    registries = result.scalars().all()
    
    # Add version count
    response = []
    for registry in registries:
        version_count = await db.execute(
            select(func.count(ModelVersion.id)).where(ModelVersion.registry_id == registry.id)
        )
        count = version_count.scalar()
        
        registry_dict = {
            "id": registry.id,
            "dataset_id": registry.dataset_id,
            "target_column": registry.target_column,
            "task_type": registry.task_type,
            "production_version_id": registry.production_version_id,
            "production_version": registry.production_version,
            "version_count": count,
            "created_at": registry.created_at,
            "updated_at": registry.updated_at,
        }
        response.append(registry_dict)
    
    return response


@router.get("/{registry_id}/versions", response_model=List[ModelVersionResponse])
async def get_model_versions(
    registry_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all versions for a model registry."""
    versions = await ModelService.get_model_versions(db, registry_id)
    return versions


@router.get("/versions/{version_id}/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(
    version_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get model metrics."""
    result = await db.execute(
        select(ModelMetrics, ModelVersion, ModelRegistry)
        .join(ModelVersion, ModelMetrics.version_id == ModelVersion.id)
        .join(ModelRegistry, ModelVersion.registry_id == ModelRegistry.id)
        .where(ModelMetrics.version_id == version_id)
    )
    row = result.first()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics not found"
        )
    
    metrics, version, registry = row
    
    # Format response based on task type
    response = {
        "id": metrics.id,
        "version_id": metrics.version_id,
        "task_type": registry.task_type,
        "train_rows": metrics.train_rows,
        "test_rows": metrics.test_rows,
        "feature_importance": metrics.feature_importance,
    }
    
    if registry.task_type == "classification":
        response["classification"] = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "auc": metrics.auc,
            "logloss": metrics.logloss,
            "confusion_matrix": metrics.confusion_matrix,
            "roc_curve": metrics.roc_curve,
            "pr_curve": metrics.pr_curve,
        }
    else:
        response["regression"] = {
            "r2": metrics.r2,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "mse": metrics.mse,
            "mape": metrics.mape,
            "residual_mean": metrics.residual_mean,
            "residual_std": metrics.residual_std,
            "prediction_vs_actual": metrics.prediction_vs_actual,
        }
    
    return response


@router.get("/versions/{version_id}/leaderboard")
async def get_leaderboard(
    version_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get model leaderboard."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == version_id)
    )
    version = result.scalar_one_or_none()
    
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Version not found"
        )
    
    return version.leaderboard


@router.post("/versions/{version_id}/promote", response_model=PromoteResponse)
async def promote_model(
    version_id: uuid.UUID,
    request: PromoteRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Promote model to production."""
    try:
        version = await ModelService.promote_version(db, version_id)
        
        return {
            "success": True,
            "message": f"Model promoted to production (version {version.version_number})",
            "new_version": version.version_number,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.post("/versions/{version_id}/archive")
async def archive_model(
    version_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Archive model version."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == version_id)
    )
    version = result.scalar_one_or_none()
    
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Version not found"
        )
    
    version.status = ModelStatus.ARCHIVED
    await db.commit()
    
    return {"message": "Model archived successfully"}


@router.get("/versions/{version_id}/shap/global")
async def get_shap_global(
    version_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get global SHAP values."""
    result = await db.execute(
        select(ModelMetrics).where(ModelMetrics.version_id == version_id)
    )
    metrics = result.scalar_one_or_none()
    
    if not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics not found"
        )
    
    # Return feature importance as SHAP proxy
    feature_importance = metrics.feature_importance or {}
    
    return {
        "feature_names": list(feature_importance.keys()),
        "shap_values": list(feature_importance.values()),
        "base_value": 0.0,
        "feature_importance": feature_importance,
    }


@router.post("/versions/{version_id}/shap/local")
async def get_shap_local(
    version_id: uuid.UUID,
    request: ShapLocalRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get local SHAP explanation."""
    from app.services.prediction_service import PredictionService
    
    result = await db.execute(
        select(ModelVersion, ModelMetrics)
        .join(ModelMetrics, ModelVersion.id == ModelMetrics.version_id)
        .where(ModelVersion.id == version_id)
    )
    row = result.first()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    version, metrics = row
    
    # Make prediction with SHAP
    pred_result = await PredictionService.predict(
        db, version_id, request.features, current_user.id, return_shap=True
    )
    
    shap_values = pred_result.get('shap_values', {})
    
    # Format contributions
    contributions = []
    for feature, value in shap_values.items():
        if feature != 'BiasTerm':
            contributions.append({
                "feature": feature,
                "value": request.features.get(feature),
                "contribution": value,
            })
    
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return {
        "prediction": pred_result['prediction'],
        "base_value": shap_values.get('BiasTerm', 0),
        "shap_values": shap_values,
        "feature_contributions": contributions[:10],
    }


@router.post("/versions/{version_id}/drift-check")
async def check_drift(
    version_id: uuid.UUID,
    batch_data: List[dict],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Check for data drift."""
    from app.services.prediction_service import PredictionService
    
    try:
        drift_report = await PredictionService.check_drift(db, version_id, batch_data)
        return drift_report
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift check failed: {str(e)}"
        )


# Need to import func for count
from sqlalchemy import func
