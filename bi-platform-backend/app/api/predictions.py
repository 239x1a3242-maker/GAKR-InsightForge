"""Predictions API."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import uuid

from app.core.database import get_db
from app.api.auth import get_current_user
from app.models.user import User
from app.models.model_registry import ModelVersion, ModelStatus
from app.services.prediction_service import PredictionService
from app.schemas.prediction import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse
)

router = APIRouter()


@router.post("/{model_id}", response_model=PredictionResponse)
async def predict(
    model_id: uuid.UUID,
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Make single prediction."""
    # Verify model exists and is production ready
    result = await db.execute(
        select(ModelVersion).where(
            ModelVersion.id == model_id
        )
    )
    version = result.scalar_one_or_none()
    
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    try:
        pred_result = await PredictionService.predict(
            db, model_id, request.features, current_user.id, request.return_shap
        )
        
        return {
            "model_id": model_id,
            "model_version": version.version_number,
            "predictions": [{
                "prediction": pred_result['prediction'],
                "confidence": pred_result.get('confidence'),
                "probabilities": pred_result.get('probabilities'),
                "shap_values": pred_result.get('shap_values'),
            }],
            "prediction_time_ms": pred_result['prediction_time_ms'],
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/{model_id}/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    model_id: uuid.UUID,
    request: BatchPredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Make batch predictions."""
    # Verify model exists
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == model_id)
    )
    version = result.scalar_one_or_none()
    
    if not version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    try:
        pred_result = await PredictionService.predict_batch(
            db, model_id, request.records, current_user.id, request.return_confidence
        )
        
        return {
            "model_id": model_id,
            "model_version": version.version_number,
            "predictions": pred_result['predictions'],
            "total_records": pred_result['total_records'],
            "prediction_time_ms": pred_result['prediction_time_ms'],
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )
