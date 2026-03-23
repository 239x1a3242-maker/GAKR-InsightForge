"""Prediction schemas."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID


class PredictionRequest(BaseModel):
    """Single prediction request."""
    features: Dict[str, Any]
    return_shap: bool = False


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    records: List[Dict[str, Any]] = Field(..., min_length=1, max_length=10000)
    return_confidence: bool = True


class PredictionResult(BaseModel):
    """Single prediction result."""
    prediction: Any
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    shap_values: Optional[Dict[str, float]] = None


class PredictionResponse(BaseModel):
    """Prediction response."""
    model_id: UUID
    model_version: int
    predictions: List[PredictionResult]
    prediction_time_ms: float
    drift_detected: Optional[bool] = None
    drift_score: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    model_id: UUID
    model_version: int
    predictions: List[PredictionResult]
    total_records: int
    prediction_time_ms: float
    drift_summary: Optional[Dict[str, Any]] = None
