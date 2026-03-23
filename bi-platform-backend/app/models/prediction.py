"""Prediction models."""
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional


class PredictionLog(BaseModel):
    """Prediction log for audit and monitoring."""
    id: str
    model_version_id: str
    
    # Input
    input_features: Dict[str, Any]
    
    # Output
    prediction: Dict[str, Any]
    confidence: Optional[float] = None
    
    # Drift detection
    drift_score: Optional[float] = None
    drift_detected: str = "unknown"  # yes, no, unknown
    
    # Metadata
    prediction_time_ms: float = 0.0
    created_at: datetime
    
    # User info (for audit)
    created_by: Optional[str] = None
