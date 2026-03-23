"""Model registry models."""
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Optional
import enum


class ModelStatus(str, enum.Enum):
    """Model status."""
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelRegistry(BaseModel):
    """Model registry - one entry per target column."""
    id: str
    dataset_id: str
    target_column: str
    task_type: str  # classification, regression
    
    # Current production version
    production_version_id: Optional[str] = None
    
    # Metadata
    created_at: datetime
    updated_at: datetime


class ModelVersion(BaseModel):
    """Model version - each training creates a version."""
    id: str
    registry_id: str
    version_number: int
    
    # Status
    status: str = ModelStatus.STAGING
    
    # Model info
    model_path: str
    model_name: str  # H2O model name
    schema_hash: str
    feature_columns: List[str] = []
    
    # Training config
    training_config: Dict[str, Any] = {}
    
    # H2O leaderboard
    leaderboard: List[Dict[str, Any]] = []
    best_algorithm: Optional[str] = None
    
    # Training metadata
    training_duration_seconds: float = 0.0
    training_rows: int = 0
    
    # Timestamps
    created_at: datetime
    promoted_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None


class ModelMetrics(BaseModel):
    """Model metrics for each version."""
    id: str
    version_id: str
    
    # Common metrics
    train_rows: int = 0
    test_rows: int = 0
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    logloss: Optional[float] = None
    
    # Confusion matrix
    confusion_matrix: Optional[Dict[str, Any]] = None
    
    # ROC curve data
    roc_curve: Optional[Dict[str, Any]] = None
    
    # PR curve data
    pr_curve: Optional[Dict[str, Any]] = None
    
    # Regression metrics
    r2: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    mape: Optional[float] = None
    
    # Residual statistics
    residual_mean: Optional[float] = None
    residual_std: Optional[float] = None
    
    # Feature importance
    feature_importance: Dict[str, Any] = {}
    
    # SHAP values (summary)
    shap_summary: Optional[Dict[str, Any]] = None
    
    # Prediction vs actual (sample)
    prediction_vs_actual: Optional[Dict[str, Any]] = None
