"""Model registry schemas."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID


class TrainingConfig(BaseModel):
    """Training configuration."""
    max_runtime_secs: int = Field(default=600, ge=60, le=36000)
    nfolds: int = Field(default=5, ge=2, le=10)
    max_models: int = Field(default=20, ge=5, le=100)
    enable_stacking: bool = True
    seed: int = 42


class TrainingRequest(BaseModel):
    """Training request."""
    dataset_id: UUID
    targets: List[str] = Field(..., min_length=1)
    config: TrainingConfig = Field(default_factory=TrainingConfig)


class TrainingResponse(BaseModel):
    """Training response."""
    job_id: str
    status: str  # queued, running, completed, failed
    message: str
    model_ids: List[UUID]


class LeaderboardEntry(BaseModel):
    """Leaderboard entry."""
    model_id: str
    algorithm: str
    auc: Optional[float] = None
    logloss: Optional[float] = None
    mean_per_class_error: Optional[float] = None
    rmse: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    rmsle: Optional[float] = None
    training_time_ms: int


class ModelVersionResponse(BaseModel):
    """Model version response."""
    id: UUID
    registry_id: UUID
    version_number: int
    status: str
    model_name: str
    best_algorithm: Optional[str]
    leaderboard: List[LeaderboardEntry]
    training_duration_seconds: float
    training_rows: int
    created_at: datetime
    promoted_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class ModelRegistryResponse(BaseModel):
    """Model registry response."""
    id: UUID
    dataset_id: UUID
    target_column: str
    task_type: str
    production_version_id: Optional[UUID]
    production_version: Optional[ModelVersionResponse]
    version_count: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ClassificationMetrics(BaseModel):
    """Classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: Optional[float] = None
    logloss: Optional[float] = None
    confusion_matrix: List[List[int]]
    roc_curve: Optional[Dict[str, List[float]]] = None
    pr_curve: Optional[Dict[str, List[float]]] = None


class RegressionMetrics(BaseModel):
    """Regression metrics."""
    r2: float
    rmse: float
    mae: float
    mse: float
    mape: Optional[float] = None
    residual_mean: float
    residual_std: float
    prediction_vs_actual: List[Dict[str, float]]


class ModelMetricsResponse(BaseModel):
    """Model metrics response."""
    id: UUID
    version_id: UUID
    task_type: str
    train_rows: int
    test_rows: int
    classification: Optional[ClassificationMetrics] = None
    regression: Optional[RegressionMetrics] = None
    feature_importance: Dict[str, float]
    
    class Config:
        from_attributes = True


class PromoteRequest(BaseModel):
    """Promote model request."""
    comment: Optional[str] = None


class PromoteResponse(BaseModel):
    """Promote model response."""
    success: bool
    message: str
    previous_version: Optional[int] = None
    new_version: int


class TrainingStatusResponse(BaseModel):
    """Training status response."""
    job_id: str
    status: str  # queued, running, completed, failed
    progress: float  # 0-100
    current_step: str
    message: Optional[str] = None
    model_ids: Optional[List[UUID]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ShapGlobalResponse(BaseModel):
    """SHAP global response."""
    feature_names: List[str]
    shap_values: List[float]
    base_value: float
    feature_importance: Dict[str, float]


class ShapLocalRequest(BaseModel):
    """SHAP local request."""
    features: Dict[str, Any]


class ShapLocalResponse(BaseModel):
    """SHAP local response."""
    prediction: float
    base_value: float
    shap_values: Dict[str, float]
    feature_contributions: List[Dict[str, Any]]


class DriftReport(BaseModel):
    """Drift report."""
    model_id: UUID
    psi_score: float
    drift_detected: bool
    threshold: float
    feature_psi: Dict[str, float]
    report_date: datetime
    recommendation: str
