"""Analysis schemas."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID


class DescriptiveAnalyticsRequest(BaseModel):
    """Descriptive analytics request."""
    columns: Optional[List[str]] = None
    include_correlation: bool = True


class DistributionData(BaseModel):
    """Distribution data."""
    bins: List[str]
    counts: List[int]


class DescriptiveAnalyticsResponse(BaseModel):
    """Descriptive analytics response."""
    dataset_id: UUID
    total_rows: int
    total_columns: int
    numeric_columns: int
    categorical_columns: int
    date_columns: int
    memory_usage_mb: float
    missing_summary: Dict[str, Any]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    distributions: Dict[str, DistributionData]
    column_stats: Dict[str, Any]
    insights: List[str]
    generated_at: datetime


class DiagnosticAnalyticsRequest(BaseModel):
    """Diagnostic analytics request."""
    target_column: Optional[str] = None
    focus_columns: Optional[List[str]] = None


class DiagnosticAnalyticsResponse(BaseModel):
    """Diagnostic analytics response."""
    dataset_id: UUID
    target_column: Optional[str]
    feature_importance: Optional[Dict[str, float]] = None
    correlations: Dict[str, List[Dict[str, Any]]]
    outliers: Dict[str, Any]
    segments: List[Dict[str, Any]]
    root_causes: List[str]
    generated_at: datetime


class PredictiveAnalyticsRequest(BaseModel):
    """Predictive analytics request."""
    target_column: str
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    algorithms: Optional[List[str]] = None


class AlgorithmResult(BaseModel):
    """Algorithm result."""
    name: str
    metrics: Dict[str, float]
    training_time_ms: int
    rank: int


class PredictiveAnalyticsResponse(BaseModel):
    """Predictive analytics response."""
    dataset_id: UUID
    target_column: str
    task_type: str
    best_algorithm: str
    algorithms: List[AlgorithmResult]
    feature_importance: Dict[str, float]
    cross_validation_score: float
    generated_at: datetime


class PrescriptiveAnalyticsRequest(BaseModel):
    """Prescriptive analytics request."""
    target_column: str
    constraints: Optional[Dict[str, Any]] = None


class Recommendation(BaseModel):
    """Recommendation."""
    action: str
    impact: str
    confidence: float
    supporting_data: Dict[str, Any]


class PrescriptiveAnalyticsResponse(BaseModel):
    """Prescriptive analytics response."""
    dataset_id: UUID
    target_column: str
    recommendations: List[Recommendation]
    scenarios: List[Dict[str, Any]]
    what_if_analysis: Dict[str, Any]
    generated_at: datetime


class AskAIRequest(BaseModel):
    """Ask AI request."""
    question: str = Field(..., min_length=5, max_length=1000)
    context: Optional[str] = None


class AskAIResponse(BaseModel):
    """Ask AI response."""
    question: str
    answer: str
    confidence: float
    sources: List[str]
    suggested_followups: List[str]
    generated_at: datetime
    tokens_used: int
