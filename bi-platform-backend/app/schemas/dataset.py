"""Dataset schemas."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID


class ColumnInfo(BaseModel):
    """Column information."""
    name: str
    dtype: str
    nullable: bool = True
    unique_count: Optional[int] = None
    missing_count: Optional[int] = None
    missing_percentage: Optional[float] = None


class DatasetCreate(BaseModel):
    """Dataset create request."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class DatasetResponse(BaseModel):
    """Dataset response."""
    id: UUID
    name: str
    description: Optional[str]
    file_type: str
    file_size_bytes: int
    row_count: int
    column_count: int
    columns: List[Dict[str, Any]]
    created_by: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    """Dataset list response."""
    items: List[DatasetResponse]
    total: int


class ColumnStats(BaseModel):
    """Column statistics."""
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    unique_count: Optional[int] = None
    top_values: Optional[List[Dict[str, Any]]] = None


class DatasetProfileResponse(BaseModel):
    """Dataset profile response."""
    id: UUID
    dataset_id: UUID
    total_rows: int
    total_columns: int
    numeric_columns: int
    categorical_columns: int
    date_columns: int
    memory_usage_mb: float
    total_missing_values: int
    missing_percentage: float
    duplicate_rows: int
    constant_columns: List[str]
    column_stats: Dict[str, ColumnStats]
    correlation_matrix: Dict[str, Dict[str, float]]
    insights: List[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True


class DataQualityIssue(BaseModel):
    """Data quality issue."""
    type: str
    column: Optional[str]
    severity: str  # low, medium, high
    description: str
    suggestion: Optional[str]


class DataQualityResponse(BaseModel):
    """Data quality response."""
    issues: List[DataQualityIssue]
    score: float  # 0-100


class TaskDetectionResponse(BaseModel):
    """Task detection response."""
    target_column: str
    task_type: str  # classification, regression
    confidence: float
    reason: str
    unique_count: int
    unique_ratio: float
    suggested_algorithms: List[str]
