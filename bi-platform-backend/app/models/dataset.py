"""Dataset models."""
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Any, Optional


class Dataset(BaseModel):
    """Dataset model."""
    id: str
    name: str
    description: Optional[str] = None
    file_path: str
    file_type: str  # csv, xlsx, json
    file_size_bytes: int = 0
    
    # Schema info
    row_count: int = 0
    column_count: int = 0
    columns: List[Dict[str, Any]] = []  # [{name, type, nullable}]
    
    # Ownership
    created_by: str
    
    # Timestamps
    created_at: datetime
    updated_at: datetime


class DatasetProfile(BaseModel):
    """Dataset profile with statistics."""
    id: str
    dataset_id: str
    
    # Overview
    total_rows: int = 0
    total_columns: int = 0
    numeric_columns: int = 0
    categorical_columns: int = 0
    date_columns: int = 0
    memory_usage_mb: float = 0.0
    
    # Quality
    total_missing_values: int = 0
    missing_percentage: float = 0.0
    duplicate_rows: int = 0
    constant_columns: List[str] = []
    
    # Column statistics
    column_stats: Dict[str, Any] = {}  # {col_name: {mean, std, min, max, ...}}
    
    # Correlation matrix
    correlation_matrix: Dict[str, Any] = {}
    
    # Insights
    insights: List[str] = []
    
    # Created at
    created_at: datetime
    updated_at: datetime
