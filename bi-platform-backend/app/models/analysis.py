"""Analysis models."""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any
import enum


class AnalysisType(str, enum.Enum):
    """Analysis type."""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"


class AnalysisResult(BaseModel):
    """Analysis results storage."""
    id: str
    dataset_id: str
    analysis_type: str
    
    # Results
    summary: Dict[str, Any] = {}
    visualizations: List[Dict[str, Any]] = []
    statistics: Dict[str, Any] = {}
    insights: List[str] = []
    
    # Created at
    created_at: datetime


class AIReport(BaseModel):
    """AI-generated reports."""
    id: str
    dataset_id: str
    analysis_type: str
    
    # Report content
    title: str
    content: str
    key_findings: List[str] = []
    recommendations: List[str] = []
    
    # Model info (if LLM was used)
    model_name: Optional[str] = None
    tokens_used: int = 0
    
    # Created at
    created_at: datetime
