"""Report and dashboard schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class VisualConfig(BaseModel):
    """Visual configuration schema."""
    visual_type: str  # bar, line, pie, table, kpi, etc.
    title: Optional[str] = None
    
    # Position and size (in grid units)
    x: int = 0
    y: int = 0
    width: int = 4
    height: int = 4
    
    # Chart-specific configuration
    chart_config: Dict[str, Any] = {}
    
    # Data binding
    data_fields: Dict[str, Any] = {}  # x_axis, y_axis, values, colors, etc.
    
    # Formatting
    formatting: Dict[str, Any] = {}
    
    # Conditional formatting rules
    conditional_formatting: List[Dict[str, Any]] = []


class PageConfig(BaseModel):
    """Report page configuration."""
    id: str
    name: str
    visuals: List[VisualConfig] = []
    filters: List[Dict[str, Any]] = []
    background: Optional[str] = None


class ReportBase(BaseModel):
    """Base report schema."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None


class ReportCreate(ReportBase):
    """Report creation schema."""
    dataset_id: str
    pages: Optional[List[PageConfig]] = None
    layout_config: Optional[Dict[str, Any]] = None
    theme: Optional[str] = "default"


class ReportUpdate(BaseModel):
    """Report update schema."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    pages: Optional[List[PageConfig]] = None
    layout_config: Optional[Dict[str, Any]] = None
    theme: Optional[str] = None
    is_shared: Optional[bool] = None


class ReportResponse(ReportBase):
    """Report response schema."""
    id: str
    owner_id: str
    workspace_id: Optional[str]
    dataset_id: str
    pages: List[PageConfig]
    layout_config: Dict[str, Any]
    theme: str
    is_active: bool
    is_shared: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ReportListResponse(BaseModel):
    """Report list response."""
    id: str
    name: str
    description: Optional[str]
    dataset_id: str
    dataset_name: Optional[str]
    page_count: int
    is_shared: bool
    created_at: datetime
    updated_at: datetime


class VisualCreate(BaseModel):
    """Visual creation schema."""
    page_id: str
    visual_type: str
    title: Optional[str] = None
    x: int = 0
    y: int = 0
    width: int = 4
    height: int = 4
    chart_config: Dict[str, Any] = {}
    data_fields: Dict[str, Any] = {}
    formatting: Dict[str, Any] = {}


class VisualUpdate(BaseModel):
    """Visual update schema."""
    title: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    chart_config: Optional[Dict[str, Any]] = None
    data_fields: Optional[Dict[str, Any]] = None
    formatting: Optional[Dict[str, Any]] = None
    conditional_formatting: Optional[List[Dict[str, Any]]] = None


class VisualResponse(VisualCreate):
    """Visual response schema."""
    id: str
    report_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DashboardWidget(BaseModel):
    """Dashboard widget schema."""
    id: str
    type: str  # pinned_visual, kpi, text, image
    
    # Source (for pinned visuals)
    report_id: Optional[str] = None
    visual_id: Optional[str] = None
    
    # Position and size
    x: int = 0
    y: int = 0
    width: int = 4
    height: int = 4
    
    # Configuration
    title: Optional[str] = None
    config: Dict[str, Any] = {}


class DashboardBase(BaseModel):
    """Base dashboard schema."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None


class DashboardCreate(DashboardBase):
    """Dashboard creation schema."""
    widgets: Optional[List[DashboardWidget]] = None
    filters: Optional[List[Dict[str, Any]]] = None
    layout_config: Optional[Dict[str, Any]] = None
    theme: Optional[str] = "default"


class DashboardUpdate(BaseModel):
    """Dashboard update schema."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    widgets: Optional[List[DashboardWidget]] = None
    filters: Optional[List[Dict[str, Any]]] = None
    layout_config: Optional[Dict[str, Any]] = None
    theme: Optional[str] = None
    is_shared: Optional[bool] = None
    is_favorite: Optional[bool] = None


class DashboardResponse(DashboardBase):
    """Dashboard response schema."""
    id: str
    owner_id: str
    workspace_id: Optional[str]
    widgets: List[DashboardWidget]
    filters: List[Dict[str, Any]]
    layout_config: Dict[str, Any]
    theme: str
    is_active: bool
    is_shared: bool
    is_favorite: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DashboardShareRequest(BaseModel):
    """Dashboard share request schema."""
    password: Optional[str] = None
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class DashboardShareResponse(BaseModel):
    """Dashboard share response schema."""
    share_url: str
    share_token: str
    expires_at: Optional[datetime]


class ExportRequest(BaseModel):
    """Export request schema."""
    format: str  # pdf, png, csv, excel, powerpoint
    page_id: Optional[str] = None
    visual_ids: Optional[List[str]] = None
    include_data: bool = False


class ExportResponse(BaseModel):
    """Export response schema."""
    download_url: str
    file_name: str
    file_size: int
    expires_at: datetime
