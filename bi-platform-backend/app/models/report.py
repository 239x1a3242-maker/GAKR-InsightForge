"""Report and visualization models."""
from sqlalchemy import Column, String, DateTime, Integer, JSON, ForeignKey, Text, Float, Boolean
from sqlalchemy.orm import relationship
from app.core.database import Base
from datetime import datetime, timezone
import uuid


class Report(Base):
    """Report model."""
    __tablename__ = "reports"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    description = Column(String(500))
    
    # Ownership
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    workspace_id = Column(String(36), ForeignKey("workspaces.id"))
    
    # Linked dataset
    dataset_id = Column(String(36), ForeignKey("datasets.id"))
    
    # Pages (JSON array of page definitions)
    pages = Column(JSON, default=list)
    
    # Layout settings
    layout_config = Column(JSON, default=dict)  # grid size, snap settings, etc.
    
    # Theme
    theme = Column(String(50), default="default")
    
    # Status
    is_active = Column(Boolean, default=True)
    is_shared = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    owner = relationship("User", back_populates="reports")
    workspace = relationship("Workspace", back_populates="reports")
    visuals = relationship("Visual", back_populates="report", cascade="all, delete-orphan")


class Visual(Base):
    """Visual (chart/table/KPI) in a report."""
    __tablename__ = "visuals"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    report_id = Column(String(36), ForeignKey("reports.id"), nullable=False)
    page_id = Column(String(36), nullable=False)  # Page this visual belongs to
    
    # Visual type
    visual_type = Column(String(50), nullable=False)  # bar, line, pie, table, kpi, etc.
    
    # Position and size (in grid units)
    position_x = Column(Integer, default=0)
    position_y = Column(Integer, default=0)
    width = Column(Integer, default=4)
    height = Column(Integer, default=4)
    
    # Layer order
    z_index = Column(Integer, default=0)
    
    # Configuration
    title = Column(String(200))
    config = Column(JSON, default=dict)  # Chart-specific configuration
    
    # Data binding
    data_config = Column(JSON, default=dict)  # Fields, aggregations, filters
    
    # Interactions
    interactions = Column(JSON, default=dict)  # Cross-filter, drill-through settings
    
    # Conditional formatting
    conditional_formatting = Column(JSON, default=list)
    
    # Is locked
    is_locked = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    report = relationship("Report", back_populates="visuals")


class Dashboard(Base):
    """Dashboard model (pinned visuals from reports)."""
    __tablename__ = "dashboards"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    description = Column(String(500))
    
    # Ownership
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    workspace_id = Column(String(36), ForeignKey("workspaces.id"))
    
    # Widgets (JSON array)
    widgets = Column(JSON, default=list)
    
    # Global filters
    filters = Column(JSON, default=list)
    
    # Layout
    layout_config = Column(JSON, default=dict)
    
    # Theme
    theme = Column(String(50), default="default")
    
    # Status
    is_active = Column(Boolean, default=True)
    is_shared = Column(Boolean, default=False)
    is_favorite = Column(Boolean, default=False)
    
    # Sharing
    share_token = Column(String(100), unique=True)
    share_password = Column(String(255))
    share_expires_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    owner = relationship("User", back_populates="dashboards")
    workspace = relationship("Workspace", back_populates="dashboards")


class VisualTemplate(Base):
    """Reusable visual templates."""
    __tablename__ = "visual_templates"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    
    # Template category
    category = Column(String(50))  # kpi, comparison, trend, distribution, etc.
    
    # Visual type
    visual_type = Column(String(50), nullable=False)
    
    # Default configuration
    default_config = Column(JSON, default=dict)
    
    # Preview image
    preview_url = Column(String(500))
    
    # Is system template
    is_system = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
