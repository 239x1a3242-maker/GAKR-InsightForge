"""Reports API routes."""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta

from app.core.database import get_db
from app.models.user import User
from app.models.report import Report, Visual, Dashboard
from app.models.dataset import Dataset
from app.schemas.report import (
    ReportCreate, ReportUpdate, ReportResponse, ReportListResponse,
    VisualCreate, VisualUpdate, VisualResponse,
    DashboardCreate, DashboardUpdate, DashboardResponse,
    DashboardShareRequest, DashboardShareResponse,
    ExportRequest, ExportResponse
)
from app.api.auth import get_current_user, log_audit

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("", response_model=List[ReportListResponse])
async def list_reports(
    workspace_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List reports accessible to user."""
    query = select(Report).where(
        and_(
            Report.is_active == True,
            Report.owner_id == current_user.id
        )
    )
    
    if workspace_id:
        query = query.where(Report.workspace_id == workspace_id)
    if dataset_id:
        query = query.where(Report.dataset_id == dataset_id)
    
    query = query.order_by(desc(Report.updated_at))
    
    result = await db.execute(query)
    reports = result.scalars().all()
    
    return [
        ReportListResponse(
            id=r.id,
            name=r.name,
            description=r.description,
            dataset_id=r.dataset_id,
            dataset_name=r.dataset.name if r.dataset else None,
            page_count=len(r.pages) if r.pages else 0,
            is_shared=r.is_shared,
            created_at=r.created_at,
            updated_at=r.updated_at
        )
        for r in reports
    ]


@router.post("", response_model=ReportResponse, status_code=201)
async def create_report(
    data: ReportCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    request: Request = None
):
    """Create a new report."""
    # Verify dataset exists and user has access
    result = await db.execute(
        select(Dataset).where(
            and_(
                Dataset.id == data.dataset_id,
                Dataset.is_active == True
            )
        )
    )
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if dataset.owner_id != current_user.id and not dataset.is_shared:
        raise HTTPException(status_code=403, detail="Access denied to dataset")
    
    # Create default page if none provided
    pages = data.pages
    if not pages:
        pages = [{
            "id": str(uuid.uuid4()),
            "name": "Page 1",
            "visuals": [],
            "filters": []
        }]
    
    report = Report(
        name=data.name,
        description=data.description,
        owner_id=current_user.id,
        workspace_id=dataset.workspace_id,
        dataset_id=data.dataset_id,
        pages=[p.model_dump() if hasattr(p, 'model_dump') else p for p in pages],
        layout_config=data.layout_config or {"columns": 12, "rowHeight": 30},
        theme=data.theme or "default"
    )
    
    db.add(report)
    await db.commit()
    await db.refresh(report)
    
    await log_audit(db, current_user.id, "create", "report", report.id, request=request)
    
    return ReportResponse.model_validate(report)


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get report details."""
    result = await db.execute(
        select(Report).where(
            and_(
                Report.id == report_id,
                Report.is_active == True
            )
        )
    )
    report = result.scalar_one_or_none()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if report.owner_id != current_user.id and not report.is_shared:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return ReportResponse.model_validate(report)


@router.put("/{report_id}", response_model=ReportResponse)
async def update_report(
    report_id: str,
    data: ReportUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update report."""
    result = await db.execute(
        select(Report).where(Report.id == report_id)
    )
    report = result.scalar_one_or_none()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if report.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if data.name:
        report.name = data.name
    if data.description is not None:
        report.description = data.description
    if data.pages:
        report.pages = [p.model_dump() if hasattr(p, 'model_dump') else p for p in data.pages]
    if data.layout_config:
        report.layout_config = data.layout_config
    if data.theme:
        report.theme = data.theme
    if data.is_shared is not None:
        report.is_shared = data.is_shared
    
    await db.commit()
    await db.refresh(report)
    
    await log_audit(db, current_user.id, "update", "report", report_id)
    
    return ReportResponse.model_validate(report)


@router.delete("/{report_id}")
async def delete_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete report."""
    result = await db.execute(
        select(Report).where(Report.id == report_id)
    )
    report = result.scalar_one_or_none()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if report.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    report.is_active = False
    await db.commit()
    
    await log_audit(db, current_user.id, "delete", "report", report_id)
    
    return {"message": "Report deleted successfully"}


@router.post("/{report_id}/pages")
async def add_page(
    report_id: str,
    page_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Add a new page to report."""
    result = await db.execute(
        select(Report).where(Report.id == report_id)
    )
    report = result.scalar_one_or_none()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if report.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    new_page = {
        "id": str(uuid.uuid4()),
        "name": page_name,
        "visuals": [],
        "filters": []
    }
    
    pages = report.pages or []
    pages.append(new_page)
    report.pages = pages
    
    await db.commit()
    await db.refresh(report)
    
    return {"page_id": new_page["id"], "message": "Page added successfully"}


@router.delete("/{report_id}/pages/{page_id}")
async def delete_page(
    report_id: str,
    page_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a page from report."""
    result = await db.execute(
        select(Report).where(Report.id == report_id)
    )
    report = result.scalar_one_or_none()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if report.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    pages = report.pages or []
    pages = [p for p in pages if p.get("id") != page_id]
    report.pages = pages
    
    await db.commit()
    
    return {"message": "Page deleted successfully"}


# Dashboard routes
@router.get("/dashboards/list", response_model=List[DashboardResponse])
async def list_dashboards(
    workspace_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List dashboards."""
    query = select(Dashboard).where(
        and_(
            Dashboard.is_active == True,
            Dashboard.owner_id == current_user.id
        )
    )
    
    if workspace_id:
        query = query.where(Dashboard.workspace_id == workspace_id)
    
    query = query.order_by(desc(Dashboard.updated_at))
    
    result = await db.execute(query)
    dashboards = result.scalars().all()
    
    return [DashboardResponse.model_validate(d) for d in dashboards]


@router.post("/dashboards", response_model=DashboardResponse, status_code=201)
async def create_dashboard(
    data: DashboardCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    request: Request = None
):
    """Create a new dashboard."""
    dashboard = Dashboard(
        name=data.name,
        description=data.description,
        owner_id=current_user.id,
        widgets=[w.model_dump() if hasattr(w, 'model_dump') else w for w in (data.widgets or [])],
        filters=data.filters or [],
        layout_config=data.layout_config or {"columns": 12, "rowHeight": 100},
        theme=data.theme or "default"
    )
    
    db.add(dashboard)
    await db.commit()
    await db.refresh(dashboard)
    
    await log_audit(db, current_user.id, "create", "dashboard", dashboard.id, request=request)
    
    return DashboardResponse.model_validate(dashboard)


@router.get("/dashboards/{dashboard_id}", response_model=DashboardResponse)
async def get_dashboard(
    dashboard_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get dashboard details."""
    result = await db.execute(
        select(Dashboard).where(
            and_(
                Dashboard.id == dashboard_id,
                Dashboard.is_active == True
            )
        )
    )
    dashboard = result.scalar_one_or_none()
    
    if not dashboard:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    if dashboard.owner_id != current_user.id and not dashboard.is_shared:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return DashboardResponse.model_validate(dashboard)


@router.put("/dashboards/{dashboard_id}", response_model=DashboardResponse)
async def update_dashboard(
    dashboard_id: str,
    data: DashboardUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update dashboard."""
    result = await db.execute(
        select(Dashboard).where(Dashboard.id == dashboard_id)
    )
    dashboard = result.scalar_one_or_none()
    
    if not dashboard:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    if dashboard.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if data.name:
        dashboard.name = data.name
    if data.description is not None:
        dashboard.description = data.description
    if data.widgets:
        dashboard.widgets = [w.model_dump() if hasattr(w, 'model_dump') else w for w in data.widgets]
    if data.filters:
        dashboard.filters = data.filters
    if data.layout_config:
        dashboard.layout_config = data.layout_config
    if data.theme:
        dashboard.theme = data.theme
    if data.is_shared is not None:
        dashboard.is_shared = data.is_shared
    if data.is_favorite is not None:
        dashboard.is_favorite = data.is_favorite
    
    await db.commit()
    await db.refresh(dashboard)
    
    await log_audit(db, current_user.id, "update", "dashboard", dashboard_id)
    
    return DashboardResponse.model_validate(dashboard)


@router.delete("/dashboards/{dashboard_id}")
async def delete_dashboard(
    dashboard_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete dashboard."""
    result = await db.execute(
        select(Dashboard).where(Dashboard.id == dashboard_id)
    )
    dashboard = result.scalar_one_or_none()
    
    if not dashboard:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    if dashboard.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    dashboard.is_active = False
    await db.commit()
    
    await log_audit(db, current_user.id, "delete", "dashboard", dashboard_id)
    
    return {"message": "Dashboard deleted successfully"}


@router.post("/dashboards/{dashboard_id}/share", response_model=DashboardShareResponse)
async def share_dashboard(
    dashboard_id: str,
    data: DashboardShareRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create shareable link for dashboard."""
    result = await db.execute(
        select(Dashboard).where(Dashboard.id == dashboard_id)
    )
    dashboard = result.scalar_one_or_none()
    
    if not dashboard:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    if dashboard.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Generate share token
    share_token = str(uuid.uuid4())
    dashboard.share_token = share_token
    
    if data.password:
        from app.core.security import get_password_hash
        dashboard.share_password = get_password_hash(data.password)
    
    if data.expires_in_days:
        dashboard.share_expires_at = datetime.now(timezone.utc) + timedelta(days=data.expires_in_days)
    
    dashboard.is_shared = True
    await db.commit()
    
    share_url = f"/share/dashboard/{share_token}"
    
    return DashboardShareResponse(
        share_url=share_url,
        share_token=share_token,
        expires_at=dashboard.share_expires_at
    )


@router.post("/export")
async def export_report(
    data: ExportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Export report or dashboard."""
    # TODO: Implement export functionality
    # This would generate PDF, PNG, or PowerPoint files
    
    return {
        "message": "Export started",
        "job_id": str(uuid.uuid4()),
        "status": "processing"
    }
