"""Reports & Dashboards API (File-based)."""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional
from pydantic import BaseModel

from app.api.auth import get_current_user
from app.core.filedb import ReportDB, DashboardDB

router = APIRouter()
dashboard_router = APIRouter()


# ── Pydantic models ──────────────────────────────────────────────────────────

class ReportCreate(BaseModel):
    name: str
    description: str = ""
    dataset_id: str = ""


class ReportUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    pages: Optional[list] = None


class DashboardCreate(BaseModel):
    name: str
    description: str = ""


class DashboardUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    widgets: Optional[list] = None
    layout: Optional[list] = None


# ── Reports ──────────────────────────────────────────────────────────────────

@router.get("")
async def list_reports(
    skip: int = 0,
    limit: int = 100,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    reports = ReportDB.get_by_user(current_user["id"])
    paginated = reports[skip: skip + limit]
    return {"items": paginated, "total": len(reports), "skip": skip, "limit": limit}


@router.post("")
async def create_report(
    data: ReportCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    report = ReportDB.create(
        user_id=current_user["id"],
        name=data.name,
        description=data.description,
        dataset_id=data.dataset_id,
    )
    return report


@router.get("/{report_id}")
async def get_report(
    report_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    report = ReportDB.get_by_id(report_id)
    if not report or report.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report


@router.put("/{report_id}")
async def update_report(
    report_id: str,
    data: ReportUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    report = ReportDB.get_by_id(report_id)
    if not report or report.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    updates = {k: v for k, v in data.dict().items() if v is not None}
    if "pages" in updates:
        updates["page_count"] = len(updates["pages"])
    updated = ReportDB.update(report_id, **updates)
    return updated


@router.delete("/{report_id}")
async def delete_report(
    report_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    report = ReportDB.get_by_id(report_id)
    if not report or report.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    ReportDB.delete(report_id)
    return {"message": "Report deleted"}


@router.post("/{report_id}/pages")
async def add_page(
    report_id: str,
    page_name: str = "New Page",
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    report = ReportDB.get_by_id(report_id)
    if not report or report.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    import uuid
    pages = report.get("pages", [])
    new_page = {"id": str(uuid.uuid4()), "name": page_name, "widgets": []}
    pages.append(new_page)
    ReportDB.update(report_id, pages=pages, page_count=len(pages))
    return new_page


@router.delete("/{report_id}/pages/{page_id}")
async def delete_page(
    report_id: str,
    page_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    report = ReportDB.get_by_id(report_id)
    if not report or report.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    pages = [p for p in report.get("pages", []) if p.get("id") != page_id]
    ReportDB.update(report_id, pages=pages, page_count=len(pages))
    return {"message": "Page deleted"}


# ── Dashboards ───────────────────────────────────────────────────────────────

@dashboard_router.get("")
async def list_dashboards(
    skip: int = 0,
    limit: int = 100,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    dashboards = DashboardDB.get_by_user(current_user["id"])
    paginated = dashboards[skip: skip + limit]
    return {"items": paginated, "total": len(dashboards), "skip": skip, "limit": limit}


@dashboard_router.post("")
async def create_dashboard(
    data: DashboardCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    dashboard = DashboardDB.create(
        user_id=current_user["id"],
        name=data.name,
        description=data.description,
    )
    return dashboard


@dashboard_router.get("/{dashboard_id}")
async def get_dashboard(
    dashboard_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    dashboard = DashboardDB.get_by_id(dashboard_id)
    if not dashboard or dashboard.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard not found")
    return dashboard


@dashboard_router.put("/{dashboard_id}")
async def update_dashboard(
    dashboard_id: str,
    data: DashboardUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    dashboard = DashboardDB.get_by_id(dashboard_id)
    if not dashboard or dashboard.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard not found")
    updates = {k: v for k, v in data.dict().items() if v is not None}
    updated = DashboardDB.update(dashboard_id, **updates)
    return updated


@dashboard_router.delete("/{dashboard_id}")
async def delete_dashboard(
    dashboard_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    dashboard = DashboardDB.get_by_id(dashboard_id)
    if not dashboard or dashboard.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard not found")
    DashboardDB.delete(dashboard_id)
    return {"message": "Dashboard deleted"}


@dashboard_router.post("/{dashboard_id}/share")
async def share_dashboard(
    dashboard_id: str,
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    dashboard = DashboardDB.get_by_id(dashboard_id)
    if not dashboard or dashboard.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard not found")
    import uuid
    share_token = str(uuid.uuid4())
    DashboardDB.update(dashboard_id, is_published=True, share_token=share_token)
    return {"share_token": share_token, "share_url": f"/shared/{share_token}"}


@router.post("/export")
async def export_report(
    data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return {"message": "Export queued", "format": data.get("format", "pdf"), "status": "pending"}
