"""API routes (File-based)."""
from fastapi import APIRouter
from app.api import auth, datasets, query
from app.api import reports_minimal as reports
from app.api import ml_minimal as ml
from app.api import analysis

# File-based API router
api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(reports.router, prefix="/reports", tags=["Reports"])
api_router.include_router(reports.dashboard_router, prefix="/dashboards", tags=["Dashboards"])
api_router.include_router(ml.router, prefix="/ml", tags=["ML Models"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
api_router.include_router(query.router, prefix="/query", tags=["Query"])
