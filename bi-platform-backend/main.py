"""Main FastAPI application."""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import structlog

from app.core.config import settings
from app.core.filedb import init_db
from app.api import api_router

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting up", app_name=settings.APP_NAME)
    init_db()  # Initialize file-based database
    logger.info("File-based database initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise BI & AutoML Platform API (File-based)",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": time.time(),
        "database": "file-based"
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/api/docs",
        "database": "file-based JSON storage"
    }


# API routes
app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
