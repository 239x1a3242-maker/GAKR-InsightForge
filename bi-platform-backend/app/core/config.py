"""Application configuration."""
from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # App
    APP_NAME: str = "Enterprise BI & AutoML Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # File-based Database (no PostgreSQL)
    FILE_BASED_DB: bool = True
    DATA_DIR: str = "./data"
    UPLOADS_DIR: str = "./uploads"
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 500
    UPLOAD_DIR: str = "./uploads"
    MODEL_DIR: str = "./data/models"
    
    # H2O
    H2O_PORT: int = 54321
    H2O_IP: str = "localhost"
    
    # LLM
    LLM_MODEL_NAME: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    LLM_MAX_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.7
    
    # Query Engine
    MAX_QUERY_ROWS: int = 1000000
    QUERY_TIMEOUT_SECONDS: int = 300
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
