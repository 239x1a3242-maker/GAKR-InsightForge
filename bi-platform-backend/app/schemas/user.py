"""User schemas."""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from uuid import UUID


class UserCreate(BaseModel):
    """User create request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=1, max_length=255)


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response."""
    id: UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenRefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str
