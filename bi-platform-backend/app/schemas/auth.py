"""Authentication schemas."""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=6)


class UserUpdate(BaseModel):
    """User update schema."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    timezone: Optional[str] = None
    language: Optional[str] = None
    avatar_url: Optional[str] = None


class UserResponse(UserBase):
    """User response schema."""
    id: str
    is_active: bool
    email_verified: bool
    timezone: str
    language: str
    avatar_url: Optional[str]
    last_login: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """Login request schema."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request schema."""
    current_password: str
    new_password: str = Field(..., min_length=6)


class WorkspaceBase(BaseModel):
    """Base workspace schema."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None


class WorkspaceCreate(WorkspaceBase):
    """Workspace creation schema."""
    pass


class WorkspaceUpdate(BaseModel):
    """Workspace update schema."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    settings: Optional[dict] = None


class WorkspaceMember(BaseModel):
    """Workspace member schema."""
    user_id: str
    name: str
    email: str
    role: str
    joined_at: datetime


class WorkspaceResponse(WorkspaceBase):
    """Workspace response schema."""
    id: str
    slug: str
    is_active: bool
    settings: dict
    created_at: datetime
    updated_at: datetime
    member_count: int
    
    class Config:
        from_attributes = True


class WorkspaceDetailResponse(WorkspaceResponse):
    """Workspace detail response with members."""
    members: List[WorkspaceMember]


class RoleBase(BaseModel):
    """Base role schema."""
    name: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = None
    permissions: List[str] = []


class RoleCreate(RoleBase):
    """Role creation schema."""
    rls_filter: Optional[str] = None


class RoleResponse(RoleBase):
    """Role response schema."""
    id: str
    workspace_id: str
    rls_filter: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class AuditLogResponse(BaseModel):
    """Audit log response schema."""
    id: str
    user_id: Optional[str]
    user_name: Optional[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    details: Optional[dict]
    ip_address: Optional[str]
    timestamp: datetime
    
    class Config:
        from_attributes = True
