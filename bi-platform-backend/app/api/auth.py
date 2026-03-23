"""Authentication API (File-based)."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import timedelta
from typing import Optional, Dict, Any
from pydantic import BaseModel

from app.core.security import (
    verify_password, get_password_hash, create_access_token,
    create_refresh_token, decode_token
)
from app.core.filedb import UserDB

router = APIRouter()
security = HTTPBearer()


# Request/Response models
class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get current user from token."""
    token = credentials.credentials
    payload = decode_token(token)
    
    if not payload or payload.get('type') != 'access':
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user = UserDB.get_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user


@router.post("/register")
async def register(request: RegisterRequest):
    """Register new user."""
    # Check if email exists
    if UserDB.get_by_email(request.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists — please login instead"
        )
    
    # Create user
    user = UserDB.create(
        email=request.email,
        password_hash=get_password_hash(request.password),
        full_name=request.name
    )
    
    # Generate tokens
    access_token = create_access_token(data={"sub": user["id"], "type": "access"})
    refresh_token = create_refresh_token(data={"sub": user["id"], "type": "refresh"})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 86400,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user.get("full_name", ""),
            "is_active": user.get("is_active", True),
            "email_verified": user.get("email_verified", False),
            "timezone": user.get("timezone", "UTC"),
            "language": user.get("language", "en"),
            "created_at": user.get("created_at", ""),
        }
    }


@router.post("/login")
async def login(request: LoginRequest):
    """Login user."""
    user = UserDB.get_by_email(request.email)
    
    if not user or not verify_password(request.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Generate tokens
    access_token = create_access_token(data={"sub": user["id"], "type": "access"})
    refresh_token = create_refresh_token(data={"sub": user["id"], "type": "refresh"})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 86400,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user.get("full_name", ""),
            "is_active": user.get("is_active", True),
            "email_verified": user.get("email_verified", False),
            "timezone": user.get("timezone", "UTC"),
            "language": user.get("language", "en"),
            "created_at": user.get("created_at", ""),
        }
    }


@router.post("/refresh")
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token."""
    payload = decode_token(request.refresh_token)
    
    if not payload or payload.get('type') != 'refresh':
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user_id = payload.get('sub')
    user = UserDB.get_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Generate new tokens
    access_token = create_access_token(data={"sub": user["id"], "type": "access"})
    new_refresh_token = create_refresh_token(data={"sub": user["id"], "type": "refresh"})
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@router.get("/me")
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user."""
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "name": current_user.get("full_name", ""),
        "is_active": current_user.get("is_active", True),
        "email_verified": current_user.get("email_verified", False),
        "timezone": current_user.get("timezone", "UTC"),
        "language": current_user.get("language", "en"),
        "created_at": current_user.get("created_at", ""),
    }


@router.post("/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logout user (invalidate token on frontend)."""
    return {"message": "Logged out successfully"}


@router.put("/me")
async def update_me(
    data: dict,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update current user profile."""
    allowed = {"full_name", "timezone", "language", "avatar_url"}
    updates = {k: v for k, v in data.items() if k in allowed}
    updated = UserDB.update(current_user["id"], **updates)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return {
        "id": updated["id"],
        "email": updated["email"],
        "name": updated.get("full_name", ""),
        "is_active": updated.get("is_active", True),
        "email_verified": updated.get("email_verified", False),
        "timezone": updated.get("timezone", "UTC"),
        "language": updated.get("language", "en"),
        "created_at": updated.get("created_at", ""),
    }


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Change user password."""
    if not verify_password(request.current_password, current_user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    new_hash = get_password_hash(request.new_password)
    UserDB.update(current_user["id"], password_hash=new_hash)
    return {"message": "Password changed successfully"}


# ── Workspace endpoints (stub - single-user mode) ────────────────────────────

@router.get("/workspaces")
async def list_workspaces(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List workspaces for current user (single default workspace)."""
    return {
        "items": [
            {
                "id": f"ws_{current_user['id'][:8]}",
                "name": "My Workspace",
                "slug": "my-workspace",
                "description": "Default workspace",
                "is_active": True,
                "settings": {},
                "created_at": current_user.get("created_at", ""),
                "updated_at": current_user.get("created_at", ""),
                "member_count": 1,
            }
        ],
        "total": 1,
    }


@router.post("/workspaces")
async def create_workspace(
    data: dict,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create workspace (stub - returns default workspace)."""
    import uuid
    from datetime import datetime
    return {
        "id": str(uuid.uuid4()),
        "name": data.get("name", "New Workspace"),
        "slug": data.get("name", "new-workspace").lower().replace(" ", "-"),
        "description": data.get("description", ""),
        "is_active": True,
        "settings": {},
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "member_count": 1,
    }


@router.get("/workspaces/{workspace_id}")
async def get_workspace(
    workspace_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get workspace by ID."""
    return {
        "id": workspace_id,
        "name": "My Workspace",
        "slug": "my-workspace",
        "description": "Default workspace",
        "is_active": True,
        "settings": {},
        "created_at": current_user.get("created_at", ""),
        "updated_at": current_user.get("created_at", ""),
        "member_count": 1,
    }


def log_audit(action: str, user_id: str, details: str = ""):
    """Log audit event."""
    pass
