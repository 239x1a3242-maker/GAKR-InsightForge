"""User model."""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class User(BaseModel):
    """User model."""
    id: str
    email: str
    full_name: str
    role: str = "analyst"  # admin, analyst, viewer
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
