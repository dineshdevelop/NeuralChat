from pydantic import BaseModel, EmailStr
from enum import Enum
from typing import Optional

class Role(str, Enum):
    admin = "admin"
    user = "user"
    anonymous = "anonymous"

class CurrentUser(BaseModel):
    user_id: str
    email: Optional[EmailStr] = None
    role: Role
    auth_method: str  # "api_key" or "cognito"

class APIKeyRecord(BaseModel):
    key_id: str
    hashed_key: str
    role: Role
    name: str
    is_active: bool = True
