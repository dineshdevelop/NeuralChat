from fastapi import Request, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from typing import Optional

from app.auth.models import CurrentUser, Role
from app.auth.api_key import verify_api_key
from app.auth.cognito import verify_cognito_jwt

# Security schemes for Swagger UI
security_bearer = HTTPBearer(auto_error=False)
security_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_current_user(
    request: Request,
    bearer: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    api_key: Optional[str] = Security(security_api_key)
) -> CurrentUser:
    """
    Extracts the current user from either an API Key or a Cognito JWT.
    """
    # 1. Try API Key first
    if api_key:
        user = verify_api_key(api_key)
        if user:
            return user
            
    # 2. Try Bearer Token (Cognito JWT)
    if bearer:
        user = verify_cognito_jwt(bearer.credentials)
        if user:
            return user
            
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )

def require_role(allowed_roles: list[Role]):
    """
    Dependency factory to enforce role-based access control.
    """
    def role_checker(user: CurrentUser = Security(get_current_user)):
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required roles: {[r.value for r in allowed_roles]}"
            )
        return user
    return role_checker
