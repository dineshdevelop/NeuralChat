from fastapi import APIRouter, Depends, HTTPException, status
from typing import Any
import time

from app.schemas.auth import TokenRequest, TokenResponse, CurrentUserResponse
from app.auth.models import CurrentUser
from app.api.deps import CurrentUserDep

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(req: TokenRequest) -> Any:
    """
    Mock login endpoint.
    In production, this would call AWS Cognito InitiateAuth.
    For local dev, we return a mock token that our cognito.py validator recognizes.
    """
    # Simple validation (mock)
    if not req.username or not req.password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
        
    # Generate a mock token
    # The mock validator in cognito.py looks for "mock_token_{email}"
    mock_token = f"mock_token_{req.username}"
    
    return TokenResponse(
        access_token=mock_token,
        refresh_token=f"mock_refresh_{req.username}",
        token_type="bearer",
        expires_in=3600
    )

@router.get("/me", response_model=CurrentUserResponse)
async def read_users_me(current_user: CurrentUser = CurrentUserDep) -> Any:
    """
    Get current user information based on the provided token or API key.
    """
    return CurrentUserResponse(
        sub=current_user.user_id,
        email=current_user.email,
        role=current_user.role,
        auth_method=current_user.auth_method
    )
