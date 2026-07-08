# =============================================================================
# app/schemas/auth.py — Authentication Request / Response Models
# =============================================================================
#
# 🧠 LEARNING NOTE — Authentication Schemas:
#
# These models represent:
#   1. TokenRequest   → credentials sent to obtain a JWT token
#   2. TokenResponse  → JWT token returned after successful auth
#   3. APIKeyCreateRequest → admin creates a new API key
#   4. APIKeyResponse → the generated API key info (shown ONCE at creation)
#   5. CurrentUserResponse → /auth/me endpoint response
#
# Security note:
#   The raw API key is only returned ONCE at creation time.
#   After that, only the hashed key is stored. The user must save it immediately.
# =============================================================================

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    """
    Request body for POST /auth/token.

    The client exchanges Cognito credentials (username + password) for a JWT.
    This proxies the Cognito InitiateAuth flow.
    """

    username: str = Field(
        ...,
        min_length=1,
        description="Cognito username (usually the user's email address).",
        examples=["user@example.com"],
    )

    password: str = Field(
        ...,
        min_length=8,
        description="Cognito account password.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "username": "user@example.com",
                "password": "SecurePassword123!",
            }
        }


class TokenResponse(BaseModel):
    """
    Response for POST /auth/token.

    Returns the JWT tokens from Cognito. The client should:
      1. Store the access_token (use in Authorization: Bearer header)
      2. Use refresh_token to get a new access_token before it expires
    """

    access_token: str = Field(
        ...,
        description="JWT access token. Include in Authorization: Bearer <token> header.",
    )

    refresh_token: Optional[str] = Field(
        default=None,
        description="Refresh token to get a new access token without re-authenticating.",
    )

    token_type: str = Field(
        default="bearer",
        description="Token type — always 'bearer' for Cognito JWTs.",
    )

    expires_in: int = Field(
        ...,
        description="Token expiry in seconds (typically 3600 = 1 hour).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJSUzI1NiIsInR...",
                "refresh_token": "eyJjdHkiOiJKV1QiLCJlbmMiOi...",
                "token_type": "bearer",
                "expires_in": 3600,
            }
        }


class APIKeyCreateRequest(BaseModel):
    """
    Request body for POST /auth/api-key.

    Admin creates a named API key with a specific role.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Human-readable name for this API key (e.g., 'CI Pipeline Key').",
        examples=["CI Pipeline Key", "Internal Service Bot"],
    )

    role: Literal["admin", "user"] = Field(
        default="user",
        description="Access role granted to this API key.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "CI Pipeline Key",
                "role": "user",
            }
        }


class APIKeyResponse(BaseModel):
    """
    Response for POST /auth/api-key.

    ⚠️ IMPORTANT: The raw `key` field is only returned ONCE at creation.
    Store it immediately — it cannot be retrieved again.
    Only the hashed version is stored server-side.
    """

    key_id: str = Field(
        ...,
        description="Unique identifier for this API key (safe to store and display).",
    )

    key: Optional[str] = Field(
        default=None,
        description=(
            "The raw API key value. ⚠️ Only shown once at creation — save it now!"
        ),
    )

    name: str = Field(..., description="Human-readable name for this key.")

    role: str = Field(..., description="Role assigned to this key.")

    created_at: datetime = Field(..., description="When this key was created.")

    is_active: bool = Field(
        default=True,
        description="Whether this key is currently active.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "key_id": "key_abc123",
                "key": "rag_sk.AbCdEfGhIjKlMnOpQrStUvWxYz1234567890",
                "name": "CI Pipeline Key",
                "role": "user",
                "created_at": "2024-01-15T10:30:00Z",
                "is_active": True,
            }
        }


class CurrentUserResponse(BaseModel):
    """
    Response for GET /auth/me.

    Returns info about the currently authenticated user.
    Works for both API key auth and Cognito JWT auth.
    """

    sub: str = Field(
        ...,
        description="Unique identifier for the user (Cognito sub or key_id).",
    )

    email: Optional[str] = Field(
        default=None,
        description="User's email address (from Cognito JWT, not available for API keys).",
    )

    role: str = Field(
        ...,
        description="User's access role (admin or user).",
    )

    auth_method: Literal["api_key", "cognito_jwt"] = Field(
        ...,
        description="How this user was authenticated.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sub": "cognito-user-uuid-1234",
                "email": "user@example.com",
                "role": "user",
                "auth_method": "cognito_jwt",
            }
        }
