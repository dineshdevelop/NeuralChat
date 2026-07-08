import secrets
from typing import Optional
from app.auth.models import Role, CurrentUser

# For this mock phase, we just store a valid hardcoded key for the admin
MOCK_API_KEYS = {
    "rag_sk.AbCdEfGhIjKlMnOp": CurrentUser(
        user_id="api_admin",
        role=Role.admin,
        auth_method="api_key"
    )
}

def verify_api_key(api_key: str) -> Optional[CurrentUser]:
    """
    Validates an API key and returns the associated user if valid.
    In production, this would hash the key and check against a DB.
    """
    if not api_key:
        return None
        
    return MOCK_API_KEYS.get(api_key)

def generate_api_key() -> str:
    """Generate a new secure API key"""
    return f"rag_sk.{secrets.token_urlsafe(32)}"
