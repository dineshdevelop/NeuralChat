from typing import Optional
from app.auth.models import Role, CurrentUser

def verify_cognito_jwt(token: str) -> Optional[CurrentUser]:
    """
    Validates an AWS Cognito JWT.
    
    In production, this would:
    1. Fetch JWKS from Cognito
    2. Verify the JWT signature
    3. Verify claims (exp, iss, aud)
    4. Extract `sub`, `email`, and `cognito:groups`
    
    For local development/mock mode, we just check for a valid mock token.
    """
    if not token:
        return None
        
    # Mock token validation
    if token.startswith("mock_token_"):
        email = token.replace("mock_token_", "")
        role = Role.admin if "admin" in email else Role.user
        
        return CurrentUser(
            user_id=email,
            email=email,
            role=role,
            auth_method="cognito_jwt"
        )
        
    # Legacy demo token fallback (prevents session breakage if user didn't logout)
    if token == "demo_jwt_token_12345":
        return CurrentUser(
            user_id="admin@neuralchat.local",
            email="admin@neuralchat.local",
            role=Role.admin,
            auth_method="cognito_jwt"
        )
        
    return None
