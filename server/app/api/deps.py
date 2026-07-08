from fastapi import Depends
from app.auth.models import CurrentUser, Role
from app.auth.rbac import get_current_user, require_role

# Alias for standard dependency
CurrentUserDep = Depends(get_current_user)

# Role-specific dependencies
require_admin = Depends(require_role([Role.admin]))
require_user_or_admin = Depends(require_role([Role.user, Role.admin]))
