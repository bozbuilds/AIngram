"""SecurityMiddleware — composes auth, RBAC, bounds, and rate limiting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aingram.security.auth import CallerContext, TokenAuthenticator
from aingram.security.bounds import InputBoundsChecker
from aingram.security.rate_limit import RateLimiter
from aingram.security.roles import Role, RoleAuthorizer

if TYPE_CHECKING:
    from aingram.storage.engine import StorageEngine


class SecurityMiddleware:
    """Compose all security concerns into a single pipeline."""

    def __init__(self, engine: StorageEngine) -> None:
        self._authenticator = TokenAuthenticator(engine)
        self._authorizer = RoleAuthorizer()
        self._bounds = InputBoundsChecker()
        self._rate_limiter = RateLimiter()

    def process(self, tool_name: str, token: str, params: dict) -> CallerContext:
        ctx = self._authenticator.authenticate(token)
        self._authorizer.check(ctx, tool_name)
        self._bounds.validate(tool_name, params)
        self._rate_limiter.check(ctx.session_id, tool_name, is_admin=(ctx.role == Role.ADMIN))
        return ctx
