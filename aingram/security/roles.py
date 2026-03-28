"""RBAC role definitions and permission enforcement."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from aingram.exceptions import AuthorizationError

if TYPE_CHECKING:
    from aingram.security.auth import CallerContext


class Role(StrEnum):
    READER = 'reader'
    CONTRIBUTOR = 'contributor'
    ADMIN = 'admin'


PERMISSIONS: dict[str, set[Role]] = {
    'remember': {Role.CONTRIBUTOR, Role.ADMIN},
    'recall': {Role.READER, Role.CONTRIBUTOR, Role.ADMIN},
    'reference': {Role.CONTRIBUTOR, Role.ADMIN},
    'get_related': {Role.CONTRIBUTOR, Role.ADMIN},
    'verify': {Role.READER, Role.CONTRIBUTOR, Role.ADMIN},
    'get_experiment_context': {Role.READER, Role.CONTRIBUTOR, Role.ADMIN},
    'consolidate': {Role.ADMIN},
    'sync': {Role.ADMIN},
    'review_memory': {Role.CONTRIBUTOR, Role.ADMIN},
    'get_due_reviews': {Role.READER, Role.CONTRIBUTOR, Role.ADMIN},
    'get_surprise': {Role.READER, Role.CONTRIBUTOR, Role.ADMIN},
}


class RoleAuthorizer:
    """Check caller role against tool permission map."""

    def check(self, caller: CallerContext, tool_name: str) -> None:
        allowed = PERMISSIONS.get(tool_name)
        if allowed is None:
            return
        if caller.role not in allowed:
            raise AuthorizationError(f'Role {caller.role!r} cannot access {tool_name!r}')
