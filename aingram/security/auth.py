"""Token authentication and CallerContext."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aingram.exceptions import AuthenticationError
from aingram.security.roles import Role

if TYPE_CHECKING:
    from aingram.storage.engine import StorageEngine


@dataclass(frozen=True)
class CallerContext:
    """Authenticated caller identity, threaded through security-sensitive methods."""

    agent_id: str
    session_id: str
    role: Role
    ed25519_verified: bool = False


class TokenAuthenticator:
    """Verify bearer tokens against agent_tokens table."""

    def __init__(self, engine: StorageEngine) -> None:
        self._engine = engine

    def authenticate(self, token: str) -> CallerContext:
        agent = self._engine.verify_agent_token(token)
        if agent is None:
            raise AuthenticationError('Invalid or revoked token')
        return CallerContext(
            agent_id=agent['agent_id'],
            session_id=agent['agent_id'],
            role=Role(agent['role']),
        )
