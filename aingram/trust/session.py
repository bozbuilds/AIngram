# aingram/trust/session.py
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from aingram.types import AgentSession


class SessionManager:
    """Manages an agent's cryptographic session and chain state.

    Private key lives in memory only — never persisted.
    """

    def __init__(
        self,
        agent_name: str,
        *,
        parent_session_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._private_key = Ed25519PrivateKey.generate()
        self._public_key_hex = self._private_key.public_key().public_bytes_raw().hex()
        self._session_id = uuid.uuid4().hex
        self._agent_name = agent_name
        self._parent_session_id = parent_session_id
        self._metadata = metadata
        self._created_at = datetime.now(UTC).isoformat()
        self._sequence_num = 0
        self._prev_entry_id: str | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def public_key_hex(self) -> str:
        return self._public_key_hex

    def sign(self, entry_id: str) -> str:
        """Sign an entry_id. Private key never leaves this object."""
        from aingram.trust.signing import sign_entry

        return sign_entry(self._private_key, entry_id)

    @property
    def next_sequence_num(self) -> int:
        return self._sequence_num + 1

    @property
    def prev_entry_id(self) -> str | None:
        return self._prev_entry_id

    def advance(self, entry_id: str) -> None:
        """Record a new entry in the chain."""
        self._sequence_num += 1
        self._prev_entry_id = entry_id

    def __repr__(self) -> str:
        return f'SessionManager(session_id={self._session_id!r}, agent={self._agent_name!r})'

    def to_agent_session(self) -> AgentSession:
        """Produce an AgentSession dataclass for storage."""
        return AgentSession(
            session_id=self._session_id,
            agent_name=self._agent_name,
            public_key=self._public_key_hex,
            created_at=self._created_at,
            parent_session_id=self._parent_session_id,
            metadata=self._metadata,
        )
