# aingram/integrations/crewai.py
from __future__ import annotations

from aingram import MemoryStore


class AIngramCrewMemory:
    """Crew-style memory facade over MemoryStore (save / search)."""

    def __init__(self, db_path: str = 'agent_memory.db', *, store: MemoryStore | None = None):
        self._store = store or MemoryStore(db_path)

    def save(self, text: str, **kwargs) -> None:
        self._store.remember(text, **kwargs)

    def search(self, query: str, limit: int = 10):
        return self._store.recall(query, limit=limit, verify=False)

    def reset(self) -> None:
        """No-op: AIngram is append-only."""
        pass
