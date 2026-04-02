# aingram/integrations/smolagents.py
from __future__ import annotations

from aingram import MemoryStore


class AIngramSmolagentsMemory:
    """Smolagents-style write_memory / retrieve_memory."""

    def __init__(self, db_path: str = 'agent_memory.db', *, store: MemoryStore | None = None):
        self._store = store or MemoryStore(db_path)

    def write_memory(self, text: str, **kwargs) -> None:
        self._store.remember(text, **kwargs)

    def retrieve_memory(self, query: str, *, limit: int = 5):
        return self._store.recall(query, limit=limit, verify=False)
