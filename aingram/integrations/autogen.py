# aingram/integrations/autogen.py
from __future__ import annotations

from aingram import MemoryStore


class AIngramAutogenMemory:
    """AutoGen-style memory with add() and query()."""

    def __init__(self, db_path: str = 'agent_memory.db', *, store: MemoryStore | None = None):
        self._store = store or MemoryStore(db_path)

    def add(self, content: str, **kwargs) -> None:
        self._store.remember(content, **kwargs)

    def query(self, query: str, *, limit: int = 5) -> str:
        rows = self._store.recall(query, limit=limit, verify=False)
        return '\n'.join(r.entry.content for r in rows)
