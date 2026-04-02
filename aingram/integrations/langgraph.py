# aingram/integrations/langgraph.py
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from aingram import MemoryStore


@dataclass
class _StoreItem:
    key: str
    value: dict
    score: float | None = None


class AIngramLangGraphStore:
    """LangGraph-style put / get / search over MemoryStore."""

    def __init__(self, db_path: str = 'agent_memory.db', *, store: MemoryStore | None = None):
        self._store = store or MemoryStore(db_path)

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict,
        *,
        index: Sequence[str] | None = None,
    ) -> None:
        del index  # optional embedding index — not used for sqlite backend
        # Embed only the value so semantic search isn't dominated by namespace/key boilerplate.
        # The full KV routing info is stored in metadata for exact lookup via get().
        embed_text = json.dumps(value, sort_keys=True) if value else key
        kv_meta = {'aingram_kv': True, 'namespace': list(namespace), 'key': key, 'value': value}
        self._store.remember(embed_text, entry_type='meta', metadata=kv_meta)

    def get(self, namespace: tuple[str, ...], key: str) -> _StoreItem | None:
        ns = list(namespace)
        with self._store._engine._lock:
            rows = self._store._engine._conn.execute(
                "SELECT metadata FROM memory_entries WHERE entry_type = 'meta' "
                'ORDER BY rowid DESC LIMIT 500'
            ).fetchall()
        for (meta_json,) in rows:
            if not meta_json:
                continue
            try:
                d = json.loads(meta_json)
            except json.JSONDecodeError:
                continue
            if (
                d.get('aingram_kv')
                and d.get('namespace') == ns
                and d.get('key') == key
            ):
                return _StoreItem(key=key, value=d.get('value') or {})
        return None

    def search(
        self,
        namespace: tuple[str, ...],
        *,
        query: str | None = None,
        filter: dict | None = None,
        limit: int = 10,
    ):
        """Semantic search across all stored KV items.

        Note: namespace scoping and filter predicates are not enforced — results
        are drawn from the full database. This is a known limitation of the
        single-file SQLite backend.
        """
        del namespace, filter  # namespace/filter not enforced on this backend
        q = (query or ' ').strip() or ' '
        results = self._store.recall(q, limit=limit, verify=False)
        return [
            _StoreItem(
                key=r.entry.entry_id,
                value={'content': r.entry.content, 'score': r.score},
                score=r.score,
            )
            for r in results
        ]
