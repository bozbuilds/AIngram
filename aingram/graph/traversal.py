# aingram/graph/traversal.py
from __future__ import annotations

import logging
import re

from aingram.storage.engine import StorageEngine
from aingram.types import Entity

logger = logging.getLogger(__name__)


class GraphTraversal:
    def __init__(self, engine: StorageEngine) -> None:
        self._engine = engine

    def detect_entities(self, text: str) -> list[Entity]:
        """Find known entities whose names appear in the text."""
        entity_scan_limit = 1000
        entities = self._engine.get_entities(limit=entity_scan_limit)
        if len(entities) >= entity_scan_limit:
            logger.warning(
                'Entity scan capped at %d — some entities may be invisible to graph search',
                entity_scan_limit,
            )
        matches = [
            e for e in entities if re.search(r'\b' + re.escape(e.name) + r'\b', text, re.IGNORECASE)
        ]
        matches.sort(key=lambda e: len(e.name), reverse=True)
        return matches

    def search(self, query: str, *, limit: int = 50) -> list[str]:
        """Find entry IDs related to entities detected in query text.

        Returns entry IDs ranked by entity link count (most linked first).
        """
        entities = self.detect_entities(query)
        if not entities:
            return []

        entity_ids = [e.entity_id for e in entities]
        traversed = self._engine.traverse_graph(entity_ids, max_hops=2)
        all_entity_ids = set(entity_ids) | {eid for eid, _hop in traversed}

        return self.get_ranked_entry_ids(list(all_entity_ids), limit=limit)

    def get_ranked_entry_ids(self, entity_ids: list[str], *, limit: int) -> list[str]:
        """Get entry IDs linked to entities, ranked by link count."""
        entity_entries = self._engine.get_entry_ids_for_entities(entity_ids)
        entry_counts: dict[str, int] = {}
        for eids in entity_entries.values():
            for eid in eids:
                entry_counts[eid] = entry_counts.get(eid, 0) + 1

        ranked = sorted(entry_counts, key=lambda eid: entry_counts[eid], reverse=True)
        return ranked[:limit]
