# aingram/graph/builder.py
from __future__ import annotations

import logging

from aingram.storage.engine import StorageEngine
from aingram.types import Entity, Relationship

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self, engine: StorageEngine) -> None:
        self._engine = engine

    def upsert_entity(
        self,
        name: str,
        entity_type: str,
        *,
        source_entry: str | None = None,
    ) -> str:
        """Insert or update entity. Optionally link to source entry."""
        entity_id = self._engine.upsert_entity(name=name, entity_type=entity_type)
        if source_entry is not None:
            self._engine.link_entity_to_mention(entity_id, source_entry)
        logger.debug('Upserted entity %s (%s) -> %s', name, entity_type, entity_id)
        return entity_id

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        *,
        fact: str | None = None,
        source_entry: str | None = None,
    ) -> str:
        return self._engine.insert_relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            fact=fact,
            source_memory=source_entry,
        )

    def get_entities(self, *, limit: int = 100) -> list[Entity]:
        return self._engine.get_entities(limit=limit)

    def get_entity_relationships(self, entity_id: str) -> list[Relationship]:
        return self._engine.get_relationships_for_entity(entity_id)
