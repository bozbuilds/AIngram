# aingram/worker.py
from __future__ import annotations

import json
import logging
import threading

from aingram.graph.builder import GraphBuilder
from aingram.processing.protocols import EntityExtractor, LLMProcessor
from aingram.storage.engine import StorageEngine
from aingram.types import ExtractedEntity, ExtractionResult

logger = logging.getLogger(__name__)

RELATIONSHIP_SYSTEM_PROMPT = (
    'You are a knowledge graph builder. Extract relationships between entities from text. '
    'Output ONLY a valid JSON array. Each element must have keys: source, target, relation, fact. '
    'If no relationships exist, output: []'
)
RELATIONSHIP_USER_PROMPT = 'Entities: {entities}\n\nText: "{content}"\n\nExtract relationships:'


class BackgroundWorker:
    def __init__(
        self,
        engine: StorageEngine | None = None,
        *,
        db_path: str | None = None,
        extractor: EntityExtractor,
        llm: LLMProcessor | None = None,
        entity_types: list[str] | None = None,
        poll_interval: float = 0.1,
        training_logger=None,
    ) -> None:
        if engine is not None:
            self._engine = engine
            self._owns_engine = False
        elif db_path is not None:
            self._engine = StorageEngine(db_path)
            self._owns_engine = True
        else:
            raise ValueError('Either engine or db_path must be provided')

        self._extractor = extractor
        self._llm = llm
        self._training_logger = training_logger
        self._entity_types = entity_types or [
            'person',
            'organization',
            'location',
            'project',
            'technology',
        ]
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._builder = GraphBuilder(self._engine)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._owns_engine:
            self._engine.close()

    def process_one(self) -> bool:
        task = self._engine.dequeue_task()
        if task is None:
            return False
        task_id, task_type, payload = task
        try:
            if task_type == 'extract_entities_v3':
                self._handle_extract_entities_v3(payload)
            elif task_type == 'link_graph_v3':
                self._handle_link_graph_v3(payload)
            else:
                logger.warning('Unknown task type: %s', task_type)
            self._engine.complete_task(task_id)
        except Exception as e:
            logger.error('Task %s failed: %s', task_id, e, exc_info=True)
            self._engine.fail_task(task_id, str(e))
        return True

    def _run(self) -> None:
        import time

        while not self._stop_event.is_set():
            if not self.process_one():
                time.sleep(self._poll_interval)

    def _handle_extract_entities_v3(self, payload: dict) -> None:
        entry_id = payload['entry_id']
        entry = self._engine.get_entry(entry_id)
        if entry is None:
            logger.warning('Entry %s not found for extraction', entry_id)
            return

        # Parse content text for extraction
        try:
            content_dict = json.loads(entry.content)
            text = content_dict.get('text', entry.content)
        except (json.JSONDecodeError, TypeError):
            text = entry.content

        entities = self._extractor.extract(text, self._entity_types)
        entity_names = []
        for extracted in entities:
            self._builder.upsert_entity(
                extracted.name,
                extracted.entity_type,
                source_entry=entry_id,
            )
            entity_names.append(extracted.name)

        # Log training pair if training logger is configured
        if self._training_logger is not None and entities:
            result = ExtractionResult(
                entry_type=str(entry.entry_type),
                confidence=entry.confidence or 0.5,
                relevance=entry.importance,
                entities=[
                    ExtractedEntity(name=e.name, entity_type=e.entity_type, score=e.score)
                    for e in entities
                ],
            )
            self._training_logger.log(text, result)

        # Enqueue graph linking if we found entities and have LLM
        if len(entity_names) >= 2 and self._llm is not None:
            self._engine.enqueue_task(
                task_type='link_graph_v3',
                payload={'entry_id': entry_id, 'entity_names': entity_names},
            )

    def _handle_link_graph_v3(self, payload: dict) -> None:
        entry_id = payload['entry_id']
        entity_names = payload.get('entity_names', [])
        entry = self._engine.get_entry(entry_id)
        if entry is None or not entity_names:
            return

        try:
            content_dict = json.loads(entry.content)
            text = content_dict.get('text', entry.content)
        except (json.JSONDecodeError, TypeError):
            text = entry.content

        prompt = RELATIONSHIP_USER_PROMPT.format(
            entities=', '.join(entity_names),
            content=text,
        )
        try:
            raw = self._llm.complete(prompt, system=RELATIONSHIP_SYSTEM_PROMPT)
        except Exception:
            logger.warning('LLM relationship extraction failed', exc_info=True)
            return

        try:
            relationships = json.loads(raw)
        except json.JSONDecodeError:
            return

        if not isinstance(relationships, list):
            return

        for rel in relationships:
            source_name = rel.get('source')
            target_name = rel.get('target')
            if not source_name or not target_name:
                continue
            found_sources = self._engine.find_entities_by_name(source_name)
            found_targets = self._engine.find_entities_by_name(target_name)
            source_entity = found_sources[0] if found_sources else None
            target_entity = found_targets[0] if found_targets else None

            if source_entity and target_entity:
                self._builder.add_relationship(
                    source_entity.entity_id,
                    target_entity.entity_id,
                    rel.get('relation', 'related'),
                    fact=rel.get('fact'),
                    source_entry=entry_id,
                )
