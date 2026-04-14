from __future__ import annotations

import json
import logging
import threading

from aingram.capture.config import CaptureConfig
from aingram.capture.queue import CaptureQueue
from aingram.capture.types import CaptureRecord

logger = logging.getLogger(__name__)


class CaptureDrain:
    def __init__(
        self,
        *,
        queue: CaptureQueue,
        config: CaptureConfig,
        store=None,
        memory_db_path: str = '',
        embedder=None,
    ) -> None:
        self._queue = queue
        self._config = config
        self._embedder = embedder
        if store is not None:
            self._store = store
            self._owns_store = False
        else:
            self._store = None
            self._owns_store = True
        self._memory_db_path = memory_db_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._records_since_consolidation: int = 0

    def _get_store(self):
        if self._store is None:
            from aingram.store import MemoryStore

            self._store = MemoryStore(
                self._memory_db_path,
                agent_name='capture-daemon',
                embedder=self._embedder,
            )
        return self._store

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def close(self) -> None:
        self.stop()
        if self._store is not None and self._owns_store:
            self._store.close()
            self._store = None

    def _run(self) -> None:
        while not self._stop.is_set():
            count = self.process_batch()
            if count == 0:
                self._stop.wait(self._config.poll_interval)

    def process_batch(self) -> int:
        batch = self._queue.dequeue_batch(self._config.drain_batch_size)
        if not batch:
            return 0

        for row_id, record in batch:
            try:
                store = self._get_store()
                content = self._format_for_remember(record)
                metadata = self._build_metadata(record)
                tags = ['captured', record.source_tool]
                store.remember(content, metadata=metadata, tags=tags)
                self._queue.mark_done(row_id)
                self._records_since_consolidation += 1
            except Exception as e:
                logger.error('Failed to drain record %s: %s', row_id, e, exc_info=True)
                self._queue.mark_error(row_id, str(e))

        if (
            self._config.consolidation_interval_records > 0
            and self._records_since_consolidation >= self._config.consolidation_interval_records
        ):
            try:
                store = self._get_store()
                store.consolidate()
                self._records_since_consolidation = 0
            except Exception as e:
                logger.error('Auto-consolidation failed: %s', e, exc_info=True)

        return len(batch)

    @staticmethod
    def _format_for_remember(record: CaptureRecord) -> str:
        parts = []
        if record.user_prompt:
            parts.append(record.user_prompt)
        if record.assistant_response:
            parts.append(record.assistant_response)
        return '\n\n---\n\n'.join(parts)

    @staticmethod
    def _build_metadata(record: CaptureRecord) -> dict:
        meta = {
            'source_tool': record.source_tool,
            'capture_session_id': record.session_id,
        }
        if record.container_tag:
            meta['container_tag'] = record.container_tag
        if record.model:
            meta['model'] = record.model
        if record.project_path:
            meta['project_path'] = record.project_path
        if record.metadata:
            try:
                extra = json.loads(record.metadata)
                if isinstance(extra, dict):
                    meta.update(extra)
            except json.JSONDecodeError:
                pass
        return meta
