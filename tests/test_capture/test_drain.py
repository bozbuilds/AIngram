from unittest.mock import MagicMock

from aingram.capture.config import CaptureConfig
from aingram.capture.drain import CaptureDrain
from aingram.capture.queue import CaptureQueue
from aingram.capture.types import CaptureRecord
from tests.conftest import MockEmbedder


def _make_record(**overrides):
    defaults = dict(
        source_tool='claude_code',
        session_id='sess-1',
        user_prompt='What is Python?',
        assistant_response='A programming language.',
        timestamp=1000.0,
        container_tag='aingram:unified',
    )
    defaults.update(overrides)
    return CaptureRecord(**defaults)


class TestCaptureDrain:
    def test_drains_record_to_memory_store(self, tmp_path):
        queue_db = str(tmp_path / 'queue.db')
        memory_db = str(tmp_path / 'memory.db')
        config = CaptureConfig(poll_interval=0.1, drain_batch_size=10)

        queue = CaptureQueue(queue_db)
        queue.insert(_make_record())

        drain = CaptureDrain(
            queue=queue,
            memory_db_path=memory_db,
            config=config,
            embedder=MockEmbedder(),
        )
        processed = drain.process_batch()
        assert processed == 1
        assert queue.pending_count() == 0
        drain.close()

    def test_marks_error_on_failure(self, tmp_path):
        queue_db = str(tmp_path / 'queue.db')
        config = CaptureConfig(poll_interval=0.1)

        queue = CaptureQueue(queue_db)
        queue.insert(_make_record())

        drain = CaptureDrain(
            queue=queue,
            memory_db_path='',
            config=config,
            embedder=MockEmbedder(),
        )
        drain.process_batch()
        drain.close()

    def test_empty_queue_returns_zero(self, tmp_path):
        queue_db = str(tmp_path / 'queue.db')
        memory_db = str(tmp_path / 'memory.db')
        config = CaptureConfig()

        queue = CaptureQueue(queue_db)
        drain = CaptureDrain(
            queue=queue,
            memory_db_path=memory_db,
            config=config,
            embedder=MockEmbedder(),
        )
        assert drain.process_batch() == 0
        drain.close()

    def test_formats_content_as_plain_text(self, tmp_path):
        queue_db = str(tmp_path / 'queue.db')
        memory_db = str(tmp_path / 'memory.db')
        config = CaptureConfig()

        queue = CaptureQueue(queue_db)
        queue.insert(
            _make_record(
                user_prompt='question here',
                assistant_response='answer here',
            )
        )

        drain = CaptureDrain(
            queue=queue,
            memory_db_path=memory_db,
            config=config,
            embedder=MockEmbedder(),
        )
        drain.process_batch()

        from aingram.store import MemoryStore

        store = MemoryStore(memory_db, embedder=MockEmbedder())
        results = store.recall('question', limit=1, verify=False)
        assert len(results) >= 1
        store.close()
        drain.close()

    def test_start_and_stop_thread(self, tmp_path):
        queue_db = str(tmp_path / 'queue.db')
        memory_db = str(tmp_path / 'memory.db')
        config = CaptureConfig(poll_interval=0.1)

        queue = CaptureQueue(queue_db)
        drain = CaptureDrain(
            queue=queue,
            memory_db_path=memory_db,
            config=config,
            embedder=MockEmbedder(),
        )
        drain.start()
        assert drain.is_running
        drain.stop(timeout=2.0)
        assert not drain.is_running
        drain.close()


class TestAutoConsolidation:
    def _make_drain(self, tmp_path, interval: int, store=None):
        queue_db = str(tmp_path / 'queue.db')
        config = CaptureConfig(
            poll_interval=0.1,
            drain_batch_size=10,
            consolidation_interval_records=interval,
        )
        queue = CaptureQueue(queue_db)
        drain = CaptureDrain(
            queue=queue,
            config=config,
            store=store,
            memory_db_path='',
            embedder=MockEmbedder(),
        )
        return drain, queue

    def test_consolidation_not_called_before_threshold(self, tmp_path):
        mock_store = MagicMock()
        drain, queue = self._make_drain(tmp_path, interval=5, store=mock_store)

        for i in range(4):
            queue.insert(_make_record(user_prompt=f'unique prompt {i}', timestamp=1000.0 + i))

        drain.process_batch()
        mock_store.consolidate.assert_not_called()
        drain.close()

    def test_consolidation_called_at_threshold(self, tmp_path):
        mock_store = MagicMock()
        drain, queue = self._make_drain(tmp_path, interval=3, store=mock_store)

        for i in range(3):
            queue.insert(_make_record(user_prompt=f'unique prompt {i}', timestamp=1000.0 + i))

        drain.process_batch()
        mock_store.consolidate.assert_called_once()
        drain.close()

    def test_counter_resets_after_successful_consolidation(self, tmp_path):
        mock_store = MagicMock()
        drain, queue = self._make_drain(tmp_path, interval=2, store=mock_store)

        for i in range(2):
            queue.insert(_make_record(user_prompt=f'unique prompt {i}', timestamp=1000.0 + i))
        drain.process_batch()
        assert drain._records_since_consolidation == 0

        queue.insert(_make_record(user_prompt='unique prompt 2', timestamp=1002.0))
        drain.process_batch()
        assert drain._records_since_consolidation == 1
        assert mock_store.consolidate.call_count == 1
        drain.close()

    def test_counter_not_reset_on_consolidation_failure(self, tmp_path):
        mock_store = MagicMock()
        mock_store.consolidate.side_effect = RuntimeError('consolidation error')
        drain, queue = self._make_drain(tmp_path, interval=2, store=mock_store)

        for i in range(2):
            queue.insert(_make_record(user_prompt=f'unique prompt {i}', timestamp=1000.0 + i))
        drain.process_batch()

        assert drain._records_since_consolidation == 2
        drain.close()

    def test_errored_records_do_not_increment_counter(self, tmp_path):
        mock_store = MagicMock()
        mock_store.remember.side_effect = RuntimeError('remember failed')
        drain, queue = self._make_drain(tmp_path, interval=2, store=mock_store)

        for i in range(3):
            queue.insert(_make_record(user_prompt=f'unique prompt {i}', timestamp=1000.0 + i))
        drain.process_batch()

        assert drain._records_since_consolidation == 0
        mock_store.consolidate.assert_not_called()
        drain.close()

    def test_zero_interval_never_consolidates(self, tmp_path):
        mock_store = MagicMock()
        drain, queue = self._make_drain(tmp_path, interval=0, store=mock_store)

        for i in range(10):
            queue.insert(_make_record(user_prompt=f'unique prompt {i}', timestamp=1000.0 + i))
        drain.process_batch()
        mock_store.consolidate.assert_not_called()
        drain.close()
