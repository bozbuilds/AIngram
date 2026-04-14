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
