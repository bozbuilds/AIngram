import threading

from aingram.capture.queue import CaptureQueue
from aingram.capture.types import CaptureRecord


def _make_record(**overrides):
    defaults = dict(
        source_tool='claude_code',
        session_id='sess-1',
        user_prompt='hello',
        timestamp=1000.0,
    )
    defaults.update(overrides)
    return CaptureRecord(**defaults)


class TestCaptureQueueInsertAndDequeue:
    def test_insert_and_dequeue(self, tmp_queue_db):
        q = CaptureQueue(tmp_queue_db)
        q.insert(_make_record())
        batch = q.dequeue_batch(10)
        assert len(batch) == 1
        _, record = batch[0]
        assert record.source_tool == 'claude_code'
        assert record.user_prompt == 'hello'
        q.close()

    def test_dequeue_empty_returns_empty(self, tmp_queue_db):
        q = CaptureQueue(tmp_queue_db)
        assert q.dequeue_batch(10) == []
        q.close()

    def test_dequeue_marks_processing(self, tmp_queue_db):
        q = CaptureQueue(tmp_queue_db)
        q.insert(_make_record())
        batch = q.dequeue_batch(10)
        assert len(batch) == 1
        assert q.dequeue_batch(10) == []
        q.close()

    def test_dequeue_respects_limit(self, tmp_queue_db):
        q = CaptureQueue(tmp_queue_db)
        for i in range(5):
            q.insert(_make_record(session_id=f'sess-{i}'))
        batch = q.dequeue_batch(3)
        assert len(batch) == 3
        q.close()

    def test_fifo_ordering(self, tmp_queue_db):
        q = CaptureQueue(tmp_queue_db)
        q.insert(_make_record(user_prompt='first'))
        q.insert(_make_record(user_prompt='second'))
        batch = q.dequeue_batch(10)
        assert batch[0][1].user_prompt == 'first'
        assert batch[1][1].user_prompt == 'second'
        q.close()


class TestCaptureQueueStateTransitions:
    def test_mark_done(self, tmp_queue_db):
        q = CaptureQueue(tmp_queue_db)
        q.insert(_make_record())
        batch = q.dequeue_batch(1)
        row_id = batch[0][0]
        q.mark_done(row_id)
        assert q.pending_count() == 0
        q.close()

    def test_mark_error(self, tmp_queue_db):
        q = CaptureQueue(tmp_queue_db)
        q.insert(_make_record())
        batch = q.dequeue_batch(1)
        row_id = batch[0][0]
        q.mark_error(row_id, 'something broke')
        assert q.pending_count() == 0
        q.close()


class TestCaptureQueueToggles:
    def test_default_toggle_on(self, tmp_queue_db):
        q = CaptureQueue(tmp_queue_db)
        q.init_toggles({'claude_code': True, 'chatgpt': False})
        assert q.get_toggle('claude_code') == 'on'
        assert q.get_toggle('chatgpt') == 'off'
        q.close()

    def test_set_toggle(self, tmp_queue_db):
        q = CaptureQueue(tmp_queue_db)
        q.init_toggles({'cursor': True})
        q.set_toggle('cursor', 'off')
        assert q.get_toggle('cursor') == 'off'
        q.close()

    def test_unknown_tool_returns_off(self, tmp_queue_db):
        q = CaptureQueue(tmp_queue_db)
        assert q.get_toggle('nonexistent') == 'off'
        q.close()


class TestCaptureQueueThreadSafety:
    def test_concurrent_inserts(self, tmp_queue_db):
        q = CaptureQueue(tmp_queue_db)
        errors = []

        def insert_records(n):
            try:
                for i in range(n):
                    q.insert(
                        _make_record(session_id=f'thread-{threading.current_thread().name}-{i}')
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=insert_records, args=(20,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert q.pending_count() == 80
        q.close()
