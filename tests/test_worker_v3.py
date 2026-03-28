"""Background worker v3 tests."""

import time

import pytest

from aingram.storage.engine import StorageEngine
from aingram.types import AgentSession
from aingram.worker import BackgroundWorker
from tests.conftest import MockExtractor, MockLLM


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / 'test.db'
    eng = StorageEngine(str(db))
    session = AgentSession(
        session_id='s1',
        agent_name='test',
        public_key='a' * 64,
        created_at='2026-01-01T00:00:00+00:00',
    )
    eng.store_session(session)
    eng.store_entry(
        entry_id='e1',
        content_hash='ch1',
        entry_type='observation',
        content='{"text":"Alice met with Bob at Acme Corp"}',
        session_id='s1',
        sequence_num=1,
        prev_entry_id=None,
        signature='sig1',
        created_at='2026-01-01T00:00:00+00:00',
        embedding=[0.1] * 768,
    )
    yield eng
    eng.close()


class TestWorkerV3:
    def test_extract_entities_v3(self, engine):
        worker = BackgroundWorker(
            engine=engine,
            extractor=MockExtractor(),
            llm=None,
        )
        engine.enqueue_task(
            task_type='extract_entities_v3',
            payload={'entry_id': 'e1'},
        )
        processed = worker.process_one()
        assert processed is True

        # Check entities were created and linked
        entity_ids = engine.get_entity_ids_for_entry('e1')
        assert len(entity_ids) >= 1

    def test_link_graph_v3(self, engine):
        # First create entities
        eid1 = engine.upsert_entity(name='Alice', entity_type='person')
        eid2 = engine.upsert_entity(name='Acme', entity_type='organization')
        engine.link_entity_to_mention(eid1, 'e1')
        engine.link_entity_to_mention(eid2, 'e1')

        rel_json = (
            '[{"source":"Alice","target":"Acme","relation":"works_at",'
            '"fact":"Alice works at Acme"}]'
        )
        worker = BackgroundWorker(
            engine=engine,
            extractor=MockExtractor(),
            llm=MockLLM(rel_json),
        )
        engine.enqueue_task(
            task_type='link_graph_v3',
            payload={'entry_id': 'e1', 'entity_names': ['Alice', 'Acme']},
        )
        processed = worker.process_one()
        assert processed is True

        # Check relationship was created
        rels = engine.get_relationships_for_entity(eid1)
        assert len(rels) >= 1


class TestWorkerProcessEdgeCases:
    """Edge cases for process_one ported from legacy worker tests."""

    def test_process_one_no_tasks(self, engine):
        worker = BackgroundWorker(engine=engine, extractor=MockExtractor())
        assert worker.process_one() is False

    def test_process_one_skips_deleted_entry(self, engine):
        engine.enqueue_task(
            task_type='extract_entities_v3',
            payload={'entry_id': 'nonexistent'},
        )
        worker = BackgroundWorker(engine=engine, extractor=MockExtractor())
        assert worker.process_one() is True
        # No entities should be created
        assert engine.get_entity_count() == 0

    def test_unknown_task_type_handled_gracefully(self, engine):
        engine.enqueue_task(task_type='unknown_type_xyz', payload={})
        worker = BackgroundWorker(engine=engine, extractor=MockExtractor())
        worker.process_one()  # should not raise
        assert engine.get_pending_task_count() == 0

    def test_extract_enqueues_link_graph_when_llm_available(self, engine):
        llm = MockLLM()
        worker = BackgroundWorker(
            engine=engine,
            extractor=MockExtractor(),
            llm=llm,
        )
        engine.enqueue_task(
            task_type='extract_entities_v3',
            payload={'entry_id': 'e1'},
        )
        worker.process_one()
        # Should have enqueued a link_graph_v3 task
        task = engine.dequeue_task()
        assert task is not None
        _, task_type, payload = task
        assert task_type == 'link_graph_v3'
        assert payload['entry_id'] == 'e1'

    def test_extract_skips_link_graph_without_llm(self, engine):
        worker = BackgroundWorker(
            engine=engine,
            extractor=MockExtractor(),
            llm=None,
        )
        engine.enqueue_task(
            task_type='extract_entities_v3',
            payload={'entry_id': 'e1'},
        )
        worker.process_one()
        assert engine.dequeue_task() is None  # no link_graph enqueued

    def test_link_graph_handles_invalid_json(self, engine):
        engine.upsert_entity(name='Alice', entity_type='person')
        engine.upsert_entity(name='Acme', entity_type='organization')
        engine.enqueue_task(
            task_type='link_graph_v3',
            payload={'entry_id': 'e1', 'entity_names': ['Alice', 'Acme']},
        )
        worker = BackgroundWorker(
            engine=engine,
            extractor=MockExtractor(),
            llm=MockLLM('not valid json'),
        )
        worker.process_one()  # should not raise
        assert engine.get_pending_task_count() == 0

    def test_link_graph_handles_non_list_json(self, engine):
        engine.enqueue_task(
            task_type='link_graph_v3',
            payload={'entry_id': 'e1', 'entity_names': ['Alice', 'Acme']},
        )
        worker = BackgroundWorker(
            engine=engine,
            extractor=MockExtractor(),
            llm=MockLLM('{"source": "Alice"}'),
        )
        worker.process_one()  # should not raise
        assert engine.get_pending_task_count() == 0


class TestWorkerLifecycle:
    """Worker thread start/stop lifecycle."""

    def test_start_and_stop(self, engine):
        worker = BackgroundWorker(engine=engine, extractor=MockExtractor())
        worker.start()
        assert worker._thread is not None
        assert worker._thread.is_alive()
        worker.stop()

    def test_background_processes_task(self, engine):
        engine.enqueue_task(
            task_type='extract_entities_v3',
            payload={'entry_id': 'e1'},
        )
        worker = BackgroundWorker(
            engine=engine,
            extractor=MockExtractor(),
            poll_interval=0.05,
        )
        worker.start()
        time.sleep(0.5)  # give worker time to process
        worker.stop()
        # Verify entities were created
        entity_ids = engine.get_entity_ids_for_entry('e1')
        assert len(entity_ids) >= 1
