"""Graph module tests adapted for v3 schema."""

import pytest

from aingram.graph.builder import GraphBuilder
from aingram.graph.traversal import GraphTraversal
from aingram.storage.engine import StorageEngine
from aingram.types import AgentSession


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / 'test.db'
    eng = StorageEngine(str(db))
    yield eng
    eng.close()


@pytest.fixture
def engine_with_data(engine):
    """Engine with a session, entries, and entities linked."""
    session = AgentSession(
        session_id='s1',
        agent_name='test',
        public_key='a' * 64,
        created_at='2026-01-01T00:00:00+00:00',
    )
    engine.store_session(session)
    engine.store_entry(
        entry_id='e1',
        content_hash='ch1',
        entry_type='observation',
        content='The connection pool causes latency',
        session_id='s1',
        sequence_num=1,
        prev_entry_id=None,
        signature='sig1',
        created_at='2026-01-01T00:00:00+00:00',
        embedding=[0.1] * 768,
    )
    engine.store_entry(
        entry_id='e2',
        content_hash='ch2',
        entry_type='hypothesis',
        content='API gateway is affected by pool exhaustion',
        session_id='s1',
        sequence_num=2,
        prev_entry_id='e1',
        signature='sig2',
        created_at='2026-01-01T00:01:00+00:00',
        embedding=[0.2] * 768,
    )
    return engine


class TestGraphBuilderV3:
    def test_upsert_entity_with_source_entry(self, engine_with_data):
        builder = GraphBuilder(engine_with_data)
        eid = builder.upsert_entity('pool', 'component', source_entry='e1')
        linked = engine_with_data.get_entry_ids_for_entity(eid)
        assert 'e1' in linked

    def test_upsert_entity_without_source(self, engine_with_data):
        builder = GraphBuilder(engine_with_data)
        eid = builder.upsert_entity('pool', 'component')
        assert eid  # just verify no error


class TestGraphTraversalV3:
    def test_search_returns_entry_ids(self, engine_with_data):
        builder = GraphBuilder(engine_with_data)
        builder.upsert_entity('connection pool', 'component', source_entry='e1')
        builder.upsert_entity('API gateway', 'system', source_entry='e2')

        traversal = GraphTraversal(engine_with_data)
        entry_ids = traversal.search('connection pool problems')
        assert 'e1' in entry_ids

    def test_search_empty_query(self, engine_with_data):
        traversal = GraphTraversal(engine_with_data)
        assert traversal.search('xyznonexistent') == []
