# tests/test_storage/test_engine_v3.py — v3 engine tests
import math
import sqlite3

import pytest

from aingram.exceptions import DatabaseError
from aingram.storage.engine import StorageEngine
from aingram.types import AgentSession, ReasoningChain
from tests.conftest import ensure_test_session


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / 'test.db'
    eng = StorageEngine(str(db))
    yield eng
    eng.close()


@pytest.fixture
def session():
    return AgentSession(
        session_id='s1',
        agent_name='test-agent',
        public_key='a' * 64,
        created_at='2026-01-01T00:00:00+00:00',
    )


@pytest.fixture
def engine_with_session(engine, session):
    engine.store_session(session)
    return engine


def _store_test_entry(engine, entry_id='e-test', content_hash='h-test'):
    """Store a minimal test entry for FK references (session s1)."""
    engine._conn.execute(
        'INSERT OR IGNORE INTO memory_entries '
        '(entry_id, content_hash, entry_type, content, session_id, sequence_num, '
        'prev_entry_id, signature, created_at, importance) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (
            entry_id,
            content_hash,
            'observation',
            '{"text":"test"}',
            's1',
            1,
            None,
            'sig' + entry_id,
            '2026-01-01T00:00:00+00:00',
            0.5,
        ),
    )
    engine._conn.commit()
    return entry_id


class TestSessionOps:
    def test_store_and_get_session(self, engine, session):
        engine.store_session(session)
        result = engine.get_session('s1')
        assert result is not None
        assert result.agent_name == 'test-agent'
        assert result.public_key == 'a' * 64

    def test_get_missing_session(self, engine):
        assert engine.get_session('nonexistent') is None


class TestEntryOps:
    def test_store_and_get_entry(self, engine_with_session):
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{"text":"hello world"}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
        entry = engine_with_session.get_entry('e1')
        assert entry is not None
        assert entry.entry_id == 'e1'
        assert entry.content == '{"text":"hello world"}'
        assert entry.session_id == 's1'
        assert entry.sequence_num == 1

    def test_get_missing_entry(self, engine):
        assert engine.get_entry('nonexistent') is None

    def test_get_entry_count(self, engine_with_session):
        assert engine_with_session.get_entry_count() == 0
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
        assert engine_with_session.get_entry_count() == 1

    def test_get_entries_by_ids(self, engine_with_session):
        for i in range(3):
            engine_with_session.store_entry(
                entry_id=f'e{i}',
                content_hash=f'ch{i}',
                entry_type='observation',
                content=f'{{"text":"entry {i}"}}',
                session_id='s1',
                sequence_num=i + 1,
                prev_entry_id=f'e{i - 1}' if i > 0 else None,
                signature=f'sig{i}',
                created_at='2026-01-01T00:00:00+00:00',
                embedding=[0.1 * (i + 1)] * 768,
            )
        entries = engine_with_session.get_entries_by_ids(['e0', 'e2'])
        assert len(entries) == 2
        assert {e.entry_id for e in entries} == {'e0', 'e2'}

    def test_get_entries_by_ids_empty(self, engine):
        assert engine.get_entries_by_ids([]) == []

    def test_update_entry_access(self, engine_with_session):
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
        engine_with_session.update_entry_access('e1')
        entry = engine_with_session.get_entry('e1')
        assert entry.access_count == 1
        assert entry.accessed_at is not None


class TestSearch:
    def test_fts_search(self, engine_with_session):
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{"text":"unique searchable content xyz"}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
        results = engine_with_session.search_fts('searchable', limit=5)
        assert len(results) == 1
        assert results[0][0] == 'e1'

    def test_vector_search(self, engine_with_session):
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[1.0] * 768,
        )
        results = engine_with_session.search_vectors([1.0] * 768, limit=5)
        assert len(results) == 1
        assert results[0][0] == 'e1'


class TestChainOps:
    def test_create_and_get_chain(self, engine_with_session):
        chain = ReasoningChain(
            chain_id='c1',
            title='experiment',
            created_by_session='s1',
            created_at='2026-01-01T00:00:00+00:00',
        )
        engine_with_session.create_chain(chain)
        result = engine_with_session.get_chain('c1')
        assert result is not None
        assert result.title == 'experiment'
        assert result.status == 'active'
        assert result.outcome is None

    def test_update_chain_status(self, engine_with_session):
        chain = ReasoningChain(
            chain_id='c1',
            title='experiment',
            created_by_session='s1',
            created_at='2026-01-01T00:00:00+00:00',
        )
        engine_with_session.create_chain(chain)
        engine_with_session.update_chain_status('c1', 'completed')
        result = engine_with_session.get_chain('c1')
        assert result.status == 'completed'

    def test_get_entries_by_chain(self, engine_with_session):
        chain = ReasoningChain(
            chain_id='c1',
            title='exp',
            created_by_session='s1',
            created_at='2026-01-01T00:00:00+00:00',
        )
        engine_with_session.create_chain(chain)
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='hypothesis',
            content='{"text":"chain entry"}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
            reasoning_chain_id='c1',
        )
        entries = engine_with_session.get_entries_by_chain('c1')
        assert len(entries) == 1
        assert entries[0].reasoning_chain_id == 'c1'

    def test_get_entries_by_session(self, engine_with_session):
        for i in range(3):
            engine_with_session.store_entry(
                entry_id=f'e{i}',
                content_hash=f'ch{i}',
                entry_type='observation',
                content='{}',
                session_id='s1',
                sequence_num=i + 1,
                prev_entry_id=f'e{i - 1}' if i > 0 else None,
                signature=f'sig{i}',
                created_at='2026-01-01T00:00:00+00:00',
                embedding=[0.1] * 768,
            )
        entries = engine_with_session.get_entries_by_session('s1')
        assert len(entries) == 3
        assert [e.sequence_num for e in entries] == [1, 2, 3]


class TestFTSPrecision:
    def test_fts_searches_text_not_json_keys(self, engine_with_session):
        """FTS should match on content text, not JSON structural keys."""
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{"text":"the weather is sunny today"}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
        # Searching for actual content should match
        results = engine_with_session.search_fts('sunny', limit=5)
        assert len(results) == 1
        # Searching for the JSON key 'text' should NOT match
        results = engine_with_session.search_fts('text', limit=5)
        assert len(results) == 0

    def test_fts_query_with_double_quote(self, engine_with_session):
        """FTS queries containing double quotes should not crash."""
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{"text":"she said hello"}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
        # Should not raise, should return results or empty list
        results = engine_with_session.search_fts('she"said', limit=5)
        assert isinstance(results, list)


class TestEmbeddingDim:
    def test_get_embedding_dim(self, engine):
        assert engine.get_embedding_dim() == 768


class TestChainCompletion:
    def _create_test_chain(self, engine, chain_id='chain-1'):
        ensure_test_session(engine)
        engine.create_chain(
            ReasoningChain(
                chain_id=chain_id,
                title='Test Experiment',
                created_by_session='test-session',
                created_at='2026-01-01T00:00:00+00:00',
            )
        )
        return chain_id

    def test_complete_chain_sets_outcome(self, engine):
        chain_id = self._create_test_chain(engine)
        engine.complete_chain(chain_id, outcome='confirmed')

        chain = engine.get_chain(chain_id)
        assert chain.status == 'completed'
        assert chain.outcome == 'confirmed'

    def test_complete_chain_invalid_outcome_raises(self, engine):
        chain_id = self._create_test_chain(engine, 'chain-bad')
        with pytest.raises(ValueError, match='Invalid outcome'):
            engine.complete_chain(chain_id, outcome='bogus')

    def test_get_completed_chains(self, engine):
        self._create_test_chain(engine, 'chain-a')
        self._create_test_chain(engine, 'chain-b')
        self._create_test_chain(engine, 'chain-c')

        engine.complete_chain('chain-a', outcome='confirmed')
        engine.complete_chain('chain-b', outcome='refuted')

        completed = engine.get_completed_chains()
        assert len(completed) == 2
        chain_ids = {c.chain_id for c in completed}
        assert chain_ids == {'chain-a', 'chain-b'}

    def test_complete_chain_nonexistent_is_silent(self, engine):
        ensure_test_session(engine)
        engine.complete_chain('nonexistent-chain', outcome='confirmed')
        assert engine.get_completed_chains() == []

    def test_get_completed_chains_empty(self, engine):
        assert engine.get_completed_chains() == []

    def test_get_chain_includes_outcome(self, engine):
        chain_id = self._create_test_chain(engine, 'chain-outcome')
        engine.complete_chain(chain_id, outcome='partial')

        chain = engine.get_chain(chain_id)
        assert chain.outcome == 'partial'


class TestKnowledgeItemsCRUD:
    def test_store_knowledge_item(self, engine):
        ensure_test_session(engine)
        item_id = engine.store_knowledge_item(
            principle='Connection pools should be 5x peak concurrent requests',
            supporting_chains=['chain-1', 'chain-2'],
            confidence=0.9,
            session_id='test-session',
        )
        assert isinstance(item_id, str)

    def test_get_knowledge_items(self, engine):
        ensure_test_session(engine)
        engine.store_knowledge_item(
            principle='Principle A',
            supporting_chains=['chain-1'],
            confidence=0.8,
            session_id='test-session',
        )
        engine.store_knowledge_item(
            principle='Principle B',
            supporting_chains=['chain-2', 'chain-3'],
            confidence=0.95,
            session_id='test-session',
        )

        items = engine.get_knowledge_items()
        assert len(items) == 2
        principles = {item['principle'] for item in items}
        assert 'Principle A' in principles
        assert 'Principle B' in principles

    def test_get_knowledge_items_empty(self, engine):
        assert engine.get_knowledge_items() == []

    def test_get_knowledge_items_returns_supporting_chains(self, engine):
        ensure_test_session(engine)
        engine.store_knowledge_item(
            principle='Test',
            supporting_chains=['c1', 'c2'],
            confidence=0.7,
            session_id='test-session',
        )
        items = engine.get_knowledge_items()
        assert items[0]['supporting_chains'] == ['c1', 'c2']

    def test_update_knowledge_item(self, engine):
        ensure_test_session(engine)
        item_id = engine.store_knowledge_item(
            principle='Original principle',
            supporting_chains=['chain-1'],
            confidence=0.7,
            session_id='test-session',
        )

        engine.update_knowledge_item(
            item_id,
            principle='Updated principle',
            supporting_chains=['chain-1', 'chain-2'],
            confidence=0.9,
        )

        items = engine.get_knowledge_items()
        assert len(items) == 1
        assert items[0]['principle'] == 'Updated principle'
        assert items[0]['supporting_chains'] == ['chain-1', 'chain-2']
        assert items[0]['confidence'] == 0.9


class TestSchemaAlignment:
    def test_reasoning_chains_has_outcome_column(self, engine_with_session):
        engine_with_session.create_chain(
            ReasoningChain(
                chain_id='test-chain',
                title='Test',
                created_by_session='s1',
                created_at='2026-01-01T00:00:00+00:00',
            )
        )
        engine_with_session._conn.execute(
            "UPDATE reasoning_chains SET outcome = 'confirmed' WHERE chain_id = ?",
            ('test-chain',),
        )
        engine_with_session._conn.commit()
        row = engine_with_session._conn.execute(
            'SELECT outcome FROM reasoning_chains WHERE chain_id = ?',
            ('test-chain',),
        ).fetchone()
        assert row[0] == 'confirmed'

    def test_reasoning_chains_outcome_check_constraint(self, engine_with_session):
        engine_with_session.create_chain(
            ReasoningChain(
                chain_id='test-chain-2',
                title='Test',
                created_by_session='s1',
                created_at='2026-01-01T00:00:00+00:00',
            )
        )
        with pytest.raises(sqlite3.IntegrityError):
            engine_with_session._conn.execute(
                "UPDATE reasoning_chains SET outcome = 'invalid' WHERE chain_id = ?",
                ('test-chain-2',),
            )

    def test_knowledge_items_table_exists(self, engine_with_session):
        engine_with_session._conn.execute(
            'INSERT INTO knowledge_items '
            '(knowledge_id, principle, supporting_chains, '
            'confidence, created_by_session, created_at) VALUES (?, ?, ?, ?, ?, ?)',
            ('ki-1', 'Test principle', '["chain-1"]', 0.9, 's1', '2026-01-01T00:00:00+00:00'),
        )
        engine_with_session._conn.commit()
        row = engine_with_session._conn.execute(
            'SELECT principle, confidence FROM knowledge_items WHERE knowledge_id = ?',
            ('ki-1',),
        ).fetchone()
        assert row[0] == 'Test principle'
        assert row[1] == 0.9

    def test_entity_has_entity_id_pk(self, engine):
        eid = engine.insert_entity(name='TestEntity', entity_type='concept')
        entity = engine.get_entity(eid)
        assert entity.entity_id == eid
        assert entity.name == 'TestEntity'
        assert entity.mention_count == 1

    def test_entity_mentions_table_has_confidence(self, engine_with_session):
        eid = engine_with_session.insert_entity(name='TestEntity2', entity_type='concept')
        entry_id = _store_test_entry(engine_with_session)
        engine_with_session.link_entity_to_mention(eid, entry_id, confidence=0.95)
        row = engine_with_session._conn.execute(
            'SELECT confidence FROM entity_mentions WHERE entity_id = ? AND entry_id = ?',
            (eid, entry_id),
        ).fetchone()
        assert row[0] == 0.95

    def test_upsert_entity_increments_mention_count(self, engine):
        eid = engine.upsert_entity(name='Counter', entity_type='concept')
        entity = engine.get_entity(eid)
        assert entity.mention_count == 1

        eid2 = engine.upsert_entity(name='Counter', entity_type='concept')
        assert eid == eid2
        entity = engine.get_entity(eid)
        assert entity.mention_count == 2

    def test_store_cross_reference(self, engine_with_session):
        for eid, ch in [('e-src', 'h-src'), ('e-tgt', 'h-tgt')]:
            engine_with_session.store_entry(
                entry_id=eid,
                content_hash=ch,
                entry_type='observation',
                content='{}',
                session_id='s1',
                sequence_num=1 if eid == 'e-src' else 2,
                prev_entry_id='e-src' if eid == 'e-tgt' else None,
                signature=f'sig-{eid}',
                created_at='2026-01-01T00:00:00+00:00',
                embedding=[0.1] * 768,
            )
        sig = 'ab' * 64
        rid = engine_with_session.store_cross_reference(
            source_entry_id='e-src',
            target_entry_id='e-tgt',
            reference_type='supports',
            session_id='s1',
            signature=sig,
        )
        assert rid >= 1
        row = engine_with_session._conn.execute(
            'SELECT reference_type FROM cross_references WHERE id = ?',
            (rid,),
        ).fetchone()
        assert row[0] == 'supports'


class TestEntityEntryLinks:
    """Entity-to-entry linking (entity_mentions table)."""

    def test_link_entity_to_mention(self, engine_with_session):
        eid = engine_with_session.upsert_entity(name='Pool', entity_type='component')
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
        engine_with_session.link_entity_to_mention(eid, 'e1')
        entry_ids = engine_with_session.get_entry_ids_for_entity(eid)
        assert 'e1' in entry_ids

    def test_get_entry_ids_for_entities(self, engine_with_session):
        eid = engine_with_session.upsert_entity(name='Pool', entity_type='component')
        for i in range(2):
            engine_with_session.store_entry(
                entry_id=f'e{i}',
                content_hash=f'ch{i}',
                entry_type='observation',
                content='{}',
                session_id='s1',
                sequence_num=i + 1,
                prev_entry_id=f'e{i - 1}' if i > 0 else None,
                signature=f'sig{i}',
                created_at='2026-01-01T00:00:00+00:00',
                embedding=[0.1] * 768,
            )
            engine_with_session.link_entity_to_mention(eid, f'e{i}')
        result = engine_with_session.get_entry_ids_for_entities([eid])
        assert eid in result
        assert set(result[eid]) == {'e0', 'e1'}

    def test_get_entity_ids_for_entry(self, engine_with_session):
        eid = engine_with_session.upsert_entity(name='Pool', entity_type='component')
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
        engine_with_session.link_entity_to_mention(eid, 'e1')
        entity_ids = engine_with_session.get_entity_ids_for_entry('e1')
        assert eid in entity_ids


class TestConsolidationSupport:
    """Engine methods for v3 consolidation."""

    def test_get_entries_for_decay(self, engine_with_session):
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
        entries = engine_with_session.get_entries_for_decay(limit=100)
        assert len(entries) == 1
        assert entries[0].entry_id == 'e1'

    def test_batch_update_entry_importance(self, engine_with_session):
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
        count = engine_with_session.batch_update_entry_importance([('e1', 0.3)])
        assert count == 1
        entry = engine_with_session.get_entry('e1')
        assert abs(entry.importance - 0.3) < 0.001

    def test_batch_update_entry_importance_empty(self, engine):
        assert engine.batch_update_entry_importance([]) == 0


# ------------------------------------------------------------------ backfill
# Tests below were ported from legacy v1/v2 engine tests to ensure no
# behavioral regressions in functionality shared between schema versions.


class TestEngineInit:
    """SQLite engine initialization guarantees."""

    def test_creates_db_file(self, tmp_path):
        db = tmp_path / 'init.db'
        eng = StorageEngine(str(db))
        assert db.exists()
        eng.close()

    def test_enables_wal_mode(self, engine):
        cursor = engine._conn.execute('PRAGMA journal_mode')
        assert cursor.fetchone()[0] == 'wal'

    def test_sets_busy_timeout(self, engine):
        cursor = engine._conn.execute('PRAGMA busy_timeout')
        assert cursor.fetchone()[0] == 5000

    def test_operations_after_close_raise(self, tmp_path):
        eng = StorageEngine(str(tmp_path / 'closed.db'))
        eng.close()
        with pytest.raises(DatabaseError, match='closed'):
            eng.get_entry_count()


class TestTaskQueue:
    """Task queue shared between v2 and v3."""

    def test_enqueue_and_dequeue(self, engine):
        task_id = engine.enqueue_task(
            task_type='extract_entities_v3',
            payload={'entry_id': 'e1'},
        )
        assert isinstance(task_id, str)
        task = engine.dequeue_task()
        assert task is not None
        tid, task_type, payload = task
        assert tid == task_id
        assert task_type == 'extract_entities_v3'
        assert payload == {'entry_id': 'e1'}

    def test_dequeue_empty(self, engine):
        assert engine.dequeue_task() is None

    def test_dequeue_claims_task(self, engine):
        engine.enqueue_task(task_type='test', payload={})
        task = engine.dequeue_task()
        assert task is not None
        assert engine.dequeue_task() is None  # claimed

    def test_dequeue_respects_priority(self, engine):
        engine.enqueue_task(task_type='low', payload={'order': 1}, priority=1)
        engine.enqueue_task(task_type='high', payload={'order': 2}, priority=10)
        task = engine.dequeue_task()
        assert task is not None
        _, task_type, _ = task
        assert task_type == 'high'

    def test_complete_task(self, engine):
        engine.enqueue_task(task_type='test', payload={})
        task = engine.dequeue_task()
        assert task is not None
        task_id, _, _ = task
        engine.complete_task(task_id)
        assert engine.dequeue_task() is None

    def test_fail_task(self, engine):
        engine.enqueue_task(task_type='test', payload={})
        task = engine.dequeue_task()
        assert task is not None
        task_id, _, _ = task
        engine.fail_task(task_id, 'something went wrong')
        assert engine.dequeue_task() is None

    def test_get_pending_task_count(self, engine):
        assert engine.get_pending_task_count() == 0
        engine.enqueue_task(task_type='a', payload={})
        engine.enqueue_task(task_type='b', payload={})
        assert engine.get_pending_task_count() == 2
        engine.dequeue_task()
        assert engine.get_pending_task_count() == 1


class TestEntityCRUD:
    """Entity CRUD using v3 upsert_entity."""

    def test_upsert_and_get(self, engine):
        eid = engine.upsert_entity(name='Alice', entity_type='person')
        assert isinstance(eid, str)
        entity = engine.get_entity(eid)
        assert entity is not None
        assert entity.name == 'Alice'
        assert entity.entity_type == 'person'

    def test_get_nonexistent_entity(self, engine):
        assert engine.get_entity('does-not-exist') is None

    def test_upsert_idempotent(self, engine):
        eid1 = engine.upsert_entity(name='Alice', entity_type='person')
        eid2 = engine.upsert_entity(name='Alice', entity_type='person')
        assert eid1 == eid2

    def test_get_entities(self, engine):
        engine.upsert_entity(name='Alice', entity_type='person')
        engine.upsert_entity(name='Acme', entity_type='organization')
        entities = engine.get_entities()
        assert len(entities) == 2
        names = {e.name for e in entities}
        assert names == {'Alice', 'Acme'}

    def test_get_entities_respects_limit(self, engine):
        for i in range(5):
            engine.upsert_entity(name=f'Entity{i}', entity_type='test')
        entities = engine.get_entities(limit=3)
        assert len(entities) == 3

    def test_get_entity_count(self, engine):
        assert engine.get_entity_count() == 0
        engine.upsert_entity(name='Alice', entity_type='person')
        engine.upsert_entity(name='Acme', entity_type='organization')
        assert engine.get_entity_count() == 2

    def test_find_entities_by_name_exact(self, engine):
        engine.upsert_entity(name='Alice', entity_type='person')
        engine.upsert_entity(name='Bob', entity_type='person')
        results = engine.find_entities_by_name('Alice')
        assert len(results) == 1
        assert results[0].name == 'Alice'

    def test_find_entities_by_name_case_insensitive(self, engine):
        engine.upsert_entity(name='Alice', entity_type='person')
        results = engine.find_entities_by_name('alice')
        assert len(results) == 1
        assert results[0].name == 'Alice'

    def test_find_entities_by_name_no_match(self, engine):
        engine.upsert_entity(name='Alice', entity_type='person')
        assert engine.find_entities_by_name('Nobody') == []


class TestRelationshipCRUD:
    """Relationship insertion and retrieval."""

    def test_insert_and_get(self, engine):
        src = engine.upsert_entity(name='Alice', entity_type='person')
        tgt = engine.upsert_entity(name='Acme', entity_type='organization')
        rel_id = engine.insert_relationship(
            source_id=src,
            target_id=tgt,
            relation_type='works_at',
            fact='Alice works at Acme',
        )
        assert isinstance(rel_id, str)
        rels = engine.get_relationships_for_entity(src)
        assert len(rels) == 1
        assert rels[0].relation_type == 'works_at'
        assert rels[0].fact == 'Alice works at Acme'

    def test_get_relationships_both_directions(self, engine):
        src = engine.upsert_entity(name='Alice', entity_type='person')
        tgt = engine.upsert_entity(name='Acme', entity_type='organization')
        engine.insert_relationship(
            source_id=src,
            target_id=tgt,
            relation_type='works_at',
        )
        rels = engine.get_relationships_for_entity(tgt)
        assert len(rels) == 1

    def test_get_relationships_empty(self, engine):
        eid = engine.upsert_entity(name='Alone', entity_type='person')
        assert engine.get_relationships_for_entity(eid) == []

    def test_invalidate_relationship(self, engine):
        e1 = engine.upsert_entity(name='Alice', entity_type='person')
        e2 = engine.upsert_entity(name='Acme', entity_type='organization')
        rel_id = engine.insert_relationship(
            source_id=e1,
            target_id=e2,
            relation_type='works_at',
        )
        engine.invalidate_relationship(rel_id)
        rels = engine.get_relationships_for_entity(e1)
        assert len(rels) == 1
        assert rels[0].t_invalid is not None


class TestGraphTraversal:
    """Recursive CTE graph traversal on the v3 engine."""

    def test_one_hop(self, engine):
        e1 = engine.upsert_entity(name='Alice', entity_type='person')
        e2 = engine.upsert_entity(name='Acme', entity_type='organization')
        engine.insert_relationship(source_id=e1, target_id=e2, relation_type='works_at')
        traversed = engine.traverse_graph([e1], max_hops=1)
        entity_ids = {eid for eid, _hop in traversed}
        assert e2 in entity_ids

    def test_two_hops(self, engine):
        e1 = engine.upsert_entity(name='Alice', entity_type='person')
        e2 = engine.upsert_entity(name='Acme', entity_type='organization')
        e3 = engine.upsert_entity(name='Project X', entity_type='project')
        engine.insert_relationship(source_id=e1, target_id=e2, relation_type='works_at')
        engine.insert_relationship(source_id=e2, target_id=e3, relation_type='owns')
        traversed = engine.traverse_graph([e1], max_hops=2)
        entity_ids = {eid for eid, _hop in traversed}
        assert e2 in entity_ids
        assert e3 in entity_ids

    def test_excludes_starting_entities(self, engine):
        e1 = engine.upsert_entity(name='Alice', entity_type='person')
        e2 = engine.upsert_entity(name='Acme', entity_type='organization')
        engine.insert_relationship(source_id=e1, target_id=e2, relation_type='works_at')
        traversed = engine.traverse_graph([e1], max_hops=1)
        entity_ids = {eid for eid, _hop in traversed}
        assert e1 not in entity_ids

    def test_respects_max_hops(self, engine):
        e1 = engine.upsert_entity(name='A', entity_type='node')
        e2 = engine.upsert_entity(name='B', entity_type='node')
        e3 = engine.upsert_entity(name='C', entity_type='node')
        engine.insert_relationship(source_id=e1, target_id=e2, relation_type='linked')
        engine.insert_relationship(source_id=e2, target_id=e3, relation_type='linked')
        traversed = engine.traverse_graph([e1], max_hops=1)
        entity_ids = {eid for eid, _hop in traversed}
        assert e2 in entity_ids
        assert e3 not in entity_ids

    def test_bidirectional(self, engine):
        e1 = engine.upsert_entity(name='Alice', entity_type='person')
        e2 = engine.upsert_entity(name='Acme', entity_type='organization')
        engine.insert_relationship(source_id=e2, target_id=e1, relation_type='employs')
        traversed = engine.traverse_graph([e1], max_hops=1)
        entity_ids = {eid for eid, _hop in traversed}
        assert e2 in entity_ids

    def test_empty_start(self, engine):
        assert engine.traverse_graph([], max_hops=2) == []

    def test_no_relationships(self, engine):
        e1 = engine.upsert_entity(name='Isolated', entity_type='node')
        assert engine.traverse_graph([e1], max_hops=2) == []

    def test_filters_expired_relationships(self, engine):
        e1 = engine.upsert_entity(name='Alice', entity_type='person')
        e2 = engine.upsert_entity(name='OldCo', entity_type='organization')
        rel_id = engine.insert_relationship(
            source_id=e1,
            target_id=e2,
            relation_type='worked_at',
        )
        engine.invalidate_relationship(rel_id)
        traversed = engine.traverse_graph([e1], max_hops=1)
        entity_ids = {eid for eid, _hop in traversed}
        assert e2 not in entity_ids


class TestFTSSanitization:
    """FTS special character handling."""

    def test_special_characters_dont_crash(self, engine_with_session):
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{"text":"Normal content here"}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
        # These should not raise
        engine_with_session.search_fts('"')
        engine_with_session.search_fts('AND OR NOT')
        engine_with_session.search_fts('test*')
        engine_with_session.search_fts('{near}')
        engine_with_session.search_fts('')


class TestVectorSearchEdgeCases:
    """Additional vector search edge cases."""

    def _make_embedding(self, seed: float, dim: int = 768) -> list[float]:
        return [math.sin(seed * (i + 1)) for i in range(dim)]

    def test_search_vectors_empty_db(self, engine):
        results = engine.search_vectors(self._make_embedding(1.0), limit=5)
        assert results == []

    def test_search_returns_distances(self, engine_with_session):
        emb = self._make_embedding(1.0)
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=emb,
        )
        results = engine_with_session.search_vectors(emb, limit=1)
        assert len(results) == 1
        _entry_id, distance = results[0]
        assert distance < 0.01  # nearly identical

    def test_similar_vectors_ranked_above_dissimilar(self, engine_with_session):
        emb1 = self._make_embedding(1.0)
        emb2 = self._make_embedding(2.0)
        emb_query = self._make_embedding(1.01)  # very similar to emb1
        engine_with_session.store_entry(
            entry_id='e1',
            content_hash='ch1',
            entry_type='observation',
            content='{}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig1',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=emb1,
        )
        engine_with_session.store_entry(
            entry_id='e2',
            content_hash='ch2',
            entry_type='observation',
            content='{}',
            session_id='s1',
            sequence_num=2,
            prev_entry_id='e1',
            signature='sig2',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=emb2,
        )
        results = engine_with_session.search_vectors(emb_query, limit=2)
        assert len(results) == 2
        assert results[0][0] == 'e1'  # closer to query
