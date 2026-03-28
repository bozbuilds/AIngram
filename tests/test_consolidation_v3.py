"""Consolidation tests adapted for v3 schema."""

import pytest

from aingram.storage.engine import StorageEngine
from aingram.types import AgentSession
from tests.conftest import MockEmbedder, MockLLM


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / 'test.db'
    eng = StorageEngine(str(db))
    yield eng
    eng.close()


@pytest.fixture
def engine_with_entries(engine):
    session = AgentSession(
        session_id='s1',
        agent_name='test',
        public_key='a' * 64,
        created_at='2026-01-01T00:00:00+00:00',
    )
    engine.store_session(session)
    for i in range(5):
        engine.store_entry(
            entry_id=f'e{i}',
            content_hash=f'ch{i}',
            entry_type='observation',
            content=f'{{"text":"entry {i}"}}',
            session_id='s1',
            sequence_num=i + 1,
            prev_entry_id=f'e{i - 1}' if i > 0 else None,
            signature=f'sig{i}',
            created_at='2025-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
    return engine


class TestDecayV3:
    def test_apply_decay_returns_count(self, engine_with_entries):
        from aingram.consolidation.decay import apply_decay

        count = apply_decay(engine_with_entries)
        # Entries from 2025 should all have decayed importance
        assert count > 0

    def test_decay_reduces_importance(self, engine_with_entries):
        from aingram.consolidation.decay import apply_decay

        before = engine_with_entries.get_entry('e0')
        assert before.importance == 0.5  # default

        apply_decay(engine_with_entries)

        after = engine_with_entries.get_entry('e0')
        assert after.importance < before.importance


class TestContradictionV3:
    def test_no_contradictions_without_entities(self, engine_with_entries):
        from aingram.consolidation.contradiction import ContradictionDetector

        detector = ContradictionDetector(engine_with_entries, llm=MockLLM('{"contradicts": false}'))
        result = detector.detect_and_resolve()
        assert result.contradictions_found == 0

    def test_entry_type_aware_supersession(self, engine_with_entries):
        """A result superseding a hypothesis should not be flagged as contradiction."""
        from aingram.consolidation.contradiction import ContradictionDetector

        # Set up: hypothesis and result about same topic, linked to same entity
        engine_with_entries.store_entry(
            entry_id='hyp1',
            content_hash='ch_hyp',
            entry_type='hypothesis',
            content='{"text":"Pool size should be 50"}',
            session_id='s1',
            sequence_num=6,
            prev_entry_id='e4',
            signature='sig_hyp',
            created_at='2025-01-01T00:00:00+00:00',
            embedding=[0.1] * 768,
        )
        engine_with_entries.store_entry(
            entry_id='res1',
            content_hash='ch_res',
            entry_type='result',
            content='{"text":"Pool size 50 confirmed as correct"}',
            session_id='s1',
            sequence_num=7,
            prev_entry_id='hyp1',
            signature='sig_res',
            created_at='2025-01-01T00:01:00+00:00',
            embedding=[0.1] * 768,
        )
        # Link both to same entity
        eid = engine_with_entries.upsert_entity(name='pool', entity_type='component')
        engine_with_entries.link_entity_to_mention(eid, 'hyp1')
        engine_with_entries.link_entity_to_mention(eid, 'res1')

        # Even if LLM says "contradicts", the type pair (hypothesis, result) should be skipped
        detector = ContradictionDetector(
            engine_with_entries,
            llm=MockLLM('{"contradicts": true, "superseded_index": 0}'),
        )
        result = detector.detect_and_resolve()
        assert result.contradictions_found == 0  # skipped because result→hypothesis is natural


class TestMergerV3:
    def test_chain_entries_never_merged(self, engine):
        """Entries within the same reasoning chain should never be merged."""
        from aingram.consolidation.merger import MemoryMerger
        from aingram.trust.session import SessionManager
        from aingram.types import ReasoningChain

        session = AgentSession(
            session_id='s1',
            agent_name='test',
            public_key='a' * 64,
            created_at='2026-01-01T00:00:00+00:00',
        )
        engine.store_session(session)

        # Create a real SessionManager so the merger doesn't early-return
        sm = SessionManager(agent_name='merger-test')
        engine.store_session(sm.to_agent_session())

        engine.create_chain(
            ReasoningChain(
                chain_id='c1',
                title='exp',
                created_by_session='s1',
                created_at='2026-01-01T00:00:00+00:00',
            )
        )
        eid = engine.upsert_entity(name='pool', entity_type='component')
        for i in range(4):
            engine.store_entry(
                entry_id=f'e{i}',
                content_hash=f'ch{i}',
                entry_type='observation',
                content=f'{{"text":"entry {i}"}}',
                session_id='s1',
                sequence_num=i + 1,
                prev_entry_id=f'e{i - 1}' if i > 0 else None,
                signature=f'sig{i}',
                created_at='2026-01-01T00:00:00+00:00',
                embedding=[0.1] * 768,
                reasoning_chain_id='c1',  # all in same chain
            )
            engine.link_entity_to_mention(eid, f'e{i}')

        merger = MemoryMerger(
            engine,
            embedder=MockEmbedder(),
            llm=MockLLM('Summary of pooled entries'),
            session=sm,  # real session — forces merger to reach chain-exclusion logic
        )
        result = merger.merge_similar(min_cluster_size=3)
        # Should NOT merge because they're all in the same chain
        assert result.summaries_created == 0

    def test_importance_reduction_order_independent(self, engine):
        """Importance reduction must pair each entry_id with its own importance,
        regardless of the order get_entries_by_ids returns rows."""
        from unittest.mock import patch

        from aingram.consolidation.merger import _MERGED_IMPORTANCE_FACTOR, MemoryMerger
        from aingram.trust.session import SessionManager

        session = AgentSession(
            session_id='s1',
            agent_name='test',
            public_key='a' * 64,
            created_at='2026-01-01T00:00:00+00:00',
        )
        engine.store_session(session)

        sm = SessionManager(agent_name='merger-test')
        engine.store_session(sm.to_agent_session())

        eid = engine.upsert_entity(name='widget', entity_type='component')

        # Create entries with deliberately different importance values
        importances = {'e0': 0.9, 'e1': 0.1, 'e2': 0.5}
        for i, (entry_id, imp) in enumerate(importances.items()):
            engine.store_entry(
                entry_id=entry_id,
                content_hash=f'ch{i}',
                entry_type='observation',
                content=f'{{"text":"entry {i}"}}',
                session_id='s1',
                sequence_num=i + 1,
                prev_entry_id=f'e{i - 1}' if i > 0 else None,
                signature=f'sig{i}',
                created_at='2026-01-01T00:00:00+00:00',
                embedding=[0.1] * 768,
                importance=imp,
            )
            engine.link_entity_to_mention(eid, entry_id)

        # Wrap get_entries_by_ids to reverse the returned order,
        # simulating a DB that returns rows in a different order than requested.
        original_get = engine.get_entries_by_ids

        def reversed_get(ids):
            return list(reversed(original_get(ids)))

        with patch.object(engine, 'get_entries_by_ids', side_effect=reversed_get):
            merger = MemoryMerger(
                engine,
                embedder=MockEmbedder(),
                llm=MockLLM('Summary of widget entries'),
                session=sm,
            )
            result = merger.merge_similar(min_cluster_size=3)

        assert result.summaries_created == 1

        # Verify each entry's importance was reduced based on its OWN original value
        for entry_id, original_imp in importances.items():
            entry = engine.get_entry(entry_id)
            expected = original_imp * _MERGED_IMPORTANCE_FACTOR
            assert abs(entry.importance - expected) < 1e-9, (
                f'{entry_id}: expected {expected}, got {entry.importance}'
            )
