"""End-to-end Phase 2 integration tests."""

import pytest

from tests.conftest import ClusterTestEmbedder, MockEmbedder, MockExtractor, MockLLM


@pytest.fixture
def mem(tmp_path):
    from aingram.store import MemoryStore

    db = tmp_path / 'integration.db'
    store = MemoryStore(str(db), agent_name='integration-test', embedder=MockEmbedder())
    yield store
    store.close()


@pytest.fixture
def mem_clustered(tmp_path):
    """MemoryStore with ClusterTestEmbedder for deterministic knowledge synthesis."""
    from aingram.store import MemoryStore

    db = tmp_path / 'integration_cluster.db'
    store = MemoryStore(str(db), agent_name='integration-test', embedder=ClusterTestEmbedder())
    yield store
    store.close()


def test_remember_enqueues_extraction(mem):
    """remember() should enqueue a background extraction task."""
    mem.remember('Alice met Bob at Acme Corp')
    count = mem._engine.get_pending_task_count()
    assert count >= 1  # extract_entities_v3 task queued


def test_worker_processes_extraction(mem):
    """Background worker extracts entities and links them."""
    from aingram.worker import BackgroundWorker

    eid = mem.remember('Alice met Bob at the Acme office')

    worker = BackgroundWorker(
        engine=mem._engine,
        extractor=MockExtractor(),
        llm=None,
    )
    # Process the queued extraction task
    worker.process_one()

    # Entities should now be linked to the entry
    entity_ids = mem._engine.get_entity_ids_for_entry(eid)
    assert len(entity_ids) >= 1


def test_graph_enriched_recall(mem):
    """After extraction, graph search should boost entity-linked entries."""
    from aingram.graph.builder import GraphBuilder

    eid1 = mem.remember('Connection pool exhaustion causes latency')
    mem.remember('Weather is sunny today')

    # Simulate entity extraction
    builder = GraphBuilder(mem._engine)
    builder.upsert_entity('connection pool', 'component', source_entry=eid1)

    results = mem.recall('connection pool')
    assert len(results) >= 1
    assert any('connection' in r.entry.content.lower() for r in results)


def test_consolidation_runs(mem):
    """Consolidation pipeline runs end-to-end."""
    for i in range(5):
        mem.remember(f'Observation number {i}')
    result = mem.consolidate()
    assert result.memories_decayed >= 0
    assert result.contradictions_found == 0


def test_consolidation_with_llm(mem):
    """consolidate(llm=...) passes the LLM to sub-components."""
    llm = MockLLM()
    for i in range(3):
        mem.remember(f'Entry {i}')
    result = mem.consolidate(llm=llm)
    assert result.memories_decayed >= 0
    # With no entity links, contradiction/merge are still no-ops,
    # but the call should succeed with an LLM provided.
    assert result.contradictions_found == 0
    assert result.memories_merged == 0


def test_full_cycle_remember_extract_recall_verify(mem):
    """Full cycle: remember → extract → recall → verify."""
    from aingram.worker import BackgroundWorker

    chain_id = mem.create_chain('Investigation')
    mem.remember('Pool size is 10', chain_id=chain_id, entry_type='observation')
    mem.remember('Pool should be 50', chain_id=chain_id, entry_type='hypothesis')
    mem.remember('Pool 50 works', chain_id=chain_id, entry_type='result')

    # Process background tasks
    worker = BackgroundWorker(
        engine=mem._engine,
        extractor=MockExtractor(),
        llm=None,
    )
    while worker.process_one():
        pass

    # Recall by chain
    results = mem.recall(chain_id=chain_id)
    assert len(results) == 3
    types = {r.entry.entry_type for r in results}
    assert types == {'observation', 'hypothesis', 'result'}

    # Verify chain integrity
    vr = mem.verify()
    assert vr.valid is True
    assert vr.entries_checked == 3


class TestPhase1AddendumIntegration:
    def test_cross_reference_workflow(self, mem):
        eid1 = mem.remember('Pool size is 10', entry_type='observation')
        eid2 = mem.remember('Pool should be 50', entry_type='hypothesis')

        mem.reference(source_id=eid2, target_id=eid1, reference_type='builds_on')

        row = mem._engine._conn.execute(
            'SELECT signature FROM cross_references WHERE source_entry_id = ?',
            (eid2,),
        ).fetchone()
        assert row is not None
        assert len(row[0]) == 128

        result = mem.verify()
        assert result.valid is True

    def test_entities_property_after_extraction(self, mem):
        from aingram.graph.builder import GraphBuilder

        builder = GraphBuilder(mem._engine)

        eid = mem.remember('Connection pool at Acme Corp', entry_type='observation')
        builder.upsert_entity('Acme Corp', 'organization', source_entry=eid)
        builder.upsert_entity('connection pool', 'component', source_entry=eid)

        entities = mem.entities
        names = {e.name for e in entities}
        assert 'Acme Corp' in names
        assert 'connection pool' in names

    def test_knowledge_items_table_writable(self, mem):
        mem._engine._conn.execute(
            'INSERT INTO knowledge_items '
            '(knowledge_id, principle, supporting_chains, confidence, '
            'created_by_session, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (
                'ki-int-1',
                'Pools need sizing',
                '["chain-1"]',
                0.85,
                mem._session.session_id,
                '2026-01-01T00:00:00+00:00',
            ),
        )
        mem._engine._conn.commit()
        row = mem._engine._conn.execute(
            'SELECT principle FROM knowledge_items WHERE knowledge_id = ?',
            ('ki-int-1',),
        ).fetchone()
        assert row[0] == 'Pools need sizing'

    def test_outcome_field_writable(self, mem):
        chain_id = mem.create_chain('Test experiment')
        mem.remember('Hypothesis', entry_type='hypothesis', chain_id=chain_id)

        mem._engine._conn.execute(
            "UPDATE reasoning_chains SET status = 'completed', outcome = 'confirmed' "
            'WHERE chain_id = ?',
            (chain_id,),
        )
        mem._engine._conn.commit()
        row = mem._engine._conn.execute(
            'SELECT outcome FROM reasoning_chains WHERE chain_id = ?',
            (chain_id,),
        ).fetchone()
        assert row[0] == 'confirmed'

    def test_entity_mention_count_increments(self, mem):
        from aingram.graph.builder import GraphBuilder

        builder = GraphBuilder(mem._engine)

        eid1 = mem.remember('First mention of SQLite', entry_type='observation')
        entity_id = builder.upsert_entity('SQLite', 'technology', source_entry=eid1)

        eid2 = mem.remember('Second mention of SQLite', entry_type='observation')
        builder.upsert_entity('SQLite', 'technology', source_entry=eid2)

        entity = mem._engine.get_entity(entity_id)
        assert entity.mention_count >= 2


class TestPhase2AddendumIntegration:
    """Verify knowledge synthesis works end-to-end."""

    def test_full_experiment_lifecycle(self, tmp_path):
        """hypothesis → method → result → complete → consolidate → knowledge item."""
        from aingram.store import MemoryStore
        from tests.conftest import MockLLM

        db = str(tmp_path / 'lifecycle.db')
        embedder = ClusterTestEmbedder()

        mem_a = MemoryStore(db, agent_name='agent-a', embedder=embedder)
        chain_id = mem_a.create_chain('Pool sizing experiment')
        mem_a.remember(
            'Connection pool exhaustion causes latency spikes',
            entry_type='hypothesis',
            chain_id=chain_id,
        )
        mem_a.remember(
            'Increase pool from 10 to 50, monitor for 1 hour',
            entry_type='method',
            chain_id=chain_id,
        )
        mem_a.remember(
            'Latency dropped from 2s to 200ms after pool increase',
            entry_type='result',
            chain_id=chain_id,
        )
        mem_a.complete_chain(chain_id, outcome='confirmed')
        mem_a.close()

        mem_b = MemoryStore(db, agent_name='agent-b', embedder=embedder)
        chain2 = mem_b.create_chain('Pool sizing experiment 2')
        mem_b.remember('Pool 100 also works', entry_type='result', chain_id=chain2)
        mem_b.complete_chain(chain2, outcome='confirmed')

        llm = MockLLM(response='Connection pools should be sized at 50-100x for optimal latency')
        result = mem_b.consolidate(llm=llm)

        assert result.chains_analyzed >= 2
        assert result.knowledge_synthesized >= 1
        assert len(mem_b.knowledge_items) >= 1

        ki = mem_b.knowledge_items[0]
        assert 'Connection pools' in ki['principle']
        assert len(ki['supporting_chains']) >= 2
        mem_b.close()

    def test_incomplete_chains_not_synthesized(self, mem):
        """Active chains should not be included in knowledge synthesis."""
        from tests.conftest import MockLLM

        chain1 = mem.create_chain('Completed')
        mem.remember('Result', entry_type='result', chain_id=chain1)
        mem.complete_chain(chain1, outcome='confirmed')

        chain2 = mem.create_chain('Still active')
        mem.remember('WIP', entry_type='observation', chain_id=chain2)

        llm = MockLLM(response='Should not synthesize with only 1 chain')
        result = mem.consolidate(llm=llm)

        assert result.knowledge_synthesized == 0

    def test_consolidation_result_has_all_fields(self, mem):
        """ConsolidationResult should have both Phase 1 and Phase 2 fields."""
        result = mem.consolidate()

        assert hasattr(result, 'memories_decayed')
        assert hasattr(result, 'contradictions_found')
        assert hasattr(result, 'memories_merged')

        assert hasattr(result, 'knowledge_synthesized')
        assert hasattr(result, 'chains_analyzed')
        assert hasattr(result, 'knowledge_reviewed')

    def test_clustered_synthesis_produces_knowledge_items(self, tmp_path):
        """Clustered synthesis should produce separate items for different topics."""
        from aingram.store import MemoryStore
        from tests.conftest import MockLLM

        db = str(tmp_path / 'clustered_item.db')
        embedder = ClusterTestEmbedder()

        mem_a = MemoryStore(db, agent_name='agent-a', embedder=embedder)
        c1 = mem_a.create_chain('Pool experiment A')
        mem_a.remember('Pool 50 works great', entry_type='result', chain_id=c1)
        mem_a.complete_chain(c1, outcome='confirmed')
        mem_a.close()

        mem_b = MemoryStore(db, agent_name='agent-b', embedder=embedder)
        c2 = mem_b.create_chain('Pool experiment B')
        mem_b.remember('Pool 50 is optimal', entry_type='result', chain_id=c2)
        mem_b.complete_chain(c2, outcome='confirmed')

        llm = MockLLM(response='Pools should be sized at 50')
        result = mem_b.consolidate(llm=llm)

        assert result.knowledge_synthesized >= 1
        assert len(mem_b.knowledge_items) >= 1
        mem_b.close()

    def test_synthesis_idempotent(self, tmp_path):
        """Running consolidate twice should update, not duplicate knowledge items."""
        from aingram.store import MemoryStore
        from tests.conftest import MockLLM

        db = str(tmp_path / 'idempotent.db')
        embedder = ClusterTestEmbedder()

        mem_a = MemoryStore(db, agent_name='agent-a', embedder=embedder)
        c1 = mem_a.create_chain('Pool run one')
        mem_a.remember('pool sizing 50 works in production', entry_type='result', chain_id=c1)
        mem_a.complete_chain(c1, outcome='confirmed')
        mem_a.close()

        mem_b = MemoryStore(db, agent_name='agent-b', embedder=embedder)
        c2 = mem_b.create_chain('Pool run two')
        mem_b.remember('pool sizing 50 confirmed again', entry_type='result', chain_id=c2)
        mem_b.complete_chain(c2, outcome='confirmed')

        llm = MockLLM(response='First principle')
        mem_b.consolidate(llm=llm)
        count_after_first = len(mem_b.knowledge_items)

        llm2 = MockLLM(response='Updated principle')
        mem_b.consolidate(llm=llm2)
        count_after_second = len(mem_b.knowledge_items)

        assert count_after_second == count_after_first
        assert count_after_first >= 1
        principles = {k['principle'] for k in mem_b.knowledge_items}
        assert 'Updated principle' in principles
        mem_b.close()
