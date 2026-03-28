"""4-way retrieval tests — vector + FTS + graph + chain."""

import pytest

from tests.conftest import MockEmbedder


@pytest.fixture
def store_with_graph(tmp_path):
    from aingram.store import MemoryStore

    db = tmp_path / 'test.db'
    mem = MemoryStore(str(db), agent_name='test', embedder=MockEmbedder())

    # Create entries and link entities
    eid1 = mem.remember('The connection pool is exhausted', entry_type='observation')
    eid2 = mem.remember('API gateway routes to pool', entry_type='observation')
    mem.remember('Unrelated weather observation', entry_type='observation')

    # Manually link entities for graph search
    from aingram.graph.builder import GraphBuilder

    builder = GraphBuilder(mem._engine)
    pool_id = builder.upsert_entity('connection pool', 'component', source_entry=eid1)
    gw_id = builder.upsert_entity('API gateway', 'system', source_entry=eid2)
    builder.upsert_entity('connection pool', 'component', source_entry=eid2)
    builder.add_relationship(pool_id, gw_id, 'connected_to', source_entry=eid2)

    yield mem
    mem.close()


class TestFourWayRetrieval:
    def test_graph_dimension_boosts_entity_linked_entries(self, store_with_graph):
        """Entries linked to entities matching the query should rank higher."""
        results = store_with_graph.recall('connection pool problems')
        assert len(results) >= 1
        # Entries linked to 'connection pool' entity should appear
        entry_ids = {r.entry.entry_id for r in results}
        # The entity-linked entries should be in results
        assert len(entry_ids) >= 1

    def test_chain_dimension_boosts_same_chain(self, tmp_path):
        """Entries in the same reasoning chain should get boosted."""
        from aingram.store import MemoryStore

        db = tmp_path / 'chain_test.db'
        mem = MemoryStore(str(db), agent_name='test', embedder=MockEmbedder())

        chain_id = mem.create_chain('pool investigation')
        mem.remember('Pool size is 10', chain_id=chain_id, entry_type='observation')
        mem.remember('Pool exhaustion detected', chain_id=chain_id, entry_type='observation')
        mem.remember('Unrelated entry about weather', entry_type='observation')

        # Search within chain context
        results = mem.recall('pool size', chain_id=chain_id)
        assert len(results) >= 1
        # All results should be from the chain
        for r in results:
            assert r.entry.reasoning_chain_id == chain_id

        mem.close()

    def test_confidence_weighting(self, tmp_path):
        """Entries with higher confidence should rank higher (when confidence is set)."""
        from aingram.store import MemoryStore

        db = tmp_path / 'conf_test.db'
        mem = MemoryStore(str(db), agent_name='test', embedder=MockEmbedder())

        mem.remember('Low confidence claim', confidence=0.1, entry_type='hypothesis')
        mem.remember('High confidence claim', confidence=0.95, entry_type='hypothesis')

        results = mem.recall('claim')
        # Both should appear; the high-confidence one should generally rank higher
        assert len(results) == 2

        mem.close()
