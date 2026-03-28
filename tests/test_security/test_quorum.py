"""Synthesis quorum tests — requires chains from 2+ sessions."""

from tests.conftest import ClusterTestEmbedder, MockLLM


class TestSynthesisQuorum:
    def test_single_session_chains_skipped(self, tmp_path):
        """Chains from a single session should NOT produce knowledge items."""
        from aingram.store import MemoryStore

        db = tmp_path / 'quorum.db'
        store = MemoryStore(str(db), agent_name='agent-a', embedder=ClusterTestEmbedder())

        chain1 = store.create_chain('Pool experiment 1')
        store.remember('Pool size 50 works', entry_type='result', chain_id=chain1)
        store.complete_chain(chain1, outcome='confirmed')

        chain2 = store.create_chain('Pool experiment 2')
        store.remember('Pool 50 confirmed', entry_type='result', chain_id=chain2)
        store.complete_chain(chain2, outcome='confirmed')

        llm = MockLLM(response='Pools should be 50')
        result = store.consolidate(llm=llm)

        assert result.knowledge_synthesized == 0
        store.close()

    def test_multi_session_chains_synthesized(self, tmp_path):
        """Chains from 2 different sessions SHOULD produce knowledge items."""
        from aingram.store import MemoryStore

        db = tmp_path / 'quorum_multi.db'
        embedder = ClusterTestEmbedder()

        store_a = MemoryStore(str(db), agent_name='agent-a', embedder=embedder)
        chain1 = store_a.create_chain('Pool experiment A')
        store_a.remember('Pool size 50 works', entry_type='result', chain_id=chain1)
        store_a.complete_chain(chain1, outcome='confirmed')
        store_a.close()

        store_b = MemoryStore(str(db), agent_name='agent-b', embedder=embedder)
        chain2 = store_b.create_chain('Pool experiment B')
        store_b.remember('Pool 50 confirmed', entry_type='result', chain_id=chain2)
        store_b.complete_chain(chain2, outcome='confirmed')

        llm = MockLLM(response='Pools should be 50')
        result = store_b.consolidate(llm=llm)

        assert result.knowledge_synthesized >= 1
        store_b.close()
