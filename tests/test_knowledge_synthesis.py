"""Knowledge synthesis tests — distill principles from completed chains."""

import json

from aingram.consolidation.knowledge import (
    KnowledgeSynthesizer,
    _cluster_chains,
    _cosine_similarity,
)
from aingram.types import ReasoningChain
from tests.conftest import ClusterTestEmbedder, MockEmbedder, MockLLM, ensure_test_session


class TestKnowledgeSynthesizer:
    def _setup_completed_chain(self, engine, chain_id, outcome='confirmed'):
        ensure_test_session(engine)
        engine.create_chain(
            ReasoningChain(
                chain_id=chain_id,
                title=f'Experiment {chain_id}',
                created_by_session='test-session',
                created_at='2026-01-01T00:00:00+00:00',
            )
        )
        engine.complete_chain(chain_id, outcome=outcome)

    def _store_chain_entry(
        self,
        engine,
        entry_id,
        chain_id,
        content,
        *,
        entry_type='result',
        sequence_num=1,
    ):
        ensure_test_session(engine)
        engine._conn.execute(
            'INSERT INTO memory_entries '
            '(entry_id, content_hash, entry_type, content, session_id, sequence_num, '
            'prev_entry_id, signature, created_at, importance, reasoning_chain_id) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                entry_id,
                f'hash-{entry_id}',
                entry_type,
                json.dumps({'text': content}),
                'test-session',
                sequence_num,
                None,
                f'sig-{entry_id}',
                '2026-01-01T00:00:00+00:00',
                0.5,
                chain_id,
            ),
        )
        engine._conn.commit()

    def test_no_synthesis_without_llm(self, engine):
        synth = KnowledgeSynthesizer(
            engine,
            llm=None,
            embedder=MockEmbedder(),
            session_id='test-session',
            fallback_to_single_cluster=True,
            require_quorum=False,
        )
        result = synth.synthesize()
        assert result.knowledge_synthesized == 0
        assert result.chains_analyzed == 0

    def test_no_synthesis_without_completed_chains(self, engine):
        llm = MockLLM(response='Some principle')
        synth = KnowledgeSynthesizer(
            engine,
            llm=llm,
            embedder=MockEmbedder(),
            session_id='test-session',
            fallback_to_single_cluster=True,
            require_quorum=False,
        )
        result = synth.synthesize()
        assert result.chains_analyzed == 0

    def test_synthesizes_from_completed_chains(self, engine):
        self._setup_completed_chain(engine, 'chain-1', 'confirmed')
        self._store_chain_entry(engine, 'e1', 'chain-1', 'Pool size 50 works', sequence_num=1)

        self._setup_completed_chain(engine, 'chain-2', 'confirmed')
        self._store_chain_entry(engine, 'e2', 'chain-2', 'Pool size 50 is optimal', sequence_num=2)

        llm = MockLLM(response='Connection pools should be sized at 50 for optimal performance')
        synth = KnowledgeSynthesizer(
            engine,
            llm=llm,
            embedder=MockEmbedder(),
            session_id='test-session',
            fallback_to_single_cluster=True,
            require_quorum=False,
        )
        result = synth.synthesize()

        assert result.chains_analyzed >= 2
        assert result.knowledge_synthesized >= 1

    def test_synthesized_item_stored_in_db(self, engine):
        self._setup_completed_chain(engine, 'chain-1', 'confirmed')
        self._store_chain_entry(engine, 'e1', 'chain-1', 'Finding A', sequence_num=1)

        self._setup_completed_chain(engine, 'chain-2', 'confirmed')
        self._store_chain_entry(engine, 'e2', 'chain-2', 'Finding B', sequence_num=2)

        llm = MockLLM(response='Synthesized principle from findings')
        synth = KnowledgeSynthesizer(
            engine,
            llm=llm,
            embedder=MockEmbedder(),
            session_id='test-session',
            fallback_to_single_cluster=True,
            require_quorum=False,
        )
        synth.synthesize()

        items = engine.get_knowledge_items()
        assert len(items) >= 1
        assert 'Synthesized principle' in items[0]['principle']

    def test_skips_chains_without_entries(self, engine):
        self._setup_completed_chain(engine, 'empty-chain', 'confirmed')

        llm = MockLLM(response='Should not be called')
        synth = KnowledgeSynthesizer(
            engine,
            llm=llm,
            embedder=MockEmbedder(),
            session_id='test-session',
            fallback_to_single_cluster=True,
            require_quorum=False,
        )
        result = synth.synthesize()
        assert result.knowledge_synthesized == 0

    def test_handles_llm_failure_gracefully(self, engine):
        self._setup_completed_chain(engine, 'chain-1', 'confirmed')
        self._store_chain_entry(engine, 'e1', 'chain-1', 'Some finding', sequence_num=1)

        self._setup_completed_chain(engine, 'chain-2', 'confirmed')
        self._store_chain_entry(engine, 'e2', 'chain-2', 'Another finding', sequence_num=2)

        llm = MockLLM(response='')
        synth = KnowledgeSynthesizer(
            engine,
            llm=llm,
            embedder=MockEmbedder(),
            session_id='test-session',
            fallback_to_single_cluster=True,
            require_quorum=False,
        )
        result = synth.synthesize()
        assert result.knowledge_synthesized == 0


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 0.001

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 0.001

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, b) == 0.0


class TestClusterChains:
    def test_similar_chains_cluster_together(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.99, 0.1, 0.0]
        v3 = [0.0, 0.0, 1.0]

        embeddings = [('c1', v1), ('c2', v2), ('c3', v3)]
        clusters = _cluster_chains(embeddings)

        assert len(clusters) == 1
        assert set(clusters[0]) == {'c1', 'c2'}

    def test_all_different_produces_no_clusters(self):
        embeddings = [
            ('c1', [1.0, 0.0, 0.0]),
            ('c2', [0.0, 1.0, 0.0]),
            ('c3', [0.0, 0.0, 1.0]),
        ]
        clusters = _cluster_chains(embeddings)
        assert clusters == []

    def test_all_similar_produces_one_cluster(self):
        embeddings = [
            ('c1', [1.0, 0.1, 0.0]),
            ('c2', [1.0, 0.0, 0.1]),
            ('c3', [1.0, 0.05, 0.05]),
        ]
        clusters = _cluster_chains(embeddings)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_two_distinct_groups(self):
        embeddings = [
            ('a1', [1.0, 0.0, 0.0]),
            ('a2', [0.98, 0.1, 0.0]),
            ('b1', [0.0, 1.0, 0.0]),
            ('b2', [0.1, 0.98, 0.0]),
        ]
        clusters = _cluster_chains(embeddings)
        assert len(clusters) == 2

    def test_empty_input(self):
        assert _cluster_chains([]) == []

    def test_single_chain_no_cluster(self):
        clusters = _cluster_chains([('c1', [1.0, 0.0, 0.0])])
        assert clusters == []


class TestClusteredSynthesis:
    def _ensure_session(self, engine):
        ensure_test_session(engine)

    def _setup_completed_chain(self, engine, chain_id, title, outcome='confirmed'):
        self._ensure_session(engine)
        engine.create_chain(
            ReasoningChain(
                chain_id=chain_id,
                title=title,
                created_by_session='test-session',
                created_at='2026-01-01T00:00:00+00:00',
            )
        )
        engine.complete_chain(chain_id, outcome=outcome)

    def _store_chain_entry(self, engine, entry_id, chain_id, content, *, sequence_num):
        self._ensure_session(engine)
        engine._conn.execute(
            'INSERT INTO memory_entries '
            '(entry_id, content_hash, entry_type, content, session_id, sequence_num, '
            'prev_entry_id, signature, created_at, importance, reasoning_chain_id) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                entry_id,
                f'hash-{entry_id}',
                'result',
                json.dumps({'text': content}),
                'test-session',
                sequence_num,
                None,
                f'sig-{entry_id}',
                '2026-01-01T00:00:00+00:00',
                0.5,
                chain_id,
            ),
        )
        engine._conn.commit()

    def test_requires_embedder(self, engine):
        llm = MockLLM(response='principle')
        embedder = ClusterTestEmbedder()
        synth = KnowledgeSynthesizer(
            engine,
            llm=llm,
            embedder=embedder,
            session_id='test-session',
            require_quorum=False,
        )
        assert synth is not None

    def test_produces_separate_items_for_different_topics(self, engine):
        self._setup_completed_chain(engine, 'pool-1', 'Pool sizing test 1')
        self._store_chain_entry(engine, 'p1', 'pool-1', 'Pool size 50 works', sequence_num=1)

        self._setup_completed_chain(engine, 'pool-2', 'Pool sizing test 2')
        self._store_chain_entry(engine, 'p2', 'pool-2', 'Pool size 50 optimal', sequence_num=2)

        self._setup_completed_chain(engine, 'cache-1', 'Redis cache experiment 1')
        self._store_chain_entry(
            engine,
            'c1',
            'cache-1',
            'Redis cache reduces latency by 80%',
            sequence_num=3,
        )

        self._setup_completed_chain(engine, 'cache-2', 'Redis cache experiment 2')
        self._store_chain_entry(
            engine,
            'c2',
            'cache-2',
            'Redis cache TTL should be 5 minutes',
            sequence_num=4,
        )

        llm = MockLLM(response='Synthesized principle')
        embedder = ClusterTestEmbedder()
        synth = KnowledgeSynthesizer(
            engine,
            llm=llm,
            embedder=embedder,
            session_id='test-session',
            require_quorum=False,
        )
        result = synth.synthesize()

        assert result.chains_analyzed >= 4
        assert result.knowledge_synthesized >= 1

    def test_upsert_updates_existing_item(self, engine):
        self._setup_completed_chain(engine, 'ch-1', 'Pool experiment 1')
        self._store_chain_entry(engine, 'e1', 'ch-1', 'Pool finding alpha', sequence_num=1)

        self._setup_completed_chain(engine, 'ch-2', 'Pool experiment 2')
        self._store_chain_entry(engine, 'e2', 'ch-2', 'Pool finding beta', sequence_num=2)

        llm = MockLLM(response='First synthesis')
        embedder = ClusterTestEmbedder()
        synth = KnowledgeSynthesizer(
            engine,
            llm=llm,
            embedder=embedder,
            session_id='test-session',
            require_quorum=False,
        )
        synth.synthesize()

        items_before = engine.get_knowledge_items()
        count_before = len(items_before)

        llm2 = MockLLM(response='Updated synthesis')
        synth2 = KnowledgeSynthesizer(
            engine,
            llm=llm2,
            embedder=ClusterTestEmbedder(),
            session_id='test-session',
            require_quorum=False,
        )
        synth2.synthesize()

        items_after = engine.get_knowledge_items()
        assert len(items_after) == count_before
        principles = {i['principle'] for i in items_after}
        assert 'Updated synthesis' in principles

    def test_new_chains_expand_existing_item(self, engine):
        self._setup_completed_chain(engine, 'ch-1', 'Experiment 1')
        self._store_chain_entry(engine, 'e1', 'ch-1', 'Pool finding 1', sequence_num=1)

        self._setup_completed_chain(engine, 'ch-2', 'Experiment 2')
        self._store_chain_entry(engine, 'e2', 'ch-2', 'Pool finding 2', sequence_num=2)

        llm = MockLLM(response='Two-chain principle')
        embedder = ClusterTestEmbedder()
        synth = KnowledgeSynthesizer(
            engine,
            llm=llm,
            embedder=embedder,
            session_id='test-session',
            require_quorum=False,
        )
        synth.synthesize()

        self._setup_completed_chain(engine, 'ch-3', 'Experiment 3')
        self._store_chain_entry(engine, 'e3', 'ch-3', 'Pool finding 3', sequence_num=3)

        llm2 = MockLLM(response='Three-chain principle')
        synth2 = KnowledgeSynthesizer(
            engine,
            llm=llm2,
            embedder=ClusterTestEmbedder(),
            session_id='test-session',
            require_quorum=False,
        )
        synth2.synthesize()

        items = engine.get_knowledge_items()
        assert len(items) == 1
        assert len(items[0]['supporting_chains']) >= 3

    def test_no_llm_still_returns_empty(self, engine):
        embedder = ClusterTestEmbedder()
        synth = KnowledgeSynthesizer(
            engine,
            llm=None,
            embedder=embedder,
            session_id='test-session',
            require_quorum=False,
        )
        result = synth.synthesize()
        assert result.knowledge_synthesized == 0
