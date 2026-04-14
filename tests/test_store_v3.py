# tests/test_store_v3.py — MemoryStore v3 tests (Tasks 7-9)
import json
import os
from unittest.mock import patch

import pytest

from tests.conftest import ClusterTestEmbedder, MockEmbedder


@pytest.fixture
def store(tmp_path):
    from aingram.store import MemoryStore

    db = tmp_path / 'test.db'
    mem = MemoryStore(str(db), agent_name='test-agent', embedder=MockEmbedder())
    yield mem
    mem.close()


class TestInit:
    def test_creates_session(self, store):
        assert store._session.agent_name == 'test-agent'
        assert len(store._session.public_key_hex) == 64

    def test_creates_db_file(self, store, tmp_path):
        assert os.path.exists(tmp_path / 'test.db')


class TestRemember:
    def test_returns_entry_id(self, store):
        entry_id = store.remember('hello world')
        assert isinstance(entry_id, str)
        assert len(entry_id) == 64

    def test_string_content_wrapped(self, store):
        entry_id = store.remember('hello world')
        entry = store._engine.get_entry(entry_id)
        content = json.loads(entry.content)
        assert content == {'text': 'hello world'}

    def test_dict_content_stored(self, store):
        entry_id = store.remember({'result': 'latency dropped', 'metric': 200})
        entry = store._engine.get_entry(entry_id)
        content = json.loads(entry.content)
        assert content['metric'] == 200
        assert content['result'] == 'latency dropped'

    def test_explicit_entry_type(self, store):
        entry_id = store.remember('test', entry_type='hypothesis')
        entry = store._engine.get_entry(entry_id)
        assert entry.entry_type == 'hypothesis'

    def test_default_entry_type_is_observation(self, store):
        entry_id = store.remember('test')
        entry = store._engine.get_entry(entry_id)
        assert entry.entry_type == 'observation'

    def test_chain_linkage(self, store):
        id1 = store.remember('first entry')
        id2 = store.remember('second entry')
        e1 = store._engine.get_entry(id1)
        e2 = store._engine.get_entry(id2)
        assert e1.prev_entry_id is None
        assert e1.sequence_num == 1
        assert e2.prev_entry_id == id1
        assert e2.sequence_num == 2

    def test_content_hash_integrity(self, store):
        """Content hash can be recomputed from stored content."""
        from aingram.trust import compute_content_hash

        entry_id = store.remember('verify me')
        entry = store._engine.get_entry(entry_id)
        content_dict = json.loads(entry.content)
        expected_hash = compute_content_hash(content_dict, entry.entry_type)
        assert entry.content_hash == expected_hash

    def test_signature_verifiable(self, store):
        """Signature can be verified with session's public key."""
        from aingram.trust import verify_signature

        entry_id = store.remember('sign me')
        entry = store._engine.get_entry(entry_id)
        session = store._engine.get_session(entry.session_id)
        assert verify_signature(session.public_key, entry.entry_id, entry.signature)

    def test_with_chain_id(self, store):
        chain_id = store.create_chain('experiment')
        entry_id = store.remember('chain entry', chain_id=chain_id)
        entry = store._engine.get_entry(entry_id)
        assert entry.reasoning_chain_id == chain_id

    def test_with_metadata(self, store):
        entry_id = store.remember('meta entry', metadata={'source': 'api'})
        entry = store._engine.get_entry(entry_id)
        assert entry.metadata == {'source': 'api'}

    def test_with_confidence(self, store):
        entry_id = store.remember('confident', confidence=0.95)
        entry = store._engine.get_entry(entry_id)
        assert entry.confidence == 0.95

    def test_extraction_runs_with_explicit_entry_type(self, store):
        """Extraction should capture confidence even when caller sets entry_type."""
        from aingram.types import ExtractionResult

        class StubExtractor:
            def extract(self, text):
                return ExtractionResult(
                    entry_type='lesson',
                    confidence=0.82,
                    relevance=0.9,
                )

        store.set_extractor(StubExtractor())
        entry_id = store.remember('important finding', entry_type='hypothesis')
        entry = store._engine.get_entry(entry_id)
        # entry_type stays as caller-provided 'hypothesis'
        assert entry.entry_type == 'hypothesis'
        # confidence was filled in by extraction since caller did not provide one
        assert entry.confidence == 0.82

    def test_extraction_overrides_default_entry_type(self, store):
        """When entry_type is the default, extraction should override it."""
        from aingram.types import ExtractionResult

        class StubExtractor:
            def extract(self, text):
                return ExtractionResult(
                    entry_type='lesson',
                    confidence=0.75,
                    relevance=0.8,
                )

        store.set_extractor(StubExtractor())
        entry_id = store.remember('a lesson was learned')
        entry = store._engine.get_entry(entry_id)
        assert entry.entry_type == 'lesson'
        assert entry.confidence == 0.75

    def test_extraction_does_not_override_caller_confidence(self, store):
        """Caller-provided confidence takes precedence over extraction."""
        from aingram.types import ExtractionResult

        class StubExtractor:
            def extract(self, text):
                return ExtractionResult(
                    entry_type='observation',
                    confidence=0.5,
                    relevance=0.9,
                )

        store.set_extractor(StubExtractor())
        entry_id = store.remember('high confidence', confidence=0.99)
        entry = store._engine.get_entry(entry_id)
        assert entry.confidence == 0.99


class TestRecall:
    def test_recall_by_entry_id(self, store):
        eid = store.remember('specific entry')
        results = store.recall(entry_id=eid)
        assert len(results) == 1
        assert results[0].entry.entry_id == eid
        assert results[0].score == 1.0

    def test_recall_by_entry_id_not_found(self, store):
        results = store.recall(entry_id='nonexistent')
        assert results == []

    def test_recall_by_chain(self, store):
        chain_id = store.create_chain('experiment')
        store.remember('hypothesis about latency', chain_id=chain_id, entry_type='hypothesis')
        store.remember('result of experiment', chain_id=chain_id, entry_type='result')
        store.remember('unrelated entry')
        results = store.recall(chain_id=chain_id)
        assert len(results) == 2

    def test_recall_hybrid_search(self, store):
        store.remember('The database connection pool is exhausted')
        store.remember('Python decorators are syntactic sugar')
        results = store.recall('connection pool')
        assert len(results) >= 1
        # The pool-related entry should rank higher
        assert (
            'connection' in results[0].entry.content.lower()
            or 'pool' in results[0].entry.content.lower()
        )

    def test_recall_filter_by_type(self, store):
        store.remember('observation one', entry_type='observation')
        store.remember('lesson learned', entry_type='lesson')
        results = store.recall('test', entry_type='lesson')
        for r in results:
            assert r.entry.entry_type == 'lesson'

    def test_recall_verify_true(self, store):
        store.remember('verifiable entry')
        results = store.recall('verifiable')
        assert all(r.verified is True for r in results)

    def test_recall_verify_false(self, store):
        store.remember('unverified entry')
        results = store.recall('unverified', verify=False)
        assert all(r.verified is None for r in results)

    def test_recall_requires_query_or_id(self, store):
        with pytest.raises(ValueError, match='query is required'):
            store.recall()

    def test_recall_respects_limit(self, store):
        for i in range(10):
            store.remember(f'entry number {i}')
        results = store.recall('entry', limit=3)
        assert len(results) <= 3


def test_recall_uses_fts_prefilter_when_above_threshold(tmp_path, mock_embedder):
    """When FTS returns >= threshold candidates, filtered vector search is used."""
    from unittest.mock import patch

    from aingram.store import MemoryStore

    db = str(tmp_path / 'test.db')
    mem = MemoryStore(db, embedder=mock_embedder)

    for i in range(60):
        mem.remember(f'gradient descent step {i} in training loop')

    with (
        patch.object(
            mem._engine,
            'search_vectors_filtered',
            wraps=mem._engine.search_vectors_filtered,
        ) as mock_filtered,
        patch.object(
            mem._engine,
            'search_vectors',
            wraps=mem._engine.search_vectors,
        ) as mock_full,
    ):
        mem.recall('gradient descent', limit=10)
        assert mock_filtered.call_count == 1
        assert mock_full.call_count == 0

    mem.close()


class TestGetContext:
    def test_returns_string(self, store):
        store.remember('context entry one')
        store.remember('context entry two')
        context = store.get_context('context')
        assert isinstance(context, str)
        assert len(context) > 0

    def test_respects_token_budget(self, store):
        for i in range(20):
            store.remember(f'This is a longer entry with content about topic {i} ' * 5)
        context = store.get_context('topic', max_tokens=100)
        # With ~4 chars per token, 100 tokens ≈ 400 chars
        assert len(context) < 2000


class TestVerify:
    def test_verify_own_chain(self, store):
        store.remember('entry 1')
        store.remember('entry 2')
        store.remember('entry 3')
        result = store.verify()
        assert result.valid is True
        assert result.entries_checked == 3
        assert result.errors == []

    def test_verify_detects_tampered_content(self, store):
        """Direct DB modification breaks verification."""
        eid = store.remember('original content')
        # Tamper with content directly in DB
        store._engine._conn.execute(
            'UPDATE memory_entries SET content = \'{"text":"tampered"}\' WHERE entry_id = ?',
            (eid,),
        )
        store._engine._conn.commit()
        result = store.verify()
        assert result.valid is False
        assert len(result.errors) > 0

    def test_verify_empty_chain(self, store):
        result = store.verify()
        assert result.valid is True
        assert result.entries_checked == 0

    def test_verify_single_gap_limited_errors(self, store):
        """A single sequence gap should not produce cascading false positives."""
        store.remember('entry 1')
        store.remember('entry 2')
        store.remember('entry 3')
        row = store._engine._conn.execute(
            'SELECT entry_id FROM memory_entries WHERE sequence_num = 2 AND session_id = ?',
            (store._session.session_id,),
        ).fetchone()
        assert row is not None
        mid = row[0]
        store._engine._conn.execute('DELETE FROM vec_entries WHERE entry_id = ?', (mid,))
        store._engine._conn.execute('DELETE FROM entries_fts WHERE entry_id = ?', (mid,))
        store._engine._conn.execute(
            'DELETE FROM memory_entries WHERE entry_id = ?',
            (mid,),
        )
        store._engine._conn.commit()
        result = store.verify()
        assert result.valid is False
        # Should have limited errors — not a cascade of false positives for every entry after
        # the gap
        seq_gap_errors = [e for e in result.errors if 'Sequence gap' in e]
        assert len(seq_gap_errors) == 1  # only ONE gap error, not two


class TestEntryTypeValidation:
    def test_invalid_entry_type_raises(self, store):
        """Invalid entry_type should raise ValueError before hitting the DB."""
        with pytest.raises(ValueError, match='invalid_type'):
            store.remember('test', entry_type='invalid_type')


class TestCreateChain:
    def test_returns_chain_id(self, store):
        chain_id = store.create_chain('experiment-001')
        assert isinstance(chain_id, str)
        assert len(chain_id) == 32  # uuid4 hex

    def test_chain_exists_after_creation(self, store):
        chain_id = store.create_chain('my experiment')
        chain = store._engine.get_chain(chain_id)
        assert chain is not None
        assert chain.title == 'my experiment'
        assert chain.status == 'active'
        assert chain.created_by_session == store._session.session_id

    def test_entries_link_to_chain(self, store):
        chain_id = store.create_chain('test chain')
        eid = store.remember('chain entry', chain_id=chain_id)
        entry = store._engine.get_entry(eid)
        assert entry.reasoning_chain_id == chain_id


class TestCompleteChain:
    def test_complete_chain_sets_outcome(self, store):
        chain_id = store.create_chain('Test experiment')
        store.remember('Hypothesis', entry_type='hypothesis', chain_id=chain_id)
        store.remember('Result', entry_type='result', chain_id=chain_id)

        store.complete_chain(chain_id, outcome='confirmed')

        chain = store._engine.get_chain(chain_id)
        assert chain.status == 'completed'
        assert chain.outcome == 'confirmed'

    def test_complete_chain_invalid_outcome_raises(self, store):
        chain_id = store.create_chain('Bad experiment')

        with pytest.raises(ValueError, match='Invalid outcome'):
            store.complete_chain(chain_id, outcome='bogus')


class TestKnowledgeItems:
    def test_knowledge_items_empty(self, store):
        assert store.knowledge_items == []

    def test_consolidate_includes_knowledge_synthesis(self, tmp_path):
        from aingram.store import MemoryStore
        from tests.conftest import MockLLM

        db = tmp_path / 'synthesis_test.db'
        embedder = ClusterTestEmbedder()
        with MemoryStore(str(db), agent_name='agent-a', embedder=embedder) as sa:
            chain1 = sa.create_chain('Pool experiment 1')
            sa.remember('Pool 50 works', entry_type='result', chain_id=chain1)
            sa.complete_chain(chain1, outcome='confirmed')

        with MemoryStore(str(db), agent_name='agent-b', embedder=embedder) as sb:
            chain2 = sb.create_chain('Pool experiment 2')
            sb.remember('Pool 50 optimal', entry_type='result', chain_id=chain2)
            sb.complete_chain(chain2, outcome='confirmed')

            llm = MockLLM(response='Pools should be sized at 50')
            result = sb.consolidate(llm=llm)

            assert result.chains_analyzed >= 2
            assert result.knowledge_synthesized >= 1
            assert len(sb.knowledge_items) >= 1

    def test_consolidate_without_llm_skips_synthesis(self, store):
        chain1 = store.create_chain('Experiment')
        store.remember('Finding', entry_type='result', chain_id=chain1)
        store.complete_chain(chain1, outcome='confirmed')

        result = store.consolidate()
        assert result.knowledge_synthesized == 0
        assert result.chains_analyzed == 0


class TestReference:
    def test_reference_creates_cross_reference(self, store):
        eid1 = store.remember('First observation', entry_type='observation')
        eid2 = store.remember('Builds on first', entry_type='hypothesis')

        store.reference(source_id=eid1, target_id=eid2, reference_type='builds_on')

        row = store._engine._conn.execute(
            'SELECT reference_type FROM cross_references '
            'WHERE source_entry_id = ? AND target_entry_id = ?',
            (eid1, eid2),
        ).fetchone()
        assert row is not None
        assert row[0] == 'builds_on'

    def test_reference_invalid_type_raises(self, store):
        eid1 = store.remember('First', entry_type='observation')
        eid2 = store.remember('Second', entry_type='observation')

        with pytest.raises(ValueError, match='Invalid reference_type'):
            store.reference(source_id=eid1, target_id=eid2, reference_type='invalid')

    def test_reference_nonexistent_entry_raises(self, store):
        eid1 = store.remember('Exists', entry_type='observation')

        with pytest.raises(ValueError, match='not found'):
            store.reference(source_id=eid1, target_id='nonexistent', reference_type='supports')

    def test_reference_is_signed(self, store):
        eid1 = store.remember('A', entry_type='observation')
        eid2 = store.remember('B', entry_type='observation')

        store.reference(source_id=eid1, target_id=eid2, reference_type='supports')

        row = store._engine._conn.execute(
            'SELECT signature, session_id FROM cross_references '
            'WHERE source_entry_id = ? AND target_entry_id = ?',
            (eid1, eid2),
        ).fetchone()
        assert row[0] is not None
        assert len(row[0]) == 128
        assert row[1] == store._session.session_id


class TestEntitiesProperty:
    def test_entities_returns_list(self, store):
        assert isinstance(store.entities, list)

    def test_entities_returns_extracted_entities(self, store):
        from aingram.graph.builder import GraphBuilder

        builder = GraphBuilder(store._engine)
        eid = store.remember('Test content', entry_type='observation')
        builder.upsert_entity('TestEntity', 'concept', source_entry=eid)

        entities = store.entities
        assert len(entities) >= 1
        assert any(e.name == 'TestEntity' for e in entities)


class TestStats:
    def test_stats_keys(self, store):
        stats = store.stats
        assert 'entry_count' in stats
        assert 'db_size_bytes' in stats
        assert 'embedding_dim' in stats

    def test_entry_count_increments(self, store):
        assert store.stats['entry_count'] == 0
        store.remember('test')
        assert store.stats['entry_count'] == 1


class TestContextManager:
    def test_context_manager(self, tmp_path):
        from aingram.store import MemoryStore

        db = tmp_path / 'ctx.db'
        with MemoryStore(str(db), embedder=MockEmbedder()) as mem:
            mem.remember('test')
        # Should not raise after close


class TestConsolidate:
    def test_consolidate_basic(self, store):
        """Consolidate should run without errors even with no data."""
        from aingram.types import ConsolidationResult

        result = store.consolidate()
        assert isinstance(result, ConsolidationResult)

    def test_consolidate_applies_decay(self, store):
        from aingram.types import ConsolidationResult

        # Add old entries
        store.remember('old observation')
        result = store.consolidate()
        assert isinstance(result, ConsolidationResult)

    def test_consolidate_passes_llm_through(self, store):
        """consolidate(llm=...) should forward LLM to detector and merger."""
        from tests.conftest import MockLLM

        llm = MockLLM()
        from aingram.consolidation.knowledge import KnowledgeSynthesizer, SynthesisResult

        with (
            patch(
                'aingram.consolidation.contradiction.LLMContradictionClassifier.__init__',
                return_value=None,
            ) as mock_classifier_init,
            patch(
                'aingram.consolidation.contradiction.ContradictionDetector.__init__',
                return_value=None,
            ) as mock_detector_init,
            patch(
                'aingram.consolidation.contradiction.ContradictionDetector.detect_and_resolve',
            ) as mock_detect,
            patch(
                'aingram.consolidation.merger.MemoryMerger.__init__',
                return_value=None,
            ) as mock_merger_init,
            patch(
                'aingram.consolidation.merger.MemoryMerger.merge_similar',
            ) as mock_merge,
            patch.object(
                KnowledgeSynthesizer,
                'synthesize',
                return_value=SynthesisResult(0, 0),
            ),
        ):
            from aingram.consolidation.contradiction import ContradictionResult
            from aingram.consolidation.merger import MergeResult

            mock_detect.return_value = ContradictionResult(
                contradictions_found=0,
                contradictions_resolved=0,
            )
            mock_merge.return_value = MergeResult(
                memories_merged=0,
                summaries_created=0,
            )

            store.consolidate(llm=llm)

            mock_classifier_init.assert_called_once_with(llm)

            _, kwargs = mock_detector_init.call_args
            assert 'classifier' in kwargs

            # Verify LLM was passed to MemoryMerger
            _, kwargs = mock_merger_init.call_args
            assert kwargs['llm'] is llm
