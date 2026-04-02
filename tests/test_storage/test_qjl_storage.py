"""Storage engine QJL operations — store, coarse search, re-rank."""

import numpy as np
import pytest

from aingram.processing.qjl import NUM_PROJECTIONS, SEED, create_projection, encode
from aingram.storage.engine import StorageEngine
from tests.conftest import ensure_test_session


@pytest.fixture
def engine_with_qjl_entries(tmp_path):
    db = tmp_path / 'test.db'
    eng = StorageEngine(str(db))
    ensure_test_session(eng, 's1')
    dim = eng.get_embedding_dim()
    projection = create_projection(dim, NUM_PROJECTIONS, SEED)

    rng = np.random.RandomState(42)
    for i in range(10):
        vec = rng.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        vec_list = vec.tolist()
        packed, _ = encode(vec_list, projection)
        eng.store_entry(
            entry_id=f'e{i}',
            content_hash=f'h{i}',
            entry_type='observation',
            content=f'{{"text":"entry {i}"}}',
            session_id='s1',
            sequence_num=i + 1,
            prev_entry_id=None if i == 0 else f'e{i - 1}',
            signature='sig',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=vec_list,
            qjl_bits=packed,
        )
    yield eng, projection
    eng.close()


class TestStoreEntryQJL:
    def test_dual_write_populates_both_tables(self, engine_with_qjl_entries):
        eng, _ = engine_with_qjl_entries
        vec_count = eng._conn.execute('SELECT COUNT(*) FROM vec_entries').fetchone()[0]
        qjl_count = eng._conn.execute('SELECT COUNT(*) FROM vec_entries_qjl').fetchone()[0]
        assert vec_count == 10
        assert qjl_count == 10

    def test_store_entry_without_qjl_bits(self, tmp_path):
        db = tmp_path / 'test.db'
        eng = StorageEngine(str(db))
        ensure_test_session(eng, 's1')
        dim = eng.get_embedding_dim()
        vec = [0.1] * dim
        eng.store_entry(
            entry_id='e-no-qjl',
            content_hash='h0',
            entry_type='observation',
            content='{"text":"no qjl"}',
            session_id='s1',
            sequence_num=1,
            prev_entry_id=None,
            signature='sig',
            created_at='2026-01-01T00:00:00+00:00',
            embedding=vec,
        )
        vec_count = eng._conn.execute('SELECT COUNT(*) FROM vec_entries').fetchone()[0]
        qjl_count = eng._conn.execute('SELECT COUNT(*) FROM vec_entries_qjl').fetchone()[0]
        assert vec_count == 1
        assert qjl_count == 0
        eng.close()


class TestStoreEntriesQJLBatch:
    def test_batch_insert(self, tmp_path):
        db = tmp_path / 'test.db'
        eng = StorageEngine(str(db))
        dim = eng.get_embedding_dim()
        projection = create_projection(dim, NUM_PROJECTIONS, SEED)
        entries = []
        for i in range(5):
            vec = [float(i * 0.1 + j * 0.01) for j in range(dim)]
            packed, _ = encode(vec, projection)
            entries.append((f'batch-e{i}', packed))
        eng.store_entries_qjl_batch(entries)
        count = eng._conn.execute('SELECT COUNT(*) FROM vec_entries_qjl').fetchone()[0]
        assert count == 5
        eng.close()


class TestSearchQJLCoarse:
    def test_returns_results_sorted_by_distance(self, engine_with_qjl_entries):
        eng, projection = engine_with_qjl_entries
        dim = eng.get_embedding_dim()
        query = np.random.RandomState(99).randn(dim).astype(np.float32)
        query = query / np.linalg.norm(query)
        packed, _ = encode(query.tolist(), projection)
        results = eng.search_qjl_coarse(packed, limit=5)
        assert len(results) == 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_returns_entry_ids(self, engine_with_qjl_entries):
        eng, projection = engine_with_qjl_entries
        dim = eng.get_embedding_dim()
        query = np.random.RandomState(99).randn(dim).astype(np.float32)
        packed, _ = encode(query.tolist(), projection)
        results = eng.search_qjl_coarse(packed, limit=3)
        ids = [eid for eid, _ in results]
        assert all(eid.startswith('e') for eid in ids)


class TestRerankByVector:
    def test_reranks_candidates_by_cosine(self, engine_with_qjl_entries):
        eng, _ = engine_with_qjl_entries
        dim = eng.get_embedding_dim()
        query = np.random.RandomState(99).randn(dim).astype(np.float32)
        query = query / np.linalg.norm(query)
        candidate_ids = [f'e{i}' for i in range(10)]
        results = eng.rerank_by_vector(query.tolist(), candidate_ids, limit=3)
        assert len(results) == 3
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_empty_candidates(self, engine_with_qjl_entries):
        eng, _ = engine_with_qjl_entries
        dim = eng.get_embedding_dim()
        query = [0.1] * dim
        results = eng.rerank_by_vector(query, [], limit=5)
        assert results == []
