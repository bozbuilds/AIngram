"""QJL two-pass search integration tests."""

import numpy as np

from aingram import MemoryStore
from tests.conftest import MockEmbedder


class _OrthogonalTopicEmbedder:
    """Nearly-orthogonal unit directions per topic so KNN orders are predictable."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        rng = np.random.RandomState(0)
        q, _ = np.linalg.qr(rng.randn(dim, dim).astype(np.float64))
        self._basis = q.astype(np.float64)

    def embed(self, text: str) -> list[float]:
        t = text.lower()
        if 'python' in t or 'programming' in t:
            i = 0
        elif 'weather' in t or 'sunny' in t:
            i = 1
        else:
            i = 2
        return self._basis[i].tolist()


class TestDualWrite:
    def test_remember_writes_to_qjl_table(self, tmp_path):
        db = tmp_path / 'test.db'
        embedder = MockEmbedder(dim=768)
        mem = MemoryStore(str(db), embedder=embedder)
        mem.remember('test entry')
        qjl_count = mem._engine._conn.execute('SELECT COUNT(*) FROM vec_entries_qjl').fetchone()[0]
        assert qjl_count == 1
        mem.close()

    def test_multiple_remembers_all_dual_written(self, tmp_path):
        db = tmp_path / 'test.db'
        embedder = MockEmbedder(dim=768)
        mem = MemoryStore(str(db), embedder=embedder)
        for i in range(5):
            mem.remember(f'entry {i}')
        vec_count = mem._engine._conn.execute('SELECT COUNT(*) FROM vec_entries').fetchone()[0]
        qjl_count = mem._engine._conn.execute('SELECT COUNT(*) FROM vec_entries_qjl').fetchone()[0]
        assert vec_count == qjl_count == 5
        mem.close()


class TestThresholdGate:
    def test_below_threshold_uses_float32(self, tmp_path):
        """With < 25K entries, recall should work via standard float32 path."""
        db = tmp_path / 'test.db'
        embedder = MockEmbedder(dim=768)
        mem = MemoryStore(str(db), embedder=embedder)
        mem.remember('the sky is blue')
        results = mem.recall('sky color', verify=False)
        assert len(results) >= 1
        mem.close()

    def test_recall_returns_results_regardless_of_path(self, tmp_path):
        """Recall works whether threshold is met or not."""
        db = tmp_path / 'test.db'
        embedder = MockEmbedder(dim=768)
        mem = MemoryStore(str(db), embedder=embedder)
        for i in range(10):
            mem.remember(f'memory about topic {i}')
        results = mem.recall('topic 5', verify=False)
        assert len(results) > 0
        mem.close()


class TestSearchQuality:
    def test_top_result_consistency(self, tmp_path):
        """Two-pass and float32 should agree on top-1 for well-separated vectors."""
        db = tmp_path / 'test.db'
        mem = MemoryStore(str(db), embedder=_OrthogonalTopicEmbedder(dim=768))
        mem.remember('Python is a programming language')
        mem.remember('The weather is sunny today')
        mem.remember('Cats are independent animals')
        results = mem.recall('programming languages', verify=False, limit=1)
        assert len(results) == 1
        assert 'Python' in results[0].entry.content or 'programming' in results[0].entry.content
        mem.close()


class TestImportJsonBackfill:
    def test_imported_entries_get_qjl_bits(self, tmp_path):
        """After import_json, vec_entries_qjl should be populated."""
        src_db = tmp_path / 'src.db'
        dst_db = tmp_path / 'dst.db'
        export_path = tmp_path / 'export.json'

        embedder = MockEmbedder(dim=768)
        src = MemoryStore(str(src_db), embedder=embedder)
        for i in range(3):
            src.remember(f'exported entry {i}')
        src.export_json(str(export_path))
        src.close()

        dst = MemoryStore(str(dst_db), embedder=embedder)
        dst.import_json(str(export_path))
        qjl_count = dst._engine._conn.execute('SELECT COUNT(*) FROM vec_entries_qjl').fetchone()[0]
        assert qjl_count == 3
        dst.close()


class TestCompactQJLRebuild:
    def test_compact_rebuilds_qjl_for_new_dim(self, tmp_path):
        """After compact(), QJL table should be rebuilt for the new dimension."""
        db = tmp_path / 'test.db'
        embedder = MockEmbedder(dim=768)
        mem = MemoryStore(str(db), embedder=embedder)
        for i in range(3):
            mem.remember(f'compact test entry {i}')

        qjl_before = mem._engine._conn.execute('SELECT COUNT(*) FROM vec_entries_qjl').fetchone()[0]
        assert qjl_before == 3

        mem.compact(confirm=True, target_dim=256)

        qjl_after = mem._engine._conn.execute('SELECT COUNT(*) FROM vec_entries_qjl').fetchone()[0]
        assert qjl_after == 3

        assert mem._qjl_projection is not None
        assert mem._qjl_projection.shape[1] == 256

        results = mem.recall('compact test', verify=False, limit=1)
        assert len(results) >= 1
        mem.close()


class TestInt8Removal:
    def test_quantize_method_removed(self):
        assert not hasattr(MemoryStore, 'quantize')

    def test_int8_table_absent(self, tmp_path):
        db = tmp_path / 'test.db'
        embedder = MockEmbedder(dim=768)
        mem = MemoryStore(str(db), embedder=embedder)
        tables = [
            r[0]
            for r in mem._engine._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        assert 'vec_entries_int8' not in tables
        mem.close()


class TestImportJsonSearch:
    def test_search_works_on_imported_data(self, tmp_path):
        """Two-pass search works on imported data (not just row counts)."""
        src_db = tmp_path / 'src.db'
        dst_db = tmp_path / 'dst.db'
        export_path = tmp_path / 'export.json'

        embedder = MockEmbedder(dim=768)
        src = MemoryStore(str(src_db), embedder=embedder)
        src.remember('Python is a programming language')
        src.remember('The weather is sunny')
        src.export_json(str(export_path))
        src.close()

        dst = MemoryStore(str(dst_db), embedder=embedder)
        dst.import_json(str(export_path))
        results = dst.recall('programming', verify=False, limit=1)
        assert len(results) >= 1
        dst.close()
