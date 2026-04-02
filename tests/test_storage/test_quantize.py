import math
import struct

import pytest

from aingram.storage.engine import StorageEngine
from tests.conftest import ensure_test_session


@pytest.fixture
def engine_with_entries(tmp_path):
    db = tmp_path / 'test.db'
    eng = StorageEngine(str(db))
    ensure_test_session(eng, 's1')
    dim = eng.get_embedding_dim()
    for i in range(5):
        vec = [math.sin(i * 0.1 + j * 0.01) for j in range(dim)]
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
            embedding=vec,
        )
    yield eng
    eng.close()


def test_quantize_round_trip_accuracy(engine_with_entries):
    eng = engine_with_entries
    dim = eng.get_embedding_dim()

    originals = {}
    for i in range(5):
        blob = eng.get_entry_embedding(f'e{i}')
        assert blob is not None
        originals[f'e{i}'] = struct.unpack(f'{dim}f', blob)

    eng.quantize_all_embeddings()
    eng.rebuild_vec_from_int8()

    for eid, original in originals.items():
        blob = eng.get_entry_embedding(eid)
        assert blob is not None
        rebuilt = struct.unpack(f'{dim}f', blob)
        for j in range(dim):
            assert abs(original[j] - rebuilt[j]) < 0.01, (
                f'Dimension {j} of {eid}: {original[j]:.6f} vs {rebuilt[j]:.6f}'
            )


def test_quantize_is_quantized(engine_with_entries):
    eng = engine_with_entries
    assert not eng.is_quantized()
    eng.quantize_all_embeddings()
    assert eng.is_quantized()


def test_quantize_zero_range_vector(tmp_path):
    db = tmp_path / 'test.db'
    eng = StorageEngine(str(db))
    ensure_test_session(eng, 's1')
    dim = eng.get_embedding_dim()
    vec = [0.42] * dim
    eng.store_entry(
        entry_id='e-zero',
        content_hash='h-zero',
        entry_type='observation',
        content='{"text":"zero range"}',
        session_id='s1',
        sequence_num=1,
        prev_entry_id=None,
        signature='sig',
        created_at='2026-01-01T00:00:00+00:00',
        embedding=vec,
    )
    eng.quantize_all_embeddings()
    eng.rebuild_vec_from_int8()
    blob = eng.get_entry_embedding('e-zero')
    rebuilt = struct.unpack(f'{dim}f', blob)
    for j in range(dim):
        assert abs(rebuilt[j] - 0.42) < 0.01
    eng.close()
