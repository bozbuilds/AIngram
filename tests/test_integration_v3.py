# tests/test_integration_v3.py — end-to-end integration tests for Phase 1
"""End-to-end integration test for the Phase 1 trust foundation."""

import json

import pytest

from tests.conftest import MockEmbedder


@pytest.fixture
def mem(tmp_path):
    from aingram.store import MemoryStore

    db = tmp_path / 'integration.db'
    store = MemoryStore(str(db), agent_name='integration-test', embedder=MockEmbedder())
    yield store
    store.close()


def test_remember_recall_verify_cycle(mem):
    """Full cycle: remember → recall → verify."""
    id1 = mem.remember('The API latency spiked to 2 seconds at 3PM', entry_type='observation')
    id2 = mem.remember(
        'The spike is caused by connection pool exhaustion',
        entry_type='hypothesis',
        parent_entry_id=id1,
    )
    mem.remember(
        'Latency dropped to 200ms after increasing pool size',
        entry_type='result',
        parent_entry_id=id2,
    )

    results = mem.recall(entry_id=id2)
    assert len(results) == 1
    assert results[0].verified is True

    results = mem.recall('connection pool latency')
    assert len(results) >= 1

    vr = mem.verify()
    assert vr.valid is True
    assert vr.entries_checked == 3
    assert vr.errors == []


def test_reasoning_chain_workflow(mem):
    """Create a chain, add entries, recall by chain."""
    chain_id = mem.create_chain('temperature-experiment-001')

    mem.remember('Room temp is 22C', chain_id=chain_id, entry_type='observation')
    mem.remember('Higher temps cause slower response', chain_id=chain_id, entry_type='hypothesis')
    mem.remember('Test at 30C showed 15% slowdown', chain_id=chain_id, entry_type='result')

    results = mem.recall(chain_id=chain_id)
    assert len(results) == 3
    types = {r.entry.entry_type for r in results}
    assert types == {'observation', 'hypothesis', 'result'}


def test_multi_agent_shared_db(tmp_path):
    """Two agents share a DB, each with independent chains."""
    from aingram.store import MemoryStore

    db = str(tmp_path / 'shared.db')
    agent_a = MemoryStore(db, agent_name='researcher', embedder=MockEmbedder())
    agent_b = MemoryStore(db, agent_name='validator', embedder=MockEmbedder())

    id_a = agent_a.remember('Pool size should be 50', entry_type='hypothesis')

    results = agent_b.recall(entry_id=id_a)
    assert len(results) == 1
    assert results[0].verified is True

    agent_b.remember('Confirmed — pool size 50 works', entry_type='result', parent_entry_id=id_a)

    assert agent_a.verify().valid is True
    assert agent_b.verify().valid is True

    agent_a.close()
    agent_b.close()


def test_tamper_detection(mem):
    """Tampered entries are detected during verification."""
    mem.remember('original observation')
    mem.remember('second observation')

    mem._engine._conn.execute(
        'UPDATE memory_entries SET content = \'{"text":"forged"}\' '
        'WHERE sequence_num = 1 AND session_id = ?',
        (mem._session.session_id,),
    )
    mem._engine._conn.commit()

    vr = mem.verify()
    assert vr.valid is False
    assert any('Integrity check failed' in e for e in vr.errors)


def test_dict_content_roundtrip(mem):
    """Dict content survives remember → recall roundtrip."""
    original = {'metric': 'latency', 'value': 200, 'unit': 'ms'}
    eid = mem.remember(original, entry_type='result')
    results = mem.recall(entry_id=eid)
    stored = json.loads(results[0].entry.content)
    assert stored['metric'] == 'latency'
    assert stored['value'] == 200


def test_stats(mem):
    """Stats reflect current state."""
    assert mem.stats['entry_count'] == 0
    mem.remember('test entry')
    assert mem.stats['entry_count'] == 1
    assert mem.stats['embedding_dim'] == 768
    assert mem.stats['db_size_bytes'] > 0
