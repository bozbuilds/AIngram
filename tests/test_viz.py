import pytest

from aingram.storage.engine import StorageEngine
from aingram.viz.server import VizHandler, create_server
from tests.conftest import ensure_test_session


@pytest.fixture
def viz_engine(tmp_path):
    db = tmp_path / 'viz_test.db'
    eng = StorageEngine(str(db))
    ensure_test_session(eng, 'test-session')
    yield eng
    eng.close()


def test_api_stats(viz_engine):
    stats = VizHandler._get_stats(viz_engine)
    assert 'entry_count' in stats
    assert 'entity_count' in stats
    assert 'chain_count' in stats


def test_api_entities(viz_engine):
    data = VizHandler._get_entities(viz_engine)
    assert 'nodes' in data
    assert 'edges' in data


def test_api_entry_not_found(viz_engine):
    result = VizHandler._get_entry(viz_engine, 'nonexistent-id')
    assert result is None


def test_create_server_returns_engine(tmp_path):
    db = tmp_path / 'srv.db'
    srv, eng = create_server(str(db), port=0)
    try:
        assert eng is not None
        assert srv.server_address[1] != 0
    finally:
        srv.server_close()
        eng.close()
