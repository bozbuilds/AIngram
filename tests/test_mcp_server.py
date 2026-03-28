"""MCP server tests — verify tool functions work via MemoryStore."""

import pytest

from tests.conftest import MockEmbedder


@pytest.fixture
def mcp_store(tmp_path):
    """Create a MemoryStore for MCP testing."""
    from aingram.store import MemoryStore

    db_path = str(tmp_path / 'mcp_test.db')
    store = MemoryStore(db_path, embedder=MockEmbedder(), agent_name='mcp-test')
    yield store
    store.close()


class TestMCPToolFunctions:
    """Test the underlying functions that MCP tools wrap."""

    def test_remember_returns_entry_id(self, mcp_store):
        entry_id = mcp_store.remember('Test memory', entry_type='observation')
        assert isinstance(entry_id, str)
        assert len(entry_id) == 64

    def test_recall_returns_results(self, mcp_store):
        mcp_store.remember('Connection pool exhaustion', entry_type='observation')
        results = mcp_store.recall('connection pool')
        assert isinstance(results, list)

    def test_verify_returns_valid(self, mcp_store):
        mcp_store.remember('Something', entry_type='observation')
        result = mcp_store.verify()
        assert result.valid is True

    def test_reference_works(self, mcp_store):
        eid1 = mcp_store.remember('A', entry_type='observation')
        eid2 = mcp_store.remember('B', entry_type='hypothesis')
        mcp_store.reference(source_id=eid1, target_id=eid2, reference_type='builds_on')

    def test_get_context_returns_string(self, mcp_store):
        mcp_store.remember('Pool sizing experiment results', entry_type='result')
        context = mcp_store.get_context('pool sizing', max_tokens=500)
        assert isinstance(context, str)

    def test_create_chain_returns_id(self, mcp_store):
        chain_id = mcp_store.create_chain('Experiment 1')
        assert isinstance(chain_id, str)


class TestMCPServerCreation:
    """Test MCP server can be instantiated."""

    def test_create_server_succeeds(self, tmp_path):
        from aingram.mcp_server import HAS_MCP, create_server

        if not HAS_MCP:
            pytest.skip('mcp package not installed')

        server = create_server(
            db_path=str(tmp_path / 'mcp_srv.db'),
            embedder=MockEmbedder(),
        )
        assert server is not None
