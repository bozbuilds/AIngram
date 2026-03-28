"""MCP server security integration tests."""

import pytest

from tests.conftest import MockEmbedder

try:
    import mcp.server.fastmcp  # noqa: F401

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

pytestmark = pytest.mark.skipif(not HAS_MCP, reason='mcp package not installed')


class TestMCPSecurity:
    def test_create_server_with_auth(self, tmp_path):
        """Server creation with require_auth=True (default) should succeed."""
        from aingram.mcp_server import create_server

        server = create_server(db_path=str(tmp_path / 'auth.db'), embedder=MockEmbedder())
        assert server is not None

    def test_create_server_without_auth(self, tmp_path):
        """Server with require_auth=False should work for local dev."""
        from aingram.mcp_server import create_server

        server = create_server(
            db_path=str(tmp_path / 'noauth.db'),
            embedder=MockEmbedder(),
            require_auth=False,
        )
        assert server is not None
