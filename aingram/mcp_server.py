"""AIngram MCP server — tools wrapping MemoryStore (Lite)."""

from __future__ import annotations

import atexit
import json
import logging

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP

    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    FastMCP = None  # type: ignore[misc, assignment]


def create_server(
    *,
    db_path: str = './agent_memory.db',
    agent_name: str = 'mcp-agent',
    embedder=None,
    require_auth: bool = True,
):
    """Create an MCP server with AIngram tools (remember, recall, reference, …).

    When require_auth=True (default), all tools require a bearer token.
    Set require_auth=False for local development without authentication.
    """
    if not HAS_MCP:
        raise ImportError(
            'mcp package required for MCP server. Install with: pip install "aingram[mcp]"'
        )

    from aingram.exceptions import (
        AuthenticationError,
        AuthorizationError,
        InputBoundsError,
        RateLimitError,
    )
    from aingram.store import MemoryStore

    mcp = FastMCP('aingram')
    store = MemoryStore(db_path, agent_name=agent_name, embedder=embedder)
    atexit.register(store.close)

    middleware = None
    if require_auth:
        from aingram.security.middleware import SecurityMiddleware

        middleware = SecurityMiddleware(store._engine)

    def _sec_err(e: Exception) -> str:
        """Format security exception as error JSON.

        Clients MUST check for the 'is_error' key to distinguish error responses
        from valid tool output.
        """
        return json.dumps({'is_error': True, 'error': type(e).__name__, 'message': str(e)})

    @mcp.tool()
    def remember(
        content: str,
        token: str = '',
        entry_type: str = 'observation',
        chain_id: str | None = None,
        parent_hash: str | None = None,
        confidence: float | None = None,
    ) -> str:
        """Store experimental reasoning with automatic hashing, signing, and extraction."""
        params = {'content': content, 'confidence': confidence}
        try:
            if middleware:
                middleware.process('remember', token, params)
                confidence = params.get('confidence', confidence)
            entry_id = store.remember(
                content,
                entry_type=entry_type,
                chain_id=chain_id,
                parent_entry_id=parent_hash,
                confidence=confidence,
            )
            return json.dumps({'entry_id': entry_id})
        except (
            AuthenticationError,
            AuthorizationError,
            InputBoundsError,
            RateLimitError,
        ) as e:
            return _sec_err(e)

    @mcp.tool()
    def recall(
        query: str | None = None,
        token: str = '',
        entry_type: str | None = None,
        chain_id: str | None = None,
        limit: int = 20,
        verify: bool = True,
    ) -> str:
        """Search memory using combined FTS5 + vector similarity + structured filters."""
        params = {'limit': limit}
        try:
            if middleware:
                middleware.process('recall', token, params)
                limit = params['limit']
            results = store.recall(
                query=query,
                entry_type=entry_type,
                chain_id=chain_id,
                limit=limit,
                verify=verify,
            )
            return json.dumps(
                [
                    {
                        'entry_id': r.entry.entry_id,
                        'entry_type': str(r.entry.entry_type),
                        'content': r.entry.content,
                        'score': r.score,
                        'confidence': r.entry.confidence,
                    }
                    for r in results
                ]
            )
        except (
            AuthenticationError,
            AuthorizationError,
            InputBoundsError,
            RateLimitError,
        ) as e:
            return _sec_err(e)

    @mcp.tool()
    def get_related(
        entry_id: str,
        token: str = '',
        depth: int = 2,
    ) -> str:
        """Find experiments that build on, contradict, or refine a given finding."""
        params = {'depth': depth}
        try:
            if middleware:
                middleware.process('get_related', token, params)
                depth = params['depth']
            entity_ids = store._engine.get_entity_ids_for_entry(entry_id)
            if not entity_ids:
                return json.dumps([])
            traversed = store._engine.traverse_graph(entity_ids, max_hops=depth)
            all_eids = set(entity_ids) | {eid for eid, _hop in traversed}

            from aingram.graph.traversal import GraphTraversal

            traversal = GraphTraversal(store._engine)
            ranked = traversal.get_ranked_entry_ids(list(all_eids), limit=20)
            return json.dumps(ranked)
        except (
            AuthenticationError,
            AuthorizationError,
            InputBoundsError,
            RateLimitError,
        ) as e:
            return _sec_err(e)

    @mcp.tool()
    def reference(
        source_id: str,
        target_id: str,
        reference_type: str,
        token: str = '',
    ) -> str:
        """Create a signed reference between two memory entries."""
        try:
            caller = None
            if middleware:
                caller = middleware.process('reference', token, {})
            store.reference(
                source_id=source_id,
                target_id=target_id,
                reference_type=reference_type,
                caller=caller,
            )
            return json.dumps({'status': 'ok'})
        except (
            AuthenticationError,
            AuthorizationError,
            InputBoundsError,
            RateLimitError,
        ) as e:
            return _sec_err(e)

    @mcp.tool()
    def verify(
        session_id: str | None = None,
        token: str = '',
    ) -> str:
        """Walk hash chain from genesis to tip, checking every link and signature."""
        try:
            caller = None
            if middleware:
                caller = middleware.process('verify', token, {})
            result = store.verify(session_id=session_id, caller=caller)
            return json.dumps(
                {
                    'valid': result.valid,
                    'session_id': result.session_id,
                    'entries_checked': result.entries_checked,
                    'errors': result.errors,
                }
            )
        except (
            AuthenticationError,
            AuthorizationError,
            InputBoundsError,
            RateLimitError,
        ) as e:
            return _sec_err(e)

    @mcp.tool()
    def get_experiment_context(
        topic: str,
        token: str = '',
        max_tokens: int = 2000,
    ) -> str:
        """Retrieve relevant prior experiments, lessons, and contradictions for injection."""
        params = {'max_tokens': max_tokens}
        try:
            if middleware:
                middleware.process('get_experiment_context', token, params)
                max_tokens = params['max_tokens']
            return store.get_context(topic, max_tokens=max_tokens)
        except (
            AuthenticationError,
            AuthorizationError,
            InputBoundsError,
            RateLimitError,
        ) as e:
            return _sec_err(e)

    return mcp
