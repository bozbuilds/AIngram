# aingram/integrations/hermes.py
"""Hermes Agent integration for AIngram.

This provides a memory provider that Hermes agents can use to store
and recall facts with semantic understanding across sessions.

Unlike FTS5-only memory, AIngram uses hybrid vector+FTS5+knowledge graph
retrieval so the agent can find past facts using different wording than
how they were stored.

Usage from a Hermes agent skill::

    from aingram import MemoryStore
    from aingram.integrations.hermes import AIngramHermesMemory

    mem = AIngramHermesMemory("~/.hermes/aingram.db")
    mem.remember("The API rate limit is 100 req/min.")
    results = mem.recall("how fast can we call the api?")

As an MCP server (recommended for Hermes)::

    from aingram.mcp_server import create_server
    server = create_server(db_path="~/.hermes/aingram.db", require_auth=False)
    server.run(transport="stdio")

Then register in Hermes config.yaml::

    mcp_servers:
      aingram-memory:
        command: /path/to/aingram-mcp.sh
        enabled: true

Works with any MCP-compatible agent gateway, including OpenClaw.
中文: Aingram 可作为 Hermes Agent / OpenClaw 的长期记忆系统。
     支持语义搜索，用不同的词也能找到之前存储的事实。
"""

from __future__ import annotations

import os
from typing import Any

from aingram import MemoryStore


class AIngramHermesMemory:
    """Hermes Agent / OpenClaw memory provider backed by AIngram's hybrid retrieval.

    Provides the same remember/recall interface used by Hermes memory tools,
    with AIngram's FTS5 + vector + knowledge graph hybrid retrieval underneath.

    ／ Hermes Agent / OpenClaw 的长期记忆提供者
    ／ 支持混合检索（全文搜索 + 向量搜索 + 知识图谱），
    ／ 即使使用不同的词语也能找到之前存储的事实。

    Args:
        db_path: Path to the SQLite database file. ``~`` is expanded.
        **kwargs: Additional arguments forwarded to ``MemoryStore``.
    """

    def __init__(self, db_path: str = "agent_memory.db", **kwargs: Any) -> None:
        expanded = os.path.expanduser(db_path)
        self._store = MemoryStore(expanded, **kwargs)

    # ── Hermes-compatible interface ──────────────────────────────────────

    def remember(self, content: str, **kwargs: Any) -> str:
        """Store a fact. Returns the entry ID.

        存储一条事实记忆，返回该条记忆的唯一 ID。

        Args:
            content: The fact text to remember.
            **kwargs: Passed to ``MemoryStore.remember()``.
        """
        return self._store.remember(content, **kwargs)

    def recall(self, query: str, *, limit: int = 5, **kwargs: Any) -> list[dict]:
        """Search memory using semantic + keyword + graph hybrid retrieval.

        Returns results ranked by relevance, even when the query uses
        completely different words than the stored fact.

        语义搜索记忆。即使使用与存储时完全不同的词语，
        也能根据语义找到相关结果。

        Args:
            query: Search query (natural language). 搜索关键词（自然语言）。
            limit: Max results to return. 最大返回条数。
            **kwargs: Passed to ``MemoryStore.recall()``.
        """
        results = self._store.recall(query, limit=limit, **kwargs)
        return [
            {
                "entry_id": r.entry.entry_id,
                "content": r.entry.content,
                "score": r.score,
                "entry_type": str(r.entry.entry_type),
            }
            for r in results
        ]

    def close(self) -> None:
        """Close the underlying database connection. 关闭数据库连接。"""
        self._store.close()

    def __enter__(self) -> AIngramHermesMemory:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
