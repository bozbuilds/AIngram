# aingram/storage/engine.py
from __future__ import annotations

import json
import logging
import math
import sqlite3
import struct
import threading
import uuid
from datetime import UTC, datetime

import sqlite_vec

from aingram.exceptions import DatabaseError
from aingram.storage.schema import SCHEMA_VERSION, apply_schema, get_schema_version, vec_entries_ddl
from aingram.types import (
    AgentSession,
    Entity,
    EntryType,
    Memory,
    MemoryEntry,
    MemoryType,
    ReasoningChain,
    Relationship,
)

logger = logging.getLogger(__name__)

# SQLite default SQLITE_LIMIT_VARIABLE_NUMBER is 999; stay well under it.
_SQLITE_VAR_LIMIT = 900


def _sanitize_fts_query(query: str) -> str:
    """Escape FTS5 metacharacters by wrapping each term in double quotes."""
    terms = query.split()
    if not terms:
        return '""'
    return ' '.join(f'"{term.replace(chr(34), "")}"' for term in terms)


_ENTRY_COLUMNS = (
    'entry_id, content_hash, entry_type, content, session_id, sequence_num, '
    'prev_entry_id, signature, created_at, reasoning_chain_id, parent_entry_id, '
    'tags, metadata, confidence, importance, accessed_at, access_count, surprise, consolidated'
)

_ENTITY_COLUMNS = 'entity_id, name, entity_type, first_seen, last_seen, mention_count'

_VALID_OUTCOMES = frozenset(
    {
        'confirmed',
        'refuted',
        'partial',
        'inconclusive',
        'error',
    }
)


class StorageEngine:
    def __init__(self, db_path: str, *, embedding_dim: int | None = None) -> None:
        self._db_path = db_path
        self._closed = False
        self._lock = threading.Lock()
        self._requested_embedding_dim = embedding_dim
        self._create_dim = embedding_dim if embedding_dim is not None else 768
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute('PRAGMA journal_mode=WAL')
            self._conn.execute('PRAGMA busy_timeout=5000')
            self._conn.execute('PRAGMA foreign_keys=ON')
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            ver_before = get_schema_version(self._conn)
            apply_schema(self._conn, enable_vec=True, vec_embedding_dim=self._create_dim)
            self._ensure_embedding_dim_meta(ver_before)
        except sqlite3.Error as e:
            raise DatabaseError(str(e)) from e

        stored_version = get_schema_version(self._conn)
        if stored_version != SCHEMA_VERSION:
            raise DatabaseError(
                f'Schema version mismatch: expected {SCHEMA_VERSION}, got {stored_version}'
            )

        stored_ed = self.get_embedding_dim()
        if self._requested_embedding_dim is not None and stored_ed != self._requested_embedding_dim:
            raise DatabaseError(
                f'Database embedding_dim is {stored_ed} but open requested '
                f'embedding_dim={self._requested_embedding_dim}. '
                'Omit embedding_dim to use the value stored in the database.'
            )

    def _check_open(self) -> None:
        if self._closed:
            raise DatabaseError('Store is closed')

    def _ensure_embedding_dim_meta(self, ver_before: int | None) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM db_metadata WHERE key='embedding_dim'"
            ).fetchone()
            if row is not None:
                return
            fallback = str(self._create_dim)
            now = datetime.now(UTC).isoformat()
            self._conn.execute(
                "INSERT INTO db_metadata (key, value, updated_at) VALUES ('embedding_dim', ?, ?)",
                (fallback, now),
            )
            self._conn.commit()

    def get_embedding_dim(self) -> int:
        self._check_open()
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM db_metadata WHERE key='embedding_dim'"
            ).fetchone()
        if row is None:
            return 768
        return int(row[0])

    def has_memory(self, memory_id: str) -> bool:
        self._check_open()
        cursor = self._conn.execute('SELECT 1 FROM memories WHERE id = ?', (memory_id,))
        return cursor.fetchone() is not None

    def store_memory(
        self,
        *,
        content: str,
        memory_type: MemoryType,
        importance: float,
        agent_id: str,
        metadata: dict,
        embedding: list[float],
        memory_id: str | None = None,
        summary: str | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        accessed_at: datetime | None = None,
        access_count: int | None = None,
    ) -> str:
        self._check_open()
        expected = self.get_embedding_dim()
        if len(embedding) != expected:
            raise DatabaseError(
                f'Embedding length {len(embedding)} does not match store dimension {expected}'
            )
        memory_id = memory_id or str(uuid.uuid4())
        c_at = (created_at or datetime.now(UTC)).isoformat()
        u_at = (updated_at or datetime.now(UTC)).isoformat()
        a_at = accessed_at.isoformat() if accessed_at else None
        ac = 0 if access_count is None else access_count

        with self._lock:
            try:
                cursor = self._conn.execute(
                    """INSERT INTO memories
             (id, content, summary, memory_type, importance, agent_id, metadata,
              created_at, updated_at, accessed_at, access_count)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        memory_id,
                        content,
                        summary,
                        memory_type.value,
                        importance,
                        agent_id,
                        json.dumps(metadata),
                        c_at,
                        u_at,
                        a_at,
                        ac,
                    ),
                )
                rowid = cursor.lastrowid
                self._conn.execute(
                    'INSERT INTO memories_fts (rowid, content, summary) VALUES (?, ?, ?)',
                    (rowid, content, summary),
                )
                emb_blob = struct.pack(f'{len(embedding)}f', *embedding)
                self._conn.execute(
                    'INSERT INTO vec_memories (memory_id, embedding) VALUES (?, ?)',
                    (memory_id, emb_blob),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

        return memory_id

    def get_memory(self, memory_id: str) -> Memory | None:
        self._check_open()
        with self._lock:
            cursor = self._conn.execute(
                """SELECT id, content, summary, memory_type, importance, agent_id,
                  metadata, created_at, updated_at, accessed_at, access_count
           FROM memories WHERE id = ?""",
                (memory_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            now = datetime.now(UTC).isoformat()
            new_count = row[10] + 1
            self._conn.execute(
                'UPDATE memories SET accessed_at = ?, access_count = ? WHERE id = ?',
                (now, new_count, memory_id),
            )
            self._conn.commit()

        return Memory(
            id=row[0],
            content=row[1],
            summary=row[2],
            memory_type=MemoryType(row[3]),
            importance=row[4],
            agent_id=row[5],
            metadata=json.loads(row[6]),
            created_at=datetime.fromisoformat(row[7]),
            updated_at=datetime.fromisoformat(row[8]),
            accessed_at=datetime.fromisoformat(now),
            access_count=new_count,
        )

    def get_memories_batch(self, memory_ids: list[str]) -> dict[str, Memory]:
        """Fetch multiple memories by ID in a single query. Does NOT update access tracking."""
        self._check_open()
        if not memory_ids:
            return {}
        placeholders = ','.join('?' * len(memory_ids))
        cursor = self._conn.execute(
            f"""SELECT id, content, summary, memory_type, importance, agent_id,
              metadata, created_at, updated_at, accessed_at, access_count
           FROM memories WHERE id IN ({placeholders})""",
            memory_ids,
        )
        results = {}
        for row in cursor.fetchall():
            results[row[0]] = Memory(
                id=row[0],
                content=row[1],
                summary=row[2],
                memory_type=MemoryType(row[3]),
                importance=row[4],
                agent_id=row[5],
                metadata=json.loads(row[6]),
                created_at=datetime.fromisoformat(row[7]),
                updated_at=datetime.fromisoformat(row[8]),
                accessed_at=datetime.fromisoformat(row[9]) if row[9] else None,
                access_count=row[10],
            )
        return results

    def delete_memory(self, memory_id: str) -> bool:
        """Legacy v2 method — operates on the old `memories` table.
        Only works for databases that still have v2 tables (pre-migration).
        For v3, entries are immutable (trust-protected) and cannot be deleted."""
        self._check_open()
        with self._lock:
            try:
                rowid_row = self._conn.execute(
                    'SELECT rowid FROM memories WHERE id = ?', (memory_id,)
                ).fetchone()
                if rowid_row is None:
                    return False

                self._conn.execute('DELETE FROM memories_fts WHERE rowid = ?', (rowid_row[0],))
                self._conn.execute('DELETE FROM vec_memories WHERE memory_id = ?', (memory_id,))
                self._conn.execute('DELETE FROM entity_mentions WHERE entry_id = ?', (memory_id,))
                self._conn.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
                self._conn.commit()
                return True
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def search_fts(self, query: str, *, limit: int = 10) -> list[tuple[str, float]]:
        """Return list of (entry_id, bm25_score) for FTS matches.

        Note: FTS5 bm25() returns negative values (more negative = more relevant).
        Results are sorted most-relevant-first (ascending, since values are negative).
        """
        self._check_open()
        sanitized = _sanitize_fts_query(query)
        try:
            with self._lock:
                cursor = self._conn.execute(
                    """SELECT entry_id, bm25(entries_fts)
                       FROM entries_fts
                       WHERE entries_fts MATCH ?
                       ORDER BY bm25(entries_fts) ASC
                       LIMIT ?""",
                    (sanitized, limit),
                )
                return cursor.fetchall()
        except sqlite3.OperationalError:
            return []

    def search_vectors(
        self, query_embedding: list[float], *, limit: int = 10
    ) -> list[tuple[str, float]]:
        """Return list of (entry_id, distance) sorted by ascending distance."""
        self._check_open()
        blob = struct.pack(f'{len(query_embedding)}f', *query_embedding)
        with self._lock:
            cursor = self._conn.execute(
                """SELECT entry_id, distance
                   FROM vec_entries
                   WHERE embedding MATCH ?
                     AND k = ?
                   ORDER BY distance""",
                (blob, limit),
            )
            return cursor.fetchall()

    def search_vectors_filtered(
        self, query_embedding: list[float], candidate_ids: list[str], *, limit: int = 10
    ) -> list[tuple[str, float]]:
        """Cosine similarity search over a pre-filtered candidate set.

        Fetches embedding blobs for candidate_ids, computes cosine similarity
        in Python, and returns top results sorted by ascending distance.
        Returns empty list if candidate_ids is empty.
        """
        self._check_open()
        if not candidate_ids:
            return []

        query_norm = math.sqrt(sum(x * x for x in query_embedding))
        if query_norm == 0:
            return []

        dim = len(query_embedding)
        rows: list[tuple] = []
        with self._lock:
            for i in range(0, len(candidate_ids), _SQLITE_VAR_LIMIT):
                chunk = candidate_ids[i : i + _SQLITE_VAR_LIMIT]
                placeholders = ','.join('?' * len(chunk))
                rows += self._conn.execute(
                    f'SELECT entry_id, embedding FROM vec_entries '
                    f'WHERE entry_id IN ({placeholders})',
                    chunk,
                ).fetchall()

        results: list[tuple[str, float]] = []
        for entry_id, blob in rows:
            vec = struct.unpack(f'{dim}f', blob)
            dot = sum(a * b for a, b in zip(query_embedding, vec))
            vec_norm = math.sqrt(sum(x * x for x in vec))
            if vec_norm == 0:
                continue
            cosine_sim = dot / (query_norm * vec_norm)
            distance = 1.0 - cosine_sim
            results.append((entry_id, distance))

        results.sort(key=lambda x: x[1])
        return results[:limit]

    _VALID_AGENT_ROLES = frozenset({'reader', 'contributor', 'admin'})

    def create_agent_token(
        self,
        agent_name: str,
        role: str,
        *,
        public_key: str | None = None,
    ) -> dict:
        """Create an agent with a bearer token. Returns {'agent_id', 'token'}."""
        import hashlib
        import secrets
        import uuid

        if role not in self._VALID_AGENT_ROLES:
            valid = ', '.join(sorted(self._VALID_AGENT_ROLES))
            raise ValueError(f'Invalid role: {role}. Must be one of: {valid}')

        self._check_open()
        agent_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        now = datetime.now(UTC).isoformat()

        with self._lock:
            try:
                self._conn.execute(
                    'INSERT INTO agent_tokens (agent_id, agent_name, token_hash, role, '
                    'public_key, created_at) VALUES (?, ?, ?, ?, ?, ?)',
                    (agent_id, agent_name, token_hash, role, public_key, now),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

        return {'agent_id': agent_id, 'token': token}

    def verify_agent_token(self, token: str) -> dict | None:
        """Verify a bearer token. Returns agent dict or None."""
        import hashlib

        self._check_open()
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self._lock:
            row = self._conn.execute(
                'SELECT agent_id, agent_name, role, public_key FROM agent_tokens '
                'WHERE token_hash = ? AND revoked_at IS NULL',
                (token_hash,),
            ).fetchone()
        if row is None:
            return None
        return {
            'agent_id': row[0],
            'agent_name': row[1],
            'role': row[2],
            'public_key': row[3],
        }

    def revoke_agent_token(self, agent_name: str) -> None:
        self._check_open()
        now = datetime.now(UTC).isoformat()
        with self._lock:
            try:
                self._conn.execute(
                    'UPDATE agent_tokens SET revoked_at = ? WHERE agent_name = ? '
                    'AND revoked_at IS NULL',
                    (now, agent_name),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def list_agent_tokens(self) -> list[dict]:
        self._check_open()
        with self._lock:
            rows = self._conn.execute(
                'SELECT agent_id, agent_name, role, public_key, created_at, revoked_at '
                'FROM agent_tokens ORDER BY created_at'
            ).fetchall()
        return [
            {
                'agent_id': r[0],
                'agent_name': r[1],
                'role': r[2],
                'public_key': r[3],
                'created_at': r[4],
                'revoked_at': r[5],
            }
            for r in rows
        ]

    # compact_embeddings removed — v2 method referenced dropped tables (vec_memories, meta)

    def vacuum(self) -> None:
        self._check_open()
        with self._lock:
            try:
                self._conn.execute('VACUUM')
            except sqlite3.Error as e:
                raise DatabaseError(str(e)) from e

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        *,
        limit: int = 10,
        agent_id: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[tuple[str, float]]:
        """Hybrid vector + FTS search with Reciprocal Rank Fusion.
        Returns list of (memory_id, rrf_score) sorted descending.

        .. deprecated::
            No longer used by MemoryPipeline after Phase 3 refactor to 3-way RRF.
            Kept as a low-level utility. Consider removing in a future cleanup pass
            (this survived Phases 1-4).
        """
        self._check_open()
        from aingram.storage.queries import reciprocal_rank_fusion

        # Fetch wider candidate sets for RRF
        candidate_limit = limit * 5

        # Vector candidates
        vec_results = self.search_vectors(query_embedding, limit=candidate_limit)
        vec_ranked = [mid for mid, _dist in vec_results]

        # FTS candidates
        fts_results = self.search_fts(query_text, limit=candidate_limit)
        fts_ranked = [mid for mid, _score in fts_results]

        # RRF merge
        rrf_scores = reciprocal_rank_fusion([vec_ranked, fts_ranked])

        # Filter by agent_id and memory_type if specified
        if agent_id is not None or memory_type is not None:
            candidate_ids = list(rrf_scores.keys())
            placeholders = ','.join('?' * len(candidate_ids))
            query_parts = [f'SELECT id FROM memories WHERE id IN ({placeholders})']
            params: list[object] = list(candidate_ids)

            if agent_id is not None:
                query_parts.append('AND agent_id = ?')
                params.append(agent_id)
            if memory_type is not None:
                query_parts.append('AND memory_type = ?')
                params.append(memory_type.value)

            cursor = self._conn.execute(' '.join(query_parts), params)
            valid_ids = {row[0] for row in cursor.fetchall()}
            rrf_scores = {mid: score for mid, score in rrf_scores.items() if mid in valid_ids}

        # Sort by score descending and limit
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]

    def get_memory_count(self) -> int:
        self._check_open()
        cursor = self._conn.execute('SELECT COUNT(*) FROM memories')
        return cursor.fetchone()[0]

    def list_all_memories(self, *, agent_id: str | None = None) -> list[Memory]:
        """Return all memories without updating access tracking."""
        self._check_open()
        if agent_id is None:
            cursor = self._conn.execute(
                """SELECT id, content, summary, memory_type, importance, agent_id, metadata,
                          created_at, updated_at, accessed_at, access_count
                   FROM memories ORDER BY created_at"""
            )
        else:
            cursor = self._conn.execute(
                """SELECT id, content, summary, memory_type, importance, agent_id, metadata,
                          created_at, updated_at, accessed_at, access_count
                   FROM memories WHERE agent_id = ? ORDER BY created_at""",
                (agent_id,),
            )
        results: list[Memory] = []
        for row in cursor.fetchall():
            results.append(
                Memory(
                    id=row[0],
                    content=row[1],
                    summary=row[2],
                    memory_type=MemoryType(row[3]),
                    importance=row[4],
                    agent_id=row[5],
                    metadata=json.loads(row[6]),
                    created_at=datetime.fromisoformat(row[7]),
                    updated_at=datetime.fromisoformat(row[8]),
                    accessed_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    access_count=row[10],
                )
            )
        return results

    def list_all_relationships(self) -> list[Relationship]:
        self._check_open()
        cursor = self._conn.execute(
            """SELECT id, source_id, target_id, relation_type, fact, weight,
                      t_valid, t_invalid, source_memory
               FROM relationships"""
        )
        return [self._row_to_relationship(row) for row in cursor.fetchall()]

    def enqueue_task(self, *, task_type: str, payload: dict, priority: int = 0) -> str:
        self._check_open()
        task_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT INTO task_queue
                       (id, task_type, payload, status, priority, created_at)
                       VALUES (?, ?, ?, 'pending', ?, ?)""",
                    (task_id, task_type, json.dumps(payload), priority, now),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e
        return task_id

    def dequeue_task(self) -> tuple[str, str, dict] | None:
        """Atomically claim the highest-priority pending task.
        Returns (task_id, task_type, payload) or None."""
        self._check_open()
        with self._lock:
            try:
                cursor = self._conn.execute(
                    """SELECT id, task_type, payload FROM task_queue
                       WHERE status = 'pending'
                       ORDER BY priority DESC, created_at ASC
                       LIMIT 1""",
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                task_id, task_type, payload_str = row
                now = datetime.now(UTC).isoformat()
                self._conn.execute(
                    "UPDATE task_queue SET status = 'claimed', claimed_at = ? WHERE id = ?",
                    (now, task_id),
                )
                self._conn.commit()
                return task_id, task_type, json.loads(payload_str)
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def complete_task(self, task_id: str) -> None:
        self._check_open()
        now = datetime.now(UTC).isoformat()
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE task_queue SET status = 'completed', completed_at = ? WHERE id = ?",
                    (now, task_id),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def fail_task(self, task_id: str, error: str) -> None:
        self._check_open()
        now = datetime.now(UTC).isoformat()
        with self._lock:
            try:
                # Persist error in the payload JSON for later inspection
                row = self._conn.execute(
                    'SELECT payload FROM task_queue WHERE id = ?', (task_id,)
                ).fetchone()
                if row:
                    payload = json.loads(row[0])
                    payload['error'] = error
                    self._conn.execute(
                        """UPDATE task_queue
                           SET status = 'failed', completed_at = ?, payload = ?
                           WHERE id = ?""",
                        (now, json.dumps(payload), task_id),
                    )
                    self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e
        logger.warning('Task %s failed: %s', task_id, error)

    def get_pending_task_count(self) -> int:
        self._check_open()
        try:
            cursor = self._conn.execute("SELECT COUNT(*) FROM task_queue WHERE status = 'pending'")
            return cursor.fetchone()[0]
        except sqlite3.Error as e:
            raise DatabaseError(str(e)) from e

    # ------------------------------------------------------------------ v3 API

    @staticmethod
    def _row_to_entry(row: tuple) -> MemoryEntry:
        tags_raw = row[11]
        meta_raw = row[12]
        return MemoryEntry(
            entry_id=row[0],
            content_hash=row[1],
            entry_type=EntryType(row[2]),
            content=row[3],
            session_id=row[4],
            sequence_num=row[5],
            prev_entry_id=row[6],
            signature=row[7],
            created_at=row[8],
            reasoning_chain_id=row[9],
            parent_entry_id=row[10],
            tags=json.loads(tags_raw) if tags_raw else None,
            metadata=json.loads(meta_raw) if meta_raw else None,
            confidence=row[13],
            importance=row[14],
            accessed_at=row[15],
            access_count=row[16] if row[16] is not None else 0,
            surprise=row[17],
            consolidated=row[18] if row[18] is not None else 0,
        )

    def store_session(self, session: AgentSession) -> None:
        self._check_open()
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO agent_sessions
                       (session_id, agent_name, public_key, parent_session_id, created_at, metadata)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        session.session_id,
                        session.agent_name,
                        session.public_key,
                        session.parent_session_id,
                        session.created_at,
                        json.dumps(session.metadata) if session.metadata else None,
                    ),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def get_session(self, session_id: str) -> AgentSession | None:
        self._check_open()
        with self._lock:
            row = self._conn.execute(
                'SELECT session_id, agent_name, public_key, '
                'parent_session_id, created_at, metadata '
                'FROM agent_sessions WHERE session_id = ?',
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return AgentSession(
            session_id=row[0],
            agent_name=row[1],
            public_key=row[2],
            parent_session_id=row[3],
            created_at=row[4],
            metadata=json.loads(row[5]) if row[5] else None,
        )

    def store_entry(
        self,
        *,
        entry_id: str,
        content_hash: str,
        entry_type: str,
        content: str,
        session_id: str,
        sequence_num: int,
        prev_entry_id: str | None,
        signature: str,
        created_at: str,
        embedding: list[float],
        reasoning_chain_id: str | None = None,
        parent_entry_id: str | None = None,
        tags: list | None = None,
        metadata: dict | None = None,
        confidence: float | None = None,
        importance: float = 0.5,
        surprise: float | None = None,
        consolidated: int = 0,
        qjl_bits: bytes | None = None,
    ) -> None:
        self._check_open()
        blob = struct.pack(f'{len(embedding)}f', *embedding)
        with self._lock:
            try:
                self._conn.execute(
                    f"""INSERT INTO memory_entries
                       ({_ENTRY_COLUMNS})
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry_id,
                        content_hash,
                        entry_type,
                        content,
                        session_id,
                        sequence_num,
                        prev_entry_id,
                        signature,
                        created_at,
                        reasoning_chain_id,
                        parent_entry_id,
                        json.dumps(tags) if tags else None,
                        json.dumps(metadata) if metadata else None,
                        confidence,
                        importance,
                        None,
                        0,
                        surprise,
                        consolidated,
                    ),
                )
                # Extract human-readable text for FTS — avoid indexing JSON structural chars
                try:
                    parsed = json.loads(content)
                    fts_text = parsed.get('text', content) if isinstance(parsed, dict) else content
                except (json.JSONDecodeError, TypeError):
                    fts_text = content
                self._conn.execute(
                    'INSERT INTO entries_fts (content, entry_id) VALUES (?, ?)',
                    (fts_text, entry_id),
                )
                self._conn.execute(
                    'INSERT INTO vec_entries (entry_id, embedding) VALUES (?, ?)',
                    (entry_id, blob),
                )
                if qjl_bits is not None:
                    self._conn.execute(
                        'INSERT INTO vec_entries_qjl (entry_id, embedding) '
                        'VALUES (?, vec_bit(?))',
                        (entry_id, qjl_bits),
                    )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def get_verify_parents_batch(self, entry_ids: list[str]) -> dict[str, list[str]]:
        """Parent IDs for verify() — lite uses linear chain (prev_entry_id) only."""
        self._check_open()
        if not entry_ids:
            return {}
        placeholders = ','.join('?' * len(entry_ids))
        with self._lock:
            rows = self._conn.execute(
                f'SELECT entry_id, prev_entry_id FROM memory_entries '
                f'WHERE entry_id IN ({placeholders})',
                entry_ids,
            ).fetchall()
        result: dict[str, list[str]] = {eid: [] for eid in entry_ids}
        for eid, prev in rows:
            if prev:
                result[eid] = [prev]
        return result

    def get_entry_embedding(self, entry_id: str) -> bytes | None:
        """Return the float32 embedding blob for a single entry."""
        self._check_open()
        with self._lock:
            row = self._conn.execute(
                'SELECT embedding FROM vec_entries WHERE entry_id = ?',
                (entry_id,),
            ).fetchone()
        return row[0] if row else None

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        self._check_open()
        with self._lock:
            row = self._conn.execute(
                f'SELECT {_ENTRY_COLUMNS} FROM memory_entries WHERE entry_id = ?',
                (entry_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def get_entries_by_ids(self, entry_ids: list[str]) -> list[MemoryEntry]:
        self._check_open()
        if not entry_ids:
            return []
        placeholders = ','.join('?' * len(entry_ids))
        with self._lock:
            rows = self._conn.execute(
                f'SELECT {_ENTRY_COLUMNS} FROM memory_entries WHERE entry_id IN ({placeholders})',
                entry_ids,
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_entries_by_chain(self, chain_id: str, *, limit: int = 100) -> list[MemoryEntry]:
        self._check_open()
        with self._lock:
            rows = self._conn.execute(
                f'SELECT {_ENTRY_COLUMNS} FROM memory_entries '
                'WHERE reasoning_chain_id = ? ORDER BY created_at ASC LIMIT ?',
                (chain_id, limit),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_entries_by_session(self, session_id: str) -> list[MemoryEntry]:
        self._check_open()
        with self._lock:
            rows = self._conn.execute(
                f'SELECT {_ENTRY_COLUMNS} FROM memory_entries '
                'WHERE session_id = ? ORDER BY sequence_num',
                (session_id,),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_recent_entry_texts(self, *, chain_id: str | None = None, limit: int = 10) -> str:
        """Return concatenated human-readable text of recent entries for surprise context."""
        self._check_open()
        with self._lock:
            if chain_id:
                rows = self._conn.execute(
                    'SELECT content FROM memory_entries '
                    'WHERE reasoning_chain_id = ? ORDER BY created_at DESC LIMIT ?',
                    (chain_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    'SELECT content FROM memory_entries ORDER BY created_at DESC LIMIT ?',
                    (limit,),
                ).fetchall()

        texts: list[str] = []
        for (content,) in rows:
            try:
                parsed = json.loads(content)
                texts.append(parsed.get('text', content) if isinstance(parsed, dict) else content)
            except (json.JSONDecodeError, TypeError):
                texts.append(content)
        return '\n'.join(texts)

    def get_entry_count(self) -> int:
        self._check_open()
        with self._lock:
            return self._conn.execute('SELECT COUNT(*) FROM memory_entries').fetchone()[0]

    def update_entry_access(self, entry_id: str) -> None:
        self._check_open()
        now = datetime.now(UTC).isoformat()
        with self._lock:
            try:
                self._conn.execute(
                    """UPDATE memory_entries
                       SET accessed_at = ?, access_count = access_count + 1
                       WHERE entry_id = ?""",
                    (now, entry_id),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def create_chain(self, chain: ReasoningChain) -> None:
        self._check_open()
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT INTO reasoning_chains
                       (chain_id, title, status, outcome, created_by_session, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        chain.chain_id,
                        chain.title,
                        chain.status,
                        chain.outcome,
                        chain.created_by_session,
                        chain.created_at,
                    ),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def get_chain(self, chain_id: str) -> ReasoningChain | None:
        self._check_open()
        with self._lock:
            row = self._conn.execute(
                'SELECT chain_id, title, status, outcome, created_by_session, created_at '
                'FROM reasoning_chains WHERE chain_id = ?',
                (chain_id,),
            ).fetchone()
        if row is None:
            return None
        return ReasoningChain(
            chain_id=row[0],
            title=row[1],
            status=row[2],
            outcome=row[3],
            created_by_session=row[4],
            created_at=row[5],
        )

    def complete_chain(self, chain_id: str, *, outcome: str) -> None:
        """Mark a reasoning chain as completed with an outcome."""
        if outcome not in _VALID_OUTCOMES:
            raise ValueError(
                f'Invalid outcome: {outcome}. Must be one of: {", ".join(sorted(_VALID_OUTCOMES))}'
            )
        self._check_open()
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE reasoning_chains SET status = 'completed', outcome = ? "
                    'WHERE chain_id = ?',
                    (outcome, chain_id),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def get_completed_chains(self) -> list[ReasoningChain]:
        """Chains with status='completed' and a non-null outcome."""
        self._check_open()
        with self._lock:
            cursor = self._conn.execute(
                'SELECT chain_id, title, status, outcome, created_by_session, created_at '
                "FROM reasoning_chains WHERE status = 'completed' AND outcome IS NOT NULL "
                'ORDER BY created_at ASC',
            )
            return [
                ReasoningChain(
                    chain_id=row[0],
                    title=row[1],
                    status=row[2],
                    outcome=row[3],
                    created_by_session=row[4],
                    created_at=row[5],
                )
                for row in cursor.fetchall()
            ]

    def get_chain_count(self) -> int:
        self._check_open()
        with self._lock:
            return self._conn.execute('SELECT COUNT(*) FROM reasoning_chains').fetchone()[0]

    def get_all_chains(self, *, limit: int = 100) -> list[ReasoningChain]:
        self._check_open()
        with self._lock:
            rows = self._conn.execute(
                'SELECT chain_id, title, status, outcome, created_by_session, created_at '
                'FROM reasoning_chains ORDER BY created_at DESC LIMIT ?',
                (limit,),
            ).fetchall()
        return [
            ReasoningChain(
                chain_id=row[0],
                title=row[1],
                status=row[2],
                outcome=row[3],
                created_by_session=row[4],
                created_at=row[5],
            )
            for row in rows
        ]

    def get_entities_for_entry(self, entry_id: str) -> list[Entity]:
        """Return all Entity objects mentioned in a given memory entry."""
        self._check_open()
        with self._lock:
            rows = self._conn.execute(
                f'SELECT {_ENTITY_COLUMNS} FROM entities '
                'JOIN entity_mentions ON entities.entity_id = entity_mentions.entity_id '
                'WHERE entity_mentions.entry_id = ?',
                (entry_id,),
            ).fetchall()
        return [self._row_to_entity(row) for row in rows]

    def store_knowledge_item(
        self,
        *,
        principle: str,
        supporting_chains: list[str],
        confidence: float,
        session_id: str,
    ) -> str:
        """Store a synthesized knowledge item. Returns the knowledge_id."""
        self._check_open()
        knowledge_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._lock:
            try:
                self._conn.execute(
                    'INSERT INTO knowledge_items '
                    '(knowledge_id, principle, supporting_chains, confidence, '
                    'created_by_session, created_at) VALUES (?, ?, ?, ?, ?, ?)',
                    (
                        knowledge_id,
                        principle,
                        json.dumps(supporting_chains),
                        confidence,
                        session_id,
                        now,
                    ),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e
        return knowledge_id

    def get_knowledge_items(self, *, limit: int = 100) -> list[dict]:
        """Knowledge items ordered by confidence descending."""
        self._check_open()
        with self._lock:
            cursor = self._conn.execute(
                'SELECT knowledge_id, principle, supporting_chains, confidence, '
                'created_by_session, created_at, stability, difficulty, due_at, '
                'fsrs_state, last_review, reps, lapses '
                'FROM knowledge_items ORDER BY confidence DESC LIMIT ?',
                (limit,),
            )
            return [self._row_to_knowledge_item(row) for row in cursor.fetchall()]

    def _row_to_knowledge_item(self, row: tuple) -> dict:
        return {
            'knowledge_id': row[0],
            'principle': row[1],
            'supporting_chains': json.loads(row[2]) if row[2] else [],
            'confidence': row[3],
            'created_by_session': row[4],
            'created_at': row[5],
            'stability': row[6],
            'difficulty': row[7],
            'due_at': row[8],
            'fsrs_state': row[9] if row[9] is not None else 0,
            'last_review': row[10],
            'reps': row[11] if row[11] is not None else 0,
            'lapses': row[12] if row[12] is not None else 0,
        }

    def get_knowledge_item(self, knowledge_id: str) -> dict | None:
        """Get a single knowledge item by ID, including FSRS columns."""
        self._check_open()
        with self._lock:
            row = self._conn.execute(
                'SELECT knowledge_id, principle, supporting_chains, confidence, '
                'created_by_session, created_at, stability, difficulty, due_at, '
                'fsrs_state, last_review, reps, lapses '
                'FROM knowledge_items WHERE knowledge_id = ?',
                (knowledge_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_knowledge_item(row)

    def update_knowledge_item(
        self,
        knowledge_id: str,
        *,
        principle: str,
        supporting_chains: list[str],
        confidence: float,
    ) -> None:
        """Update an existing knowledge item's principle, chains, and confidence."""
        self._check_open()
        with self._lock:
            try:
                self._conn.execute(
                    'UPDATE knowledge_items SET principle = ?, supporting_chains = ?, '
                    'confidence = ? WHERE knowledge_id = ?',
                    (principle, json.dumps(supporting_chains), confidence, knowledge_id),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def update_chain_status(self, chain_id: str, status: str) -> None:
        self._check_open()
        with self._lock:
            try:
                self._conn.execute(
                    'UPDATE reasoning_chains SET status = ? WHERE chain_id = ?',
                    (status, chain_id),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    # ------------------------------------------------------------------ entity/relationship

    def _row_to_entity(self, row: tuple) -> Entity:
        return Entity(
            entity_id=row[0],
            name=row[1],
            entity_type=row[2],
            first_seen=datetime.fromisoformat(row[3]),
            last_seen=datetime.fromisoformat(row[4]),
            mention_count=row[5],
        )

    def _row_to_relationship(self, row: tuple) -> Relationship:
        return Relationship(
            id=row[0],
            source_id=row[1],
            target_id=row[2],
            relation_type=row[3],
            fact=row[4],
            weight=row[5],
            t_valid=datetime.fromisoformat(row[6]) if row[6] else None,
            t_invalid=datetime.fromisoformat(row[7]) if row[7] else None,
            source_memory=row[8],
        )

    def insert_entity(self, *, name: str, entity_type: str) -> str:
        self._check_open()
        entity_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._lock:
            try:
                self._conn.execute(
                    'INSERT INTO entities (entity_id, name, entity_type, first_seen, last_seen) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (entity_id, name, entity_type, now, now),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e
        return entity_id

    def upsert_entity(self, *, name: str, entity_type: str) -> str:
        """Atomically insert or update an entity by (name, entity_type).

        If the entity already exists, updates last_seen, increments mention_count,
        and returns the existing ID. If it does not exist, inserts a new row.
        """
        self._check_open()
        entity_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._lock:
            try:
                cursor = self._conn.execute(
                    'INSERT OR IGNORE INTO entities '
                    '(entity_id, name, entity_type, first_seen, last_seen, mention_count) '
                    'VALUES (?, ?, ?, ?, ?, 1)',
                    (entity_id, name, entity_type, now, now),
                )
                if cursor.rowcount == 0:
                    self._conn.execute(
                        'UPDATE entities SET last_seen = ?, mention_count = mention_count + 1 '
                        'WHERE name = ? AND entity_type = ?',
                        (now, name, entity_type),
                    )
                self._conn.commit()
                row = self._conn.execute(
                    'SELECT entity_id FROM entities WHERE name = ? AND entity_type = ?',
                    (name, entity_type),
                ).fetchone()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e
        return row[0]

    def get_entity(self, entity_id: str) -> Entity | None:
        self._check_open()
        cursor = self._conn.execute(
            f'SELECT {_ENTITY_COLUMNS} FROM entities WHERE entity_id = ?',
            (entity_id,),
        )
        row = cursor.fetchone()
        return self._row_to_entity(row) if row else None

    def get_entity_by_name(self, name: str, entity_type: str) -> Entity | None:
        self._check_open()
        cursor = self._conn.execute(
            f'SELECT {_ENTITY_COLUMNS} FROM entities WHERE name = ? AND entity_type = ?',
            (name, entity_type),
        )
        row = cursor.fetchone()
        return self._row_to_entity(row) if row else None

    def get_entities(self, *, limit: int = 100) -> list[Entity]:
        self._check_open()
        cursor = self._conn.execute(
            f'SELECT {_ENTITY_COLUMNS} FROM entities ORDER BY last_seen DESC LIMIT ?',
            (limit,),
        )
        return [self._row_to_entity(row) for row in cursor.fetchall()]

    def get_entity_count(self) -> int:
        self._check_open()
        cursor = self._conn.execute('SELECT COUNT(*) FROM entities')
        return cursor.fetchone()[0]

    def replace_entity(self, entity: Entity) -> None:
        """Upsert entity by primary key without DELETE+INSERT (preserves FK references)."""
        self._check_open()
        with self._lock:
            try:
                self._conn.execute(
                    'INSERT OR REPLACE INTO entities '
                    '(entity_id, name, entity_type, first_seen, last_seen, mention_count) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (
                        entity.entity_id,
                        entity.name,
                        entity.entity_type,
                        entity.first_seen.isoformat(),
                        entity.last_seen.isoformat(),
                        entity.mention_count,
                    ),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def replace_relationship(self, rel: Relationship) -> None:
        self._check_open()
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO relationships
                       (id, source_id, target_id, relation_type, fact, weight,
                        t_valid, t_invalid, source_memory)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        rel.id,
                        rel.source_id,
                        rel.target_id,
                        rel.relation_type,
                        rel.fact,
                        rel.weight,
                        rel.t_valid.isoformat() if rel.t_valid else None,
                        rel.t_invalid.isoformat() if rel.t_invalid else None,
                        rel.source_memory,
                    ),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def update_entity_last_seen(self, entity_id: str) -> None:
        self._check_open()
        now = datetime.now(UTC).isoformat()
        with self._lock:
            try:
                self._conn.execute(
                    'UPDATE entities SET last_seen = ? WHERE entity_id = ?',
                    (now, entity_id),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def insert_relationship(
        self,
        *,
        source_id: str,
        target_id: str,
        relation_type: str,
        fact: str | None = None,
        weight: float = 1.0,
        source_memory: str | None = None,
    ) -> str:
        self._check_open()
        rel_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT INTO relationships
                       (id, source_id, target_id, relation_type, fact, weight,
                        t_valid, source_memory)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (rel_id, source_id, target_id, relation_type, fact, weight, now, source_memory),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e
        return rel_id

    def get_relationships_for_entity(self, entity_id: str) -> list[Relationship]:
        self._check_open()
        cursor = self._conn.execute(
            """SELECT id, source_id, target_id, relation_type, fact, weight,
                      t_valid, t_invalid, source_memory
               FROM relationships
               WHERE source_id = ? OR target_id = ?""",
            (entity_id, entity_id),
        )
        return [self._row_to_relationship(row) for row in cursor.fetchall()]

    def link_entity_to_mention(
        self,
        entity_id: str,
        entry_id: str,
        confidence: float = 1.0,
    ) -> None:
        """Link an entity to a memory entry via entity_mentions."""
        self._check_open()
        with self._lock:
            try:
                self._conn.execute(
                    'INSERT OR IGNORE INTO entity_mentions '
                    '(entity_id, entry_id, confidence) VALUES (?, ?, ?)',
                    (entity_id, entry_id, confidence),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def get_entry_ids_for_entity(self, entity_id: str) -> list[str]:
        self._check_open()
        with self._lock:
            cursor = self._conn.execute(
                'SELECT entry_id FROM entity_mentions WHERE entity_id = ?',
                (entity_id,),
            )
            return [row[0] for row in cursor.fetchall()]

    def get_entry_ids_for_entities(self, entity_ids: list[str]) -> dict[str, list[str]]:
        self._check_open()
        if not entity_ids:
            return {}
        placeholders = ','.join('?' * len(entity_ids))
        with self._lock:
            cursor = self._conn.execute(
                'SELECT entity_id, entry_id FROM entity_mentions '
                f'WHERE entity_id IN ({placeholders})',
                entity_ids,
            )
            result: dict[str, list[str]] = {eid: [] for eid in entity_ids}
            for ent_id, entry_id in cursor.fetchall():
                result[ent_id].append(entry_id)
            return result

    def get_entity_ids_for_entry(self, entry_id: str) -> list[str]:
        self._check_open()
        with self._lock:
            cursor = self._conn.execute(
                'SELECT entity_id FROM entity_mentions WHERE entry_id = ?',
                (entry_id,),
            )
            return [row[0] for row in cursor.fetchall()]

    def find_entities_by_name(self, name: str) -> list[Entity]:
        """Find all entities matching a name (case-insensitive exact match)."""
        self._check_open()
        cursor = self._conn.execute(
            f'SELECT {_ENTITY_COLUMNS} FROM entities WHERE LOWER(name) = LOWER(?)',
            (name,),
        )
        return [self._row_to_entity(row) for row in cursor.fetchall()]

    def traverse_graph(
        self, start_entity_ids: list[str], max_hops: int = 2
    ) -> list[tuple[str, int]]:
        """Traverse knowledge graph via recursive CTE.

        Returns list of (entity_id, hop_distance) excluding starting entities.
        Only follows relationships where t_invalid IS NULL.
        """
        self._check_open()
        if not start_entity_ids:
            return []

        placeholders = ','.join('?' * len(start_entity_ids))
        sql = f"""
        WITH RECURSIVE graph_walk(entity_id, hop) AS (
            SELECT entity_id, 0 FROM entities WHERE entity_id IN ({placeholders})

            UNION

            SELECT
                CASE
                    WHEN r.source_id = gw.entity_id THEN r.target_id
                    ELSE r.source_id
                END,
                gw.hop + 1
            FROM graph_walk gw
            JOIN relationships r ON (r.source_id = gw.entity_id OR r.target_id = gw.entity_id)
            WHERE gw.hop < ?
            AND r.t_invalid IS NULL
        )
        SELECT entity_id, MIN(hop) as min_hop
        FROM graph_walk
        WHERE entity_id NOT IN ({placeholders})
        GROUP BY entity_id
        ORDER BY min_hop ASC
        """
        params = list(start_entity_ids) + [max_hops] + list(start_entity_ids)
        cursor = self._conn.execute(sql, params)
        return [(row[0], row[1]) for row in cursor.fetchall()]

    def filter_memory_ids(
        self,
        memory_ids: list[str],
        *,
        agent_id: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> set[str]:
        """Return subset of memory_ids matching the given filters."""
        self._check_open()
        if not memory_ids:
            return set()
        if agent_id is None and memory_type is None:
            return set(memory_ids)

        placeholders = ','.join('?' * len(memory_ids))
        sql = f'SELECT id FROM memories WHERE id IN ({placeholders})'
        params: list[object] = list(memory_ids)

        if agent_id is not None:
            sql += ' AND agent_id = ?'
            params.append(agent_id)
        if memory_type is not None:
            sql += ' AND memory_type = ?'
            params.append(memory_type.value)

        cursor = self._conn.execute(sql, params)
        return {row[0] for row in cursor.fetchall()}

    def batch_update_importance(self, updates: list[tuple[str, float]]) -> int:
        """Update importance scores for multiple memories in one transaction.
        Each tuple is (memory_id, new_importance). Returns count of rows updated.
        Missing or unknown memory IDs are silently skipped; check the return value
        against len(updates) if full application is required."""
        self._check_open()
        if not updates:
            return 0
        with self._lock:
            try:
                now = datetime.now(UTC).isoformat()
                count = 0
                for memory_id, importance in updates:
                    cursor = self._conn.execute(
                        'UPDATE memories SET importance = ?, updated_at = ? WHERE id = ?',
                        (importance, now, memory_id),
                    )
                    count += cursor.rowcount
                self._conn.commit()
                return count
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def get_memories_for_decay(self, *, limit: int = 5000) -> list[Memory]:
        """Get memories ordered by oldest access first, for decay processing.
        Does NOT update access tracking."""
        self._check_open()
        cursor = self._conn.execute(
            """SELECT id, content, summary, memory_type, importance, agent_id, metadata,
                      created_at, updated_at, accessed_at, access_count
               FROM memories
               ORDER BY COALESCE(accessed_at, created_at) ASC
               LIMIT ?""",
            (limit,),
        )
        results = []
        for row in cursor.fetchall():
            results.append(
                Memory(
                    id=row[0],
                    content=row[1],
                    summary=row[2],
                    memory_type=MemoryType(row[3]),
                    importance=row[4],
                    agent_id=row[5],
                    metadata=json.loads(row[6]),
                    created_at=datetime.fromisoformat(row[7]),
                    updated_at=datetime.fromisoformat(row[8]),
                    accessed_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    access_count=row[10],
                )
            )
        return results

    def invalidate_relationship(self, relationship_id: str) -> None:
        """Set t_invalid on a relationship (bi-temporal invalidation)."""
        self._check_open()
        with self._lock:
            try:
                now = datetime.now(UTC).isoformat()
                self._conn.execute(
                    'UPDATE relationships SET t_invalid = ? WHERE id = ?',
                    (now, relationship_id),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def store_cross_reference(
        self,
        *,
        source_entry_id: str,
        target_entry_id: str,
        reference_type: str,
        session_id: str,
        signature: str,
    ) -> int:
        """Store a signed cross-reference. Returns the auto-incremented id."""
        self._check_open()
        with self._lock:
            try:
                now = datetime.now(UTC).isoformat()
                cursor = self._conn.execute(
                    'INSERT INTO cross_references '
                    '(source_entry_id, target_entry_id, reference_type, '
                    'session_id, signature, created_at) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (source_entry_id, target_entry_id, reference_type, session_id, signature, now),
                )
                self._conn.commit()
                return int(cursor.lastrowid)
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def get_entity_entry_pairs(self) -> list[tuple[str, str]]:
        """Return all (entity_id, entry_id) pairs from entity_mentions."""
        self._check_open()
        with self._lock:
            cursor = self._conn.execute('SELECT entity_id, entry_id FROM entity_mentions')
            return cursor.fetchall()

    def get_entries_for_decay(self, *, limit: int = 5000) -> list[MemoryEntry]:
        """Get entries ordered by oldest access first, for decay processing."""
        self._check_open()
        with self._lock:
            cursor = self._conn.execute(
                f'SELECT {_ENTRY_COLUMNS} FROM memory_entries '
                'ORDER BY COALESCE(accessed_at, created_at) ASC LIMIT ?',
                (limit,),
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]

    def batch_update_entry_importance(self, updates: list[tuple[str, float]]) -> int:
        """Batch-update importance for multiple entries. Returns count updated."""
        self._check_open()
        if not updates:
            return 0
        with self._lock:
            try:
                count = 0
                for entry_id, new_importance in updates:
                    cursor = self._conn.execute(
                        'UPDATE memory_entries SET importance = ? WHERE entry_id = ?',
                        (new_importance, entry_id),
                    )
                    count += cursor.rowcount
                self._conn.commit()
                return count
            except sqlite3.Error as e:
                self._conn.rollback()
                raise DatabaseError(str(e)) from e

    def get_incomplete_task_count(self) -> int:
        """Count tasks that are still pending or claimed (in-progress).

        Use this instead of get_pending_task_count when polling for work completion.
        """
        self._check_open()
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM task_queue WHERE status IN ('pending', 'claimed')"
        )
        return cursor.fetchone()[0]

    def truncate_all_embeddings_to_dim(self, new_dim: int) -> int:
        """Destructively rewrites vec_entries blobs to the first new_dim floats.

        sqlite-vec vec0 tables fix float[N] at creation time; shrinking N requires
        dropping and recreating the virtual table, then re-inserting rows.
        """
        self._check_open()
        old = self.get_embedding_dim()
        if new_dim > old:
            raise DatabaseError(f'new_dim {new_dim} exceeds stored dimension {old}')
        if new_dim < 1:
            raise DatabaseError('new_dim must be >= 1')
        count = 0
        with self._lock:
            rows = self._conn.execute('SELECT entry_id, embedding FROM vec_entries').fetchall()
            rebuilt: list[tuple[str, bytes]] = []
            for entry_id, blob in rows:
                cur = len(blob) // 4
                if cur != old:
                    raise DatabaseError(
                        f'Embedding for {entry_id[:16]}… has length {cur}, expected {old}'
                    )
                vec = struct.unpack(f'{old}f', blob)
                new_blob = struct.pack(f'{new_dim}f', *vec[:new_dim])
                rebuilt.append((entry_id, new_blob))
                count += 1

            self._conn.execute('DROP TABLE IF EXISTS vec_entries')
            self._conn.execute(vec_entries_ddl(new_dim))
            for entry_id, new_blob in rebuilt:
                self._conn.execute(
                    'INSERT INTO vec_entries (entry_id, embedding) VALUES (?, ?)',
                    (entry_id, new_blob),
                )

            now = datetime.now(UTC).isoformat()
            self._conn.execute(
                'INSERT OR REPLACE INTO db_metadata (key, value, updated_at) VALUES (?, ?, ?)',
                ('embedding_dim', str(new_dim), now),
            )
            self._conn.commit()
        return count

    def store_entries_qjl_batch(self, entries: list[tuple[str, bytes]]) -> None:
        """Batch insert QJL-encoded rows into vec_entries_qjl."""
        self._check_open()
        with self._lock:
            for entry_id, packed_bits in entries:
                self._conn.execute(
                    'INSERT OR REPLACE INTO vec_entries_qjl (entry_id, embedding) '
                    'VALUES (?, vec_bit(?))',
                    (entry_id, packed_bits),
                )
            self._conn.commit()

    def search_qjl_coarse(
        self, query_bits: bytes, *, limit: int = 10
    ) -> list[tuple[str, float]]:
        """Hamming KNN on QJL bit vectors (coarse pass)."""
        self._check_open()
        with self._lock:
            cursor = self._conn.execute(
                """SELECT entry_id, distance
                   FROM vec_entries_qjl
                   WHERE embedding MATCH vec_bit(?)
                     AND k = ?
                   ORDER BY distance""",
                (query_bits, limit),
            )
            return cursor.fetchall()

    def rerank_by_vector(
        self, query_embedding: list[float], candidate_ids: list[str], *, limit: int = 10
    ) -> list[tuple[str, float]]:
        """Re-rank candidates by float32 cosine distance (pass 2).

        Delegates to search_vectors_filtered() — semantic wrapper for the
        QJL two-pass pipeline so callers express intent (re-rank) not mechanism.
        """
        return self.search_vectors_filtered(query_embedding, candidate_ids, limit=limit)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._conn.close()
