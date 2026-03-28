# aingram/storage/migration.py — v2 → v3 migration
"""Migrate v2 (memories-based) schema to v3 (trust-aware entries)."""

from __future__ import annotations

import json
import logging
import sqlite3
import struct

import rfc8785
import sqlite_vec

from aingram.trust.hashing import compute_content_hash
from aingram.trust.session import SessionManager
from aingram.trust.signing import compute_entry_id

logger = logging.getLogger(__name__)

TYPE_MAP = {
    'semantic': 'observation',
    'episodic': 'observation',
    'procedural': 'method',
    'entity': 'observation',
}


def migrate_v2_to_v3(db_path: str) -> int:
    """Migrate a v2 database to v3. Returns number of entries migrated.

    Reads all v2 data, transforms to v3 entries with a migration session,
    drops v2 tables, applies v3 schema, and inserts entries.
    """
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if 'memories' not in tables:
        conn.close()
        return 0

    # Read v2 data
    memories = conn.execute(
        'SELECT id, content, summary, memory_type, importance, agent_id, '
        'metadata, created_at, updated_at, accessed_at, access_count '
        'FROM memories ORDER BY created_at'
    ).fetchall()

    vectors: dict[str, bytes] = {}
    try:
        for row in conn.execute('SELECT memory_id, embedding FROM vec_memories'):
            vectors[row[0]] = row[1]
    except sqlite3.OperationalError:
        pass

    # v2→v3 migration: entity_memories was renamed to entity_mentions in schema.py
    old_links: list[tuple[str, str]] = []
    if 'entity_memories' in tables:
        old_links = conn.execute('SELECT entity_id, memory_id FROM entity_memories').fetchall()

    if not memories:
        conn.close()
        return 0

    # All v2 data is now in memory — safe to drop old tables before applying v3 schema
    for t in ['memories_fts', 'vec_memories', 'memories', 'entity_memories', 'meta']:
        conn.execute(f'DROP TABLE IF EXISTS {t}')
    conn.commit()
    conn.close()

    session = SessionManager(agent_name='migration', metadata={'source': 'v2'})

    entries: list[dict] = []
    uuid_to_entry_id: dict[str, str] = {}

    for mem in memories:
        (
            old_id,
            content,
            _summary,
            memory_type,
            importance,
            agent_id,
            metadata_json,
            created_at,
            _updated_at,
            accessed_at,
            access_count,
        ) = mem

        entry_type = TYPE_MAP.get(memory_type, 'observation')
        content_payload = {'text': content}
        content_hash = compute_content_hash(content_payload, entry_type)

        seq_num = session.next_sequence_num
        prev_id = session.prev_entry_id

        canonical_bytes = rfc8785.dumps(content_payload)
        dag_parents = [prev_id] if prev_id else []
        entry_id = compute_entry_id(
            content_data=canonical_bytes,
            parent_ids=dag_parents,
            pubkey_hex=session.public_key_hex,
        )
        signature = session.sign(entry_id)

        old_meta = json.loads(metadata_json) if metadata_json else {}
        old_meta['migrated_from'] = old_id
        if agent_id != 'default':
            old_meta['original_agent_id'] = agent_id

        canonical_content = rfc8785.dumps(content_payload).decode('utf-8')

        entries.append(
            {
                'entry_id': entry_id,
                'content_hash': content_hash,
                'entry_type': entry_type,
                'content': canonical_content,
                'session_id': session.session_id,
                'sequence_num': seq_num,
                'prev_entry_id': prev_id,
                'signature': signature,
                'created_at': created_at,
                'importance': importance,
                'metadata': old_meta,
                'old_id': old_id,
            }
        )

        uuid_to_entry_id[old_id] = entry_id
        session.advance(entry_id)

    from aingram.storage.engine import StorageEngine

    engine = StorageEngine(db_path)
    engine.store_session(session.to_agent_session())

    for entry_data in entries:
        old_id = entry_data.pop('old_id')
        vec_bytes = vectors.get(old_id)
        if vec_bytes:
            dim = len(vec_bytes) // 4
            embedding = list(struct.unpack(f'{dim}f', vec_bytes))
        else:
            embedding = [0.0] * engine.get_embedding_dim()

        engine.store_entry(embedding=embedding, **entry_data)

    for entity_id, old_memory_id in old_links:
        new_entry_id = uuid_to_entry_id.get(old_memory_id)
        if new_entry_id:
            try:
                engine._conn.execute(
                    'INSERT OR IGNORE INTO entity_mentions (entity_id, entry_id) VALUES (?, ?)',
                    (entity_id, new_entry_id),
                )
            except sqlite3.Error as exc:
                logger.warning(
                    'Could not link entity %s to entry %s: %s',
                    entity_id,
                    new_entry_id,
                    exc,
                )
    engine._conn.commit()

    logger.info('Migrated %d v2 memories to v3 entries', len(entries))
    engine.close()
    return len(entries)
