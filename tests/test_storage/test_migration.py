# tests/test_storage/test_migration.py
import sqlite3
import struct

import sqlite_vec

from aingram.storage.migration import migrate_v2_to_v3


def _create_v2_db(path, num_memories=3):
    """Create a v2-schema database with test data."""
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute('CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)')
    conn.execute("INSERT INTO meta VALUES ('schema_version', '2')")
    conn.execute("INSERT INTO meta VALUES ('embedding_dim', '768')")

    conn.execute("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, content TEXT NOT NULL, summary TEXT,
            memory_type TEXT NOT NULL DEFAULT 'semantic',
            importance REAL NOT NULL DEFAULT 0.5,
            agent_id TEXT NOT NULL DEFAULT 'default',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            accessed_at TEXT, access_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute('CREATE VIRTUAL TABLE memories_fts USING fts5(content, summary)')
    conn.execute(
        'CREATE VIRTUAL TABLE vec_memories USING vec0('
        'memory_id TEXT PRIMARY KEY, embedding float[768])'
    )
    conn.execute("""
        CREATE TABLE entities (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, entity_type TEXT,
            description TEXT, properties TEXT NOT NULL DEFAULT '{}',
            importance REAL NOT NULL DEFAULT 0.5,
            first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE entity_memories (
            entity_id TEXT NOT NULL, memory_id TEXT NOT NULL,
            PRIMARY KEY (entity_id, memory_id)
        )
    """)
    conn.execute("""
        CREATE TABLE relationships (
            id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT,
            relation_type TEXT, fact TEXT, weight REAL DEFAULT 1.0,
            t_valid TEXT, t_invalid TEXT, source_memory TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE task_queue (
            id TEXT PRIMARY KEY, task_type TEXT NOT NULL,
            payload TEXT DEFAULT '{}', status TEXT DEFAULT 'pending',
            priority INTEGER DEFAULT 0, created_at TEXT,
            claimed_at TEXT, completed_at TEXT
        )
    """)

    for i in range(num_memories):
        mid = f'mem-{i:03d}'
        conn.execute(
            "INSERT INTO memories VALUES (?, ?, NULL, 'semantic', 0.5, 'default', "
            "'{}', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', NULL, 0)",
            (mid, f'Memory content {i}'),
        )
        conn.execute(
            'INSERT INTO memories_fts (content, summary) VALUES (?, NULL)',
            (f'Memory content {i}',),
        )
        embedding = [0.1 * (i + 1)] * 768
        vec_data = struct.pack('768f', *embedding)
        conn.execute(
            'INSERT INTO vec_memories (memory_id, embedding) VALUES (?, ?)',
            (mid, vec_data),
        )

    # Add an entity and link
    conn.execute(
        "INSERT INTO entities VALUES ('ent-001', 'TestEntity', 'concept', NULL, "
        "'{}', 0.5, '2026-01-01', '2026-01-01', 0)"
    )
    conn.execute("INSERT INTO entity_memories VALUES ('ent-001', 'mem-000')")

    conn.commit()
    conn.close()


def test_migrate_v2_to_v3(tmp_path):
    db_path = tmp_path / 'legacy.db'
    _create_v2_db(db_path, num_memories=3)

    count = migrate_v2_to_v3(str(db_path))
    assert count == 3


def test_migration_produces_valid_entries(tmp_path):
    db_path = tmp_path / 'legacy.db'
    _create_v2_db(db_path, num_memories=2)
    migrate_v2_to_v3(str(db_path))

    from aingram.storage.engine import StorageEngine

    engine = StorageEngine(str(db_path))
    assert engine.get_entry_count() == 2
    engine.close()


def test_migration_preserves_old_id_in_metadata(tmp_path):
    db_path = tmp_path / 'legacy.db'
    _create_v2_db(db_path, num_memories=1)
    migrate_v2_to_v3(str(db_path))

    from aingram.storage.engine import StorageEngine

    engine = StorageEngine(str(db_path))
    entries = engine.get_entries_by_session(
        engine._conn.execute('SELECT session_id FROM agent_sessions LIMIT 1').fetchone()[0]
    )
    assert len(entries) == 1
    assert entries[0].metadata['migrated_from'] == 'mem-000'
    engine.close()


def test_migration_maps_memory_types(tmp_path):
    db_path = tmp_path / 'legacy.db'
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute('CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)')
    conn.execute("INSERT INTO meta VALUES ('schema_version', '2')")
    conn.execute("INSERT INTO meta VALUES ('embedding_dim', '768')")
    conn.execute("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, content TEXT NOT NULL, summary TEXT,
            memory_type TEXT NOT NULL, importance REAL DEFAULT 0.5,
            agent_id TEXT DEFAULT 'default', metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            accessed_at TEXT, access_count INTEGER DEFAULT 0
        )
    """)
    conn.execute('CREATE VIRTUAL TABLE memories_fts USING fts5(content, summary)')
    conn.execute(
        'CREATE VIRTUAL TABLE vec_memories USING vec0('
        'memory_id TEXT PRIMARY KEY, embedding float[768])'
    )

    types = [
        ('procedural', 'method'),
        ('episodic', 'observation'),
        ('semantic', 'observation'),
        ('entity', 'observation'),
    ]
    for i, (old_type, _expected) in enumerate(types):
        mid = f'm{i}'
        conn.execute(
            "INSERT INTO memories VALUES (?, ?, NULL, ?, 0.5, 'default', "
            "'{}', '2026-01-01', '2026-01-01', NULL, 0)",
            (mid, f'content {i}', old_type),
        )
        vec_data = struct.pack('768f', *([0.1] * 768))
        conn.execute('INSERT INTO vec_memories VALUES (?, ?)', (mid, vec_data))
    conn.commit()
    conn.close()

    migrate_v2_to_v3(str(db_path))

    from aingram.storage.engine import StorageEngine

    engine = StorageEngine(str(db_path))
    entries = engine.get_entries_by_session(
        engine._conn.execute('SELECT session_id FROM agent_sessions LIMIT 1').fetchone()[0]
    )
    entry_types = [e.entry_type for e in entries]
    assert 'method' in entry_types  # procedural → method
    engine.close()


def test_migration_no_v2_tables_returns_zero(tmp_path):
    """Fresh DB (no v2 tables) → migration does nothing."""
    db_path = tmp_path / 'fresh.db'
    conn = sqlite3.connect(str(db_path))
    conn.close()
    assert migrate_v2_to_v3(str(db_path)) == 0


def test_migration_chain_is_verifiable(tmp_path):
    """Migrated entries form a valid chain."""
    db_path = tmp_path / 'legacy.db'
    _create_v2_db(db_path, num_memories=5)
    migrate_v2_to_v3(str(db_path))

    from aingram.store import MemoryStore
    from tests.conftest import MockEmbedder

    mem = MemoryStore(str(db_path), embedder=MockEmbedder(), agent_name='verifier')
    migration_sid = mem._engine._conn.execute(
        "SELECT session_id FROM agent_sessions WHERE agent_name = 'migration'"
    ).fetchone()[0]
    result = mem.verify(session_id=migration_sid)
    assert result.valid is True
    assert result.entries_checked == 5
    mem.close()


def test_migrate_v8_to_v9_drops_int8_creates_qjl(tmp_path):
    """v8→v9: drop vec_entries_int8, create vec_entries_qjl, backfill QJL bits."""
    from aingram.storage.schema import SCHEMA_VERSION, apply_schema

    db = tmp_path / 'test.db'
    conn = sqlite3.connect(str(db))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    apply_schema(conn, enable_vec=True, vec_embedding_dim=16)

    for i in range(3):
        vec = [float(i * 0.1 + j * 0.01) for j in range(16)]
        blob = struct.pack(f'{len(vec)}f', *vec)
        conn.execute(
            'INSERT INTO vec_entries (entry_id, embedding) VALUES (?, ?)',
            (f'e{i}', blob),
        )

    conn.execute(
        'CREATE TABLE IF NOT EXISTS vec_entries_int8 ('
        '    entry_id TEXT PRIMARY KEY,'
        '    quantized BLOB NOT NULL,'
        '    scale REAL NOT NULL,'
        '    min_val REAL NOT NULL'
        ')'
    )
    conn.execute(
        'INSERT OR REPLACE INTO vec_entries_int8 (entry_id, quantized, scale, min_val) '
        'VALUES (?, ?, ?, ?)',
        ('e0', b'\x00' * 16, 1.0, 0.0),
    )
    conn.execute(
        "INSERT OR REPLACE INTO db_metadata (key, value, updated_at) "
        "VALUES ('quantized_version', '1', '2026-01-01')"
    )
    conn.execute("UPDATE db_metadata SET value = '8' WHERE key = 'schema_version'")
    conn.commit()

    apply_schema(conn, enable_vec=True, vec_embedding_dim=16)

    version = conn.execute(
        "SELECT value FROM db_metadata WHERE key = 'schema_version'"
    ).fetchone()[0]
    assert version == str(SCHEMA_VERSION)

    tables = [
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    assert 'vec_entries_int8' not in tables

    qjl_count = conn.execute('SELECT COUNT(*) FROM vec_entries_qjl').fetchone()[0]
    assert qjl_count == 3

    seed_row = conn.execute(
        "SELECT value FROM db_metadata WHERE key = 'qjl_seed'"
    ).fetchone()
    assert seed_row is not None
    assert seed_row[0] == '42'

    qv = conn.execute(
        "SELECT value FROM db_metadata WHERE key = 'quantized_version'"
    ).fetchone()
    assert qv is None

    conn.close()
