# tests/test_storage/test_schema.py — v3 schema tests
import sqlite3

import pytest
import sqlite_vec

from aingram.storage.schema import SCHEMA_VERSION, apply_schema, get_schema_version


def _make_conn(tmp_path, dim=768):
    db = tmp_path / 'test.db'
    conn = sqlite3.connect(str(db))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


class TestSchemaLite:
    def test_no_dag_or_sync_tables(self, tmp_path):
        import sqlite3

        import sqlite_vec

        from aingram.storage.schema import apply_schema

        conn = sqlite3.connect(str(tmp_path / 'lite.db'))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        apply_schema(conn, enable_vec=True)

        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert 'dag_parents' not in tables
        assert 'trusted_peers' not in tables
        assert 'causal_nodes' not in tables
        conn.close()

    def test_schema_version_is_7(self):
        assert SCHEMA_VERSION == 7


def test_apply_creates_core_tables(tmp_path):
    conn = _make_conn(tmp_path)
    apply_schema(conn, enable_vec=True)
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    for t in [
        'agent_sessions',
        'memory_entries',
        'reasoning_chains',
        'cross_references',
        'entities',
        'relationships',
        'entity_mentions',
        'knowledge_items',
        'task_queue',
        'db_metadata',
        'agent_tokens',
    ]:
        assert t in tables, f'Missing table: {t}'
    conn.close()


def test_apply_creates_virtual_tables(tmp_path):
    conn = _make_conn(tmp_path)
    apply_schema(conn, enable_vec=True)
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert 'vec_entries' in tables
    assert 'entries_fts' in tables
    conn.close()


def test_schema_version_stored(tmp_path):
    conn = _make_conn(tmp_path)
    apply_schema(conn, enable_vec=True)
    assert get_schema_version(conn) == SCHEMA_VERSION
    conn.close()


def test_apply_is_idempotent(tmp_path):
    conn = _make_conn(tmp_path)
    apply_schema(conn, enable_vec=True)
    apply_schema(conn, enable_vec=True)
    assert get_schema_version(conn) == SCHEMA_VERSION
    conn.close()


class TestSchemaV6:
    def test_memory_entries_has_surprise_column(self, tmp_path):
        conn = _make_conn(tmp_path)
        apply_schema(conn, enable_vec=True)
        cursor = conn.execute('PRAGMA table_info(memory_entries)')
        columns = {row[1] for row in cursor.fetchall()}
        assert 'surprise' in columns
        conn.close()


class TestSchemaV7:
    def test_memory_entries_has_consolidated_column(self, tmp_path):
        conn = _make_conn(tmp_path)
        apply_schema(conn, enable_vec=True)
        cursor = conn.execute('PRAGMA table_info(memory_entries)')
        columns = {row[1] for row in cursor.fetchall()}
        assert 'consolidated' in columns
        conn.close()


class TestSchemaV5:
    def test_knowledge_items_has_fsrs_columns(self, tmp_path):
        conn = _make_conn(tmp_path)
        apply_schema(conn, enable_vec=True)
        cursor = conn.execute('PRAGMA table_info(knowledge_items)')
        columns = {row[1] for row in cursor.fetchall()}
        assert 'stability' in columns
        assert 'difficulty' in columns
        assert 'due_at' in columns
        assert 'fsrs_state' in columns
        assert 'last_review' in columns
        assert 'reps' in columns
        assert 'lapses' in columns
        conn.close()

    def test_migration_existing_items_get_fsrs_defaults(self, tmp_path):
        """v4 knowledge_items table gains FSRS columns; new rows get defaults."""
        db_path = str(tmp_path / 'migrate.db')
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE db_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO db_metadata VALUES (
                'schema_version', '4', '2026-01-01T00:00:00+00:00'
            );
            CREATE TABLE agent_sessions (
                session_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                public_key TEXT NOT NULL,
                parent_session_id TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT
            );
            CREATE TABLE knowledge_items (
                knowledge_id TEXT PRIMARY KEY,
                principle TEXT NOT NULL,
                supporting_chains TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_by_session TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (created_by_session) REFERENCES agent_sessions(session_id)
            );
            """
        )
        conn.commit()
        apply_schema(conn, enable_vec=False)
        assert get_schema_version(conn) == SCHEMA_VERSION
        conn.execute(
            'INSERT INTO agent_sessions (session_id, agent_name, public_key, created_at) '
            "VALUES ('s1', 'test', 'pk', '2026-01-01T00:00:00+00:00')"
        )
        conn.execute(
            'INSERT INTO knowledge_items (knowledge_id, principle, supporting_chains, '
            'confidence, created_by_session, created_at) '
            "VALUES ('ki-old', 'Old principle', '[\"c1\"]', 0.8, 's1', '2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
        row = conn.execute(
            'SELECT fsrs_state, stability, difficulty FROM knowledge_items WHERE knowledge_id = ?',
            ('ki-old',),
        ).fetchone()
        assert row[0] == 0
        assert row[1] is not None
        conn.close()


def test_entry_type_check_constraint(tmp_path):
    conn = _make_conn(tmp_path)
    apply_schema(conn, enable_vec=True)
    conn.execute('PRAGMA foreign_keys=OFF')
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            'INSERT INTO memory_entries '
            '(entry_id, content_hash, entry_type, content, session_id, '
            'sequence_num, signature, created_at, importance) '
            "VALUES ('e1','ch','invalid_type','{}','s1',1,'sig','2026-01-01',0.5)"
        )
    conn.close()


def test_session_sequence_unique(tmp_path):
    conn = _make_conn(tmp_path)
    apply_schema(conn, enable_vec=True)
    conn.execute(
        'INSERT INTO agent_sessions (session_id, agent_name, public_key, created_at) '
        "VALUES ('s1', 'test', 'aaa', '2026-01-01')"
    )
    conn.execute(
        'INSERT INTO memory_entries '
        '(entry_id, content_hash, entry_type, content, session_id, '
        'sequence_num, signature, created_at, importance) '
        "VALUES ('e1','ch1','observation','{}','s1',1,'sig1','2026-01-01',0.5)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            'INSERT INTO memory_entries '
            '(entry_id, content_hash, entry_type, content, session_id, '
            'sequence_num, signature, created_at, importance) '
            "VALUES ('e2','ch2','observation','{}','s1',1,'sig2','2026-01-01',0.5)"
        )
    conn.close()


def test_chain_status_check_constraint(tmp_path):
    conn = _make_conn(tmp_path)
    apply_schema(conn, enable_vec=True)
    conn.execute('PRAGMA foreign_keys=OFF')
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            'INSERT INTO reasoning_chains '
            '(chain_id, title, status, created_by_session, created_at) '
            "VALUES ('c1','test','bogus','s1','2026-01-01')"
        )
    conn.close()


def test_creates_indexes(tmp_path):
    conn = _make_conn(tmp_path)
    apply_schema(conn, enable_vec=True)
    index_names = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
    }
    assert 'idx_entries_session' in index_names
    assert 'idx_entries_type' in index_names
    assert 'idx_entries_chain' in index_names
    assert 'idx_entries_importance' in index_names
    assert 'idx_relationships_source_id' in index_names
    assert 'idx_task_queue_status_priority' in index_names
    assert 'idx_ki_due' in index_names
    conn.close()


def test_get_schema_version_returns_none_for_empty_db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / 'empty.db'))
    assert get_schema_version(conn) is None
    conn.close()
