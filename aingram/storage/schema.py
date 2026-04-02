# aingram/storage/schema.py — Lite (open-source) schema
from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# Lite v8: vec_entries_int8 for optional embedding quantization.
SCHEMA_VERSION = 8

AGENT_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id        TEXT PRIMARY KEY,
    agent_name        TEXT NOT NULL,
    public_key        TEXT NOT NULL,
    parent_session_id TEXT,
    created_at        TEXT NOT NULL,
    metadata          TEXT,
    FOREIGN KEY (parent_session_id) REFERENCES agent_sessions(session_id)
)
"""

MEMORY_ENTRIES_TABLE = """
CREATE TABLE IF NOT EXISTS memory_entries (
    entry_id           TEXT PRIMARY KEY,
    content_hash       TEXT NOT NULL,
    entry_type         TEXT NOT NULL CHECK (entry_type IN (
        'hypothesis','method','result','lesson','observation','decision','meta'
    )),
    content            TEXT NOT NULL,
    session_id         TEXT NOT NULL,
    sequence_num       INTEGER NOT NULL,
    prev_entry_id      TEXT,
    signature          TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    reasoning_chain_id TEXT,
    parent_entry_id    TEXT,
    tags               TEXT,
    metadata           TEXT,
    confidence         REAL,
    importance         REAL NOT NULL DEFAULT 0.5,
    accessed_at        TEXT,
    access_count       INTEGER NOT NULL DEFAULT 0,
    surprise           REAL,
    consolidated       INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES agent_sessions(session_id),
    FOREIGN KEY (reasoning_chain_id) REFERENCES reasoning_chains(chain_id),
    FOREIGN KEY (parent_entry_id) REFERENCES memory_entries(entry_id),
    UNIQUE(session_id, sequence_num)
)
"""

REASONING_CHAINS_TABLE = """
CREATE TABLE IF NOT EXISTS reasoning_chains (
    chain_id           TEXT PRIMARY KEY,
    title              TEXT NOT NULL,
    status             TEXT DEFAULT 'active' CHECK (status IN (
        'active','completed','abandoned','superseded'
    )),
    outcome            TEXT CHECK (outcome IN (
        'confirmed','refuted','partial','inconclusive','error'
    )),
    created_by_session TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    FOREIGN KEY (created_by_session) REFERENCES agent_sessions(session_id)
)
"""

CROSS_REFERENCES_TABLE = """
CREATE TABLE IF NOT EXISTS cross_references (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entry_id TEXT NOT NULL,
    target_entry_id TEXT NOT NULL,
    reference_type  TEXT NOT NULL CHECK (reference_type IN (
        'builds_on','contradicts','supports','refines','supersedes'
    )),
    session_id      TEXT NOT NULL,
    signature       TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (source_entry_id) REFERENCES memory_entries(entry_id),
    FOREIGN KEY (target_entry_id) REFERENCES memory_entries(entry_id),
    FOREIGN KEY (session_id) REFERENCES agent_sessions(session_id),
    UNIQUE(source_entry_id, target_entry_id, reference_type)
)
"""

ENTITIES_TABLE = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    mention_count INTEGER DEFAULT 1,
    UNIQUE(name, entity_type)
)
"""

RELATIONSHIPS_TABLE = """
CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES entities(entity_id),
    target_id TEXT NOT NULL REFERENCES entities(entity_id),
    relation_type TEXT NOT NULL,
    fact TEXT,
    weight REAL NOT NULL DEFAULT 1.0,
    t_valid TEXT,
    t_invalid TEXT,
    source_memory TEXT
)
"""

ENTITY_MENTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id    TEXT NOT NULL,
    entry_id     TEXT NOT NULL,
    confidence   REAL DEFAULT 1.0,
    PRIMARY KEY (entity_id, entry_id),
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id),
    FOREIGN KEY (entry_id) REFERENCES memory_entries(entry_id)
)
"""

KNOWLEDGE_ITEMS_TABLE = """
CREATE TABLE IF NOT EXISTS knowledge_items (
    knowledge_id       TEXT PRIMARY KEY,
    principle          TEXT NOT NULL,
    supporting_chains  TEXT NOT NULL,
    confidence         REAL NOT NULL,
    created_by_session TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    stability          REAL DEFAULT 3.17,
    difficulty         REAL DEFAULT 5.0,
    due_at             TEXT DEFAULT (datetime('now')),
    fsrs_state         INTEGER DEFAULT 0,
    last_review        TEXT,
    reps               INTEGER DEFAULT 0,
    lapses             INTEGER DEFAULT 0,
    FOREIGN KEY (created_by_session) REFERENCES agent_sessions(session_id)
)
"""

TASK_QUEUE_TABLE = """
CREATE TABLE IF NOT EXISTS task_queue (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    claimed_at TEXT,
    completed_at TEXT
)
"""

DB_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS db_metadata (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

VEC_ENTRIES_INT8_TABLE = """
CREATE TABLE IF NOT EXISTS vec_entries_int8 (
    entry_id TEXT PRIMARY KEY,
    quantized BLOB NOT NULL,
    scale REAL NOT NULL,
    min_val REAL NOT NULL
)
"""

AGENT_TOKENS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_tokens (
    agent_id    TEXT PRIMARY KEY,
    agent_name  TEXT NOT NULL UNIQUE,
    token_hash  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('reader', 'contributor', 'admin')),
    public_key  TEXT,
    created_at  TEXT NOT NULL,
    revoked_at  TEXT
)
"""

ENTRIES_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    content,
    entry_id UNINDEXED
)
"""

INDEXES = [
    'CREATE INDEX IF NOT EXISTS idx_entries_session ON memory_entries(session_id, sequence_num)',
    'CREATE INDEX IF NOT EXISTS idx_entries_type ON memory_entries(entry_type)',
    'CREATE INDEX IF NOT EXISTS idx_entries_chain ON memory_entries(reasoning_chain_id)',
    'CREATE INDEX IF NOT EXISTS idx_entries_created ON memory_entries(created_at)',
    'CREATE INDEX IF NOT EXISTS idx_entries_importance ON memory_entries(importance)',
    'CREATE INDEX IF NOT EXISTS idx_entries_content_hash ON memory_entries(content_hash)',
    'CREATE INDEX IF NOT EXISTS idx_relationships_source_id ON relationships(source_id)',
    'CREATE INDEX IF NOT EXISTS idx_relationships_target_id ON relationships(target_id)',
    'CREATE INDEX IF NOT EXISTS idx_task_queue_status_priority ON task_queue(status, priority)',
    'CREATE INDEX IF NOT EXISTS idx_entity_mentions_entry ON entity_mentions(entry_id)',
    'CREATE INDEX IF NOT EXISTS idx_entities_name_lower ON entities(LOWER(name))',
]


def _migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    """Add FSRS columns to knowledge_items."""
    alterations = [
        'ALTER TABLE knowledge_items ADD COLUMN stability REAL DEFAULT 3.17',
        'ALTER TABLE knowledge_items ADD COLUMN difficulty REAL DEFAULT 5.0',
        "ALTER TABLE knowledge_items ADD COLUMN due_at TEXT DEFAULT (datetime('now'))",
        'ALTER TABLE knowledge_items ADD COLUMN fsrs_state INTEGER DEFAULT 0',
        'ALTER TABLE knowledge_items ADD COLUMN last_review TEXT',
        'ALTER TABLE knowledge_items ADD COLUMN reps INTEGER DEFAULT 0',
        'ALTER TABLE knowledge_items ADD COLUMN lapses INTEGER DEFAULT 0',
    ]
    for sql in alterations:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass
    conn.execute('UPDATE knowledge_items SET due_at = created_at WHERE due_at IS NULL')


def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    """Add surprise column to memory_entries."""
    try:
        conn.execute('ALTER TABLE memory_entries ADD COLUMN surprise REAL')
    except sqlite3.OperationalError:
        pass


def _migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    """Add consolidated flag to memory_entries."""
    try:
        conn.execute(
            'ALTER TABLE memory_entries ADD COLUMN consolidated INTEGER NOT NULL DEFAULT 0'
        )
    except sqlite3.OperationalError:
        pass


def vec_entries_ddl(dim: int, *, if_not_exists: bool = True) -> str:
    clause = 'IF NOT EXISTS ' if if_not_exists else ''
    return (
        f'CREATE VIRTUAL TABLE {clause}vec_entries USING vec0(\n'
        f'    entry_id TEXT PRIMARY KEY,\n'
        f'    embedding float[{dim}]\n'
        f')'
    )


def get_schema_version(conn: sqlite3.Connection) -> int | None:
    """Read schema version from db_metadata (v3) or meta (v2 fallback)."""
    try:
        row = conn.execute("SELECT value FROM db_metadata WHERE key = 'schema_version'").fetchone()
        if row:
            return int(row[0])
    except sqlite3.OperationalError:
        pass
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return int(row[0]) if row else None
    except sqlite3.OperationalError:
        return None


def apply_schema(
    conn: sqlite3.Connection,
    *,
    enable_vec: bool = False,
    vec_embedding_dim: int = 768,
) -> None:
    from datetime import UTC, datetime

    version = get_schema_version(conn)
    if version == SCHEMA_VERSION:
        return

    if version is not None:
        logger.info('Migrating schema from v%s to v%s', version, SCHEMA_VERSION)
    else:
        logger.debug('Initializing schema v%s', SCHEMA_VERSION)

    conn.execute(DB_METADATA_TABLE)
    conn.execute(AGENT_SESSIONS_TABLE)
    conn.execute(REASONING_CHAINS_TABLE)
    conn.execute(MEMORY_ENTRIES_TABLE)
    conn.execute(CROSS_REFERENCES_TABLE)
    conn.execute(ENTITIES_TABLE)
    conn.execute(RELATIONSHIPS_TABLE)
    conn.execute(ENTITY_MENTIONS_TABLE)
    conn.execute(KNOWLEDGE_ITEMS_TABLE)
    conn.execute(TASK_QUEUE_TABLE)
    conn.execute(AGENT_TOKENS_TABLE)
    conn.execute(VEC_ENTRIES_INT8_TABLE)
    conn.execute(ENTRIES_FTS)

    if enable_vec:
        conn.execute(vec_entries_ddl(vec_embedding_dim))

    for idx in INDEXES:
        conn.execute(idx)

    if version is not None and version < 5:
        _migrate_v4_to_v5(conn)

    if version is not None and version < 6:
        _migrate_v5_to_v6(conn)

    if version is not None and version < 7:
        _migrate_v6_to_v7(conn)

    conn.execute('CREATE INDEX IF NOT EXISTS idx_ki_due ON knowledge_items(due_at, fsrs_state)')

    now = datetime.now(UTC).isoformat()
    conn.execute(
        'INSERT OR REPLACE INTO db_metadata (key, value, updated_at) '
        "VALUES ('schema_version', ?, ?)",
        (str(SCHEMA_VERSION), now),
    )
    row = conn.execute("SELECT value FROM db_metadata WHERE key='embedding_dim'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO db_metadata (key, value, updated_at) VALUES ('embedding_dim', ?, ?)",
            (str(vec_embedding_dim), now),
        )
    conn.commit()
