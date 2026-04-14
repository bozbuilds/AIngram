from __future__ import annotations

import sqlite3

CAPTURE_SCHEMA_VERSION = 1

CAPTURE_QUEUE_TABLE = """
CREATE TABLE IF NOT EXISTS capture_queue (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    source_tool        TEXT NOT NULL,
    session_id         TEXT NOT NULL,
    turn_number        INTEGER,
    user_prompt        TEXT NOT NULL,
    assistant_response TEXT,
    tool_calls         TEXT,
    model              TEXT,
    project_path       TEXT,
    timestamp          REAL NOT NULL,
    metadata           TEXT,
    container_tag      TEXT,
    state              TEXT DEFAULT 'pending'
                       CHECK(state IN ('pending','processing','done','error')),
    error_reason       TEXT,
    created_at         REAL DEFAULT (unixepoch())
)
"""

CAPTURE_QUEUE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_capture_state ON capture_queue(state)
"""

TOGGLE_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS toggle_state (
    tool_name TEXT PRIMARY KEY,
    state     TEXT NOT NULL DEFAULT 'on'
              CHECK(state IN ('on','off','auto'))
)
"""

CAPTURE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS capture_schema_version (
    version INTEGER NOT NULL
)
"""


def apply_capture_schema(conn: sqlite3.Connection) -> None:
    row = None
    try:
        row = conn.execute('SELECT version FROM capture_schema_version').fetchone()
    except sqlite3.OperationalError:
        pass

    if row is not None and row[0] >= CAPTURE_SCHEMA_VERSION:
        return

    conn.execute(CAPTURE_SCHEMA_VERSION_TABLE)
    conn.execute(CAPTURE_QUEUE_TABLE)
    conn.execute(CAPTURE_QUEUE_INDEX)
    conn.execute(TOGGLE_STATE_TABLE)

    if row is None:
        conn.execute(
            'INSERT INTO capture_schema_version (version) VALUES (?)',
            (CAPTURE_SCHEMA_VERSION,),
        )
    else:
        conn.execute(
            'UPDATE capture_schema_version SET version = ?',
            (CAPTURE_SCHEMA_VERSION,),
        )
    conn.commit()
