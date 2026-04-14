from __future__ import annotations

import sqlite3
import threading

from aingram.capture.schema import apply_capture_schema
from aingram.capture.types import CaptureRecord


class CaptureQueue:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        conn = self._get_conn()
        apply_capture_schema(conn)

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            conn = sqlite3.connect(self._db_path)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA busy_timeout=5000')
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            with self._conns_lock:
                self._all_conns.append(conn)
        return self._local.conn

    def insert(self, record: CaptureRecord) -> int:
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT INTO capture_queue
               (source_tool, session_id, turn_number, user_prompt,
                assistant_response, tool_calls, model, project_path,
                timestamp, metadata, container_tag, state)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
            (
                record.source_tool,
                record.session_id,
                record.turn_number,
                record.user_prompt,
                record.assistant_response,
                record.tool_calls,
                record.model,
                record.project_path,
                record.timestamp,
                record.metadata,
                record.container_tag,
            ),
        )
        conn.commit()
        return cur.lastrowid

    def dequeue_batch(self, limit: int) -> list[tuple[int, CaptureRecord]]:
        conn = self._get_conn()
        try:
            conn.execute('BEGIN IMMEDIATE')
            rows = conn.execute(
                """SELECT id, source_tool, session_id, turn_number, user_prompt,
                          assistant_response, tool_calls, model, project_path,
                          timestamp, metadata, container_tag
                   FROM capture_queue
                   WHERE state = 'pending'
                   ORDER BY id
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            if not rows:
                conn.execute('COMMIT')
                return []
            ids = [row['id'] for row in rows]
            placeholders = ','.join('?' for _ in ids)
            conn.execute(
                f"UPDATE capture_queue SET state = 'processing' WHERE id IN ({placeholders})",
                ids,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        result = []
        for row in rows:
            record = CaptureRecord(
                source_tool=row['source_tool'],
                session_id=row['session_id'],
                user_prompt=row['user_prompt'],
                timestamp=row['timestamp'],
                turn_number=row['turn_number'],
                assistant_response=row['assistant_response'],
                tool_calls=row['tool_calls'],
                model=row['model'],
                project_path=row['project_path'],
                metadata=row['metadata'],
                container_tag=row['container_tag'],
                state='processing',
            )
            result.append((row['id'], record))
        return result

    def mark_done(self, row_id: int) -> None:
        conn = self._get_conn()
        conn.execute("UPDATE capture_queue SET state = 'done' WHERE id = ?", (row_id,))
        conn.commit()

    def mark_error(self, row_id: int, reason: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE capture_queue SET state = 'error', error_reason = ? WHERE id = ?",
            (reason, row_id),
        )
        conn.commit()

    def pending_count(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM capture_queue WHERE state = 'pending'").fetchone()
        return row[0]

    def init_toggles(self, tool_states: dict[str, bool]) -> None:
        conn = self._get_conn()
        for tool_name, enabled in tool_states.items():
            state = 'on' if enabled else 'off'
            conn.execute(
                """INSERT INTO toggle_state (tool_name, state) VALUES (?, ?)
                   ON CONFLICT(tool_name) DO NOTHING""",
                (tool_name, state),
            )
        conn.commit()

    def get_toggle(self, tool_name: str) -> str:
        conn = self._get_conn()
        row = conn.execute(
            'SELECT state FROM toggle_state WHERE tool_name = ?', (tool_name,)
        ).fetchone()
        return row['state'] if row else 'off'

    def set_toggle(self, tool_name: str, state: str) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO toggle_state (tool_name, state) VALUES (?, ?)
               ON CONFLICT(tool_name) DO UPDATE SET state = excluded.state""",
            (tool_name, state),
        )
        conn.commit()

    def last_capture_times(self) -> dict[str, float]:
        conn = self._get_conn()
        rows = conn.execute(
            'SELECT source_tool, MAX(timestamp) AS last_ts FROM capture_queue GROUP BY source_tool'
        ).fetchall()
        return {row['source_tool']: row['last_ts'] for row in rows if row['last_ts'] is not None}

    def close(self) -> None:
        with self._conns_lock:
            for conn in self._all_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_conns.clear()
        if hasattr(self._local, 'conn'):
            del self._local.conn
